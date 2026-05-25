//
//  PartitionTable.swift
//  database-server
//
//  Created by Ritesh Pakala on 11/15/25.
//

import Foundation
import Vapor

typealias PartitionSearchResult = (scores: [Float], partitions: [Database.Partition])

/// A table stores the documents with their per-document PQ indices and a global HNSW index.
///
/// Multi-shard design: instead of one monolithic HNSW graph, the global corpus is split across
/// multiple `HNSWShard` instances. New shards are spawned by `TableMutator` when the active shard
/// hits `DatabaseConfig.shardSizeThreshold` nodes. Queries fan out across all trained shards; results
/// are merged by distance before Sinatra adjustment.
struct PartitionTable: Codable {
    /// All indexed document IDs. Set for O(1) insert and remove.
    var keys: Set<DocumentID> = []
    /// All global HNSW shards. Starts as a single shard; `TableMutator` appends new shards when
    /// the active shard reaches `DatabaseConfig.shardSizeThreshold` nodes.
    var shards: [HNSWShard] = [.init()]
    /// Maps each document ID to the index in `shards` that holds its partitions.
    /// Populated on put; used for O(1) shard routing on remove.
    var documentShardIndex: [DocumentID: Int] = [:]

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case keys
        case shards
        case documentShardIndex
        // `indices` is intentionally absent from the topology file.
        // It lives in `shard-<nodeId>-indices` and is loaded/saved separately.
        case shard  // Legacy: single-shard encoding from before multi-shard
    }

    init(from decoder: Decoder) throws {
        let c              = try decoder.container(keyedBy: CodingKeys.self)
        keys               = try c.decodeIfPresent(Set<DocumentID>.self, forKey: .keys) ?? []
        documentShardIndex = try c.decodeIfPresent([DocumentID: Int].self, forKey: .documentShardIndex) ?? [:]
        if let legacyShard = try c.decodeIfPresent(HNSWShard.self, forKey: .shard) {
            // Migrate: lift legacy single shard into shards[0]
            shards = [legacyShard]
            if documentShardIndex.isEmpty {
                for key in keys { documentShardIndex[key] = 0 }
            }
        } else {
            shards = try c.decodeIfPresent([HNSWShard].self, forKey: .shards) ?? [.init()]
        }
    }

    /// Custom encoder that intentionally omits `indices`.
    /// PQ codebooks live in the companion `shard-<nodeId>-indices` file.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(keys,               forKey: .keys)
        try c.encode(shards,             forKey: .shards)
        try c.encode(documentShardIndex, forKey: .documentShardIndex)
    }

    init() {}

    // MARK: - Computed index access

    /// Aggregated view of all per-shard PQ indices. Use `index(for:)` for O(1) point lookups.
    var indices: [DocumentID: PartitionIndex] {
        var result = [DocumentID: PartitionIndex]()
        result.reserveCapacity(keys.count)
        for shard in shards { result.merge(shard.indices, uniquingKeysWith: { a, _ in a }) }
        return result
    }

    /// O(1) lookup: routes to the correct shard via `documentShardIndex`.
    func index(for docId: DocumentID) -> PartitionIndex? {
        guard let si = documentShardIndex[docId], si < shards.count else { return nil }
        return shards[si].indices[docId]
    }

    // MARK: - Active shard convenience

    /// Index of the currently active (write-target) shard.
    var activeShardIndex: Int { shards.count - 1 }

    /// The shard new partitions are inserted into.
    var activeShard: HNSWShard {
        get { shards[activeShardIndex] }
        set { shards[activeShardIndex] = newValue }
    }

    // MARK: - Mutations

    /// Puts a new document and its partitions into the table.
    /// Shard selection and spawning are handled by `TableMutator` before this call.
    /// `targetShard` explicitly routes the insertion; defaults to `activeShardIndex`
    /// (the newest shard) when nil.
    mutating func put(id: DocumentID,
                      partitions: [Database.Partition],
                      tags: [String] = [],
                      tagsEmbedding: [Float]? = nil,
                      metadata: Data? = nil,
                      request: DatabaseRequest,
                      logger: TotemLogger,
                      targetShard: Int? = nil) {
        let si = targetShard ?? activeShardIndex
        // Cross-shard upsert: if this document was previously routed to a different (now-sealed)
        // shard, mark its old nodes deleted there. Must be read before documentShardIndex[id]
        // is overwritten — otherwise the routing entry is lost.
        if let prevSI = documentShardIndex[id], prevSI != si, prevSI < shards.count {
            shards[prevSI].indices.removeValue(forKey: id)
            shards[prevSI].remove(documentId: id)
        }
        documentShardIndex[id] = si

        // Insert raw (hint) embeddings into the active HNSW shard BEFORE per-doc training clears them.
        // Track which partitions were actually inserted so PQ training matches HNSW exactly.
        let prevInsertions = shards[si].totalInsertions
        let prevMaxLevel   = shards[si].maxLevel
        var insertedPartitions: [Database.Partition] = []
        insertedPartitions.reserveCapacity(partitions.count)
        let insertStart = Date()
        for partition in partitions {
            let before = shards[si].totalInsertions
            shards[si].add(partition: partition)
            if shards[si].totalInsertions > before {
                insertedPartitions.append(partition)
            }
        }
        shards[si].trainIfReady()
        let insertMs = Date().timeIntervalSince(insertStart) * 1000

        // ── HNSW build logging ─────────────────────────────────────────────────────────
        let added = shards[si].totalInsertions - prevInsertions
        if added > 0 {
            if prevInsertions == 0 {
                logger.info(
                    "HNSW",
                    "Shard \(si) initialized — \(added) partition(s) inserted, maxLevel=\(shards[si].maxLevel) ms=\(String(format: "%.1f", insertMs))",
                    service: .database,
                    request: request,
                    flow: .embed(documentId: id)
                )
            } else if shards[si].maxLevel > prevMaxLevel {
                logger.info(
                    "HNSW",
                    "Shard \(si) new layer — maxLevel=\(shards[si].maxLevel) nodes=\(shards[si].nodes.count) threshold=\(String(format: "%.4f", shards[si].effectiveThreshold)) ms=\(String(format: "%.1f", insertMs))",
                    service: .database,
                    request: request,
                    flow: .embed(documentId: id)
                )
            } else if shards[si].totalInsertions % 500 == 0 {
                let stats = shards[si].graphStats
                logger.info(
                    "HNSW",
                    "Shard \(si) health — live=\(stats.liveNodes) deleted=\(stats.deletedNodes) maxLevel=\(stats.maxLevel) avgDegree=\(String(format: "%.1f", stats.avgLayer0Degree)) threshold=\(String(format: "%.4f", shards[si].effectiveThreshold)) ms=\(String(format: "%.1f", insertMs))",
                    service: .database,
                    request: request,
                    flow: .embed(documentId: id)
                )
            } else {
                logger.info(
                    "HNSW",
                    "Shard \(si) inserted \(added) partition(s) — nodes=\(shards[si].nodes.count) maxLevel=\(shards[si].maxLevel) threshold=\(String(format: "%.4f", shards[si].effectiveThreshold)) ms=\(String(format: "%.1f", insertMs))",
                    service: .database,
                    request: request,
                    flow: .embed(documentId: id)
                )
            }
        }

        guard !insertedPartitions.isEmpty else {
            logger.warning(
                "PartitionTable.put: no partitions inserted for \(id) — all embeddings may have failed; skipping index creation",
                service: .database,
                request: request,
                flow: .embed(documentId: id)
            )
            return
        }
        var index = PartitionIndex()
        index.train(insertedPartitions, tags: tags, tagsEmbedding: tagsEmbedding,
                    documentId: id, logger: logger)
        index.metadata = metadata
        shards[si].indices[id] = index
        keys.insert(id)

    }

    /// Updates a document and its partitions within the table.
    mutating func update(id: DocumentID,
                         partitions: [Database.Partition],
                         tags: [String] = [],
                         tagsEmbedding: [Float]? = nil,
                         request: DatabaseRequest,
                         logger: TotemLogger) {
        put(id: id, partitions: partitions, tags: tags, tagsEmbedding: tagsEmbedding,
            request: request, logger: logger)
    }

    /// Removes an index for a DocumentID. O(1) when `documentShardIndex` is populated.
    mutating func remove(id: DocumentID) {
        keys.remove(id)
        if let si = documentShardIndex.removeValue(forKey: id) {
            shards[si].indices.removeValue(forKey: id)
            shards[si].remove(documentId: id)
        } else {
            // Legacy path: document predates documentShardIndex — scan all shards.
            for i in shards.indices {
                shards[i].indices.removeValue(forKey: id)
                shards[i].remove(documentId: id)
            }
        }
    }

    // MARK: - Search

    /// A search request over the corpus.
    ///
    /// **Global scope (HNSW path, once any shard has at least one partition):**
    /// Fans out to all trained shards, merges results by distance, applies Sinatra.
    ///
    /// **Personal / group scope (owner-filtered global HNSW path, when any shard is trained):**
    /// Fans out to all trained global shards with a combined filter of
    /// `ownerDocIds ∩ groupDocIds ∩ tagFilter`, merges results, applies Sinatra.
    ///
    /// **Linear fallback (no shard trained):**
    /// Iterates the relevant document set via per-document PQ scan. O(D × P).
    mutating func search(embedding: [Float],
                queryTagEmbedding: [Float]? = nil,
                k: Int = 3,
                sinatra: Sinatra,
                registry: TotemRegistry,
                request: DatabaseRequest,
                metadataLoader: PartitionDataLoader? = nil,
                logger: TotemLogger) -> (partitions: [PartitionSearchResult], adjustments: [SinatraAdjustment], shardStats: [SearchShardStat]) {
        var aggregated: [PartitionSearchResult] = []
        var adjustments: [SinatraAdjustment] = []
        var shardStats: [SearchShardStat] = []
        let sinatraRegistry = sinatra.registry
        let startTime = Date()

        let ownerKey = TotemRegistry.Owner(id: request.ownerId)
        let anyShardTrained = shards.contains { $0.isTrained }

        switch request.scope {
        case .global where anyShardTrained:
            // ── Global HNSW path (fan-out across all trained shards) ──────────
            let ownerDocs = Set(registry.ownersDocuments[ownerKey] ?? [])

            // Build group predicate — Loom-filtered requests carry groups even on global scope.
            let globalGroupIds: Set<GroupID>
            if let gs = request.groups, !gs.isEmpty {
                globalGroupIds = Set(gs.map(\.id))

                logger.info(
                    "HNSW Search",
                    "Predicate — groups=\(gs.count)",
                    service: .database,
                    request: request,
                    flow: .chat
                )
            } else if let g = request.group, request.aggregate != true {
                globalGroupIds = [g.id]
            } else {
                globalGroupIds = []
            }
            let groupDocIds: Set<String>? = globalGroupIds.isEmpty ? nil :
                Set(globalGroupIds.flatMap { registry.groups[$0] ?? [] })

            // Access predicate: O(1) per node at HNSW traversal time. Captures the
            // two existing sets by reference — no union is ever materialised.
            // Owner docs and publicly available docs are the only accessible classes;
            // third-party private docs are excluded here so HNSW never wastes candidate
            // budget on them.
            let accessFilter: (DocumentID) -> Bool = { docId in
                ownerDocs.contains(docId) || registry.availableDocumentIds.contains(docId)
            }

            // Fan out to every trained shard; each returns k*10 candidates.
            // Merge by deduplicated partitionId using threshold-relative distance (norm):
            //   norm = raw / shard.effectiveThreshold
            // This makes distances comparable across shards with different density profiles —
            // a "rare find" in a sparse shard (large threshold) isn't buried by mediocre
            // results from a dense shard just because its raw distance is larger.
            // Raw distance is preserved separately so Sinatra receives calibrated values.
            var mergedByPartitionId: [String: (partition: Database.Partition, raw: Float, norm: Float)] = [:]
            for i in shards.indices {
                guard shards[i].isTrained else { continue }
                let shardThreshold = shards[i].effectiveThreshold
                let (shardResults, trace) = shards[i].search(
                    shardIndex: i,
                    queryEmbedding: embedding,
                    queryTagEmbedding: queryTagEmbedding,
                    k: k * 10,
                    groupFilter: groupDocIds,
                    accessFilter: accessFilter,
                    sinatra: sinatra,
                    ownerKey: ownerKey,
                    sinatraRegistry: sinatraRegistry,
                    registry: registry,
                    metadataLoader: metadataLoader,
                    request: request,
                    logger: logger
                )
                shardStats.append(SearchShardStat(
                    shardIndex:  i,
                    nodes:       trace.graphNodes,
                    maxLevel:    trace.graphMaxLevel,
                    efUsed:      trace.efUsed,
                    explored:    trace.layer0Explored,
                    candidates:  trace.candidatesBeforeFilter,
                    threshold:   shardThreshold.isFinite ? shardThreshold : -1
                ))
                for r in shardResults {
                    // Falls back to raw distance for uncalibrated shards (threshold == .infinity).
                    let norm = shardThreshold.isFinite && shardThreshold > 0
                        ? r.distance / shardThreshold : r.distance
                    if let existing = mergedByPartitionId[r.partition.id] {
                        if norm < existing.norm {
                            mergedByPartitionId[r.partition.id] = (r.partition, r.distance, norm)
                        }
                    } else {
                        mergedByPartitionId[r.partition.id] = (r.partition, r.distance, norm)
                    }
                }
            }

            // ── Phase 1: Distance-ranked merge ───────────────────────────────
            // All results are access-controlled at HNSW time via accessFilter.
            // Owner-only docs receive a sort discount so closely-matching private
            // content surfaces above equally-distant public content — no hard cap.
            // The stored .norm is left unmodified (Sinatra sees calibrated raw distances);
            // only the sort key is adjusted.
            let ownerBoostFactor: Float = 0.9
            var merged = mergedByPartitionId.values.sorted { lhs, rhs in
                let lhsOwnerOnly = ownerDocs.contains(lhs.partition.documentId)
                    && !registry.availableDocumentIds.contains(lhs.partition.documentId)
                let rhsOwnerOnly = ownerDocs.contains(rhs.partition.documentId)
                    && !registry.availableDocumentIds.contains(rhs.partition.documentId)
                let lNorm = lhsOwnerOnly ? lhs.norm * ownerBoostFactor : lhs.norm
                let rNorm = rhsOwnerOnly ? rhs.norm * ownerBoostFactor : rhs.norm
                return lNorm < rNorm
            }

            // ── Phase 2: Source diversity floor ──────────────────────────────
            let representedGroups = Set(
                merged.flatMap { registry.documentGroups[$0.partition.documentId] ?? [] }
            )
            let allAvailableGroups = Set(
                registry.availableDocumentIds.flatMap { registry.documentGroups[$0] ?? [] }
            )
            let availableGroups = globalGroupIds.isEmpty
                ? allAvailableGroups
                : allAvailableGroups.intersection(globalGroupIds)
            let unrepresentedGroups = availableGroups.subtracting(representedGroups)
            if !unrepresentedGroups.isEmpty {
                // Only inject available (public) docs for unrepresented groups.
                let availableInResults = merged.filter {
                    registry.availableDocumentIds.contains($0.partition.documentId)
                }
                var injected = 0
                for groupId in unrepresentedGroups where injected < k {
                    if let best = availableInResults.first(where: {
                        registry.documentGroups[$0.partition.documentId]?.contains(groupId) == true
                    }) {
                        merged.append(best)
                        injected += 1
                    } else {
                        let groupDocIds = (registry.groups[groupId] ?? []).filter {
                            registry.availableDocumentIds.contains($0)
                        }
                        let candidate = groupDocIds.compactMap { docId
                            -> (partition: Database.Partition, raw: Float, norm: Float)? in
                            guard let idx = index(for: docId) else { return nil }
                            return idx.searchWithScores(queryEmbedding: embedding, k: 1,
                                                        metadataLoader: metadataLoader).first
                                .map { (partition: $0.0, raw: $0.1, norm: $0.1) }
                        }.min { $0.norm < $1.norm }
                        if let candidate {
                            merged.append(candidate)
                            injected += 1
                        }
                    }
                }
                merged.sort { $0.norm < $1.norm }
            }

            // Feed Sinatra up to k*3 candidates — larger pool lets engagement history
            // rescue rare finds that raw distance alone would have buried.
            let sinatraPool = Array(merged.prefix(min(k * 3, merged.count)))
            guard !sinatraPool.isEmpty else { break }

            // Sinatra adjustment: raw distance is passed so the GBT model operates on
            // calibrated PQ values (not the shard-normalized sort key).
            let adjustedGlobal: [(Database.Partition, Float)] = sinatraPool.map { result in
                let inference = SinatraInference(
                    partitionId: result.partition.id,
                    documentId: result.partition.documentId,
                    distance: result.raw
                )
                let prediction = sinatra.infer(inference, registry: sinatraRegistry, documentStats: registry.documentStats, request: request)
                return (result.partition, prediction.adjustedDistance)
            }.sorted { $0.1 < $1.1 }

            // Max threshold across all trained shards: results from any shard get a fair
            // filter relative to the widest calibrated distance scale in the corpus.
            let maxThresholdGlobal = shards
                .filter { $0.isTrained }
                .compactMap { $0.effectiveThreshold.isFinite ? $0.effectiveThreshold : nil }
                .max() ?? Float.infinity
            let filteredGlobal = adjustedGlobal.filter { $0.1 < maxThresholdGlobal }
            let finalGlobal    = filteredGlobal.isEmpty ? adjustedGlobal : filteredGlobal

            let originalGlobal = sinatraPool.map    { String(format: "%.4f", $0.raw) }
            let inferredGlobal = adjustedGlobal.map { String(format: "%.4f", $0.1) }
            let adjByIdGlobal: [String: Float] = Dictionary(adjustedGlobal.map { ($0.0.id, $0.1) }, uniquingKeysWith: min)
            let entriesGlobal: [SinatraAdjustment.Entry] = sinatraPool.map { orig in
                SinatraAdjustment.Entry(
                    partitionId:      orig.partition.id,
                    originalDistance: orig.raw,
                    adjustedDistance: adjByIdGlobal[orig.partition.id] ?? orig.raw,
                    threshold:        maxThresholdGlobal
                )
            }
            adjustments.append(SinatraAdjustment(
                partitionCount:      sinatraPool.count,
                original:            originalGlobal,
                inferred:            inferredGlobal,
                pqDistanceThreshold: maxThresholdGlobal,
                entries:             entriesGlobal
            ))
            aggregated.append((
                scores:     finalGlobal.map { $0.1 },
                partitions: finalGlobal.map { $0.0 }
            ))

        case _ where anyShardTrained:
            // ── Owner-filtered global HNSW path ───────────────────────────────
            // Uses the global shards with combinedFilter = ownerDocIds ∩ groupDocIds ∩ tagFilter.
            // Replaces the old per-owner personal HNSW: same search quality, half the disk usage.
            let ownerDocIds = Set(registry.ownersDocuments[ownerKey] ?? [])

            let ownerGroupIds: Set<GroupID>
            if let gs = request.groups, !gs.isEmpty {
                ownerGroupIds = Set(gs.map(\.id))
                logger.info(
                    "HNSW Search",
                    "Predicate — groups=\(gs.count)",
                    service: .database,
                    request: request,
                    flow: .chat
                )
            } else if let g = request.group, request.aggregate != true {
                ownerGroupIds = [g.id]
            } else {
                ownerGroupIds = []
            }
            let ownerGroupDocIds: Set<String>? = ownerGroupIds.isEmpty ? nil :
                Set(ownerGroupIds.flatMap { registry.groups[$0] ?? [] })

            // Tag pre-filter scoped to the owner's documents only.
            let ownerTaggedIndices = indices.filter { $0.value.tagsEmbedding != nil && ownerDocIds.contains($0.key) }
            let ownerTagFilter: Set<DocumentID>? = {
                guard !ownerTaggedIndices.isEmpty, let tagQueryEmbedding = queryTagEmbedding else { return nil }
                var passing: [DocumentID] = []
                for (docId, idx) in ownerTaggedIndices {
                    guard let dist = idx.tagDistance(queryEmbedding: tagQueryEmbedding) else {
                        passing.append(docId); continue
                    }
                    let threshold = sinatra.inferTagThreshold(
                        documentId: docId,
                        owner: ownerKey,
                        registry: sinatraRegistry,
                        documentStats: registry.documentStats
                    )
                    if dist < threshold { passing.append(docId) }
                }
                let untagged = indices.filter { $0.value.tagsEmbedding == nil && ownerDocIds.contains($0.key) }.map { $0.key }
                return Set(passing + untagged)
            }()

            // combinedFilter = ownerDocIds ∩ groupDocIds ∩ tagFilter
            let baseOwnerFilter: Set<String>? = switch (ownerGroupDocIds, ownerTagFilter) {
                case (.some(let a), .some(let b)): a.intersection(b)
                case (.some(let a), nil):           a
                case (nil, .some(let b)):           b
                case (nil, nil):                    nil
            }
            let combinedFilter: Set<String> = baseOwnerFilter.map { ownerDocIds.intersection($0) } ?? ownerDocIds

            // Fan out to global shards with the combined owner+group+tag filter.
            var ownerMergedByPartitionId: [String: (partition: Database.Partition, raw: Float, norm: Float)] = [:]
            for i in shards.indices {
                guard shards[i].isTrained else { continue }
                let shardThreshold = shards[i].effectiveThreshold
                let (shardResults, trace) = shards[i].search(
                    shardIndex: i,
                    queryEmbedding: embedding,
                    queryTagEmbedding: queryTagEmbedding,
                    k: k * 10,
                    groupFilter: combinedFilter,
                    sinatra: sinatra,
                    ownerKey: ownerKey,
                    sinatraRegistry: sinatraRegistry,
                    registry: registry,
                    metadataLoader: metadataLoader,
                    request: request,
                    logger: logger
                )
                shardStats.append(SearchShardStat(
                    shardIndex:  i,
                    nodes:       trace.graphNodes,
                    maxLevel:    trace.graphMaxLevel,
                    efUsed:      trace.efUsed,
                    explored:    trace.layer0Explored,
                    candidates:  trace.candidatesBeforeFilter,
                    threshold:   shardThreshold.isFinite ? shardThreshold : -1
                ))
                for r in shardResults {
                    let norm = shardThreshold.isFinite && shardThreshold > 0
                        ? r.distance / shardThreshold : r.distance
                    if let existing = ownerMergedByPartitionId[r.partition.id] {
                        if norm < existing.norm {
                            ownerMergedByPartitionId[r.partition.id] = (r.partition, r.distance, norm)
                        }
                    } else {
                        ownerMergedByPartitionId[r.partition.id] = (r.partition, r.distance, norm)
                    }
                }
            }

            let ownerRawSorted = ownerMergedByPartitionId.values.sorted { $0.norm < $1.norm }
            let filterLabel = combinedFilter.count < ownerDocIds.count ? "[Owner+filter]" : "[Owner]"
            logger.info(
                "HNSW Search",
                "\(filterLabel) shards=\(shards.filter { $0.isTrained }.count) candidates=\(ownerRawSorted.count) → topK=\(min(ownerRawSorted.count, k))",
                service: .database,
                request: request,
                flow: .chat
            )

            let sinatraPoolOwner = Array(ownerRawSorted.prefix(min(k * 3, ownerRawSorted.count)))
            guard !sinatraPoolOwner.isEmpty else { break }

            let adjustedOwner: [(Database.Partition, Float)] = sinatraPoolOwner.map { result in
                let inference = SinatraInference(
                    partitionId: result.partition.id,
                    documentId: result.partition.documentId,
                    distance: result.raw
                )
                let prediction = sinatra.infer(inference, registry: sinatraRegistry, documentStats: registry.documentStats, request: request)
                return (result.partition, prediction.adjustedDistance)
            }.sorted { $0.1 < $1.1 }

            let maxThresholdOwner = shards
                .filter { $0.isTrained }
                .compactMap { $0.effectiveThreshold.isFinite ? $0.effectiveThreshold : nil }
                .max() ?? Float.infinity
            let filteredOwner = adjustedOwner.filter { $0.1 < maxThresholdOwner }
            let finalOwner    = filteredOwner.isEmpty ? adjustedOwner : filteredOwner

            let originalOwner = sinatraPoolOwner.map { String(format: "%.4f", $0.raw) }
            let inferredOwner = adjustedOwner.map    { String(format: "%.4f", $0.1) }
            let adjByIdOwner: [String: Float] = Dictionary(adjustedOwner.map { ($0.0.id, $0.1) }, uniquingKeysWith: min)
            let entriesOwner: [SinatraAdjustment.Entry] = sinatraPoolOwner.map { orig in
                SinatraAdjustment.Entry(
                    partitionId:      orig.partition.id,
                    originalDistance: orig.raw,
                    adjustedDistance: adjByIdOwner[orig.partition.id] ?? orig.raw,
                    threshold:        maxThresholdOwner
                )
            }
            adjustments.append(SinatraAdjustment(
                partitionCount:      sinatraPoolOwner.count,
                original:            originalOwner,
                inferred:            inferredOwner,
                pqDistanceThreshold: maxThresholdOwner,
                entries:             entriesOwner
            ))
            aggregated.append((
                scores:     finalOwner.map { $0.1 },
                partitions: finalOwner.map { $0.0 }
            ))

        default:
            // ── Per-document linear scan fallback ─────────────────────────────
            var candidateIds: Set<DocumentID>

            switch request.scope {
            case .global:
                candidateIds = registry.availableDocumentIds.union(
                    Set(registry.ownersDocuments[ownerKey] ?? [])
                )
            default:
                if request.aggregate == true {
                    candidateIds = Set(registry.ownersDocuments[ownerKey] ?? [])
                } else if let gs = request.groups, !gs.isEmpty {
                    candidateIds = Set(gs.flatMap { registry.groups[$0.id] ?? [] })
                } else if let groupId = request.group?.id {
                    candidateIds = Set(registry.groups[groupId] ?? [])
                } else {
                    candidateIds = Set(registry.ownersDocuments[ownerKey] ?? [])
                }
            }

            // Tag pre-filter for linear scan path. Skip when no queryTagEmbedding provided.
            if let linearTagQueryEmbedding = queryTagEmbedding {
                let linearTaggedIndices = indices.filter { $0.value.tagsEmbedding != nil }
                if !linearTaggedIndices.isEmpty {
                    var passing = Set<DocumentID>()
                    for (docId, idx) in linearTaggedIndices {
                        guard let dist = idx.tagDistance(queryEmbedding: linearTagQueryEmbedding) else {
                            passing.insert(docId); continue
                        }
                        let threshold = sinatra.inferTagThreshold(
                            documentId: docId, owner: ownerKey,
                            registry: sinatraRegistry, documentStats: registry.documentStats
                        )
                        if dist < threshold { passing.insert(docId) }
                    }
                    let untagged = Set(indices.filter { $0.value.tagsEmbedding == nil }.map { $0.key })
                    candidateIds = candidateIds.intersection(passing.union(untagged))
                }
            }

            let candidateArray = Array(candidateIds)
            var rawLinearResults = [(PartitionSearchResult, SinatraAdjustment?)?](
                repeating: nil, count: candidateArray.count
            )
            let tableSnapshot = self
            let sendableLoader = SendableValue(metadataLoader)
            rawLinearResults.withUnsafeMutableBufferPointer { buffer in
                let sendableBuffer = SendableValue(buffer)
                DispatchQueue.concurrentPerform(iterations: candidateArray.count) { i in
                    let id = candidateArray[i]
                    guard let index = tableSnapshot.index(for: id) else { return }
                    sendableBuffer.value[i] = index.search(
                        queryEmbedding: embedding,
                        k: k,
                        sinatra: sinatra,
                        sinatraRegistry: sinatraRegistry,
                        documentStats: registry.documentStats,
                        request: request,
                        metadataLoader: sendableLoader.value,
                        logger: logger
                    )
                }
            }
            for case let (result, adjustment)? in rawLinearResults {
                aggregated.append(result)
                if let adjustment { adjustments.append(adjustment) }
            }
        }

        let elapsedTime = Date().timeIntervalSince(startTime) * 1000
        let totalPartitions = aggregated.reduce(0) { $0 + $1.partitions.count }
        let hnswLabel: String
        if anyShardTrained && request.scope != .global {
            hnswLabel = " [Owner HNSW \(shards.filter { $0.isTrained }.count) shard(s)]"
        } else if anyShardTrained {
            hnswLabel = " [Global HNSW \(shards.filter { $0.isTrained }.count) shard(s)]"
        } else {
            hnswLabel = ""
        }
        logger.info(
            "Table Search",
            "Search completed in \(String(format: "%.1f", elapsedTime))ms\(hnswLabel) — \(totalPartitions) partitions retrieved",
            service: .database,
            request: request,
            flow: .chat
        )

        if !adjustments.isEmpty {
            let totalAdjusted = adjustments.reduce(0) { $0 + $1.partitionCount }
            let details = adjustments.map { "\($0.original) → \($0.inferred)" }.joined(separator: " | ")
            logger.info(
                "Infer",
                "⚜️ Adjusted \(totalAdjusted) partitions across \(adjustments.count) indices — distances: \(details)",
                service: .sinatra,
                request: request,
                externalOnly: true,
                flow: .chat
            )
        }

        return (aggregated, adjustments, shardStats)
    }
}
