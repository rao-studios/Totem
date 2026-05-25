//
//  HNSWShard.swift
//  database-server
//
//  Two-tier global partition search encapsulation.
//
//  Tier 1 — Tag pre-filter: uses `queryTagEmbedding` to restrict candidate documents
//  based on semantic tag similarity. Sinatra's adaptive `inferTagThreshold()` widens
//  the gate for high-engagement documents. Results are parked via `parkIndices()`.
//
//  Tier 2 — HNSW slot search: uses `queryEmbedding` to rank partition nodes within
//  the tag-filtered + access-controlled candidate set, then resolves each
//  `SearchResult { partitionId, documentId, distance }` into a `Database.Partition`
//  via the per-document `PartitionIndex`.
//

import Foundation

struct HNSWShard: Codable {
    var graph: HNSWGraph = .init()
    /// Per-document PQ index for this shard. Excluded from the topology plist — loaded
    /// separately from `shard-<nodeId>-<i>-indices` and merged in at startup.
    var indices: [DocumentID: PartitionIndex] = [:]

    // MARK: - Codable (transparent — same on-disk format as HNSWGraph; zero migration)
    // `indices` is intentionally excluded: topology and PQ data persist in separate files.

    init() {}

    init(from decoder: Decoder) throws {
        graph = try HNSWGraph(from: decoder)
    }

    func encode(to encoder: Encoder) throws {
        try graph.encode(to: encoder)
    }

    // MARK: - Forwarded read properties

    var isTrained: Bool            { graph.isTrained }
    var isEmpty:   Bool            { graph.isEmpty }
    var totalInsertions: Int       { graph.totalInsertions }
    var effectiveThreshold: Float  { graph.effectiveThreshold }
    var partitionLookup: [String: Int] { graph.partitionLookup }
    var graphStats: (liveNodes: Int, deletedNodes: Int, maxLevel: Int, avgLayer0Degree: Float) {
        graph.graphStats
    }

    // MARK: - Forwarded read-write properties

    var nodes: [HNSWGraph.Node] {
        get { graph.nodes }
        set { graph.nodes = newValue }
    }
    var maxLevel: Int {
        get { graph.maxLevel }
        set { graph.maxLevel = newValue }
    }
    var entryPoint: Int {
        get { graph.entryPoint }
        set { graph.entryPoint = newValue }
    }
    var efSearch: Int {
        get { graph.efSearch }
        set { graph.efSearch = newValue }
    }
    var emaExplored: Float {
        get { graph.emaExplored }
        set { graph.emaExplored = newValue }
    }
    var pendingWALRecords: [TopologyWALRecord] {
        get { graph.pendingWALRecords }
        set { graph.pendingWALRecords = newValue }
    }
    var vectorStore: HNSWVectorStore? {
        get { graph.vectorStore }
        set { graph.vectorStore = newValue }
    }

    // MARK: - Forwarded mutations

    mutating func add(partition: Database.Partition, efOverride: Int? = nil) { graph.add(partition: partition, efOverride: efOverride) }
    mutating func remove(documentId: String) { graph.remove(documentId: documentId) }
    @discardableResult

    mutating func trainIfReady() { graph.trainIfReady() }
    mutating func addWithoutUpsert(_ partition: Database.Partition) { graph.addWithoutUpsert(partition) }
    mutating func apply(_ record: TopologyWALRecord) { graph.apply(record) }
    @discardableResult
    mutating func compact() -> HNSWGraph.CompactionResult { graph.compact() }

    // MARK: - Forwarded diagnostics

    func readEmbedding(at vectorIndex: Int) -> [Float] { graph.readEmbedding(at: vectorIndex) }
}

// MARK: - Two-tier search

extension HNSWShard {

    /// Search for the k nearest partitions using a two-tier pipeline.
    ///
    /// **Tier 1 — Tag pre-filter (document selection)**
    /// Compares `queryTagEmbedding` against each document's `PartitionIndex.tagsEmbedding`
    /// using an exact dot-product distance. Sinatra's `inferTagThreshold()` widens the
    /// acceptance gate for documents with strong positive engagement history. Untagged
    /// documents always pass. Parking side effects are emitted here via `parkIndices()`.
    ///
    /// **Tier 2 — HNSW slot search + resolution**
    /// `queryEmbedding` drives proximity ranking over the filtered candidate set. Each
    /// `HNSWGraph.SearchResult` is resolved to a `Database.Partition` via `indices`. Results
    /// are deduplicated by `partitionId` (closest distance wins) and sorted ascending.
    ///
    /// - Parameters:
    ///   - queryEmbedding: Drives HNSW proximity ranking over partition slots (tier 2).
    ///   - queryTagEmbedding: Drives document-level tag pre-filter (tier 1).
    ///     Falls back to `queryEmbedding` when `nil`.
    ///   - k: Size of the HNSW candidate pool (typically `outerK * 10`). The caller
    ///     applies the final top-k cut after access-class filtering and Sinatra.
    ///   - groupFilter: Access-control predicate. Intersected with the tag filter.
    mutating func search(
        shardIndex: Int,
        queryEmbedding: [Float],
        queryTagEmbedding: [Float]?,
        k: Int,
        groupFilter: Set<DocumentID>?,
        accessFilter: ((DocumentID) -> Bool)? = nil,
        sinatra: Sinatra,
        ownerKey: TotemRegistry.Owner,
        sinatraRegistry: SinatraRegistry?,
        registry: TotemRegistry,
        metadataLoader: PartitionDataLoader?,
        request: DatabaseRequest,
        logger: TotemLogger
    ) -> (partitions: [(partition: Database.Partition, distance: Float)],
          trace: HNSWGraph.SearchTrace) {

        // ── Tier 1: Tag pre-filter ───────────────────────────────────────────────
        // Only apply when the caller explicitly provides a tag embedding. Falling back
        // to the query embedding as a proxy produces false exclusions: a natural-language
        // query rarely matches short tag keywords at the required distance threshold.
        let taggedIndices = self.indices.filter { $0.value.tagsEmbedding != nil }
        var tagPreFilterData: [(documentId: DocumentID, tagDistance: Float?, wasIncluded: Bool)] = []

        let tagFilteredDocIds: Set<DocumentID>? = {
            guard !taggedIndices.isEmpty, let tagQueryEmbedding = queryTagEmbedding else { return nil }
            var passing: [DocumentID] = []
            for (docId, idx) in taggedIndices {
                guard let dist = idx.tagDistance(queryEmbedding: tagQueryEmbedding) else {
                    passing.append(docId)
                    continue
                }
                let threshold = sinatra.inferTagThreshold(
                    documentId: docId,
                    owner: ownerKey,
                    registry: sinatraRegistry,
                    documentStats: registry.documentStats
                )
                let included = dist < threshold
                tagPreFilterData.append((docId, dist, included))
                if included { passing.append(docId) }
            }
            let untagged = self.indices.filter { $0.value.tagsEmbedding == nil }.map { $0.key }
            return Set(passing + untagged)
        }()

        if queryTagEmbedding == nil {
            logger.info(
                "Tag Pre-Filter",
                "[shard=\(shardIndex)] skipped — no tag embedding provided (docs tagged=\(taggedIndices.count))",
                service: .database, request: request, flow: .chat
            )
        } else {
            logger.info(
                "Tag Pre-Filter",
                "[shard=\(shardIndex)] docs tagged=\(taggedIndices.count) passed=\(tagPreFilterData.filter(\.wasIncluded).count) excluded=\(tagPreFilterData.filter { !$0.wasIncluded }.count)",
                service: .database, request: request, flow: .chat
            )
        }
        if !tagPreFilterData.isEmpty {
            sinatra.parkIndices(data: tagPreFilterData, request: request)
        }

        // ── Tier 2: HNSW slot search ─────────────────────────────────────────────
        // Combine tag pre-filter (Set), group restriction (Set), and access predicate
        // (closure) into a single nodeFilter closure — no intermediate set union is
        // ever constructed, so the per-node check is O(1) regardless of corpus size.
        let nodeFilter: ((DocumentID) -> Bool)? = switch (tagFilteredDocIds, groupFilter, accessFilter) {
            case (.some(let t), .some(let g), .some(let a)): { t.contains($0) && g.contains($0) && a($0) }
            case (.some(let t), .some(let g), nil):           { t.contains($0) && g.contains($0) }
            case (.some(let t), nil,          .some(let a)): { t.contains($0) && a($0) }
            case (.some(let t), nil,          nil):           { t.contains($0) }
            case (nil,          .some(let g), .some(let a)): { g.contains($0) && a($0) }
            case (nil,          .some(let g), nil):           { g.contains($0) }
            case (nil,          nil,          .some(let a)): a
            case (nil,          nil,          nil):           nil
        }

        let (hnswResults, trace) = graph.search(
            queryEmbedding: queryEmbedding,
            k: k,
            nodeFilter: nodeFilter
        )

        logger.info(
            "HNSW Search",
            "[shard=\(shardIndex)] layers=\(trace.graphMaxLevel + 1) hops=\(trace.upperLayerHops) ef=\(trace.efUsed) explored=\(trace.layer0Explored) candidates=\(trace.candidatesBeforeFilter) threshold=\(String(format: "%.4f", trace.threshold))",
            service: .database, request: request, flow: .chat
        )

        // ── Slot resolution + deduplication ─────────────────────────────────────
        var rawByPartitionId: [String: (partition: Database.Partition, distance: Float)] = [:]
        for r in hnswResults {
            guard let idx = self.indices[r.documentId],
                  let slot = idx.slots.first(where: { $0.id == r.partitionId })
            else { continue }
            let p = slot.toPartition(metadata: metadataLoader?(r.documentId, r.partitionId), indexMetadata: idx.metadata)
            if let existing = rawByPartitionId[p.id] {
                if r.distance < existing.distance { rawByPartitionId[p.id] = (p, r.distance) }
            } else {
                rawByPartitionId[p.id] = (p, r.distance)
            }
        }
        let resolved = rawByPartitionId.values.sorted { $0.distance < $1.distance }

        let filterLabel = nodeFilter != nil ? "filter" : "global"
        logger.info(
            "HNSW Search",
            "[shard=\(shardIndex)/\(filterLabel)] resolved=\(resolved.count)/\(hnswResults.count)",
            service: .database, request: request, flow: .chat
        )

        return (Array(resolved), trace)
    }
}
