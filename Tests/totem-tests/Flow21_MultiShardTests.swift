//
//  Flow21_MultiShardTests.swift
//  database-serverTests
//
//  Tests for multi-shard HNSW correctness, routing, and recall accuracy across
//  the global PartitionTable, TableMutator, and owner-filtered global HNSW.
//
//  Sections
//    1. PartitionTable — fan-out search + documentShardIndex routing (synchronous)
//    2. TableMutator — shard spawn at threshold, per-shard routing (async)
//    3. Owner-filtered global HNSW — owner/group scoping via combinedFilter
//    4. DatabaseConfig — default threshold and custom threshold wiring
//    5. Branch additions — shard stats, oldestAvailableShard backfill, heuristic recall
//

import XCTest
@testable import totem

final class Flow21_MultiShardTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("database-flow21-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        wipeTotemPersistenceFiles()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
        wipeTotemPersistenceFiles()
    }

    // =========================================================================
    // MARK: - Helpers
    // =========================================================================

    /// Manually builds a 2-shard PartitionTable with populated vector stores
    /// and indices, bypassing TableMutator so tests are synchronous.
    private func makeTwoShardTable(
        shardACount: Int = 10,
        shardBCount: Int = 10,
        shardASeed:  UInt64 = 1_000,
        shardBSeed:  UInt64 = 2_000
    ) throws -> (table: PartitionTable, registry: TotemRegistry, aVecs: [[Float]], bVecs: [[Float]]) {
        var table = PartitionTable()
        let storeA = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("shard-a-\(UUID().uuidString)"),
            nodeCount: 0)
        table.shards[0].vectorStore = storeA

        var aVecs: [[Float]] = []
        for i in 0..<shardACount {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: shardASeed + UInt64(i))
            let p = Database.Partition.test(id: "pa\(i)", documentId: "docA\(i)", embedding: v)
            table.put(id: "docA\(i)", partitions: [p], request: .test(scope: .global), logger: .test)
            aVecs.append(v)
        }

        var shard1 = HNSWShard()
        let storeB = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("shard-b-\(UUID().uuidString)"),
            nodeCount: 0)
        shard1.vectorStore = storeB
        table.shards.append(shard1)

        var bVecs: [[Float]] = []
        for i in 0..<shardBCount {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: shardBSeed + UInt64(i))
            let p = Database.Partition.test(id: "pb\(i)", documentId: "docB\(i)", embedding: v)
            table.put(id: "docB\(i)", partitions: [p], request: .test(scope: .global), logger: .test)
            bVecs.append(v)
        }

        var registry = TotemRegistry()
        for i in 0..<shardACount { registry.updateDocumentAccess(for: "docA\(i)", state: .available) }
        for i in 0..<shardBCount { registry.updateDocumentAccess(for: "docB\(i)", state: .available) }

        return (table, registry, aVecs, bVecs)
    }

    /// Creates a TableMutator with a temporary vector store and custom shard threshold.
    private func makeTableMutator(threshold: Int = 10_000) throws -> TableMutator {
        let store = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("vec-\(UUID().uuidString)"),
            nodeCount: 0)
        var table = PartitionTable()
        table.shards[0].vectorStore = store
        let mutator = TableMutator(
            nodeId: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
            logger: .test,
            shardSizeThreshold: threshold)
        mutator.seed(table)
        mutator.seedVectorStore(store)
        return mutator
    }

    /// Brute-force true top-k by L2 distance (used as recall oracle).
    private func trueTopK(_ vecs: [(id: String, v: [Float])], query: [Float], k: Int) -> Set<String> {
        Set(vecs.sorted { VectorFixtures.l2($0.v, query) < VectorFixtures.l2($1.v, query) }
                .prefix(k).map(\.id))
    }

    private func makePutItems(
        count: Int,
        seed: UInt64 = 40_000,
        idPrefix: String = "doc",
        ownerId: String = "test-owner"
    ) -> [(id: DocumentID, partitions: [Database.Partition], tags: [String], tagsEmbedding: [Float]?, metadata: Data?, request: DatabaseRequest)] {
        (0..<count).map { i in
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: seed + UInt64(i))
            let p = Database.Partition.test(id: "\(idPrefix)p\(i)", documentId: "\(idPrefix)\(i)",
                                        embedding: v, ownerId: ownerId)
            return ("\(idPrefix)\(i)", [p], [], nil, nil, .test(scope: .global))
        }
    }

    // =========================================================================
    // MARK: - Section 1: PartitionTable fan-out search
    // =========================================================================

    func testPartitionTableSearchSpansBothShards() throws {
        var (table, registry, _, bVecs) = try makeTwoShardTable()
        let sinatra = Sinatra(logger: .test)

        let query = VectorFixtures.near(bVecs[0], seed: 99_000)
        let (buckets, _, _) = table.search(
            embedding: query, k: 5,
            sinatra: sinatra,
            registry: registry,
            request: .test(scope: .global),
            logger: .test)

        let resultDocIds = buckets.flatMap { $0.partitions }.map { $0.documentId }
        XCTAssertTrue(resultDocIds.contains(where: { $0.hasPrefix("docB") }),
            "A query near shard-B must return at least one result from shard B — fan-out must cover all trained shards")
    }

    func testPartitionTableSearchNotLimitedToActiveShard() throws {
        var (table, registry, aVecs, _) = try makeTwoShardTable()
        let sinatra = Sinatra(logger: .test)

        let query = VectorFixtures.near(aVecs[0], seed: 99_001)
        let (buckets, _, _) = table.search(
            embedding: query, k: 5,
            sinatra: sinatra,
            registry: registry,
            request: .test(scope: .global),
            logger: .test)

        let resultDocIds = buckets.flatMap { $0.partitions }.map { $0.documentId }
        XCTAssertTrue(resultDocIds.contains(where: { $0.hasPrefix("docA") }),
            "Non-active shard 0 must still be searched — fan-out covers all trained shards")
    }

    func testDocumentShardIndexPopulatedOnPut() throws {
        var table = PartitionTable()
        let store = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("idx-test"),
            nodeCount: 0)
        table.shards[0].vectorStore = store

        for i in 0..<3 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i + 20_000))
            let p = Database.Partition.test(id: "p\(i)", documentId: "doc\(i)", embedding: v)
            table.put(id: "doc\(i)", partitions: [p], request: .test(scope: .global), logger: .test)
        }

        for i in 0..<3 {
            XCTAssertEqual(table.documentShardIndex["doc\(i)"], 0,
                "doc\(i) must be routed to shard 0 — documentShardIndex must be populated by put()")
        }
    }

    func testRemoveTargetsCorrectShardViaIndex() throws {
        var (table, _, _, _) = try makeTwoShardTable(shardACount: 5, shardBCount: 5)

        let beforeShard1 = table.shards[1].graphStats.liveNodes
        table.remove(id: "docA0")

        XCTAssertEqual(table.shards[0].graphStats.deletedNodes, 1,
            "Removing docA0 must mark exactly 1 node deleted in shard 0")
        XCTAssertEqual(table.shards[1].graphStats.liveNodes, beforeShard1,
            "Shard 1 must be unaffected when a shard-0 document is removed")
    }

    func testSearchSkipsUntrainedShard() throws {
        var table = PartitionTable()
        let store = try HNSWVectorStore(url: tempDir.appendingPathComponent("s0"), nodeCount: 0)
        table.shards[0].vectorStore = store

        let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 30_000)
        let p = Database.Partition.test(id: "p0", documentId: "doc0", embedding: v)
        table.put(id: "doc0", partitions: [p], request: .test(scope: .global), logger: .test)

        table.shards.append(HNSWShard())
        XCTAssertFalse(table.shards[1].isTrained,
            "Freshly appended shard must report isTrained == false")

        var registry = TotemRegistry()
        registry.updateDocumentAccess(for: "doc0", state: .available)
        let sinatra = Sinatra(logger: .test)

        let (buckets, _, _) = table.search(
            embedding: v, k: 3,
            sinatra: sinatra,
            registry: registry,
            request: .test(scope: .global),
            logger: .test)

        XCTAssertFalse(buckets.flatMap { $0.partitions }.isEmpty,
            "Search must succeed and return results from the trained shard even when an untrained shard is present")
    }

    func testFanoutRecallAtK() throws {
        var (table, registry, aVecs, bVecs) = try makeTwoShardTable(
            shardACount: 15, shardBCount: 15,
            shardASeed: 5_000, shardBSeed: 6_000)
        let sinatra = Sinatra(logger: .test)

        var allVecs: [(id: String, v: [Float])] = []
        for i in 0..<15 { allVecs.append((id: "pa\(i)", v: aVecs[i])) }
        for i in 0..<15 { allVecs.append((id: "pb\(i)", v: bVecs[i])) }

        let targetSeeds: [UInt64] = [5_000, 5_003, 5_007, 5_011, 5_014,
                                     6_001, 6_004, 6_008, 6_010, 6_013]
        var totalRecall: Float = 0
        for (qi, seed) in targetSeeds.enumerated() {
            let base  = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: seed)
            let query = VectorFixtures.near(base, seed: seed &+ UInt64(qi + 1) &* 90_001)

            let trueIds = trueTopK(allVecs, query: query, k: 5)

            let (buckets, _, _) = table.search(
                embedding: query, k: 5,
                sinatra: sinatra,
                registry: registry,
                request: .test(scope: .global),
                logger: .test)

            let hnswIds = Set(buckets.flatMap { $0.partitions }.map { $0.id })
            totalRecall += Float(hnswIds.intersection(trueIds).count) / 5.0
        }

        let avgRecall = totalRecall / Float(targetSeeds.count)
        XCTAssertGreaterThanOrEqual(avgRecall, 0.80,
            "Fan-out recall@5 across 2 shards must be ≥ 0.80 — got \(String(format: "%.2f", avgRecall))")
    }

    // =========================================================================
    // MARK: - Section 2: TableMutator shard spawn & routing
    // =========================================================================

    func testShardSpawnedWhenThresholdReached() async throws {
        let mutator = try makeTableMutator(threshold: 5)
        await mutator.putBatch(items: makePutItems(count: 6))
        XCTAssertEqual(mutator.snapshot?.shards.count, 2,
            "Inserting 6 docs with threshold=5 must spawn a second shard")
    }

    func testSeventhDocumentGoesIntoActiveShard() async throws {
        let mutator = try makeTableMutator(threshold: 5)
        await mutator.putBatch(items: makePutItems(count: 7))
        XCTAssertEqual(mutator.snapshot?.documentShardIndex["doc6"], 1,
            "doc6 (7th doc, threshold=5) must be routed to shard 1 after spawn")
        XCTAssertGreaterThanOrEqual(mutator.snapshot?.shards[1].nodes.count ?? 0, 1,
            "Shard 1 must have at least 1 node after the spawn")
    }

    func testRemoveAfterSpawnTargetsCorrectShard() async throws {
        let mutator = try makeTableMutator(threshold: 5)
        await mutator.putBatch(items: makePutItems(count: 8))

        await mutator.remove(id: "doc0")

        let snap = mutator.snapshot
        XCTAssertEqual(snap?.shards[0].graphStats.deletedNodes, 1,
            "Removing doc0 (shard 0) must mark exactly 1 node deleted in shard 0")
        XCTAssertEqual(snap?.shards[1].graphStats.deletedNodes, 0,
            "Shard 1 must be untouched when a shard-0 document is removed")
    }

    func testDocumentShardIndexConsistentAfterSpawn() async throws {
        let mutator = try makeTableMutator(threshold: 5)
        await mutator.putBatch(items: makePutItems(count: 8))

        let snap = mutator.snapshot
        for i in 0..<8 {
            XCTAssertNotNil(snap?.documentShardIndex["doc\(i)"],
                "doc\(i) must have a documentShardIndex entry after putBatch")
        }
        for i in 0..<5 {
            XCTAssertEqual(snap?.documentShardIndex["doc\(i)"], 0,
                "doc\(i) (docs 0–4, threshold=5) must map to shard 0")
        }
        for i in 5..<8 {
            XCTAssertEqual(snap?.documentShardIndex["doc\(i)"], 1,
                "doc\(i) (docs 5–7, after spawn) must map to shard 1")
        }
    }

    func testCompactAcrossMultipleShards() async throws {
        let mutator = try makeTableMutator(threshold: 5)
        await mutator.putBatch(items: makePutItems(count: 8))

        await mutator.remove(id: "doc0")
        await mutator.remove(id: "doc1")
        await mutator.remove(id: "doc5")
        await mutator.remove(id: "doc6")

        await mutator.compact()

        let snap = mutator.snapshot
        XCTAssertEqual(snap?.shards[0].graphStats.deletedNodes, 0,
            "Shard 0 must have 0 deleted nodes after compact()")
        XCTAssertEqual(snap?.shards[1].graphStats.deletedNodes, 0,
            "Shard 1 must have 0 deleted nodes after compact()")
    }

    // =========================================================================
    // MARK: - Section 3: Owner-filtered global HNSW
    // =========================================================================

    func testOwnerFilteredSearchReturnsOnlyOwnerDocs() throws {
        var table = PartitionTable()
        let store = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("owner-filter-\(UUID().uuidString)"),
            nodeCount: 0)
        table.shards[0].vectorStore = store

        var ownerADocIds: [String] = []
        var ownerAVecs: [[Float]] = []
        for i in 0..<4 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i + 91_000))
            let docId = "oa-doc\(i)"
            let p = Database.Partition.test(id: "oa-p\(i)", documentId: docId, embedding: v, ownerId: "owner-a")
            table.put(id: docId, partitions: [p], request: .test(scope: .global), logger: .test)
            ownerADocIds.append(docId)
            ownerAVecs.append(v)
        }
        for i in 0..<4 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i + 92_000))
            let docId = "ob-doc\(i)"
            let p = Database.Partition.test(id: "ob-p\(i)", documentId: docId, embedding: v, ownerId: "owner-b")
            table.put(id: docId, partitions: [p], request: .test(scope: .global), logger: .test)
        }

        var registry = TotemRegistry()
        registry.ownersDocuments[TotemRegistry.Owner(id: "owner-a")] = ownerADocIds

        let sinatra = Sinatra(logger: .test)
        let query = VectorFixtures.near(ownerAVecs[0], seed: 93_000)
        let (buckets, _, _) = table.search(
            embedding: query,
            k: 5,
            sinatra: sinatra,
            registry: registry,
            request: DatabaseRequest(ownerId: "owner-a", scope: .personal),
            logger: .test)

        let resultDocIds = buckets.flatMap { $0.partitions }.map { $0.documentId }
        XCTAssertFalse(resultDocIds.isEmpty,
            "Owner-filtered search must return at least one result")
        XCTAssertTrue(resultDocIds.allSatisfy { ownerADocIds.contains($0) },
            "All results must belong to owner-a — no cross-owner leakage from owner-b")
    }

    func testOwnerFilteredSearchSpanningMultipleShards() throws {
        var table = PartitionTable()
        let storeA = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("owner-ms-a-\(UUID().uuidString)"),
            nodeCount: 0)
        table.shards[0].vectorStore = storeA

        var ownerDocIds: [String] = []
        var ownerVecs: [[Float]] = []

        for i in 0..<8 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i + 94_000))
            let docId = "ms-doc\(i)"
            let p = Database.Partition.test(id: "ms-p\(i)", documentId: docId, embedding: v, ownerId: "ms-owner")
            table.put(id: docId, partitions: [p], request: .test(scope: .global), logger: .test)
            ownerDocIds.append(docId)
            ownerVecs.append(v)
        }

        var shard1 = HNSWShard()
        let storeB = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("owner-ms-b-\(UUID().uuidString)"),
            nodeCount: 0)
        shard1.vectorStore = storeB
        table.shards.append(shard1)

        for i in 8..<16 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i + 94_000))
            let docId = "ms-doc\(i)"
            let p = Database.Partition.test(id: "ms-p\(i)", documentId: docId, embedding: v, ownerId: "ms-owner")
            table.put(id: docId, partitions: [p], request: .test(scope: .global), logger: .test)
            ownerDocIds.append(docId)
            ownerVecs.append(v)
        }

        var registry = TotemRegistry()
        registry.ownersDocuments[TotemRegistry.Owner(id: "ms-owner")] = ownerDocIds

        let sinatra = Sinatra(logger: .test)
        let query = VectorFixtures.near(ownerVecs[0], seed: 95_000)
        let (buckets, _, _) = table.search(
            embedding: query,
            k: 5,
            sinatra: sinatra,
            registry: registry,
            request: DatabaseRequest(ownerId: "ms-owner", scope: .personal),
            logger: .test)

        let resultDocIds = buckets.flatMap { $0.partitions }.map { $0.documentId }
        XCTAssertFalse(resultDocIds.isEmpty,
            "Owner-filtered search across 2 shards must return at least one result")
        XCTAssertTrue(resultDocIds.allSatisfy { ownerDocIds.contains($0) },
            "All results from a multi-shard owner-filtered search must belong to the owner's documents")
    }

    func testGroupPredicateWithinOwnerDocs() throws {
        var table = PartitionTable()
        let store = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("group-pred-\(UUID().uuidString)"),
            nodeCount: 0)
        table.shards[0].vectorStore = store

        let groupAId = "group-a"
        let groupBId = "group-b"
        let ownerId = "group-owner"

        var groupADocIds: [String] = []
        var groupAVecs: [[Float]] = []
        var groupBDocIds: [String] = []
        var allOwnerDocIds: [String] = []

        for i in 0..<3 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i + 96_000))
            let docId = "ga-doc\(i)"
            let p = Database.Partition.test(id: "ga-p\(i)", documentId: docId, embedding: v, ownerId: ownerId)
            table.put(id: docId, partitions: [p], request: .test(scope: .global), logger: .test)
            groupADocIds.append(docId)
            groupAVecs.append(v)
            allOwnerDocIds.append(docId)
        }
        for i in 0..<3 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i + 97_000))
            let docId = "gb-doc\(i)"
            let p = Database.Partition.test(id: "gb-p\(i)", documentId: docId, embedding: v, ownerId: ownerId)
            table.put(id: docId, partitions: [p], request: .test(scope: .global), logger: .test)
            groupBDocIds.append(docId)
            allOwnerDocIds.append(docId)
        }

        var registry = TotemRegistry()
        registry.ownersDocuments[TotemRegistry.Owner(id: ownerId)] = allOwnerDocIds
        registry.groups[groupAId] = groupADocIds
        registry.groups[groupBId] = groupBDocIds

        let sinatra = Sinatra(logger: .test)
        let query = VectorFixtures.near(groupAVecs[0], seed: 98_000)
        let groupA = Database.Group(id: groupAId, label: "Group A", ownerId: ownerId, documents: [])
        let (buckets, _, _) = table.search(
            embedding: query,
            k: 3,
            sinatra: sinatra,
            registry: registry,
            request: DatabaseRequest(ownerId: ownerId, group: groupA, scope: .personal),
            logger: .test)

        let resultDocIds = buckets.flatMap { $0.partitions }.map { $0.documentId }
        XCTAssertFalse(resultDocIds.isEmpty,
            "Group-predicate search within owner docs must return at least one result")
        XCTAssertTrue(resultDocIds.allSatisfy { groupADocIds.contains($0) },
            "All results from a group-A scoped search must come from group A")
        XCTAssertFalse(resultDocIds.contains(where: { groupBDocIds.contains($0) }),
            "Group B documents must not appear in a group-A scoped search")
    }

    // =========================================================================
    // MARK: - Section 4: DatabaseConfig integration
    // =========================================================================

    func testDefaultShardThresholdIs2500() {
        XCTAssertEqual(DatabaseConfig().shardSizeThreshold, 2_500,
            "Default DatabaseConfig.shardSizeThreshold must be 2_500")
    }

    func testCustomThresholdThreadedToTableMutator() async throws {
        let mutator = try makeTableMutator(threshold: 3)
        await mutator.putBatch(items: makePutItems(count: 4, seed: 90_000))
        XCTAssertEqual(mutator.snapshot?.shards.count, 2,
            "A custom threshold of 3 must cause a spawn when the 4th doc is inserted")
    }

    // =========================================================================
    // MARK: - Section 5: Branch additions
    // =========================================================================

    // MARK: Shard stats

    func testSearchReturnsShardStatsForEachTrainedShard() throws {
        var (table, registry, _, _) = try makeTwoShardTable()
        let sinatra = Sinatra(logger: .test)
        let query = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 77_000)

        let (_, _, shardStats) = table.search(
            embedding: query, k: 3,
            sinatra: sinatra,
            registry: registry,
            request: .test(scope: .global),
            logger: .test)

        XCTAssertEqual(shardStats.count, 2,
            "Search across 2 trained shards must return exactly 2 SearchShardStat entries")
        XCTAssertTrue(shardStats.allSatisfy { $0.nodes > 0 },
            "Each shard stat must report at least 1 node")
        let shardIndices = Set(shardStats.map { $0.shardIndex })
        XCTAssertEqual(shardIndices, [0, 1],
            "Shard stat indices must cover both shards and be unique")
    }

    func testSearchShardStatsExploredAndEfPopulated() throws {
        var (table, registry, _, _) = try makeTwoShardTable(shardACount: 15, shardBCount: 15)
        let sinatra = Sinatra(logger: .test)
        let query = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 78_000)

        let (_, _, shardStats) = table.search(
            embedding: query, k: 5,
            sinatra: sinatra,
            registry: registry,
            request: .test(scope: .global),
            logger: .test)

        for stat in shardStats {
            XCTAssertGreaterThan(stat.explored, 0,
                "Shard \(stat.shardIndex): explored must be > 0 after a real search")
            XCTAssertGreaterThan(stat.efUsed, 0,
                "Shard \(stat.shardIndex): efUsed must be > 0 after a real search")
        }
    }

    // MARK: oldestAvailableShard backfill

    func testOldestAvailableShardBackfillsAfterCompaction() async throws {
        // docs 0-4 fill shard 0 (threshold=5), docs 5-7 go to spawned shard 1.
        let mutator = try makeTableMutator(threshold: 5)
        await mutator.putBatch(items: makePutItems(count: 8))

        // Remove 4 from shard 0, dropping its live node count below threshold.
        for i in 0..<4 { await mutator.remove(id: "doc\(i)") }
        let compactResult = await mutator.compact()

        XCTAssertGreaterThan(compactResult.removedNodes, 0,
            "Precondition: compact must remove the soft-deleted nodes so physical count drops")

        let snapBefore = mutator.snapshot
        XCTAssertLessThan(snapBefore?.shards[0].nodes.count ?? 5, 5,
            "Shard 0 must have physical capacity after compaction")

        // New insertion must backfill into shard 0 — not spawn shard 2.
        let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 80_001)
        let p = Database.Partition.test(id: "backfill-p0", documentId: "backfill-doc", embedding: v)
        await mutator.put(id: "backfill-doc", partitions: [p], request: .test())

        XCTAssertEqual(mutator.snapshot?.documentShardIndex["backfill-doc"], 0,
            "After compaction frees capacity in shard 0, new insertions must route to shard 0 (oldestAvailableShard)")
        XCTAssertEqual(mutator.snapshot?.shards.count, 2,
            "No third shard must be spawned when an older shard has capacity after compaction")
    }

    func testNoSpawnWhenAllShardsHaveCapacity() async throws {
        // threshold=10: insert 8 docs — all fit in shard 0, no spawn needed.
        let mutator = try makeTableMutator(threshold: 10)
        await mutator.putBatch(items: makePutItems(count: 8, seed: 81_000))

        XCTAssertEqual(mutator.snapshot?.shards.count, 1,
            "No spawn must occur when the single shard has not reached threshold")
        XCTAssertEqual(mutator.snapshot?.shards[0].nodes.count, 8,
            "All 8 documents must land in shard 0")
    }

    // MARK: Heuristic neighbor selection recall

    func testHeuristicSelectionPreservesRecallAtK() throws {
        // Build a single-shard graph with 40 random 1024-dim vectors and verify
        // that Algorithm 4 heuristic neighbor selection does not degrade recall.
        var graph = HNSWGraph()
        graph.vectorStore = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("heur-\(UUID().uuidString)"),
            nodeCount: 0)

        var allVecs: [(id: String, v: [Float])] = []
        for i in 0..<40 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i) + 10_000)
            graph.add(partition: .test(id: "h\(i)", documentId: "hd\(i)", embedding: v))
            allVecs.append(("h\(i)", v))
        }

        var totalRecall: Float = 0
        let querySeeds: [UInt64] = [50_000, 50_001, 50_002, 50_003, 50_004,
                                    50_005, 50_006, 50_007, 50_008, 50_009]
        for seed in querySeeds {
            let query = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: seed)
            let trueIds = Set(allVecs.sorted { VectorFixtures.l2($0.v, query) < VectorFixtures.l2($1.v, query) }
                              .prefix(3).map(\.id))
            let (results, _) = graph.search(queryEmbedding: query, k: 3)
            let hnswIds = Set(results.map { $0.partitionId })
            totalRecall += Float(hnswIds.intersection(trueIds).count) / 3.0
        }

        let avgRecall = totalRecall / Float(querySeeds.count)
        XCTAssertGreaterThanOrEqual(avgRecall, 0.80,
            "Algorithm 4 heuristic selection must maintain recall@3 ≥ 0.80 — got \(String(format: "%.2f", avgRecall))")
    }

    func testHeuristicSelectionDoesNotUnderconnectGraph() throws {
        // After inserting N nodes, no live node should have zero neighbors at layer 0.
        // The keepPrunedConnections fallback ensures this even in sparse embedding regions.
        var graph = HNSWGraph()
        graph.vectorStore = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("heur-conn-\(UUID().uuidString)"),
            nodeCount: 0)

        for i in 0..<30 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i) + 20_000)
            graph.add(partition: .test(id: "c\(i)", documentId: "cd\(i)", embedding: v))
        }

        let underconnected = graph.nodes.filter { !$0.isDeleted && ($0.neighbors.first?.isEmpty ?? true) }
        XCTAssertTrue(underconnected.isEmpty,
            "No live node must have zero layer-0 neighbors after heuristic selection — keepPrunedConnections must backfill")
    }
}
