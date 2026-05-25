//
//  Flow3_HNSWInvariantsTests.swift
//  database-serverTests
//
//  Tests for HNSWGraph: insertion, search correctness, deletion,
//  structural invariants, compaction, Phase 3 vector persistence,
//  and Phase 4 append-only topology (WAL).
//

import XCTest
@testable import totem

final class Flow3_HNSWInvariantsTests: XCTestCase {

    // MARK: - Setup

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("database-flow3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Helpers

    /// Creates a fresh mmap'd vector store at a unique path inside `tempDir`.
    private func makeVectorStore() throws -> HNSWVectorStore {
        try HNSWVectorStore(
            url: tempDir.appendingPathComponent("vec-\(UUID().uuidString)"),
            nodeCount: 0
        )
    }

    /// Creates an empty graph with a vector store attached.
    private func makeGraph() throws -> HNSWGraph {
        var graph = HNSWGraph()
        graph.vectorStore = try makeVectorStore()
        return graph
    }

    /// Creates a graph pre-loaded with `count` random 1024-dim nodes.
    private func makeGraph(count: Int, seed: UInt64 = 0) throws -> (HNSWGraph, [[Float]]) {
        var graph = try makeGraph()
        var vecs: [[Float]] = []
        for i in 0..<count {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: seed + UInt64(i))
            graph.add(partition: .test(id: "p\(i)", documentId: "doc\(i)", embedding: v))
            vecs.append(v)
        }
        return (graph, vecs)
    }

    private func populatedGraph(partitionCount: Int, seed: UInt64 = 0) throws -> HNSWGraph {
        let (graph, _) = try makeGraph(count: partitionCount, seed: seed)
        return graph
    }

    private func makePartition(
        id: String,
        documentId: String,
        embedding: [Float],
        ownerId: String = "owner1"
    ) -> Database.Partition {
        Database.Partition.test(id: id, documentId: documentId, embedding: embedding, ownerId: ownerId)
    }

    // MARK: - Insertion

    func testGraphIsEmptyBeforeFirstInsert() {
        let graph = HNSWGraph()
        XCTAssertTrue(graph.isEmpty)
        XCTAssertFalse(graph.isTrained)
        XCTAssertEqual(graph.entryPoint, -1)
    }

    func testGraphBecomesTrainedAfterFirstInsert() throws {
        var graph = try makeGraph()
        graph.add(partition: makePartition(
            id: "p0", documentId: "d0",
            embedding: VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 1)
        ))
        XCTAssertTrue(graph.isTrained)
        XCTAssertNotEqual(graph.entryPoint, -1)
    }

    func testInsertedPartitionFoundInLookup() throws {
        var graph = try makeGraph()
        graph.add(partition: makePartition(
            id: "p0", documentId: "d0",
            embedding: VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 1)
        ))
        XCTAssertNotNil(graph.partitionLookup["p0"])
    }

    func testNodeCountMatchesInsertions() throws {
        var graph = try populatedGraph(partitionCount: 10)
        XCTAssertEqual(graph.nodes.count, 10)
        XCTAssertEqual(graph.totalInsertions, 10)
    }

    func testEmptyEmbeddingIsSkipped() throws {
        var graph = try makeGraph()
        graph.add(partition: makePartition(id: "empty", documentId: "d0", embedding: []))
        XCTAssertFalse(graph.isTrained)
        XCTAssertEqual(graph.nodes.count, 0)
    }

    // MARK: - Search

    func testSearchReturnsUpToKResults() throws {
        var graph = try populatedGraph(partitionCount: 20)
        let (results, _) = graph.search(
            queryEmbedding: VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 999), k: 5
        )
        XCTAssertLessThanOrEqual(results.count, 5)
        XCTAssertGreaterThan(results.count, 0)
    }

    func testSearchResultsSortedAscendingByDistance() throws {
        var graph = try populatedGraph(partitionCount: 30)
        let (results, _) = graph.search(
            queryEmbedding: VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 777), k: 10
        )
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(results[i-1].distance, results[i].distance,
                "Results must be sorted ascending by distance")
        }
    }

    func testSearchNearestPartitionIsCorrect() throws {
        var graph = try makeGraph()
        let vectors = (0..<10).map { VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64($0 + 200)) }
        for (i, v) in vectors.enumerated() {
            graph.add(partition: makePartition(id: "p\(i)", documentId: "d\(i)", embedding: v))
        }
        let (results, _) = graph.search(queryEmbedding: vectors[3], k: 1)
        XCTAssertEqual(results.first?.partitionId, "p3",
            "Nearest neighbor must be the partition with identical embedding")
    }

    func testSearchReturnsEmptyForEmptyGraph() throws {
        var graph = try makeGraph()
        let (results, _) = graph.search(
            queryEmbedding: VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 1), k: 3
        )
        XCTAssertTrue(results.isEmpty)
    }

    func testSearchTraceGraphNodesMatchNodeCount() throws {
        var graph = try populatedGraph(partitionCount: 15)
        let (_, trace) = graph.search(
            queryEmbedding: VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 55), k: 3
        )
        XCTAssertEqual(trace.graphNodes, graph.nodes.count)
    }

    // MARK: - Deletion

    func testDeletedNodeExcludedFromSearch() throws {
        var graph = try makeGraph()
        let target = [Float](repeating: 0, count: HNSWVectorStore.vectorDim)
        graph.add(partition: makePartition(id: "target", documentId: "docTarget", embedding: target))
        for i in 0..<15 {
            graph.add(partition: makePartition(
                id: "p\(i)", documentId: "d\(i)",
                embedding: VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i + 300))
            ))
        }
        let (before, _) = graph.search(queryEmbedding: target, k: 1)
        XCTAssertEqual(before.first?.partitionId, "target")

        graph.remove(documentId: "docTarget")
        let (after, _) = graph.search(queryEmbedding: target, k: 5)
        XCTAssertFalse(after.map { $0.partitionId }.contains("target"))
    }

    func testRemoveReelectsEntryPointWhenCurrentIsDeleted() throws {
        var graph = try populatedGraph(partitionCount: 10)
        let epDocId = graph.nodes[graph.entryPoint].documentId
        graph.remove(documentId: epDocId)
        if graph.entryPoint != -1 {
            XCTAssertFalse(graph.nodes[graph.entryPoint].isDeleted,
                "Entry point must point to a live node after re-election")
        }
    }

    func testTotalDeletionsTracked() throws {
        var graph = try populatedGraph(partitionCount: 10)
        graph.remove(documentId: "doc0")
        graph.remove(documentId: "doc1")
        XCTAssertEqual(graph.totalDeletions, 2)
    }

    // MARK: - Structural invariants

    func testLayer0NeighborCountWithinBound() throws {
        var graph = try populatedGraph(partitionCount: 50)
        let maxLayer0 = graph.M * 2
        for node in graph.nodes where !node.isDeleted {
            let layer0Count = node.neighbors.first?.count ?? 0
            XCTAssertLessThanOrEqual(layer0Count, maxLayer0,
                "Layer-0 neighbors \(layer0Count) exceed 2×M=\(maxLayer0)")
        }
    }

    func testUpperLayerNeighborCountWithinBound() throws {
        var graph = try populatedGraph(partitionCount: 50)
        for node in graph.nodes where !node.isDeleted && node.level > 0 {
            for l in 1...node.level {
                let count = node.neighbors[l].count
                XCTAssertLessThanOrEqual(count, graph.M,
                    "Layer-\(l) neighbors \(count) exceed M=\(graph.M)")
            }
        }
    }

    func testEntryPointHasHighestLevel() throws {
        var graph = try populatedGraph(partitionCount: 50)
        guard graph.entryPoint != -1 else { return }
        let epLevel = graph.nodes[graph.entryPoint].level
        XCTAssertEqual(epLevel, graph.maxLevel,
            "Entry point level \(epLevel) must equal maxLevel \(graph.maxLevel)")
    }

    func testNeighborIndicesAreValidNodeIndices() throws {
        var graph = try populatedGraph(partitionCount: 30)
        let nodeCount = graph.nodes.count
        for node in graph.nodes {
            for layer in node.neighbors {
                for idx in layer {
                    XCTAssertTrue(idx >= 0 && idx < nodeCount,
                        "Neighbor index \(idx) out of bounds [0, \(nodeCount))")
                }
            }
        }
    }

    // MARK: - Compaction

    func testCompactionRemovesAllDeletedNodes() throws {
        var graph = try populatedGraph(partitionCount: 20)
        for i in 0..<5 { graph.remove(documentId: "doc\(i)") }
        graph.compact()
        XCTAssertTrue(graph.nodes.allSatisfy { !$0.isDeleted },
            "No deleted nodes should remain after compaction")
    }

    func testCompactionReducesNodeCount() throws {
        var graph = try populatedGraph(partitionCount: 20)
        let before = graph.nodes.count
        for i in 0..<5 { graph.remove(documentId: "doc\(i)") }
        let result = graph.compact()
        XCTAssertEqual(result.removedNodes, 5)
        XCTAssertEqual(graph.nodes.count, before - 5)
    }

    func testCompactionResultReportsCorrectCounts() throws {
        var graph = try populatedGraph(partitionCount: 10)
        for i in 0..<3 { graph.remove(documentId: "doc\(i)") }
        let result = graph.compact()
        XCTAssertEqual(result.beforeNodes, 10)
        XCTAssertEqual(result.afterNodes, 7)
        XCTAssertEqual(result.removedNodes, 3)
    }

    func testCompactionUpdatesPartitionLookup() throws {
        var graph = try populatedGraph(partitionCount: 10)
        graph.remove(documentId: "doc0")
        graph.compact()
        for (_, idx) in graph.partitionLookup {
            XCTAssertFalse(graph.nodes[idx].isDeleted,
                "partitionLookup must not point to deleted nodes after compaction")
        }
    }

    func testCompactionEntryPointIsLiveNode() throws {
        var graph = try populatedGraph(partitionCount: 15)
        let epDocId = graph.nodes[graph.entryPoint].documentId
        graph.remove(documentId: epDocId)
        graph.compact()
        if graph.entryPoint != -1 {
            XCTAssertFalse(graph.nodes[graph.entryPoint].isDeleted,
                "Entry point must be a live node after compaction")
        }
    }

    func testSearchStillWorksAfterCompaction() throws {
        var graph = try populatedGraph(partitionCount: 20)
        for i in 0..<3 { graph.remove(documentId: "doc\(i)") }
        graph.compact()
        let deletedDocIds = Set((0..<3).map { "doc\($0)" })
        let (results, _) = graph.search(
            queryEmbedding: VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 42), k: 5
        )
        for r in results {
            XCTAssertFalse(deletedDocIds.contains(r.documentId),
                "Deleted documents must not appear in results after compaction")
        }
    }

    // MARK: - Duplicate partition safety

    func testReindexSamePartitionEvictsOldNode() throws {
        var graph = try makeGraph()
        let v1 = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 1)
        let v2 = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 2)
        graph.add(partition: makePartition(id: "dup", documentId: "doc1", embedding: v1))
        let oldIdx = graph.partitionLookup["dup"]!
        graph.add(partition: makePartition(id: "dup", documentId: "doc1", embedding: v2))
        let newIdx = graph.partitionLookup["dup"]!
        XCTAssertTrue(graph.nodes[oldIdx].isDeleted,
            "Old node must be marked deleted after re-indexing the same partitionId")
        XCTAssertNotEqual(oldIdx, newIdx, "Re-indexing must create a new node at a new index")
        XCTAssertFalse(graph.nodes[newIdx].isDeleted, "New node must be live")
        XCTAssertEqual(graph.totalDeletions, 1)
        let (results, _) = graph.search(queryEmbedding: v2, k: 1)
        XCTAssertEqual(results.first?.partitionId, "dup")
    }

    func testSearchDeduplicatesPhantomPartitionNodes() throws {
        var table = PartitionTable()
        table.shards[0].vectorStore = try makeVectorStore()
        let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 100)
        let p = makePartition(id: "p0", documentId: "doc0", embedding: v)
        table.put(id: "doc0", partitions: [p], request: .test(scope: .global), logger: .test)
        table.shards[0].addWithoutUpsert(p)

        var registry = TotemRegistry()
        registry.updateDocumentAccess(for: "doc0", state: .available)
        let sinatra = Sinatra(logger: .test)

        let (buckets, _, _) = table.search(
            embedding: v, k: 5,
            sinatra: sinatra, registry: registry,
            request: .test(scope: .global), logger: .test
        )
        let allPartitions = buckets.flatMap { $0.partitions }
        let ids = allPartitions.map { $0.id }
        XCTAssertEqual(Set(ids).count, ids.count,
            "Search results must have unique partitionIds even when the graph has phantom nodes")
    }

    func testSearchPhantomNodeMinDistanceIsPreserved() throws {
        var table = PartitionTable()
        table.shards[0].vectorStore = try makeVectorStore()
        let query  = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 50)
        let vClose = VectorFixtures.near(query, seed: 1)
        let vFar   = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 200)
        let pClose = makePartition(id: "dup", documentId: "doc0", embedding: vClose)
        let pFar   = makePartition(id: "dup", documentId: "doc0", embedding: vFar)
        table.put(id: "doc0", partitions: [pClose], request: .test(scope: .global), logger: .test)
        table.shards[0].addWithoutUpsert(pFar)

        var registry = TotemRegistry()
        registry.updateDocumentAccess(for: "doc0", state: .available)
        let sinatra = Sinatra(logger: .test)

        let (buckets, _, _) = table.search(
            embedding: query, k: 5,
            sinatra: sinatra, registry: registry,
            request: .test(scope: .global), logger: .test
        )
        let allPartitions = buckets.flatMap { $0.partitions }
        let dupPartitions = allPartitions.filter { $0.id == "dup" }
        XCTAssertEqual(dupPartitions.count, 1, "Deduplicated partition must appear exactly once")

        let dFar = VectorFixtures.l2(query, vFar)
        if let bucket = buckets.first(where: { $0.partitions.contains(where: { $0.id == "dup" }) }),
           let idx = bucket.partitions.firstIndex(where: { $0.id == "dup" }) {
            XCTAssertLessThan(bucket.scores[idx], dFar + 1e-4,
                "Surviving result must not be the farther phantom node")
        }
    }

    // MARK: - Phase 3: Vector persistence invariants

    /// vectorIndex values must remain dense 0..<liveNodeCount after compaction.
    func testVectorIndicesAreDenseAfterCompact() throws {
        var (graph, _) = try makeGraph(count: 10)
        graph.remove(documentId: "doc3")
        graph.remove(documentId: "doc7")
        let result = graph.compact()
        let indices = graph.nodes.map { $0.vectorIndex }
        XCTAssertEqual(Set(indices), Set(0..<graph.nodes.count),
            "vectorIndex values must be dense 0..<nodeCount after compaction")
        XCTAssertEqual(result.survivingVectorIndices.count, graph.nodes.count,
            "survivingVectorIndices must cover exactly the live nodes")
    }

    /// Topology encode → decode → reattach same vector store → search returns correct result.
    func testSearchCorrectAfterTopologyRoundTrip() throws {
        // Use a named store URL so it can be reopened after decode.
        let storeURL = tempDir.appendingPathComponent("rt-vectors")
        var graph = HNSWGraph()
        graph.vectorStore = try HNSWVectorStore(url: storeURL, nodeCount: 0)
        var vecs: [[Float]] = []
        for i in 0..<10 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i) + 42)
            graph.add(partition: .test(id: "rt\(i)", documentId: "rtDoc\(i)", embedding: v))
            vecs.append(v)
        }

        let data     = try PropertyListEncoder().encode(graph)
        var restored = try PropertyListDecoder().decode(HNSWGraph.self, from: data)
        restored.vectorStore = try HNSWVectorStore(url: storeURL, nodeCount: restored.nodes.count)

        let (results, _) = restored.search(queryEmbedding: vecs[4], k: 1)
        XCTAssertEqual(results.first?.partitionId, "rt4",
            "Exact-match vector must be the nearest neighbor after topology round-trip")
    }

    // MARK: - Phase 4: WAL record encoding round-trip

    func testWALRecordRoundTrip_nodeInserted() throws {
        let record = TopologyWALRecord.nodeInserted(
            partitionId: "p0", documentId: "doc0", vectorIndex: 7, level: 1,
            neighborsByLayer: [[1, 2, 3], [4, 5]]
        )
        let payload = record.encodePayload()
        let decoded = try TopologyWALRecord.decodePayload(typeCode: record.typeCode, data: payload)
        XCTAssertEqual(decoded, record)
    }

    func testWALRecordRoundTrip_neighborsUpdated() throws {
        let record = TopologyWALRecord.neighborsUpdated(nodeIndex: 7, layer: 0, neighbors: [1, 3, 5, 9])
        let payload = record.encodePayload()
        let decoded = try TopologyWALRecord.decodePayload(typeCode: record.typeCode, data: payload)
        XCTAssertEqual(decoded, record)
    }

    func testWALRecordRoundTrip_nodeDeleted() throws {
        let record = TopologyWALRecord.nodeDeleted(nodeIndex: 42)
        let payload = record.encodePayload()
        let decoded = try TopologyWALRecord.decodePayload(typeCode: record.typeCode, data: payload)
        XCTAssertEqual(decoded, record)
    }

    func testWALRecordRoundTrip_entryPointChanged() throws {
        let record = TopologyWALRecord.entryPointChanged(nodeIndex: 5, maxLevel: 2)
        let payload = record.encodePayload()
        let decoded = try TopologyWALRecord.decodePayload(typeCode: record.typeCode, data: payload)
        XCTAssertEqual(decoded, record)
    }

    func testWALRecordRoundTrip_commit() throws {
        let record = TopologyWALRecord.commit
        let payload = record.encodePayload()
        XCTAssertTrue(payload.isEmpty, ".commit must encode to an empty payload")
        let decoded = try TopologyWALRecord.decodePayload(typeCode: record.typeCode, data: payload)
        XCTAssertEqual(decoded, record)
    }

    // MARK: - Phase 4: WAL file I/O

    func testWALAppendAndReadAll() throws {
        let wal = try HNSWTopologyWAL(url: tempDir.appendingPathComponent("test.wal"))
        let r1  = TopologyWALRecord.nodeDeleted(nodeIndex: 0)
        let r2  = TopologyWALRecord.entryPointChanged(nodeIndex: 1, maxLevel: 0)
        try wal.append(r1)
        try wal.append(r2)
        try wal.append(.commit)
        let records = try wal.readAll()
        XCTAssertEqual(records.count, 2)
        XCTAssertEqual(records[0], r1)
        XCTAssertEqual(records[1], r2)
    }

    func testWALCommitGroupSemantics() throws {
        let wal = try HNSWTopologyWAL(url: tempDir.appendingPathComponent("commit-group.wal"))
        let r1 = TopologyWALRecord.nodeDeleted(nodeIndex: 0)
        let r2 = TopologyWALRecord.entryPointChanged(nodeIndex: 1, maxLevel: 0)
        let r3 = TopologyWALRecord.nodeDeleted(nodeIndex: 2)

        // Group 1: committed
        try wal.append(r1)
        try wal.append(r2)
        try wal.append(.commit)
        // Group 2: no commit — simulates crash mid-flush
        try wal.append(r3)

        let records = try wal.readAll()
        XCTAssertEqual(records.count, 2, "Uncommitted tail must be discarded")
        XCTAssertEqual(records[0], r1)
        XCTAssertEqual(records[1], r2)
        XCTAssertFalse(records.contains(r3), "Uncommitted record must not be returned")
    }

    func testWALBackwardCompatFallback() throws {
        // Old-format WAL: no .commit records at all.
        // readAll() must return all valid checksummed records (lossless first restart after upgrade).
        let wal = try HNSWTopologyWAL(url: tempDir.appendingPathComponent("compat.wal"))
        let r1 = TopologyWALRecord.nodeDeleted(nodeIndex: 0)
        let r2 = TopologyWALRecord.entryPointChanged(nodeIndex: 1, maxLevel: 0)
        try wal.append(r1)
        try wal.append(r2)

        let records = try wal.readAll()
        XCTAssertEqual(records.count, 2,
            "Backward-compat fallback must return all valid records when no .commit records exist")
        XCTAssertEqual(records[0], r1)
        XCTAssertEqual(records[1], r2)
    }

    func testWALTruncateClearsRecords() throws {
        let wal = try HNSWTopologyWAL(url: tempDir.appendingPathComponent("trunc.wal"))
        try wal.append(.nodeDeleted(nodeIndex: 0))
        XCTAssertFalse(try wal.readAll().isEmpty)
        try wal.truncate()
        XCTAssertTrue(try wal.readAll().isEmpty)
    }

    func testCorruptRecordStopsReplayCleanly() throws {
        let walURL = tempDir.appendingPathComponent("corrupt.wal")
        let wal    = try HNSWTopologyWAL(url: walURL)
        let r1 = TopologyWALRecord.nodeDeleted(nodeIndex: 0)
        let r2 = TopologyWALRecord.nodeDeleted(nodeIndex: 1)
        try wal.append(r1)
        try wal.append(r2)

        var corrupted = try Data(contentsOf: walURL)
        corrupted[corrupted.count - 1] ^= 0xFF
        try corrupted.write(to: walURL)

        let wal2    = try HNSWTopologyWAL(url: walURL)
        let records = try wal2.readAll()
        XCTAssertEqual(records.count, 1, "Replay must stop before the corrupt record")
        XCTAssertEqual(records[0], r1)
    }

    // MARK: - Phase 4: Graph apply() — idempotency

    func testApplyNodeInsertedIsIdempotent() throws {
        var graph = try makeGraph()
        let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 0)
        graph.add(partition: .test(id: "p0", documentId: "d0", embedding: v))
        let records = graph.pendingWALRecords

        var fresh = HNSWGraph()
        fresh.vectorStore = graph.vectorStore
        for r in records { fresh.apply(r) }
        let countAfterFirst = fresh.nodes.count
        for r in records { fresh.apply(r) }  // second replay — must be no-op
        XCTAssertEqual(fresh.nodes.count, countAfterFirst)
        XCTAssertEqual(fresh.nodes.count, 1)
    }

    func testApplyNeighborsUpdatedIsIdempotent() throws {
        var graph = try makeGraph()
        for i in 0..<3 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i))
            graph.add(partition: .test(id: "p\(i)", documentId: "doc\(i)", embedding: v))
        }
        let updateRecords = graph.pendingWALRecords.filter {
            if case .neighborsUpdated = $0 { return true }
            return false
        }
        guard let record = updateRecords.first else { return }
        graph.apply(record)
        guard case .neighborsUpdated(let idx, let layer, _) = record else { return }
        let neighborsBefore = graph.nodes[idx].neighbors[layer]
        graph.apply(record)
        XCTAssertEqual(neighborsBefore, graph.nodes[idx].neighbors[layer])
    }

    // MARK: - Phase 4: WAL round-trip

    func testWALReplayPreservesSearch() throws {
        let storeURL = tempDir.appendingPathComponent("replay-vectors")
        var original = HNSWGraph()
        original.vectorStore = try HNSWVectorStore(url: storeURL, nodeCount: 0)

        var allRecords: [TopologyWALRecord] = []
        var queryVec: [Float] = []
        for i in 0..<12 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i) + 100)
            original.add(partition: .test(id: "p\(i)", documentId: "doc\(i)", embedding: v))
            allRecords.append(contentsOf: original.pendingWALRecords)
            original.pendingWALRecords = []
            if i == 6 { queryVec = v }
        }

        var replayed = HNSWGraph()
        replayed.vectorStore = try HNSWVectorStore(url: storeURL, nodeCount: original.nodes.count)
        for record in allRecords { replayed.apply(record) }

        let (origResults,   _) = original.search(queryEmbedding: queryVec, k: 3)
        let (replayResults, _) = replayed.search(queryEmbedding: queryVec, k: 3)
        XCTAssertEqual(
            origResults.map { $0.partitionId },
            replayResults.map { $0.partitionId },
            "Replayed graph must return the same top-3 results as the original"
        )
    }

    func testCrashBetweenCheckpointAndTruncateIsRecoverable() throws {
        let storeURL = tempDir.appendingPathComponent("crash-vectors")
        var graph = HNSWGraph()
        graph.vectorStore = try HNSWVectorStore(url: storeURL, nodeCount: 0)
        let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 42)
        graph.add(partition: .test(id: "p0", documentId: "doc0", embedding: v))
        let walRecords = graph.pendingWALRecords
        graph.pendingWALRecords = []

        // Checkpoint written (topology has p0), but WAL was NOT truncated.
        let checkpointData = try PropertyListEncoder().encode(graph)
        var recovered = try PropertyListDecoder().decode(HNSWGraph.self, from: checkpointData)
        recovered.vectorStore = try HNSWVectorStore(url: storeURL, nodeCount: recovered.nodes.count)
        for record in walRecords { recovered.apply(record) }

        XCTAssertEqual(recovered.nodes.count, 1,
            "Idempotent apply must not duplicate p0 on crash-between-checkpoint-and-truncate")
        XCTAssertEqual(recovered.partitionLookup["p0"], 0)
    }

    // MARK: - Phase 4: Checkpoint + WAL integration

    func testCompactCheckpointsTopology() throws {
        var (graph, _) = try makeGraph(count: 8)
        let walURL = tempDir.appendingPathComponent("compact.wal")
        let wal    = try HNSWTopologyWAL(url: walURL)
        for r in graph.pendingWALRecords { try wal.append(r) }
        graph.pendingWALRecords = []
        XCTAssertGreaterThan(wal.byteSize, 0)

        graph.remove(documentId: "doc2")
        graph.remove(documentId: "doc5")
        let result = graph.compact()
        XCTAssertGreaterThan(result.removedNodes, 0)

        graph.pendingWALRecords = []
        try wal.truncate()
        XCTAssertEqual(wal.byteSize, 0)
        XCTAssertTrue(try wal.readAll().isEmpty)
    }

    func testEmptyWALReplayEqualsCheckpointLoad() throws {
        let storeURL = tempDir.appendingPathComponent("ckpt-vectors")
        var graph = HNSWGraph()
        graph.vectorStore = try HNSWVectorStore(url: storeURL, nodeCount: 0)
        for i in 0..<6 {
            let v = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: UInt64(i) + 77)
            graph.add(partition: .test(id: "p\(i)", documentId: "doc\(i)", embedding: v))
        }
        graph.pendingWALRecords = []

        let checkpointData = try PropertyListEncoder().encode(graph)
        var loaded = try PropertyListDecoder().decode(HNSWGraph.self, from: checkpointData)
        loaded.vectorStore = try HNSWVectorStore(url: storeURL, nodeCount: loaded.nodes.count)

        let queryVec = VectorFixtures.random(dim: HNSWVectorStore.vectorDim, seed: 77 + 3)
        let (origResults,   _) = graph.search(queryEmbedding: queryVec, k: 3)
        let (loadedResults, _) = loaded.search(queryEmbedding: queryVec, k: 3)
        XCTAssertEqual(origResults.map { $0.partitionId }, loadedResults.map { $0.partitionId })
    }

    // MARK: - Phase 4: WAL performance

    func testWALAppendPerformance() throws {
        let wal    = try HNSWTopologyWAL(url: tempDir.appendingPathComponent("perf.wal"))
        let record = TopologyWALRecord.nodeInserted(
            partitionId: UUID().uuidString, documentId: UUID().uuidString,
            vectorIndex: 0, level: 0, neighborsByLayer: [Array(0..<16)]
        )
        measure { try? wal.append(record) }
    }

    func testWALReplayThroughput_1000Records() throws {
        let wal = try HNSWTopologyWAL(url: tempDir.appendingPathComponent("throughput.wal"))
        for i in 0..<1000 {
            try wal.append(.neighborsUpdated(nodeIndex: i % 100, layer: 0, neighbors: Array(0..<16)))
        }
        measure { _ = try? wal.readAll() }
    }
}
