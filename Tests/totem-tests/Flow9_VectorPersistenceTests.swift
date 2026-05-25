//
//  Flow9_VectorPersistenceTests.swift
//  database-serverTests
//
//  Tests for the Phase 3 mmap-based vector persistence lifecycle:
//    put (with live HNSWVectorStore) → encode topology → decode topology
//    → reattach HNSWVectorStore → search gives identical results.
//
//  Phase 3 replaced the in-memory `vectorBuffer: Data` + `reattachVectors(from:)` path
//  (Phase 1/2) with a memory-mapped file managed by `HNSWVectorStore`. These tests guard
//  against regressions in that design.
//
//  Each test creates a throwaway temp directory that is deleted in `tearDownWithError`.
//

import XCTest
@testable import totem

final class Flow9_VectorPersistenceTests: XCTestCase {

    private let dim = HNSWVectorStore.vectorDim  // 1024

    // Temp directory for vector store files. One directory per test, cleaned up after.
    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("database-flow9-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    // Convenience: create an HNSWVectorStore backed by a temp file inside tempDir.
    private func makeStore(name: String = "vectors", nodeCount: Int = 0) throws -> HNSWVectorStore {
        try HNSWVectorStore(url: tempDir.appendingPathComponent(name), nodeCount: nodeCount)
    }

    // Convenience: build a graph with a live vector store and insert `count` nodes.
    private func makeGraph(count: Int, seed: UInt64 = 0) throws -> (HNSWGraph, [([Float])]) {
        var graph = HNSWGraph()
        graph.vectorStore = try makeStore()
        var vecs: [[Float]] = []
        for i in 0..<count {
            let v = VectorFixtures.random(dim: dim, seed: seed + UInt64(i))
            let p = Database.Partition.test(id: "p\(i)", documentId: "doc\(i)", embedding: v)
            graph.add(partition: p)
            vecs.append(v)
        }
        return (graph, vecs)
    }

    // MARK: - Topology encoding

    func testEmbeddingsNotStoredInTopology() throws {
        let (graph, _) = try makeGraph(count: 5)

        let data     = try PropertyListEncoder().encode(graph)
        let restored = try PropertyListDecoder().decode(HNSWGraph.self, from: data)

        // vectorStore is not Codable — it must be nil after decode.
        XCTAssertNil(restored.vectorStore,
            "vectorStore must be absent from the topology PropertyList — it lives only in the mmap'd vector file")

        // Node count and structure must survive.
        XCTAssertEqual(restored.nodes.count, graph.nodes.count,
            "Node count must be preserved in the topology file")
        XCTAssertEqual(restored.entryPoint, graph.entryPoint,
            "Entry point must be preserved")
        XCTAssertEqual(restored.partitionLookup.count, graph.partitionLookup.count,
            "partitionLookup must be preserved")
    }

    func testTopologyPreservesGraphStructure() throws {
        let (graph, _) = try makeGraph(count: 8, seed: 100)

        let data     = try PropertyListEncoder().encode(graph)
        let restored = try PropertyListDecoder().decode(HNSWGraph.self, from: data)

        XCTAssertEqual(restored.nodes.count, graph.nodes.count)
        XCTAssertEqual(restored.entryPoint,  graph.entryPoint)
        XCTAssertEqual(restored.maxLevel,    graph.maxLevel)
        for i in restored.nodes.indices {
            XCTAssertEqual(restored.nodes[i].partitionId,  graph.nodes[i].partitionId)
            XCTAssertEqual(restored.nodes[i].documentId,   graph.nodes[i].documentId)
            XCTAssertEqual(restored.nodes[i].vectorIndex,  graph.nodes[i].vectorIndex)
            XCTAssertEqual(restored.nodes[i].neighbors,    graph.nodes[i].neighbors)
            XCTAssertEqual(restored.nodes[i].isDeleted,    graph.nodes[i].isDeleted)
        }
    }

    // MARK: - Phase 3 vector store lifecycle

    func testVectorStoreAppendedDuringInsertion() throws {
        var graph = HNSWGraph()
        let store = try makeStore()
        graph.vectorStore = store

        let v = VectorFixtures.random(dim: dim, seed: 42)
        let p = Database.Partition.test(id: "p0", documentId: "doc0", embedding: v)
        graph.add(partition: p)

        XCTAssertEqual(store.nodeCount, 1,
            "vectorStore.nodeCount must equal 1 after one insertion")
        XCTAssertEqual(graph.nodes.count, 1,
            "graph.nodes must have exactly one entry after insertion")
    }

    func testVectorStoreContainsCorrectFloatValues() throws {
        var graph = HNSWGraph()
        let store = try makeStore()
        graph.vectorStore = store

        let known = VectorFixtures.random(dim: dim, seed: 7)
        let p = Database.Partition.test(id: "p0", documentId: "doc0", embedding: known)
        graph.add(partition: p)

        let restored = graph.readEmbedding(at: 0)
        XCTAssertEqual(restored.count, dim,
            "readEmbedding must return a full-dim vector")
        for (i, (r, o)) in zip(restored, known).enumerated() {
            XCTAssertEqual(r, o, accuracy: 1e-6,
                "Float at index \(i) must survive the mmap round-trip without loss")
        }
    }

    func testMultipleNodesHaveDistinctVectorIndices() throws {
        let (graph, _) = try makeGraph(count: 10, seed: 200)

        let indices = graph.nodes.map { $0.vectorIndex }
        let unique  = Set(indices)
        XCTAssertEqual(unique.count, indices.count,
            "Every node must have a unique vectorIndex")
        XCTAssertEqual(Set(0..<graph.nodes.count), unique,
            "vectorIndex values must be dense 0..<nodeCount")
    }

    // MARK: - Full round-trip: insert → encode topology → decode → reattach store → search

    func testFullRoundTripPreservesNearestNeighbor() throws {
        let count = 10
        let (graph, vecs) = try makeGraph(count: count, seed: 300)
        XCTAssertTrue(graph.isTrained, "Precondition: graph must be trained")

        // Encode topology (no embeddings inside).
        let topologyData = try PropertyListEncoder().encode(graph)
        var restored     = try PropertyListDecoder().decode(HNSWGraph.self, from: topologyData)
        XCTAssertNil(restored.vectorStore, "vectorStore must be nil after topology decode")
        XCTAssertFalse(restored.isTrained == false,
            "entryPoint must survive topology round-trip")  // isTrained == true

        // Reattach the existing vector file (wasCreatedFresh == false, data valid).
        let store2 = try HNSWVectorStore(
            url:       tempDir.appendingPathComponent("vectors"),
            nodeCount: restored.nodes.count
        )
        XCTAssertFalse(store2.wasCreatedFresh,
            "Reopening an existing vector file must not set wasCreatedFresh")
        restored.vectorStore = store2

        // Search with the exact vector for node "p3".
        let (results, _) = restored.search(queryEmbedding: vecs[3], k: 1)
        XCTAssertEqual(results.first?.partitionId, "p3",
            "After vector store reattachment, search must return the correct nearest neighbor")
    }

    func testFullRoundTripPreservesTopKOrder() throws {
        let count = 20
        let (graph, vecs) = try makeGraph(count: count, seed: 400)
        let topologyData  = try PropertyListEncoder().encode(graph)
        var restored      = try PropertyListDecoder().decode(HNSWGraph.self, from: topologyData)
        restored.vectorStore = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("vectors"),
            nodeCount: restored.nodes.count
        )

        let (results, _) = restored.search(queryEmbedding: vecs[7], k: 5)
        XCTAssertGreaterThan(results.count, 0, "Search must return results after round-trip")
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(
                results[i - 1].distance, results[i].distance,
                "Results must be sorted ascending by distance after vector store reattachment"
            )
        }
    }

    func testRoundTripAfterDeletion() throws {
        let (graph, vecs) = try makeGraph(count: 8, seed: 500)
        var g = graph

        g.remove(documentId: "doc0")
        g.remove(documentId: "doc1")

        let topologyData = try PropertyListEncoder().encode(g)
        var restored     = try PropertyListDecoder().decode(HNSWGraph.self, from: topologyData)
        restored.vectorStore = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("vectors"),
            nodeCount: restored.nodes.count
        )

        let (results, _) = restored.search(queryEmbedding: vecs[5], k: 5)
        let ids = results.map(\.partitionId)
        XCTAssertFalse(ids.contains("p0"), "Deleted partition must not appear in results")
        XCTAssertFalse(ids.contains("p1"), "Deleted partition must not appear in results")
    }

    // MARK: - wasCreatedFresh guard (Phase 3 migration safety)

    func testWasCreatedFreshTrueForNewFile() throws {
        let store = try makeStore(name: "fresh-vectors", nodeCount: 0)
        XCTAssertTrue(store.wasCreatedFresh,
            "A newly created vector file must set wasCreatedFresh = true")
    }

    func testWasCreatedFreshFalseForExistingFile() throws {
        // Write something to the file first, then reopen.
        let url = tempDir.appendingPathComponent("existing-vectors")
        let s1  = try HNSWVectorStore(url: url, nodeCount: 0)
        let p   = Database.Partition.test(id: "p0", documentId: "d0",
                                      embedding: VectorFixtures.random(dim: dim, seed: 1))
        var g = HNSWGraph()
        g.vectorStore = s1
        g.add(partition: p)

        // Reopen the same file — must NOT be fresh.
        let s2 = try HNSWVectorStore(url: url, nodeCount: 1)
        XCTAssertFalse(s2.wasCreatedFresh,
            "Reopening an existing non-empty vector file must not set wasCreatedFresh")
    }

    func testIsValidForCorrectNodeCount() throws {
        let (graph, _) = try makeGraph(count: 4, seed: 600)
        let store = graph.vectorStore!

        XCTAssertTrue(store.isValidFor(nodeCount: 4),
            "isValidFor must return true when the file covers all nodes")
        XCTAssertTrue(store.isValidFor(nodeCount: 0),
            "isValidFor must return true for nodeCount 0")
        XCTAssertFalse(store.isValidFor(nodeCount: 1_000_000),
            "isValidFor must return false when nodeCount exceeds the file's capacity")
    }

    // MARK: - Topology encode performance (Phase 4 baseline)
    //
    // These tests establish the PropertyList encode cost at various node counts.
    // Phase 4 (WAL) will replace the per-second full encode with WAL appends;
    // the numbers below are the improvement target.

    func testTopologyEncodePerformance_10Nodes() throws {
        var table = PartitionTable()
        let store = try makeStore(name: "bench-10")
        table.shards[0].vectorStore = store
        for i in 0..<10 {
            let p = Database.Partition.test(
                id: "p\(i)", documentId: "doc\(i)",
                embedding: VectorFixtures.random(dim: dim, seed: UInt64(i))
            )
            table.put(id: "doc\(i)", partitions: [p], request: .test(), logger: .test)
        }

        measure {
            _ = try! PropertyListEncoder().encode(table)
        }
    }

    func testTopologyEncodePerformance_100Nodes() throws {
        var table = PartitionTable()
        let store = try makeStore(name: "bench-100")
        table.shards[0].vectorStore = store
        for i in 0..<100 {
            let p = Database.Partition.test(
                id: "p\(i)", documentId: "doc\(i)",
                embedding: VectorFixtures.random(dim: dim, seed: UInt64(i + 1000))
            )
            table.put(id: "doc\(i)", partitions: [p], request: .test(), logger: .test)
        }

        measure {
            _ = try! PropertyListEncoder().encode(table)
        }
    }

    func testVectorStoreAppendThroughput() throws {
        // Measures raw mmap append speed — the Phase 4 WAL record write will be
        // comparable in size (~1-2 KB) but even cheaper (sequential, no mmap remap).
        let store = try makeStore(name: "throughput", nodeCount: 0)
        let v     = VectorFixtures.random(dim: dim, seed: 99)

        measure {
            store.append(embedding: v)
        }
    }
}
