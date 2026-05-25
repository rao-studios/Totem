//
//  Flow3_HNSWSearchPredicateTests.swift
//  database-serverTests
//
//  Tests for HNSWGraph.search(queryEmbedding:k:nodeFilter:).
//
//  Strategy: build small graphs with documents assigned to named groups, then
//  verify that a nodeFilter predicate confines results to those documents
//  while still finding the geometrically nearest neighbours within the allowed set.
//

import XCTest
@testable import totem

final class Flow3_HNSWSearchPredicateTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("database-pred-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Helpers

    private func makeGraph() throws -> HNSWGraph {
        var graph = HNSWGraph()
        graph.vectorStore = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("vec-\(UUID().uuidString)"),
            nodeCount: 0
        )
        return graph
    }

    private func dim() -> Int { HNSWVectorStore.vectorDim }

    // MARK: - nil filter (backward compatibility)

    func testNilFilterReturnsAllLiveResults() throws {
        var graph = try makeGraph()
        for i in 0..<20 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i))
            graph.add(partition: .test(id: "p\(i)", documentId: "doc\(i)", embedding: v))
        }
        let query = VectorFixtures.random(dim: dim(), seed: 999)
        let (withNil, _)  = graph.search(queryEmbedding: query, k: 5, nodeFilter: nil)
        let (noParam, _)  = graph.search(queryEmbedding: query, k: 5)
        XCTAssertEqual(withNil.map(\.partitionId).sorted(),
                       noParam.map(\.partitionId).sorted(),
                       "nil filter must be identical to calling without filter parameter")
    }

    // MARK: - Basic filtering

    func testFilterConfinesResultsToAllowedDocumentIds() throws {
        var graph = try makeGraph()
        // Two groups of 15 docs each. groupA uses even seeds, groupB odd.
        var groupA = Set<String>()
        var groupB = Set<String>()
        for i in 0..<30 {
            let docId = "doc\(i)"
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i * 7 + 1))
            graph.add(partition: .test(id: "p\(i)", documentId: docId, embedding: v))
            if i % 2 == 0 { groupA.insert(docId) } else { groupB.insert(docId) }
        }
        let query = VectorFixtures.random(dim: dim(), seed: 42)
        let (results, _) = graph.search(queryEmbedding: query, k: 10, nodeFilter: { groupA.contains($0) })

        XCTAssertFalse(results.isEmpty, "filter search must return results when allowed set is non-empty")
        for r in results {
            XCTAssertTrue(groupA.contains(r.documentId),
                "result \(r.documentId) must belong to the allowed set")
        }
    }

    func testFilterReturnsNothingForEmptyAllowedSet() throws {
        var graph = try makeGraph()
        for i in 0..<20 {
            graph.add(partition: .test(
                id: "p\(i)", documentId: "doc\(i)",
                embedding: VectorFixtures.random(dim: dim(), seed: UInt64(i))
            ))
        }
        let (results, _) = graph.search(
            queryEmbedding: VectorFixtures.random(dim: dim(), seed: 1),
            k: 5, nodeFilter: { _ in false }
        )
        XCTAssertTrue(results.isEmpty, "empty allowed set must produce no results")
    }

    func testFilterHonoursKBound() throws {
        var graph = try makeGraph()
        var allowed = Set<String>()
        for i in 0..<30 {
            let docId = "doc\(i)"
            graph.add(partition: .test(
                id: "p\(i)", documentId: docId,
                embedding: VectorFixtures.random(dim: dim(), seed: UInt64(i + 10))
            ))
            allowed.insert(docId)
        }
        let (results, _) = graph.search(
            queryEmbedding: VectorFixtures.random(dim: dim(), seed: 77),
            k: 5, nodeFilter: { allowed.contains($0) }
        )
        XCTAssertLessThanOrEqual(results.count, 5, "filter search must respect the k limit")
    }

    func testFilterResultsSortedByDistanceAscending() throws {
        var graph = try makeGraph()
        var allowed = Set<String>()
        for i in 0..<30 {
            let docId = "doc\(i)"
            graph.add(partition: .test(
                id: "p\(i)", documentId: docId,
                embedding: VectorFixtures.random(dim: dim(), seed: UInt64(i + 50))
            ))
            if i % 3 != 0 { allowed.insert(docId) }
        }
        let (results, _) = graph.search(
            queryEmbedding: VectorFixtures.random(dim: dim(), seed: 123),
            k: 8, nodeFilter: { allowed.contains($0) }
        )
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(results[i - 1].distance, results[i].distance,
                "filtered results must be sorted ascending by distance")
        }
    }

    // MARK: - Nearest-neighbor correctness under filter

    func testFilterFindsNearestNeighbourWithinAllowedSet() throws {
        // Place one document very close to the query and exclude it via the filter.
        // The filter must return the nearest ALLOWED document instead.
        var graph = try makeGraph()
        let queryVec = VectorFixtures.unit(axis: 0).map { $0 * 0.5 }.padded(to: dim())
        let nearVec  = queryVec  // identical — but excluded from filter
        let farVec   = VectorFixtures.unit(axis: 1).padded(to: dim())

        graph.add(partition: .test(id: "near", documentId: "docNear", embedding: nearVec))
        graph.add(partition: .test(id: "far",  documentId: "docFar",  embedding: farVec))
        // pad with noise so the graph has more nodes and beam search actually runs
        for i in 0..<20 {
            graph.add(partition: .test(
                id: "noise\(i)", documentId: "docNoise\(i)",
                embedding: VectorFixtures.random(dim: dim(), seed: UInt64(i + 100))
            ))
        }

        let groupFilter: Set<String> = ["docFar"]
        let (results, _) = graph.search(queryEmbedding: queryVec, k: 1, nodeFilter: { groupFilter.contains($0) })
        XCTAssertEqual(results.first?.documentId, "docFar",
            "filter must return the nearest ALLOWED document, not the globally nearest")
    }

    func testFilterReturnsNearestAllowedAmongMany() throws {
        // Insert 50 docs. The query vector is a perturbed copy of doc42's vector.
        // Filter allows only docs 40–49; doc42 must top the results.
        var graph = try makeGraph()
        var vecs = [String: [Float]]()
        for i in 0..<50 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 200))
            vecs["doc\(i)"] = v
            graph.add(partition: .test(id: "p\(i)", documentId: "doc\(i)", embedding: v))
        }
        let query  = VectorFixtures.near(vecs["doc42"]!, seed: 7)
        let groupFilter = Set((40..<50).map { "doc\($0)" })
        let (results, _) = graph.search(queryEmbedding: query, k: 3, nodeFilter: { groupFilter.contains($0) })

        XCTAssertFalse(results.isEmpty)
        XCTAssertEqual(results.first?.documentId, "doc42",
            "nearest allowed document must be doc42 when query is perturbed from its vector")
        for r in results {
            XCTAssertTrue(groupFilter.contains(r.documentId))
        }
    }

    // MARK: - Unfiltered nodes as bridges

    func testFilterFindsResultsThroughUnfilteredBridgeNodes() throws {
        // Build a graph where the globally nearest node is excluded from the filter,
        // and the allowed node is reachable only through that excluded node.
        // The algorithm must still find the allowed node.
        var graph = try makeGraph()
        let base = VectorFixtures.unit(axis: 0).padded(to: dim())

        // Insert 20 "bridge" nodes near the query to establish dense connectivity.
        for i in 0..<20 {
            let v = VectorFixtures.near(base, seed: UInt64(i + 1)).padded(to: dim())
            graph.add(partition: .test(id: "bridge\(i)", documentId: "bridgeDoc\(i)", embedding: v))
        }
        // Insert the target allowed node near the bridges.
        let targetVec = VectorFixtures.near(base, seed: 999).padded(to: dim())
        graph.add(partition: .test(id: "target", documentId: "targetDoc", embedding: targetVec))

        let groupFilter: Set<String> = ["targetDoc"]
        let (results, _) = graph.search(queryEmbedding: base, k: 1, nodeFilter: { groupFilter.contains($0) })

        XCTAssertEqual(results.first?.documentId, "targetDoc",
            "HNSW must traverse bridge nodes to find the only allowed document")
    }

    // MARK: - Interaction with deletion

    func testFilterExcludesDeletedDocumentsEvenIfInAllowedSet() throws {
        var graph = try makeGraph()
        let target = [Float](repeating: 0, count: dim())
        graph.add(partition: .test(id: "target", documentId: "docTarget", embedding: target))
        for i in 0..<15 {
            graph.add(partition: .test(
                id: "p\(i)", documentId: "d\(i)",
                embedding: VectorFixtures.random(dim: dim(), seed: UInt64(i + 300))
            ))
        }
        // Delete docTarget, then search with a filter that includes it.
        graph.remove(documentId: "docTarget")
        let groupFilter: Set<String> = ["docTarget"]
        let (results, _) = graph.search(queryEmbedding: target, k: 3, nodeFilter: { groupFilter.contains($0) })

        XCTAssertFalse(results.contains(where: { $0.documentId == "docTarget" }),
            "deleted documents must not appear even if they are in the allowed set")
    }

    func testFilterWithMixedDeletedAndLiveDocuments() throws {
        var graph = try makeGraph()
        var allowed = Set<String>()
        for i in 0..<20 {
            let docId = "doc\(i)"
            graph.add(partition: .test(
                id: "p\(i)", documentId: docId,
                embedding: VectorFixtures.random(dim: dim(), seed: UInt64(i + 400))
            ))
            allowed.insert(docId)
        }
        // Delete half the allowed set.
        for i in 0..<10 { graph.remove(documentId: "doc\(i)") }
        let deleted = Set((0..<10).map { "doc\($0)" })

        let (results, _) = graph.search(
            queryEmbedding: VectorFixtures.random(dim: dim(), seed: 55),
            k: 10, nodeFilter: { allowed.contains($0) }
        )
        for r in results {
            XCTAssertFalse(deleted.contains(r.documentId),
                "deleted docs must not appear in filtered results")
            XCTAssertTrue(allowed.contains(r.documentId),
                "results must still be within the allowed set")
        }
    }

    // MARK: - Single-document filter

    func testSingleDocumentFilterReturnsThatDocumentWhenItExists() throws {
        var graph = try makeGraph()
        for i in 0..<20 {
            graph.add(partition: .test(
                id: "p\(i)", documentId: "doc\(i)",
                embedding: VectorFixtures.random(dim: dim(), seed: UInt64(i + 500))
            ))
        }
        let groupFilter: Set<String> = ["doc7"]
        let (results, _) = graph.search(
            queryEmbedding: VectorFixtures.random(dim: dim(), seed: 11),
            k: 5, nodeFilter: { groupFilter.contains($0) }
        )
        XCTAssertEqual(results.count, 1, "single-document filter must return at most 1 result")
        XCTAssertEqual(results.first?.documentId, "doc7")
    }
}

// MARK: - Array padding helper

private extension Array where Element == Float {
    /// Right-pad (or truncate) to `length`, filling with zeros.
    func padded(to length: Int) -> [Float] {
        if count >= length { return Array(prefix(length)) }
        return self + [Float](repeating: 0, count: length - count)
    }
}
