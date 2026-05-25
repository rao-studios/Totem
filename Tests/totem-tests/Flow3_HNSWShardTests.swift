//
//  Flow3_HNSWShardTests.swift
//  database-serverTests
//
//  Tests for HNSWShard: transparent Codable, forwarded properties,
//  two-tier search (tag pre-filter + HNSW slot resolution), and Sinatra side effects.
//

import XCTest
@testable import totem

final class Flow3_HNSWShardTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("database-gpt-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Helpers

    private func dim() -> Int { HNSWVectorStore.vectorDim }

    private func makeTable() throws -> HNSWShard {
        var table = HNSWShard()
        table.vectorStore = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("vec-\(UUID().uuidString)"),
            nodeCount: 0
        )
        return table
    }

    private func makeIndex(
        partitions: [Database.Partition],
        docId: String,
        tags: [String] = [],
        tagsEmbedding: [Float]? = nil
    ) -> PartitionIndex {
        var index = PartitionIndex()
        index.train(partitions, tags: tags, tagsEmbedding: tagsEmbedding, documentId: docId, logger: .test)
        return index
    }

    private func callSearch(
        table: inout HNSWShard,
        queryEmbedding: [Float],
        queryTagEmbedding: [Float]? = nil,
        k: Int = 10,
        groupFilter: Set<DocumentID>? = nil,
        indices: [DocumentID: PartitionIndex],
        sinatra: Sinatra,
        ownerId: String = "test-owner",
        registry: TotemRegistry = TotemRegistry()
    ) -> [(partition: Database.Partition, distance: Float)] {
        table.indices = indices
        let (results, _) = table.search(
            shardIndex: 0,
            queryEmbedding: queryEmbedding,
            queryTagEmbedding: queryTagEmbedding,
            k: k,
            groupFilter: groupFilter,
            sinatra: sinatra,
            ownerKey: TotemRegistry.Owner(id: ownerId),
            sinatraRegistry: sinatra.registry,
            registry: registry,
            metadataLoader: nil,
            request: .test(ownerId: ownerId),
            logger: .test
        )
        return results
    }

    // MARK: - Transparent Codable

    func testEncodesAsHNSWGraphFormat() throws {
        var table = try makeTable()
        for i in 0..<5 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i))
            table.add(partition: .test(id: "p\(i)", documentId: "doc\(i)", embedding: v))
        }
        let data = try PropertyListEncoder().encode(table)
        let graph = try PropertyListDecoder().decode(HNSWGraph.self, from: data)
        XCTAssertEqual(graph.totalInsertions, table.totalInsertions,
            "Decoded HNSWGraph must have same totalInsertions as HNSWShard")
        XCTAssertEqual(graph.nodes.count, table.nodes.count,
            "Decoded HNSWGraph must have same node count")
    }

    func testDecodesFromHNSWGraphFormat() throws {
        var graph = HNSWGraph()
        graph.vectorStore = try HNSWVectorStore(
            url: tempDir.appendingPathComponent("vec-decode-test"),
            nodeCount: 0
        )
        for i in 0..<5 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 100))
            graph.add(partition: .test(id: "p\(i)", documentId: "doc\(i)", embedding: v))
        }
        let data = try PropertyListEncoder().encode(graph)
        let table = try PropertyListDecoder().decode(HNSWShard.self, from: data)
        XCTAssertEqual(table.isTrained, graph.isTrained)
        XCTAssertEqual(table.totalInsertions, graph.totalInsertions)
        XCTAssertEqual(table.nodes.count, graph.nodes.count)
    }

    // MARK: - Forwarded properties

    func testForwardedPropertiesMatchUnderlyingGraph() throws {
        var table = try makeTable()
        for i in 0..<10 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 50))
            table.add(partition: .test(id: "p\(i)", documentId: "doc\(i)", embedding: v))
        }
        XCTAssertEqual(table.isTrained, table.graph.isTrained)
        XCTAssertEqual(table.totalInsertions, table.graph.totalInsertions)
        XCTAssertEqual(table.nodes.count, table.graph.nodes.count)
        XCTAssertEqual(table.effectiveThreshold, table.graph.effectiveThreshold)
        XCTAssertEqual(table.graphStats.liveNodes, table.graph.graphStats.liveNodes)
        XCTAssertEqual(table.isEmpty, table.graph.isEmpty)
    }

    // MARK: - Tier-1 tag pre-filter: untagged documents always pass

    func testUntaggedDocumentsAlwaysPass() throws {
        var table = try makeTable()
        var indices: [DocumentID: PartitionIndex] = [:]

        var partitions: [Database.Partition] = []
        for i in 0..<5 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 10))
            let p = Database.Partition.test(id: "p\(i)", documentId: "doc\(i)", embedding: v)
            table.add(partition: p)
            partitions.append(p)
        }
        for i in 0..<5 {
            indices["doc\(i)"] = makeIndex(partitions: [partitions[i]], docId: "doc\(i)")
        }

        let sinatra = Sinatra(logger: .test)
        let results = callSearch(
            table: &table,
            queryEmbedding: VectorFixtures.random(dim: dim(), seed: 999),
            queryTagEmbedding: VectorFixtures.random(dim: dim(), seed: 111),
            indices: indices,
            sinatra: sinatra
        )

        XCTAssertFalse(results.isEmpty, "Untagged documents must not be excluded by the tag filter")
    }

    // MARK: - Tier-1: tagged doc excluded when tag embedding is far

    func testTaggedDocumentExcludedWhenTagEmbeddingFar() throws {
        var table = try makeTable()

        // Orthogonal unit vectors → tagDistance = 1.0 − 0.0 = 1.0 > threshold (0.85) → excluded
        let queryTag = [Float](repeating: 0, count: dim())
            .with(index: 0, value: 1.0)
        let tagsEmb  = [Float](repeating: 0, count: dim())
            .with(index: 1, value: 1.0)

        let vTagged = VectorFixtures.random(dim: dim(), seed: 1)
        let pTagged = Database.Partition.test(id: "tagged", documentId: "docTagged", embedding: vTagged)
        table.add(partition: pTagged)

        // Untagged noise docs — must still pass through
        var partitions: [Database.Partition] = []
        for i in 0..<8 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 20))
            let p = Database.Partition.test(id: "u\(i)", documentId: "docU\(i)", embedding: v)
            table.add(partition: p)
            partitions.append(p)
        }

        var indices: [DocumentID: PartitionIndex] = [:]
        indices["docTagged"] = makeIndex(partitions: [pTagged], docId: "docTagged",
                                         tags: ["swift"], tagsEmbedding: tagsEmb)
        for i in 0..<8 {
            indices["docU\(i)"] = makeIndex(partitions: [partitions[i]], docId: "docU\(i)")
        }

        let sinatra = Sinatra(logger: .test)
        let results = callSearch(
            table: &table,
            queryEmbedding: vTagged,
            queryTagEmbedding: queryTag,
            indices: indices,
            sinatra: sinatra
        )

        XCTAssertFalse(results.isEmpty, "Untagged documents must still appear in results")
        XCTAssertTrue(
            results.allSatisfy { $0.partition.documentId != "docTagged" },
            "Tagged doc must be excluded when its tag embedding is orthogonal to the query tag"
        )
    }

    // MARK: - Tier-1: tagged doc included when tag embedding is close

    func testTaggedDocumentIncludedWhenTagEmbeddingClose() throws {
        var table = try makeTable()

        // Identical tag vectors → tagDistance = 1.0 − 1.0 = 0.0 < threshold → included
        let tagVec = [Float](repeating: 0, count: dim()).with(index: 0, value: 1.0)

        let vTagged = VectorFixtures.random(dim: dim(), seed: 2)
        let pTagged = Database.Partition.test(id: "tagged", documentId: "docTagged", embedding: vTagged)
        table.add(partition: pTagged)

        for i in 0..<10 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 50))
            table.add(partition: .test(id: "n\(i)", documentId: "docN\(i)", embedding: v))
        }

        var indices: [DocumentID: PartitionIndex] = [:]
        indices["docTagged"] = makeIndex(partitions: [pTagged], docId: "docTagged",
                                         tags: ["swift"], tagsEmbedding: tagVec)
        for i in 0..<10 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 50))
            let p = Database.Partition.test(id: "n\(i)", documentId: "docN\(i)", embedding: v)
            indices["docN\(i)"] = makeIndex(partitions: [p], docId: "docN\(i)")
        }

        let sinatra = Sinatra(logger: .test)
        let results = callSearch(
            table: &table,
            queryEmbedding: vTagged,
            queryTagEmbedding: tagVec,
            k: 5,
            groupFilter: Set(["docTagged"]),
            indices: indices,
            sinatra: sinatra
        )

        XCTAssertTrue(
            results.contains(where: { $0.partition.documentId == "docTagged" }),
            "Tagged doc must appear in results when its tag embedding matches the query tag"
        )
    }

    // MARK: - Tier-1: nil queryTagEmbedding falls back to queryEmbedding

    func testNilQueryTagEmbeddingSkipsTagFilter() throws {
        var table = try makeTable()
        let query = VectorFixtures.random(dim: dim(), seed: 42)

        let p = Database.Partition.test(id: "pt", documentId: "docT", embedding: query)
        table.add(partition: p)
        for i in 0..<8 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 80))
            table.add(partition: .test(id: "n\(i)", documentId: "docN\(i)", embedding: v))
        }

        // nil queryTagEmbedding skips the tag filter entirely — tagged documents are not excluded.
        var indices: [DocumentID: PartitionIndex] = [:]
        indices["docT"] = makeIndex(partitions: [p], docId: "docT",
                                    tags: ["x"], tagsEmbedding: query)
        for i in 0..<8 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 80))
            let np = Database.Partition.test(id: "n\(i)", documentId: "docN\(i)", embedding: v)
            indices["docN\(i)"] = makeIndex(partitions: [np], docId: "docN\(i)")
        }

        let sinatra = Sinatra(logger: .test)
        var tableCopy = table
        let withNil = callSearch(
            table: &table,
            queryEmbedding: query,
            queryTagEmbedding: nil,
            indices: indices,
            sinatra: sinatra
        )
        let withExplicit = callSearch(
            table: &tableCopy,
            queryEmbedding: query,
            queryTagEmbedding: query,
            indices: indices,
            sinatra: sinatra
        )

        XCTAssertEqual(
            Set(withNil.map { $0.partition.id }),
            Set(withExplicit.map { $0.partition.id }),
            "nil queryTagEmbedding must skip tag filtering, producing the same candidate set as an explicit matching vector"
        )
    }

    // MARK: - Tier-1 + Tier-2 intersection

    func testGroupFilterIntersectsWithTagFilter() throws {
        var table = try makeTable()
        let tagVec = [Float](repeating: 0, count: dim()).with(index: 0, value: 1.0)
        let farTag  = [Float](repeating: 0, count: dim()).with(index: 1, value: 1.0)

        // docs 0-4:  in groupFilter, close tagsEmbedding → pass both
        // docs 5-9:  not in groupFilter, close tagsEmbedding → pass tag but not group
        // docs 10-14: in groupFilter, far tagsEmbedding → fail tag filter
        var groupFilter = Set<DocumentID>()
        var indices: [DocumentID: PartitionIndex] = [:]
        for i in 0..<15 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 300))
            let p = Database.Partition.test(id: "p\(i)", documentId: "doc\(i)", embedding: v)
            table.add(partition: p)
            if i < 5 {
                groupFilter.insert("doc\(i)")
                indices["doc\(i)"] = makeIndex(partitions: [p], docId: "doc\(i)",
                                               tags: ["swift"], tagsEmbedding: tagVec)
            } else if i < 10 {
                indices["doc\(i)"] = makeIndex(partitions: [p], docId: "doc\(i)",
                                               tags: ["swift"], tagsEmbedding: tagVec)
            } else {
                groupFilter.insert("doc\(i)")
                indices["doc\(i)"] = makeIndex(partitions: [p], docId: "doc\(i)",
                                               tags: ["swift"], tagsEmbedding: farTag)
            }
        }

        let sinatra = Sinatra(logger: .test)
        let results = callSearch(
            table: &table,
            queryEmbedding: VectorFixtures.random(dim: dim(), seed: 77),
            queryTagEmbedding: tagVec,
            k: 15,
            groupFilter: groupFilter,
            indices: indices,
            sinatra: sinatra
        )

        for r in results {
            XCTAssertTrue(groupFilter.contains(r.partition.documentId),
                "Result \(r.partition.documentId) must be within groupFilter")
        }
        // docs 10-14 fail the tag filter; docs 5-9 fail the group filter
        let forbiddenDocs = Set(["doc5","doc6","doc7","doc8","doc9",
                                 "doc10","doc11","doc12","doc13","doc14"])
        for r in results {
            XCTAssertFalse(forbiddenDocs.contains(r.partition.documentId),
                "Doc outside intersection must not appear: \(r.partition.documentId)")
        }
    }

    // MARK: - Tier-2 slot resolution

    func testSearchResolvesPartitionSlotsFromIndices() throws {
        var table = try makeTable()
        var indices: [DocumentID: PartitionIndex] = [:]

        var partitions: [Database.Partition] = []
        for i in 0..<5 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 200))
            let p = Database.Partition.test(id: "p\(i)", documentId: "doc\(i)", embedding: v)
            table.add(partition: p)
            partitions.append(p)
            indices["doc\(i)"] = makeIndex(partitions: [p], docId: "doc\(i)")
        }

        let sinatra = Sinatra(logger: .test)
        let results = callSearch(
            table: &table,
            queryEmbedding: VectorFixtures.random(dim: dim(), seed: 1),
            indices: indices,
            sinatra: sinatra
        )

        let insertedIds = Set(partitions.map(\.id))
        XCTAssertFalse(results.isEmpty, "search must return resolved partitions")
        for r in results {
            XCTAssertTrue(insertedIds.contains(r.partition.id),
                "Returned partition \(r.partition.id) must be one we inserted")
        }
    }

    func testSearchDeduplicatesByPartitionId() throws {
        var table = try makeTable()
        let query    = VectorFixtures.random(dim: dim(), seed: 10)
        let vClose   = VectorFixtures.near(query, seed: 1)
        let vFar     = VectorFixtures.random(dim: dim(), seed: 500)

        let pClose = Database.Partition.test(id: "dup", documentId: "docDup", embedding: vClose)
        let pFar   = Database.Partition.test(id: "dup", documentId: "docDup", embedding: vFar)

        table.add(partition: pClose)
        table.addWithoutUpsert(pFar)

        for i in 0..<10 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 60))
            table.add(partition: .test(id: "n\(i)", documentId: "docN\(i)", embedding: v))
        }

        var indices: [DocumentID: PartitionIndex] = [:]
        indices["docDup"] = makeIndex(partitions: [pClose], docId: "docDup")
        for i in 0..<10 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 60))
            let np = Database.Partition.test(id: "n\(i)", documentId: "docN\(i)", embedding: v)
            indices["docN\(i)"] = makeIndex(partitions: [np], docId: "docN\(i)")
        }

        let sinatra = Sinatra(logger: .test)
        let results = callSearch(
            table: &table,
            queryEmbedding: query,
            indices: indices,
            sinatra: sinatra
        )

        let dupResults = results.filter { $0.partition.id == "dup" }
        XCTAssertLessThanOrEqual(dupResults.count, 1,
            "Partition 'dup' must appear at most once even when inserted twice via addWithoutUpsert")
    }

    func testSearchResultsSortedByDistanceAscending() throws {
        var table = try makeTable()
        var indices: [DocumentID: PartitionIndex] = [:]

        for i in 0..<20 {
            let v = VectorFixtures.random(dim: dim(), seed: UInt64(i + 400))
            let p = Database.Partition.test(id: "p\(i)", documentId: "doc\(i)", embedding: v)
            table.add(partition: p)
            indices["doc\(i)"] = makeIndex(partitions: [p], docId: "doc\(i)")
        }

        let sinatra = Sinatra(logger: .test)
        let results = callSearch(
            table: &table,
            queryEmbedding: VectorFixtures.random(dim: dim(), seed: 77),
            indices: indices,
            sinatra: sinatra
        )

        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(results[i-1].distance, results[i].distance,
                "Results must be sorted by ascending distance")
        }
    }

    // MARK: - Sinatra side effects

    func testParkIndicesCalledForTaggedDocuments() throws {
        // Totem stubs SinatraRegistry without parkedIndices — skip Sinatra side-effect assertion.
        throw XCTSkip("SinatraRegistry is a stub in Totem; parkedIndices not available")
    }

    func testParkIndicesNotCalledWhenNoTaggedDocuments() throws {
        // Totem stubs SinatraRegistry without parkedIndices — skip Sinatra side-effect assertion.
        throw XCTSkip("SinatraRegistry is a stub in Totem; parkedIndices not available")
    }

    // MARK: - Metadata propagation

    func testSearchResultPartitionCarriesMetadata() throws {
        var table = try makeTable()
        let payload = Data("partition-meta".utf8)
        let dim = dim()
        let docVec = VectorFixtures.random(dim: dim, seed: 9001)
        let partition = Database.Partition.test(id: "p-meta", documentId: "doc-meta", embedding: docVec)
        table.add(partition: partition)

        var index = PartitionIndex()
        index.train([partition], documentId: "doc-meta", logger: .test)
        index.metadata = payload
        let indices: [DocumentID: PartitionIndex] = ["doc-meta": index]

        let sinatra = Sinatra(logger: .test)
        let results = callSearch(
            table: &table,
            queryEmbedding: docVec,
            indices: indices,
            sinatra: sinatra
        )

        XCTAssertFalse(results.isEmpty, "Search must return at least one result")
        let found = results.first { $0.partition.documentId == "doc-meta" }
        XCTAssertNotNil(found, "Search must return a partition from doc-meta")
        XCTAssertEqual(found?.partition.metadata, payload,
            "Returned partition must carry the PartitionIndex.metadata payload")
    }
}

// MARK: - Array helpers

private extension Array where Element == Float {
    func with(index: Int, value: Float) -> [Float] {
        var copy = self
        copy[index % count] = value
        return copy
    }
}
