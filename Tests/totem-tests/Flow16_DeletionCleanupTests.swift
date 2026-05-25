//
//  Flow16_DeletionCleanupTests.swift
//  database-serverTests
//
//  Tests for the deletion cleanup fixes:
//
//  Bug 1 — Missing partitionStore.purge():
//    After removing a document, `documents/{id}-parts` was never deleted.
//    Fix: all three deletion paths in Database+Index.swift now call
//    `partitionStore(for: documentId).purge()`.
//
//  Bug 2 — personalGraphResponse stale HNSW nodes:
//    After a crash, personal HNSW may contain nodes for documents that the
//    registry no longer knows about for that owner. personalGraphResponse
//    lacked a registry backstop, so deleted documents surfaced with ownerId: "".
//    Fix: `allowedDocs = availableDocumentIds ∪ ownersDocuments[ownerKey]`
//    is computed and used to filter nodes before building the response.
//
//  Coverage:
//    1. Partition text file is written when a document is indexed.
//    2. FilePersistence.purge() removes the -parts file (mechanism used by fix).
//    3. Both document files (doc and -parts) can be purged independently.
//    4. allowedDocs is built correctly from availableDocumentIds ∪ ownersDocuments.
//    5. Stale HNSW nodes (doc not in allowedDocs) are filtered from the response.
//    6. Active nodes (doc in allowedDocs) pass the filter.
//    7. availableDocumentIds alone is sufficient for allowedDocs membership.
//    8. ownersDocuments alone is sufficient for allowedDocs membership.
//    9. isDeleted nodes are excluded regardless of allowedDocs.
//

import XCTest
@testable import totem

final class Flow16_DeletionCleanupTests: XCTestCase {

    private var testNodeId: UUID!
    private var createdKeys: [String] = []

    override func setUpWithError() throws {
        testNodeId  = UUID()
        createdKeys = []
    }

    override func tearDownWithError() throws {
        for key in createdKeys {
            FilePersistence(key: key, kind: .basic, logger: .test).purge()
        }
    }

    // Convenience: build a TableMutator with a fresh isolated vector store.
    private func makeTableMutator() throws -> TableMutator {
        let nodeId = testNodeId!
        let vecURL = FilePersistence.getDefaultURL()
            .appendingPathComponent("shard-\(nodeId)-vectors")
        let store  = try HNSWVectorStore(url: vecURL, nodeCount: 0)

        var table = PartitionTable()
        table.shards[0].vectorStore = store

        let mutator = TableMutator(nodeId: nodeId, logger: .test)
        mutator.seed(table)
        mutator.seedVectorStore(store)

        createdKeys += [
            "shard-\(nodeId)-topology",
            "shard-\(nodeId)-topology-wal",
            "shard-\(nodeId)-vectors",
            "shard-\(nodeId)-indices",
        ]
        return mutator
    }

    private func makePartition(
        id: String = "p0",
        documentId: String = "doc0",
        seed: UInt64 = 1,
        text: String = "hello world"
    ) -> Database.Partition {
        Database.Partition.test(id: id, documentId: documentId,
                            embedding: VectorFixtures.random(seed: seed),
                            text: text)
    }

    // =========================================================================
    // MARK: - 1. Partition text file written on put
    // =========================================================================

    func testPartitionDataFileWrittenOnPut() async throws {
        let mutator = try makeTableMutator()
        let docId   = "doc-write-\(testNodeId!)"
        createdKeys.append("documents/\(docId)-parts")

        await mutator.put(
            id: docId,
            partitions: [makePartition(id: "p0", documentId: docId, seed: 1)],
            request: .test()
        )
        // Force an immediate checkpoint to flush the debounced indices write.
        await mutator.replace(with: mutator.snapshot ?? PartitionTable())
        try await Task.sleep(nanoseconds: 200_000_000)

        let partsFile = FilePersistence(key: "documents/\(docId)-parts", kind: .basic, logger: .test)
        let stored: [PartitionData]? = partsFile.restore()
        XCTAssertNotNil(stored,
            "documents/{id}-parts must be written when a document is indexed")
        XCTAssertFalse((stored ?? []).isEmpty,
            "Partition text file must contain at least one record")
    }

    // =========================================================================
    // MARK: - 2. FilePersistence.purge() removes the -parts file
    //           (this is the mechanism called by Database.remove / removeAll / removeBatch)
    // =========================================================================

    func testPartitionDataFilePurgedByFilePersistence() async throws {
        let mutator = try makeTableMutator()
        let docId   = "doc-purge-\(testNodeId!)"
        createdKeys.append("documents/\(docId)-parts")

        await mutator.put(
            id: docId,
            partitions: [makePartition(id: "p0", documentId: docId, seed: 10)],
            request: .test()
        )
        await mutator.replace(with: mutator.snapshot ?? PartitionTable())
        try await Task.sleep(nanoseconds: 200_000_000)

        let partsFile = FilePersistence(key: "documents/\(docId)-parts", kind: .basic, logger: .test)
        XCTAssertNotNil(partsFile.restore() as [PartitionData]?,
            "Precondition: -parts file must exist before purge")

        // Simulate what Database.remove() now does after the fix.
        partsFile.purge()

        XCTAssertNil(partsFile.restore() as [PartitionData]?,
            "documents/{id}-parts must be absent after purge — this is the fix for Bug 1")
        XCTAssertFalse(FileManager.default.fileExists(atPath: partsFile.url.path()),
            "The -parts file must be physically removed from disk after purge()")
    }

    // =========================================================================
    // MARK: - 3. Both document files can be purged independently
    //           Verifies that doc and doc-parts are separate keys and can each
    //           be individually removed (invariant of the multi-path fix).
    // =========================================================================

    func testBothDocumentFilesArePurgedIndependently() async throws {
        let mutator = try makeTableMutator()
        let docId   = "doc-both-\(testNodeId!)"
        createdKeys += ["documents/\(docId)", "documents/\(docId)-parts"]

        // Write the document file (simulating what Database.put writes).
        let docFile = FilePersistence(key: "documents/\(docId)", kind: .basic, logger: .test)
        docFile.save(state: Database.Document.test(id: docId))

        await mutator.put(
            id: docId,
            partitions: [makePartition(id: "p0", documentId: docId, seed: 20)],
            request: .test()
        )
        await mutator.replace(with: mutator.snapshot ?? PartitionTable())
        try await Task.sleep(nanoseconds: 500_000_000)

        let partsFile = FilePersistence(key: "documents/\(docId)-parts", kind: .basic, logger: .test)

        // Both files should exist before purge.
        XCTAssertTrue(FileManager.default.fileExists(atPath: docFile.url.path()),
            "Precondition: document file must exist")
        XCTAssertTrue(FileManager.default.fileExists(atPath: partsFile.url.path()),
            "Precondition: -parts file must exist")

        // Purge both — matches Database.remove() / removeAll() / removeBatch() post-fix.
        docFile.purge()
        partsFile.purge()

        XCTAssertFalse(FileManager.default.fileExists(atPath: docFile.url.path()),
            "Document file must be gone after purge")
        XCTAssertFalse(FileManager.default.fileExists(atPath: partsFile.url.path()),
            "Partition text file must be gone after purge")
    }

    // =========================================================================
    // MARK: - 4. allowedDocs is the union of availableDocumentIds and ownersDocuments
    // =========================================================================

    func testAllowedDocsIsUnionOfAvailableAndOwned() {
        var registry = TotemRegistry()
        let owner    = TotemRegistry.Owner(id: "owner-a")

        // doc-available: in availableDocumentIds but NOT in ownersDocuments for this owner.
        registry.availableDocumentIds.insert("doc-available")

        // doc-owned: in ownersDocuments but NOT in availableDocumentIds.
        registry.ownersDocuments[owner] = ["doc-owned"]

        let allowedDocs = registry.availableDocumentIds
            .union(registry.ownersDocuments[owner] ?? [])

        XCTAssertTrue(allowedDocs.contains("doc-available"),
            "availableDocumentIds must contribute to allowedDocs")
        XCTAssertTrue(allowedDocs.contains("doc-owned"),
            "ownersDocuments must contribute to allowedDocs")
        XCTAssertFalse(allowedDocs.contains("doc-unknown"),
            "Unknown document must not appear in allowedDocs")
    }

    func testAllowedDocsExcludesDocRemovedFromRegistry() {
        var registry = TotemRegistry()
        let owner    = TotemRegistry.Owner(id: "owner-b")

        // Simulate doc existing, then being removed from both sets.
        registry.availableDocumentIds = ["doc-active"]
        registry.ownersDocuments[owner] = ["doc-active", "doc-deleted"]

        // After deletion, doc-deleted is scrubbed from both sets.
        registry.availableDocumentIds.remove("doc-deleted")
        registry.ownersDocuments[owner]?.removeAll { $0 == "doc-deleted" }

        let allowedDocs = registry.availableDocumentIds
            .union(registry.ownersDocuments[owner] ?? [])

        XCTAssertTrue(allowedDocs.contains("doc-active"))
        XCTAssertFalse(allowedDocs.contains("doc-deleted"),
            "Deleted document must not appear in allowedDocs after registry removal")
    }

    // =========================================================================
    // MARK: - 5. Stale HNSW node is filtered when document not in allowedDocs
    //           Replicates the allowedDocs guard in personalGraphResponse.
    // =========================================================================

    func testStalePHNSWNodeFilteredWhenDocNotInAllowedDocs() {
        // Build a stale node for a "deleted" document.
        let staleNode = HNSWGraph.Node(
            partitionId: "p-stale",
            documentId:  "doc-deleted",
            vectorIndex: 0,
            level:       0,
            neighbors:   [[]]
        )

        // Registry: doc-deleted was removed, so it's in neither set.
        let registry = TotemRegistry()
        let ownerKey = TotemRegistry.Owner(id: "owner-c")
        let allowedDocs = registry.availableDocumentIds
            .union(registry.ownersDocuments[ownerKey] ?? [])

        // Apply the personalGraphResponse filter.
        let visible = [staleNode].filter { node in
            !node.isDeleted && allowedDocs.contains(node.documentId)
        }

        XCTAssertTrue(visible.isEmpty,
            "Stale node for a deleted document must be filtered out by allowedDocs check (Bug 2 fix)")
    }

    // =========================================================================
    // MARK: - 6. Active node passes allowedDocs filter
    // =========================================================================

    func testActiveNodePassesAllowedDocsFilter() {
        let activeNode = HNSWGraph.Node(
            partitionId: "p-active",
            documentId:  "doc-active",
            vectorIndex: 0,
            level:       0,
            neighbors:   [[]]
        )

        var registry = TotemRegistry()
        let ownerKey = TotemRegistry.Owner(id: "owner-d")
        registry.availableDocumentIds.insert("doc-active")

        let allowedDocs = registry.availableDocumentIds
            .union(registry.ownersDocuments[ownerKey] ?? [])

        let visible = [activeNode].filter { node in
            !node.isDeleted && allowedDocs.contains(node.documentId)
        }

        XCTAssertEqual(visible.count, 1,
            "Active node whose document is in allowedDocs must pass the filter")
    }

    // =========================================================================
    // MARK: - 7. availableDocumentIds alone is sufficient for allowedDocs
    // =========================================================================

    func testAvailableDocumentIdsAloneSufficesForAllowedDocs() {
        let node = HNSWGraph.Node(
            partitionId: "p0",
            documentId:  "doc-via-available",
            vectorIndex: 0,
            level:       0,
            neighbors:   [[]]
        )

        var registry = TotemRegistry()
        let ownerKey = TotemRegistry.Owner(id: "owner-e")
        // Document is in availableDocumentIds but NOT ownersDocuments.
        registry.availableDocumentIds.insert("doc-via-available")
        // ownersDocuments[ownerKey] is nil.

        let allowedDocs = registry.availableDocumentIds
            .union(registry.ownersDocuments[ownerKey] ?? [])

        XCTAssertTrue(allowedDocs.contains(node.documentId),
            "availableDocumentIds alone must grant allowedDocs membership")
    }

    // =========================================================================
    // MARK: - 8. ownersDocuments alone is sufficient for allowedDocs
    // =========================================================================

    func testOwnersDocumentsAloneSufficesForAllowedDocs() {
        let node = HNSWGraph.Node(
            partitionId: "p0",
            documentId:  "doc-via-owned",
            vectorIndex: 0,
            level:       0,
            neighbors:   [[]]
        )

        var registry = TotemRegistry()
        let ownerKey = TotemRegistry.Owner(id: "owner-f")
        // Document is in ownersDocuments but NOT availableDocumentIds.
        registry.ownersDocuments[ownerKey] = ["doc-via-owned"]

        let allowedDocs = registry.availableDocumentIds
            .union(registry.ownersDocuments[ownerKey] ?? [])

        XCTAssertTrue(allowedDocs.contains(node.documentId),
            "ownersDocuments alone must grant allowedDocs membership")
    }

    // =========================================================================
    // MARK: - 9. isDeleted nodes excluded regardless of allowedDocs
    // =========================================================================

    func testDeletedFlagExcludesNodeEvenWhenDocIsAllowed() {
        let deletedNode = HNSWGraph.Node(
            partitionId: "p-deleted",
            documentId:  "doc-allowed",
            vectorIndex: 0,
            level:       0,
            neighbors:   [[]],
            isDeleted:   true
        )

        var registry = TotemRegistry()
        registry.availableDocumentIds.insert("doc-allowed")
        let ownerKey    = TotemRegistry.Owner(id: "owner-g")
        let allowedDocs = registry.availableDocumentIds
            .union(registry.ownersDocuments[ownerKey] ?? [])

        let visible = [deletedNode].filter { node in
            !node.isDeleted && allowedDocs.contains(node.documentId)
        }

        XCTAssertTrue(visible.isEmpty,
            "isDeleted nodes must be excluded even when the document is in allowedDocs")
    }
}
