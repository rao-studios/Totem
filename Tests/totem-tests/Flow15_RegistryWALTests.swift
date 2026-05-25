//
//  Flow15_RegistryWALTests.swift
//  database-serverTests
//
//  Tests for the RegistryWAL introduced in Phase 5:
//
//    Binary round-trip
//      • documentRegistered — with and without group, group with full metadata
//      • ownerLinked — with and without group
//      • earningsAccumulated — multi-document batch
//      • performanceAccumulated — minimal (no partitions) and full (partitions + sentiments)
//
//    WAL file operations
//      • readAll() on a fresh (empty) file returns []
//      • Multiple appended records are returned in insertion order
//      • byteSize tracks cumulative bytes written
//      • truncate() resets byteSize to 0 and subsequent readAll() returns []
//
//    Crash / corruption recovery
//      • readAll() stops and returns the clean prefix when a checksum is bad
//      • readAll() stops at a record whose declared payload length exceeds the file
//
//    Registry replay (RegistryWALRecord.apply(to:))
//      • documentRegistered — no group: owner/document/stats wired up, access .restricted
//      • documentRegistered — with group: group associations + access propagated
//      • ownerLinked — second owner linked to an already-registered document
//      • earningsAccumulated — credits added to existing documentStats
//      • performanceAccumulated — retrieval count, sentiment, and partition stats merged
//      • sequence — register → earnings → performance applied in order
//

import XCTest
@testable import totem

final class Flow15_RegistryWALTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("database-flow16-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    private func makeWAL(name: String = "test-wal") throws -> RegistryWAL {
        try RegistryWAL(url: tempDir.appendingPathComponent(name))
    }

    // MARK: - Round-trip helpers

    /// Encode a record to its payload bytes and decode back, returning the decoded record.
    private func roundTrip(_ record: RegistryWALRecord) throws -> RegistryWALRecord {
        let payload = record.encodePayload()
        return try RegistryWALRecord.decodePayload(typeCode: record.typeCode, data: payload)
    }

    // MARK: - Round-trip: documentRegistered

    func testDocumentRegistered_noGroup_roundTrip() throws {
        let original = RegistryWALRecord.documentRegistered(
            documentId: "doc-abc", ownerId: "owner-1", group: nil
        )
        let decoded = try roundTrip(original)

        guard case .documentRegistered(let dId, let oId, let group) = decoded else {
            return XCTFail("Wrong case")
        }
        XCTAssertEqual(dId, "doc-abc")
        XCTAssertEqual(oId, "owner-1")
        XCTAssertNil(group)
    }

    func testDocumentRegistered_withGroup_noMetadata_roundTrip() throws {
        let walGroup = RegistryWALRecord.WALGroup(
            id: "grp-1", label: "My Group", ownerId: "owner-1",
            access: .restricted, metadata: nil
        )
        let original = RegistryWALRecord.documentRegistered(
            documentId: "doc-1", ownerId: "owner-1", group: walGroup
        )
        let decoded = try roundTrip(original)

        guard case .documentRegistered(let dId, let oId, let g) = decoded else {
            return XCTFail("Wrong case")
        }
        XCTAssertEqual(dId, "doc-1")
        XCTAssertEqual(oId, "owner-1")
        XCTAssertEqual(g?.id, "grp-1")
        XCTAssertEqual(g?.label, "My Group")
        XCTAssertEqual(g?.ownerId, "owner-1")
        XCTAssertEqual(g?.access, .restricted)
        XCTAssertNil(g?.metadata)
    }

    func testDocumentRegistered_withGroup_fullMetadata_roundTrip() throws {
        let meta = Database.Group.Metadata(description: "A detailed description", tags: ["ai", "swift", "database"])
        let walGroup = RegistryWALRecord.WALGroup(
            id: "grp-meta", label: "Meta Group", ownerId: "owner-2",
            access: .available, metadata: meta
        )
        let original = RegistryWALRecord.documentRegistered(
            documentId: "doc-meta", ownerId: "owner-2", group: walGroup
        )
        let decoded = try roundTrip(original)

        guard case .documentRegistered(_, _, let g) = decoded else {
            return XCTFail("Wrong case")
        }
        XCTAssertEqual(g?.access, .available)
        XCTAssertEqual(g?.metadata?.description, "A detailed description")
        XCTAssertEqual(g?.metadata?.tags, ["ai", "swift", "database"])
    }

    func testDocumentRegistered_groupWithNilAccess_roundTrip() throws {
        let walGroup = RegistryWALRecord.WALGroup(
            id: "grp-nil-access", label: "Label", ownerId: "owner-3",
            access: nil, metadata: nil
        )
        let decoded = try roundTrip(
            .documentRegistered(documentId: "doc-x", ownerId: "owner-3", group: walGroup)
        )
        guard case .documentRegistered(_, _, let g) = decoded else {
            return XCTFail("Wrong case")
        }
        XCTAssertNil(g?.access)
    }

    // MARK: - Round-trip: ownerLinked

    func testOwnerLinked_noGroup_roundTrip() throws {
        let decoded = try roundTrip(
            .ownerLinked(documentId: "doc-shared", ownerId: "owner-B", group: nil)
        )
        guard case .ownerLinked(let dId, let oId, let g) = decoded else {
            return XCTFail("Wrong case")
        }
        XCTAssertEqual(dId, "doc-shared")
        XCTAssertEqual(oId, "owner-B")
        XCTAssertNil(g)
    }

    func testOwnerLinked_withGroup_roundTrip() throws {
        let walGroup = RegistryWALRecord.WALGroup(
            id: "grp-linked", label: "Linked Group", ownerId: "owner-B",
            access: .restricted, metadata: Database.Group.Metadata(description: nil, tags: ["tag1"])
        )
        let decoded = try roundTrip(
            .ownerLinked(documentId: "doc-shared", ownerId: "owner-B", group: walGroup)
        )
        guard case .ownerLinked(_, _, let g) = decoded else {
            return XCTFail("Wrong case")
        }
        XCTAssertEqual(g?.id, "grp-linked")
        XCTAssertEqual(g?.metadata?.tags, ["tag1"])
        XCTAssertNil(g?.metadata?.description)
    }

    // MARK: - Round-trip: earningsAccumulated

    func testEarningsAccumulated_roundTrip() throws {
        let items: [(documentId: DocumentID, credits: Double)] = [
            ("doc-a", 1.5),
            ("doc-b", 0.25),
            ("doc-c", 100.0),
        ]
        let decoded = try roundTrip(.earningsAccumulated(items))

        guard case .earningsAccumulated(let out) = decoded else {
            return XCTFail("Wrong case")
        }
        XCTAssertEqual(out.count, 3)
        XCTAssertEqual(out[0].documentId, "doc-a")
        XCTAssertEqual(out[0].credits, 1.5, accuracy: 1e-9)
        XCTAssertEqual(out[1].documentId, "doc-b")
        XCTAssertEqual(out[1].credits, 0.25, accuracy: 1e-9)
        XCTAssertEqual(out[2].credits, 100.0, accuracy: 1e-9)
    }

    // MARK: - Round-trip: performanceAccumulated

    func testPerformanceAccumulated_minimal_roundTrip() throws {
        let stats = Database.DocumentStats(
            id: "doc-perf",
            retrievalCount: 5,
            sentimentSum: 3.75,
            lastRetrieved: nil,
            partitionRetrievalCount: [:],
            partitionSentiments: [:]
        )
        let decoded = try roundTrip(.performanceAccumulated([.init(from: stats)]))

        guard case .performanceAccumulated(let out) = decoded else {
            return XCTFail("Wrong case")
        }
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].documentId, "doc-perf")
        XCTAssertEqual(out[0].retrievalCount, 5)
        XCTAssertEqual(out[0].sentimentSum, 3.75, accuracy: 1e-9)
        XCTAssertNil(out[0].lastRetrieved)
        XCTAssertTrue(out[0].partitionRetrievalCount.isEmpty)
        XCTAssertTrue(out[0].partitionSentiments.isEmpty)
    }

    func testPerformanceAccumulated_full_roundTrip() throws {
        let refDate = Date(timeIntervalSince1970: 1_700_000_000.5)
        var ps = Database.DocumentStats.PartitionSentiment()
        ps.retrievalCount = 4
        ps.sentimentSum   = 2.8
        ps.lastRetrieved  = refDate

        let stats = Database.DocumentStats(
            id: "doc-full",
            retrievalCount: 12,
            sentimentSum: 8.4,
            lastRetrieved: refDate,
            partitionRetrievalCount: ["p-1": 7, "p-2": 5],
            partitionSentiments: ["p-1": ps]
        )
        let decoded = try roundTrip(.performanceAccumulated([.init(from: stats)]))

        guard case .performanceAccumulated(let out) = decoded else {
            return XCTFail("Wrong case")
        }
        let s = out[0]
        XCTAssertEqual(s.retrievalCount, 12)
        XCTAssertEqual(s.sentimentSum, 8.4, accuracy: 1e-9)
        XCTAssertEqual(s.lastRetrieved?.timeIntervalSince1970 ?? 0,
                       refDate.timeIntervalSince1970, accuracy: 1e-6)
        XCTAssertEqual(s.partitionRetrievalCount["p-1"], 7)
        XCTAssertEqual(s.partitionRetrievalCount["p-2"], 5)
        XCTAssertEqual(s.partitionSentiments["p-1"]?.retrievalCount, 4)
        XCTAssertEqual(s.partitionSentiments["p-1"]?.sentimentSum ?? 0, 2.8, accuracy: 1e-9)
        XCTAssertEqual(
            s.partitionSentiments["p-1"]?.lastRetrieved?.timeIntervalSince1970 ?? 0,
            refDate.timeIntervalSince1970, accuracy: 1e-6
        )
    }

    // MARK: - WAL file operations

    func testEmptyWALReadsEmptyArray() throws {
        let wal = try makeWAL()
        let records = try wal.readAll()
        XCTAssertTrue(records.isEmpty)
        XCTAssertEqual(wal.byteSize, 0)
    }

    func testAppendedRecordsReadBackInOrder() throws {
        let wal = try makeWAL()

        try wal.append(.earningsAccumulated([("doc-1", 1.0)]))
        try wal.append(.earningsAccumulated([("doc-2", 2.0)]))
        try wal.append(.earningsAccumulated([("doc-3", 3.0)]))

        let records = try wal.readAll()
        XCTAssertEqual(records.count, 3)

        for (i, record) in records.enumerated() {
            guard case .earningsAccumulated(let items) = record else {
                return XCTFail("Wrong case at index \(i)")
            }
            XCTAssertEqual(items[0].documentId, "doc-\(i + 1)")
            XCTAssertEqual(items[0].credits, Double(i + 1), accuracy: 1e-9)
        }
    }

    func testByteSizeIncreasesAfterEachAppend() throws {
        let wal = try makeWAL()
        XCTAssertEqual(wal.byteSize, 0)

        try wal.append(.earningsAccumulated([("doc-1", 1.0)]))
        let after1 = wal.byteSize
        XCTAssertGreaterThan(after1, 0)

        try wal.append(.earningsAccumulated([("doc-2", 2.0)]))
        XCTAssertGreaterThan(wal.byteSize, after1)
    }

    func testTruncateClearsWALAndByteSize() throws {
        let wal = try makeWAL()
        try wal.append(.earningsAccumulated([("doc-1", 1.0)]))
        try wal.append(.earningsAccumulated([("doc-2", 2.0)]))
        XCTAssertGreaterThan(wal.byteSize, 0)

        try wal.truncate()
        XCTAssertEqual(wal.byteSize, 0)

        let records = try wal.readAll()
        XCTAssertTrue(records.isEmpty)
    }

    func testAppendAfterTruncateWritesFromStart() throws {
        let wal = try makeWAL()
        try wal.append(.earningsAccumulated([("doc-before", 9.0)]))
        try wal.truncate()

        try wal.append(.earningsAccumulated([("doc-after", 7.0)]))
        let records = try wal.readAll()

        XCTAssertEqual(records.count, 1)
        guard case .earningsAccumulated(let items) = records[0] else {
            return XCTFail("Wrong case")
        }
        XCTAssertEqual(items[0].documentId, "doc-after")
    }

    // MARK: - Crash / corruption recovery

    func testReadAllStopsAtBadChecksum() throws {
        let url = tempDir.appendingPathComponent("wal-bad-checksum")

        // Append two valid records, then close.
        do {
            let wal = try RegistryWAL(url: url)
            try wal.append(.earningsAccumulated([("doc-1", 1.0)]))
            try wal.append(.earningsAccumulated([("doc-2", 2.0)]))
        }

        // Overwrite the last 4 bytes (checksum of the second record).
        let fh = try FileHandle(forUpdating: url)
        defer { try? fh.close() }
        let fileSize = try fh.seekToEnd()
        try fh.seek(toOffset: fileSize - 4)
        fh.write(Data([0xFF, 0xFF, 0xFF, 0xFF]))

        // Re-open with fresh byteSize from fstat.
        let wal2 = try RegistryWAL(url: url)
        let records = try wal2.readAll()

        // Only the first (valid) record should be returned.
        XCTAssertEqual(records.count, 1)
        guard case .earningsAccumulated(let items) = records[0] else {
            return XCTFail("Wrong case")
        }
        XCTAssertEqual(items[0].documentId, "doc-1")
    }

    func testReadAllStopsAtTruncatedPayload() throws {
        let url = tempDir.appendingPathComponent("wal-truncated-payload")

        // Append one valid record, then close.
        do {
            let wal = try RegistryWAL(url: url)
            try wal.append(.earningsAccumulated([("doc-1", 1.0)]))
        }

        // Append a partial record header: typeCode + payloadLen claiming 1 000 bytes,
        // but no payload or checksum follows — simulates a crash mid-write.
        let fh = try FileHandle(forUpdating: url)
        defer { try? fh.close() }
        try fh.seekToEnd()
        var partial = Data()
        partial.append(0x03)                                  // typeCode: earningsAccumulated
        partial.append(contentsOf: [0xe8, 0x03, 0x00, 0x00]) // payloadLen = 1 000 LE
        fh.write(partial)

        let wal2 = try RegistryWAL(url: url)
        let records = try wal2.readAll()

        XCTAssertEqual(records.count, 1)
        guard case .earningsAccumulated(let items) = records[0] else {
            return XCTFail("Wrong case")
        }
        XCTAssertEqual(items[0].documentId, "doc-1")
    }

    // MARK: - Registry replay

    func testReplay_documentRegistered_noGroup() throws {
        var registry = TotemRegistry()
        let record = RegistryWALRecord.documentRegistered(
            documentId: "doc-r1", ownerId: "owner-r1", group: nil
        )
        record.apply(to: &registry)

        let owner = TotemRegistry.Owner(id: "owner-r1")
        XCTAssertTrue(registry.documentOwners["doc-r1"]?.contains(owner) == true)
        XCTAssertTrue(registry.ownersDocuments[owner]?.contains("doc-r1") == true)
        XCTAssertNotNil(registry.documentStats["doc-r1"])
        // No group → access defaults to .restricted
        XCTAssertEqual(registry.documentAccess["doc-r1"], .restricted)
        XCTAssertTrue(registry.groups.isEmpty)
    }

    func testReplay_documentRegistered_withGroup() throws {
        var registry = TotemRegistry()
        let walGroup = RegistryWALRecord.WALGroup(
            id: "grp-r", label: "Replay Group", ownerId: "owner-r2",
            access: .available, metadata: nil
        )
        let record = RegistryWALRecord.documentRegistered(
            documentId: "doc-r2", ownerId: "owner-r2", group: walGroup
        )
        record.apply(to: &registry)

        let owner = TotemRegistry.Owner(id: "owner-r2")
        XCTAssertTrue(registry.documentOwners["doc-r2"]?.contains(owner) == true)
        XCTAssertEqual(registry.groupOwners["grp-r"], owner)
        XCTAssertTrue(registry.groups["grp-r"]?.contains("doc-r2") == true)
        XCTAssertTrue(registry.documentGroups["doc-r2"]?.contains("grp-r") == true)
        XCTAssertEqual(registry.ownerDocumentGroup["owner-r2"]?["doc-r2"], "grp-r")
        // Group access .available propagates to document
        XCTAssertEqual(registry.documentAccess["doc-r2"], .available)
        XCTAssertTrue(registry.availableDocumentIds.contains("doc-r2"))
    }

    func testReplay_ownerLinked() throws {
        var registry = TotemRegistry()

        // First owner registers the document.
        RegistryWALRecord.documentRegistered(
            documentId: "doc-shared", ownerId: "owner-A", group: nil
        ).apply(to: &registry)

        // Second owner is linked via WAL replay.
        RegistryWALRecord.ownerLinked(
            documentId: "doc-shared", ownerId: "owner-B", group: nil
        ).apply(to: &registry)

        let ownerA = TotemRegistry.Owner(id: "owner-A")
        let ownerB = TotemRegistry.Owner(id: "owner-B")
        XCTAssertTrue(registry.documentOwners["doc-shared"]?.contains(ownerA) == true)
        XCTAssertTrue(registry.documentOwners["doc-shared"]?.contains(ownerB) == true)
        XCTAssertTrue(registry.ownersDocuments[ownerB]?.contains("doc-shared") == true)
    }

    func testReplay_earningsAccumulated() throws {
        var registry = TotemRegistry()

        // Register so documentStats entry exists.
        RegistryWALRecord.documentRegistered(
            documentId: "doc-earn", ownerId: "owner-e", group: nil
        ).apply(to: &registry)
        XCTAssertEqual(registry.documentStats["doc-earn"]?.totalEarned, 0)

        RegistryWALRecord.earningsAccumulated([("doc-earn", 12.5)]).apply(to: &registry)
        XCTAssertEqual(registry.documentStats["doc-earn"]?.totalEarned ?? 0, 12.5, accuracy: 1e-9)

        // Accumulating again is additive.
        RegistryWALRecord.earningsAccumulated([("doc-earn", 7.5)]).apply(to: &registry)
        XCTAssertEqual(registry.documentStats["doc-earn"]?.totalEarned ?? 0, 20.0, accuracy: 1e-9)
    }

    func testReplay_performanceAccumulated() throws {
        var registry = TotemRegistry()

        RegistryWALRecord.documentRegistered(
            documentId: "doc-perf-r", ownerId: "owner-p", group: nil
        ).apply(to: &registry)

        var ps = Database.DocumentStats.PartitionSentiment()
        ps.retrievalCount = 3
        ps.sentimentSum   = 2.1
        let stats = Database.DocumentStats(
            id: "doc-perf-r",
            retrievalCount: 5,
            sentimentSum: 3.5,
            partitionRetrievalCount: ["part-x": 5],
            partitionSentiments: ["part-x": ps]
        )
        RegistryWALRecord.performanceAccumulated([.init(from: stats)]).apply(to: &registry)

        let result = registry.documentStats["doc-perf-r"]
        XCTAssertEqual(result?.retrievalCount, 5)
        XCTAssertEqual(result?.sentimentSum ?? 0, 3.5, accuracy: 1e-9)
        XCTAssertEqual(result?.partitionRetrievalCount["part-x"], 5)
        XCTAssertEqual(result?.partitionSentiments["part-x"]?.retrievalCount, 3)
        XCTAssertEqual(result?.partitionSentiments["part-x"]?.sentimentSum ?? 0, 2.1, accuracy: 1e-9)
    }

    func testReplay_sequence_registerEarningsPerformance() throws {
        var registry = TotemRegistry()

        // 1. Register
        RegistryWALRecord.documentRegistered(
            documentId: "doc-seq", ownerId: "owner-s", group: nil
        ).apply(to: &registry)

        // 2. Accumulate earnings twice
        RegistryWALRecord.earningsAccumulated([("doc-seq", 5.0)]).apply(to: &registry)
        RegistryWALRecord.earningsAccumulated([("doc-seq", 3.0)]).apply(to: &registry)

        // 3. Accumulate performance
        let stats = Database.DocumentStats(
            id: "doc-seq", retrievalCount: 4, sentimentSum: 2.8
        )
        RegistryWALRecord.performanceAccumulated([.init(from: stats)]).apply(to: &registry)

        let owner = TotemRegistry.Owner(id: "owner-s")
        XCTAssertTrue(registry.documentOwners["doc-seq"]?.contains(owner) == true)
        XCTAssertEqual(registry.documentStats["doc-seq"]?.totalEarned ?? 0, 8.0, accuracy: 1e-9)
        XCTAssertEqual(registry.documentStats["doc-seq"]?.retrievalCount, 4)
        XCTAssertEqual(registry.documentStats["doc-seq"]?.sentimentSum ?? 0, 2.8, accuracy: 1e-9)
    }

    // MARK: - WAL file + replay integration

    func testAppendAndReplayViaNewWALObject() throws {
        let url = tempDir.appendingPathComponent("wal-integration")

        // Session 1: append a registration and earnings record.
        do {
            let wal = try RegistryWAL(url: url)
            try wal.append(.documentRegistered(documentId: "doc-int", ownerId: "owner-int", group: nil))
            try wal.append(.earningsAccumulated([("doc-int", 9.0)]))
        }

        // Session 2: re-open, read all, replay onto a fresh registry.
        let wal2 = try RegistryWAL(url: url)
        let records = try wal2.readAll()
        XCTAssertEqual(records.count, 2)

        var registry = TotemRegistry()
        for record in records { record.apply(to: &registry) }

        XCTAssertNotNil(registry.documentOwners["doc-int"])
        XCTAssertEqual(registry.documentStats["doc-int"]?.totalEarned ?? 0, 9.0, accuracy: 1e-9)
    }
}
