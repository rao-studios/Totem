//
//  HNSWTopologyWAL.swift
//  database-server
//
//  Created by Ritesh Pakala on 4/3/26.
//

import Foundation

#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

// MARK: - TopologyWALRecord

/// One mutation to the HNSW graph topology. Emitted by `HNSWGraph` during every
/// structural change and replayed at startup to restore mutations since the last checkpoint.
///
/// **Equatable** so tests can compare decoded records against originals.
enum TopologyWALRecord: Equatable {
    /// A new node was inserted. `neighborsByLayer[i]` is the final neighbor list
    /// for layer i after both the forward links and all backlinks were established.
    case nodeInserted(
        partitionId:      String,
        documentId:       String,
        vectorIndex:      Int,
        level:            Int,
        neighborsByLayer: [[Int]]
    )
    /// The neighbor list for one layer of an existing node was updated (backlink pass).
    case neighborsUpdated(nodeIndex: Int, layer: Int, neighbors: [Int])
    /// A node was soft-deleted.
    case nodeDeleted(nodeIndex: Int)
    /// The global entry point changed. Use nodeIndex = -1 / maxLevel = -1 for empty graph.
    case entryPointChanged(nodeIndex: Int, maxLevel: Int)
    /// Marks the end of one atomic mutation group. All preceding records since the last
    /// `.commit` form a single transaction; `readAll()` discards any uncommitted tail
    /// left by a crash mid-flush.
    case commit
}

// MARK: - TopologyWALRecord binary encoding

extension TopologyWALRecord {
    // Type codes written to the WAL file header byte.
    fileprivate enum TypeCode: UInt8 {
        case nodeInserted      = 0x01
        case neighborsUpdated  = 0x02
        case nodeDeleted       = 0x03
        case entryPointChanged = 0x04
        case commit            = 0x05
    }

    var typeCode: UInt8 {
        switch self {
        case .nodeInserted:      return TypeCode.nodeInserted.rawValue
        case .neighborsUpdated:  return TypeCode.neighborsUpdated.rawValue
        case .nodeDeleted:       return TypeCode.nodeDeleted.rawValue
        case .entryPointChanged: return TypeCode.entryPointChanged.rawValue
        case .commit:            return TypeCode.commit.rawValue
        }
    }

    /// Encodes the record payload (excludes the type byte, length, and checksum).
    func encodePayload() -> Data {
        var buf = Data()
        switch self {
        case .nodeInserted(let partitionId, let documentId, let vectorIndex, let level, let neighborsByLayer):
            buf.walString(partitionId)
            buf.walString(documentId)
            buf.walInt32(Int32(vectorIndex))
            buf.walUInt8(UInt8(clamping: level))
            buf.walUInt8(UInt8(clamping: neighborsByLayer.count))
            for layer in neighborsByLayer {
                buf.walUInt16(UInt16(clamping: layer.count))
                for n in layer { buf.walInt32(Int32(clamping: n)) }
            }

        case .neighborsUpdated(let nodeIndex, let layer, let neighbors):
            buf.walInt32(Int32(nodeIndex))
            buf.walUInt8(UInt8(clamping: layer))
            buf.walUInt16(UInt16(clamping: neighbors.count))
            for n in neighbors { buf.walInt32(Int32(clamping: n)) }

        case .nodeDeleted(let nodeIndex):
            buf.walInt32(Int32(nodeIndex))

        case .entryPointChanged(let nodeIndex, let maxLevel):
            buf.walInt32(Int32(nodeIndex))
            buf.walInt32(Int32(maxLevel))

        case .commit:
            break  // empty payload
        }
        return buf
    }

    /// Decodes a record from a raw payload `Data` blob.
    /// - Throws: `HNSWTopologyWAL.WALError` on truncated or unrecognised data.
    static func decodePayload(typeCode: UInt8, data: Data) throws -> TopologyWALRecord {
        var r = BinaryReader(data: data)
        switch typeCode {
        case TypeCode.nodeInserted.rawValue:
            let partitionId = try r.string()
            let documentId  = try r.string()
            let vectorIndex = Int(try r.int32())
            let level       = Int(try r.uint8())
            let layerCount  = Int(try r.uint8())
            var layers: [[Int]] = []
            for _ in 0..<layerCount {
                let count = Int(try r.uint16())
                var layer: [Int] = []
                layer.reserveCapacity(count)
                for _ in 0..<count { layer.append(Int(try r.int32())) }
                layers.append(layer)
            }
            return .nodeInserted(partitionId: partitionId, documentId: documentId,
                                 vectorIndex: vectorIndex, level: level,
                                 neighborsByLayer: layers)

        case TypeCode.neighborsUpdated.rawValue:
            let nodeIndex = Int(try r.int32())
            let layer     = Int(try r.uint8())
            let count     = Int(try r.uint16())
            var neighbors: [Int] = []
            neighbors.reserveCapacity(count)
            for _ in 0..<count { neighbors.append(Int(try r.int32())) }
            return .neighborsUpdated(nodeIndex: nodeIndex, layer: layer, neighbors: neighbors)

        case TypeCode.nodeDeleted.rawValue:
            return .nodeDeleted(nodeIndex: Int(try r.int32()))

        case TypeCode.entryPointChanged.rawValue:
            let nodeIndex = Int(try r.int32())
            let maxLevel  = Int(try r.int32())
            return .entryPointChanged(nodeIndex: nodeIndex, maxLevel: maxLevel)

        case TypeCode.commit.rawValue:
            return .commit

        default:
            throw HNSWTopologyWAL.WALError.unknownTypeCode(typeCode)
        }
    }
}

// MARK: - HNSWTopologyWAL

/// Append-only WAL file for HNSW graph topology mutations.
///
/// **Format (per record):**
/// ```
/// [typeCode: UInt8][payloadLen: UInt32 LE][payload: bytes][checksum: UInt32 LE]
/// ```
/// 9 bytes overhead per record. Adler-32 checksum over the payload detects truncation
/// and bit-flip corruption. `readAll()` stops at the first invalid record; prior records
/// are returned as-is (partial-write crash recovery).
///
/// **Concurrency:** All writes happen from within actor-isolated mutators (TableMutator,
/// PersonalHNSWMutator), so no additional locking is needed. The file is opened with
/// `O_APPEND` for atomic position-independent writes.
final class HNSWTopologyWAL: @unchecked Sendable {

    enum WALError: Error {
        case openFailed(Int32)
        case writeFailed(Int32)
        case truncateFailed(Int32)
        case truncatedRecord
        case checksumMismatch
        case unknownTypeCode(UInt8)
    }

    private let fd:  Int32
    /// Current byte length of the WAL file. Updated after every successful append.
    private(set) var byteSize: Int

    // MARK: - Init / deinit

    /// Open or create the WAL file.
    ///
    /// - Throws: `WALError.openFailed` if the OS rejects the file open (permissions, path).
    init(url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let rawFd = open(url.path, O_RDWR | O_CREAT | O_APPEND,
                         S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
        guard rawFd >= 0 else { throw WALError.openFailed(errno) }
        fd = rawFd

        var s = stat()
        fstat(fd, &s)
        byteSize = Int(s.st_size)
    }

    deinit { close(fd) }

    // MARK: - Write

    /// Appends one record to the WAL. `O_APPEND` makes this atomic for the ~1-5 KB
    /// records emitted per HNSW insert — well within PIPE_BUF on Darwin and Linux.
    func append(_ record: TopologyWALRecord) throws {
        let payload  = record.encodePayload()
        let checksum = adler32(payload)

        var buf = Data(capacity: 9 + payload.count)
        buf.walUInt8(record.typeCode)
        buf.walUInt32(UInt32(payload.count))
        buf.append(payload)
        buf.walUInt32(checksum)

        let n = buf.withUnsafeBytes { ptr -> Int in
            Foundation.write(fd, ptr.baseAddress!, ptr.count)
        }
        guard n == buf.count else { throw WALError.writeFailed(errno) }
        byteSize += buf.count
    }

    // MARK: - Read

    /// Reads all committed records in insertion order.
    ///
    /// **Commit-group semantics:** Records are accumulated in a `pending` buffer and moved
    /// to `committed` only when a `.commit` record is encountered. Any uncommitted tail
    /// (records after the last `.commit`) is silently discarded — this is the normal result
    /// of a crash mid-flush and guarantees that only fully-written mutation groups are replayed.
    ///
    /// **Backward compatibility:** WAL files written before `.commit` was introduced contain
    /// no commit records. If the full decode loop yields no committed records but does yield
    /// pending records, the old-format fallback applies all checksummed records as before.
    /// This provides a lossless first restart after an upgrade; subsequent WALs use the new
    /// format from the next checkpoint onward.
    ///
    /// Stops — without throwing — at the first record with a bad checksum or a truncated
    /// payload (crash-during-write recovery).
    func readAll() throws -> [TopologyWALRecord] {
        guard byteSize > 0 else { return [] }

        // Read the whole file from offset 0 without disturbing the write position.
        var fileData = Data(count: byteSize)
        let n = fileData.withUnsafeMutableBytes { ptr -> Int in
            pread(fd, ptr.baseAddress!, byteSize, 0)
        }
        guard n > 0 else { return [] }
        let available = n   // might be less than byteSize if file was truncated concurrently

        var committed: [TopologyWALRecord] = []
        var pending:   [TopologyWALRecord] = []
        var offset = 0

        while offset + 9 <= available {               // 9 = typeCode(1) + length(4) + checksum(4)
            let typeCode   = fileData[offset]
            let payloadLen = Int(
                UInt32(fileData[offset + 1])        |
                (UInt32(fileData[offset + 2]) << 8) |
                (UInt32(fileData[offset + 3]) << 16) |
                (UInt32(fileData[offset + 4]) << 24)
            )
            offset += 5

            guard offset + payloadLen + 4 <= available else { break }  // truncated

            let payload = fileData[offset ..< (offset + payloadLen)]
            offset += payloadLen

            let stored = UInt32(fileData[offset])        |
                        (UInt32(fileData[offset + 1]) << 8)  |
                        (UInt32(fileData[offset + 2]) << 16) |
                        (UInt32(fileData[offset + 3]) << 24)
            offset += 4

            guard adler32(payload) == stored else { break }     // corrupt record — stop here

            guard let rec = try? TopologyWALRecord.decodePayload(typeCode: typeCode, data: payload)
            else { break }                                       // unknown type — stop here

            if rec == .commit {
                committed.append(contentsOf: pending)
                pending = []
            } else {
                pending.append(rec)
            }
        }

        // Backward-compat: old WALs have no .commit records. If nothing was committed but
        // there are valid pending records, treat the whole file as one implicit commit group.
        if committed.isEmpty && !pending.isEmpty { return pending }
        return committed
    }

    // MARK: - Checkpoint truncation

    /// Truncates the WAL to zero bytes after a checkpoint has been written to disk.
    /// `ftruncate(fd, 0)` is an O(1) metadata-only operation; no data copy occurs.
    ///
    /// - Note: With `O_APPEND`, subsequent writes will go to offset 0 (end of empty file)
    ///   automatically — no additional `lseek` is required.
    func truncate() throws {
        guard ftruncate(fd, 0) == 0 else { throw WALError.truncateFailed(errno) }
        byteSize = 0
    }
}

// MARK: - Private helpers

/// Adler-32 checksum — fast, no external dependency, sufficient for single-bit flip and
/// truncation detection in a local WAL file.
private func adler32(_ data: Data) -> UInt32 {
    var a: UInt32 = 1
    var b: UInt32 = 0
    for byte in data {
        a = (a &+ UInt32(byte)) % 65521
        b = (b &+ a) % 65521
    }
    return (b << 16) | a
}

// MARK: - BinaryReader

private struct BinaryReader {
    private let data: Data
    private var offset: Int = 0

    init(data: Data) { self.data = data }

    mutating func uint8() throws -> UInt8 {
        guard offset < data.count else { throw HNSWTopologyWAL.WALError.truncatedRecord }
        defer { offset += 1 }
        return data[data.startIndex + offset]
    }

    mutating func uint16() throws -> UInt16 {
        guard offset + 2 <= data.count else { throw HNSWTopologyWAL.WALError.truncatedRecord }
        defer { offset += 2 }
        let i = data.startIndex + offset
        return UInt16(data[i]) | (UInt16(data[i + 1]) << 8)
    }

    mutating func uint32() throws -> UInt32 {
        guard offset + 4 <= data.count else { throw HNSWTopologyWAL.WALError.truncatedRecord }
        defer { offset += 4 }
        let i = data.startIndex + offset
        return UInt32(data[i]) | (UInt32(data[i+1]) << 8) | (UInt32(data[i+2]) << 16) | (UInt32(data[i+3]) << 24)
    }

    mutating func int32() throws -> Int32 { Int32(bitPattern: try uint32()) }

    mutating func string() throws -> String {
        let len = Int(try uint16())
        guard offset + len <= data.count else { throw HNSWTopologyWAL.WALError.truncatedRecord }
        defer { offset += len }
        let i = data.startIndex + offset
        return String(data: data[i ..< (i + len)], encoding: .utf8) ?? ""
    }
}

// MARK: - Data write helpers (little-endian)

private extension Data {
    mutating func walUInt8(_ v: UInt8)  { append(v) }
    mutating func walUInt16(_ v: UInt16) {
        append(UInt8(v & 0xff)); append(UInt8((v >> 8) & 0xff))
    }
    mutating func walUInt32(_ v: UInt32) {
        append(UInt8( v        & 0xff))
        append(UInt8((v >>  8) & 0xff))
        append(UInt8((v >> 16) & 0xff))
        append(UInt8((v >> 24) & 0xff))
    }
    mutating func walInt32(_ v: Int32)  { walUInt32(UInt32(bitPattern: v)) }
    mutating func walString(_ s: String) {
        let bytes = Array(s.utf8.prefix(65535))
        walUInt16(UInt16(bytes.count))
        append(contentsOf: bytes)
    }
}
