import Foundation

struct HNSWRequest: Codable {
    let seer: DatabaseRequest
    let shardIndex: Int?
    let documentId: String?
    let documentIds: [String]?
    let partitionIds: [String]?
    let partitionId: String?

    enum CodingKeys: String, CodingKey {
        case seer
        case shardIndex   = "shard_index"
        case documentId   = "document_id"
        case documentIds  = "document_ids"
        case partitionIds = "partition_ids"
        case partitionId  = "partition_id"
    }
}
