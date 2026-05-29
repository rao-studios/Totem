import Foundation

struct HNSWGraphStats: Codable {
    var liveNodes: Int
    var maxLevel: Int
    var isTrained: Bool
    var shardCount: Int

    enum CodingKeys: String, CodingKey {
        case liveNodes  = "live_nodes"
        case maxLevel   = "max_level"
        case isTrained  = "is_trained"
        case shardCount = "shard_count"
    }
}

struct HNSWStatsResponse: Codable {
    var personal: HNSWGraphStats
    var global: HNSWGraphStats
}

struct HNSWNodeResponse: Codable {
    var partitionId: String
    var document: Database.Document
    var text: String
    var url: URL
    var level: Int
    var neighborIds: [String]
    var isDeleted: Bool

    enum CodingKeys: String, CodingKey {
        case partitionId = "partition_id"
        case neighborIds = "neighbor_ids"
        case document, text, url, level, isDeleted
    }
}

struct HNSWGraphResponse: Codable {
    var nodes: [HNSWNodeResponse]
    var totalNodes: Int
    var liveNodes: Int
    var maxLevel: Int
    var shardIndex: Int?
    var shardCount: Int

    enum CodingKeys: String, CodingKey {
        case totalNodes = "total_nodes"
        case liveNodes  = "live_nodes"
        case maxLevel   = "max_level"
        case shardIndex = "shard_index"
        case shardCount = "shard_count"
        case nodes
    }
}

struct HNSWNodeBatchResponse: Codable {
    var nodes: [HNSWNodeResponse]
}

struct HNSWNodeDeleteResponse: Codable {
    var removed: Bool
    var documentId: String

    enum CodingKeys: String, CodingKey {
        case documentId = "document_id"
        case removed
    }
}
