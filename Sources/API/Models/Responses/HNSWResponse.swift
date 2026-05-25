import Vapor

struct HNSWGraphStats: Content {
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

struct HNSWStatsResponse: Content {
    var personal: HNSWGraphStats
    var global: HNSWGraphStats
}

struct HNSWNodeResponse: Content {
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

struct HNSWGraphResponse: Content {
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

struct HNSWNodeBatchResponse: Content {
    var nodes: [HNSWNodeResponse]
}

struct HNSWNodeDeleteResponse: Content {
    var removed: Bool
    var documentId: String

    enum CodingKeys: String, CodingKey {
        case documentId = "document_id"
        case removed
    }
}
