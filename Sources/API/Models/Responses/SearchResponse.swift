import Foundation

struct SearchShardStat: Codable {
    let shardIndex: Int
    let nodes: Int
    let maxLevel: Int
    let efUsed: Int
    let explored: Int
    let candidates: Int
    let threshold: Float

    enum CodingKeys: String, CodingKey {
        case shardIndex  = "shard_index"
        case nodes
        case maxLevel    = "max_level"
        case efUsed      = "ef_used"
        case explored
        case candidates
        case threshold
    }
}

struct SearchResponse: Codable {
    var object: String = "list"
    let texts: [String]
    let references: [Database.DocumentReference]
    let shardStats: [SearchShardStat]?

    enum CodingKeys: String, CodingKey {
        case object
        case texts
        case references
        case shardStats = "shard_stats"
    }
}
