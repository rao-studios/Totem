import Foundation

extension Database {
    struct SearchResult {
        var data: [PartitionSearchResult]
        var adjustments: [SinatraAdjustment]
        var shardStats: [SearchShardStat]

        var partitionWithScores: [(score: Float, partition: Database.Partition)] {
            data.flatMap { zip($0.scores, $0.partitions) }
        }

        var partitions: [Database.Partition] {
            data.flatMap { $0.partitions }
        }

        var asDocumentReference: [Database.DocumentReference] {
            partitions.map {
                .init(id: $0.documentId, partitionId: $0.id, ownerId: $0.ownerId)
            }
        }
    }

    struct SearchChatResult {
        var context: [String]
        var adjustments: [SinatraAdjustment]
        var references: [Database.DocumentReference]
        var partitions: [Database.Partition]
        var shardStats: [SearchShardStat]

        init(
            context: [String],
            adjustments: [SinatraAdjustment],
            references: [Database.DocumentReference],
            partitions: [Database.Partition] = [],
            shardStats: [SearchShardStat] = []
        ) {
            self.context = context
            self.adjustments = adjustments
            self.references = references
            self.partitions = partitions
            self.shardStats = shardStats
        }
    }
}
