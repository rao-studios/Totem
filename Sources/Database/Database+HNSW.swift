//
//  Database+HNSW.swift
//  database-server
//
//  Created by Ritesh Pakala on 3/13/26.
//

import Foundation

extension Database {
    // MARK: - Startup

    func initializeHNSW() async {
        await self.deduplicateGlobalHNSW()
        await self.compactGlobalHNSW()
    }

    // MARK: - Dedup

    @discardableResult
    func deduplicateGlobalHNSW() async -> Int {
        let removed = await tableMutator.deduplicateCrossShardNodes()
        if removed > 0 {
            logger.info(
                "HNSW Dedup",
                "⚜️ Global — removed \(removed) cross-shard duplicate document(s)",
                service: .database
            )
        }
        return removed
    }

    // MARK: - Global HNSW

    func compactGlobalHNSW() async {
        let result = await tableMutator.compact()
        let didChange = result.removedNodes > 0 || result.demotedEmptyHubs > 0
        guard didChange else {
            let snapshot  = table
            let liveNodes = snapshot?.shards.reduce(0) { acc, shard in acc + shard.nodes.filter { !$0.isDeleted }.count } ?? 0
            let maxLevel  = snapshot?.shards.map { $0.maxLevel }.max() ?? 0
            logger.debug(
                "HNSW Compact",
                "⚜️ Global graph is clean — nodes=\(liveNodes) maxLevel=\(maxLevel) (nothing compacted)",
                service: .database
            )
            return
        }
        logger.info(
            "HNSW Compact",
            "⚜️ Global — compacted \(result.removedNodes) deleted node(s) (\(result.removedHubs) hub(s)), demoted \(result.demotedEmptyHubs) empty hub(s) — \(result.beforeNodes) → \(result.afterNodes) nodes, maxLevel=\(result.afterMaxLevel)",
            service: .database
        )
    }

    // MARK: - Node removal

    func removeNode(partitionId: String, ownerId: String) async {
        guard let table = self.table else {
            logger.warning(
                "⚠️ Partition \(partitionId) not found in HNSW — nothing to remove",
                service: .database
            )
            return
        }

        var documentId: String?
        for shard in table.shards {
            if let nodeIndex = shard.partitionLookup[partitionId] {
                documentId = shard.nodes[nodeIndex].documentId
                break
            }
        }

        guard let documentId else {
            logger.warning(
                "⚠️ Partition \(partitionId) not found in HNSW — nothing to remove",
                service: .database
            )
            return
        }

        logger.info(
            "Remove Node",
            "⚜️ Removing node for partition \(partitionId) and document \(documentId)",
            service: .database
        )

        await remove(documentId: documentId, ownerId: ownerId)

        await compactGlobalHNSW()
    }
}

