//
//  PersistenceActor.swift
//  database-server
//
//  Created by Ritesh Pakala on 3/20/26.
//

import Foundation
import Logging

/// Serializes all disk I/O for a single file through Swift's actor model.
///
/// `FilePersistence.save` is not thread-safe: two concurrent calls writing to
/// the same URL race on `data.write(to:)` and on the `fileExists → createFile`
/// branch. Wrapping it in an actor guarantees serial execution per file without
/// blocking any thread — callers simply `await` the actor method and the Swift
/// runtime handles the hop.
///
/// One `PersistenceActor` is created per logical file:
///   - `TotemCache<Value>` owns one for the table and one for the registry.
///   - `PersonalHNSWMutator` owns one per `ownerId`.
actor PersistenceActor {
    private let persistence: FilePersistence

    init(persistence: FilePersistence) {
        self.persistence = persistence
    }

    /// Serialized write. Runs on this actor's executor; no two saves overlap.
    func save<T: Codable>(_ value: T) {
        persistence.save(state: value)
    }

    /// Serialized read. Waits for any in-flight save to complete before reading,
    /// so the caller always sees the latest committed state.
    func restore<T: Codable>() -> T? {
        persistence.restore()
    }

    /// Removes the backing file from disk.
    func purge() {
        persistence.purge()
    }
}

// MARK: - IndicesPersistenceActor

/// Serializes all shard-indices saves for a single node.
///
/// `TableMutator.saveIndicesAsync()` is called from seven sites — `scheduleSave`,
/// `flushIndicesIfDirty`, `checkpoint`, `flushIfDirty`, `remove`, `removeAll`,
/// and `replace`. Before this actor was introduced, each call spawned an
/// independent `Task.detached` that would encode and write all N shard-indices
/// files concurrently. With 59 shards at ~500 KB each, multiple concurrent tasks
/// allocated ~29 MB of PropertyList buffers simultaneously, causing OOM crashes
/// inside `FilePersistence.save → data.write(to:options:.atomic)`.
///
/// Routing through this actor ensures at most one save runs at a time and that
/// later calls automatically queue behind the current one.
actor IndicesPersistenceActor {
    private let nodeId: UUID
    private let logger: Logger

    init(nodeId: UUID, logger: Logger) {
        self.nodeId  = nodeId
        self.logger  = logger
    }

    /// Writes each shard's PQ-index dictionary to its own file sequentially.
    /// The actor serialises concurrent callers, so no two writes ever race on
    /// the same `shard-<nodeId>-<i>-indices` path.
    func save(shards: [HNSWShard]) {
        for i in shards.indices {
            FilePersistence(
                key:    "shard-\(nodeId)-\(i)-indices",
                kind:   .basic,
                logger: logger
            ).save(state: shards[i].indices)
        }
    }
}
