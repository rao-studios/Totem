//
//  RegistryMutator.swift
//  database-server
//
//  Created by Ritesh Pakala Rao on 3/13/26.
//

import Foundation

/// Serializes all read-modify-write operations on the TotemRegistry file.
///
/// Without this actor, concurrent embedding requests each load the same registry
/// state, apply their mutations locally, and save — last write wins, silently
/// dropping every other concurrent registration. Groups and documents registered
/// in the "lost" writes disappear without any purge being called.
///
/// Mirrors `TableMutator`: every method that mutates the registry must go through
/// this actor so each operation sees the full result of the previous one.
actor RegistryMutator {
    private let cache: TotemCache<TotemRegistry>
    private let logger: TotemLogger

    // MARK: - WAL (Phase 5 — append-only registry mutations)
    //
    // High-frequency mutations (register, linkOwner, accumulateEarnings,
    // accumulatePerformance) append a small binary record to the WAL instead of
    // triggering a full PropertyList rewrite of the entire registry.
    //
    // A checkpoint (full plist save + WAL truncation) fires when:
    //   - the WAL grows beyond `walCheckpointThreshold`, or
    //   - a cold-path mutation (remove, access update, group rename) runs — these
    //     are infrequent and correctness matters more than write latency there.
    //
    // If the WAL cannot be opened, all paths fall back to the Phase 4 1-second
    // debounced full-save behaviour transparently.

    private var wal:          RegistryWAL?
    private var walByteCount: Int = 0
    /// Maximum WAL file size before a checkpoint is forced (default: 16 MB).
    static let walCheckpointThreshold = 16 * 1024 * 1024

    // MARK: - Debounced disk saves (WAL-unavailable fallback)

    private var registryDirty = false
    private var flushTask: Task<Void, Never>?

    init(logger: TotemLogger, walURL: URL? = FilePersistence.getDefaultURL().appendingPathComponent("registry-wal")) {
        self.cache = TotemCache(
            persistence: FilePersistence(key: "registry", kind: .basic, logger: logger.base)
        )
        self.logger = logger
        self.wal          = walURL.flatMap { try? RegistryWAL(url: $0) }
        self.walByteCount = wal?.byteSize ?? 0
    }

    // MARK: - Startup seeding

    /// Seeds the lock-protected snapshot synchronously at startup — no actor hop required.
    nonisolated func seed(_ initial: TotemRegistry) { cache.seed(initial) }

    // MARK: - Snapshot

    /// Synchronous, lock-protected read of the latest registry state.
    /// Never hops the actor queue — safe to call from any context.
    nonisolated var snapshot: TotemRegistry? { cache.snapshot }

    // MARK: - Private

    private func loadedRegistry() async -> TotemRegistry {
        // WAL is replayed once at startup inside Database.initializeRegistry() before the
        // snapshot is seeded — so the cache is always fully current here.
        return await cache.load { .init() }
    }

    /// Appends `record` to the WAL. If the WAL is unavailable, falls back to a
    /// 1-second debounced full save. Schedules a checkpoint when the WAL crosses
    /// `walCheckpointThreshold`.
    private func appendWAL(_ record: RegistryWALRecord) {
        if let w = wal {
            try? w.append(record)
            walByteCount = w.byteSize
            if walByteCount >= Self.walCheckpointThreshold { checkpoint() }
        } else {
            scheduleSave()
        }
    }

    /// Appends multiple WAL records in one pass, then checks the threshold once.
    private func appendWALBatch(_ records: [RegistryWALRecord]) {
        if let w = wal {
            for r in records { try? w.append(r) }
            walByteCount = w.byteSize
            if walByteCount >= Self.walCheckpointThreshold { checkpoint() }
        } else {
            scheduleSave()
        }
    }

    /// Full checkpoint: save the complete registry to disk, then truncate the WAL.
    /// WAL truncation is deferred until after the save so a crash between the two
    /// leaves the WAL intact — the next startup replays it and the state is recovered.
    private func checkpoint() {
        guard let registry = cache.snapshot else { return }
        walByteCount  = 0
        registryDirty = false
        flushTask?.cancel()
        flushTask     = nil
        let capturedWal = wal
        Task {
            await self.cache.saveNow(registry)
            try? capturedWal?.truncate()
        }
    }

    /// Durably flushes the registry to disk and truncates the WAL.
    /// Call from `Database.shutdown()` to ensure no in-flight checkpoint is lost.
    func flushForShutdown() async {
        guard let registry = cache.snapshot else { return }
        await cache.saveNow(registry)
        try? wal?.truncate()
        walByteCount  = 0
        registryDirty = false
        flushTask?.cancel()
        flushTask     = nil
    }

    private func scheduleSave() {
        registryDirty = true
        guard flushTask == nil else { return }
        flushTask = Task {
            do {
                try await Task.sleep(nanoseconds: 1_000_000_000)
            } catch {
                return
            }
            self.flushIfDirty()
        }
    }

    private func flushIfDirty() {
        guard registryDirty, let registry = cache.snapshot else {
            flushTask = nil
            return
        }
        cache.saveAsync(registry)
        registryDirty = false
        flushTask = nil
    }

    // MARK: - Register

    func register(
        _ document: Database.Document,
        group: Database.Group?,
        ownerId: String
    ) async {
        var registry = await loadedRegistry()
        registry.applyRegister(documentId: document.id, ownerId: ownerId, group: group)
        cache.update(registry)
        appendWAL(.documentRegistered(
            documentId: document.id,
            ownerId:    ownerId,
            group:      group.map { .init(from: $0) }
        ))
    }

    /// Registers multiple documents in a single actor invocation.
    /// Loads the registry once, applies all mutations, then appends one WAL record
    /// per document and checks the threshold once.
    func registerBatch(items: [(document: Database.Document, group: Database.Group?, ownerId: String)]) async {
        var registry = await loadedRegistry()
        for (document, group, ownerId) in items {
            registry.applyRegister(documentId: document.id, ownerId: ownerId, group: group)
        }
        cache.update(registry)
        appendWALBatch(items.map { (document, group, ownerId) in
            .documentRegistered(
                documentId: document.id,
                ownerId:    ownerId,
                group:      group.map { .init(from: $0) }
            )
        })
    }

    // MARK: - Link Owner

    /// Links a new owner to an already-indexed document without re-embedding.
    /// The physical document, vectors, and partition table entry are shared as-is.
    func linkOwner(documentId: DocumentID, group: Database.Group?, ownerId: String) async {
        var registry = await loadedRegistry()
        registry.linkOwner(documentId: documentId, ownerId: ownerId, group: group)
        cache.update(registry)
        appendWAL(.ownerLinked(
            documentId: documentId,
            ownerId:    ownerId,
            group:      group.map { .init(from: $0) }
        ))
    }

    /// Links multiple new owners to existing documents in a single actor invocation.
    func linkOwnerBatch(items: [(documentId: DocumentID, group: Database.Group?, ownerId: String)]) async {
        guard !items.isEmpty else { return }
        var registry = await loadedRegistry()
        for (documentId, group, ownerId) in items {
            registry.linkOwner(documentId: documentId, ownerId: ownerId, group: group)
        }
        cache.update(registry)
        appendWALBatch(items.map { (documentId, group, ownerId) in
            .ownerLinked(
                documentId: documentId,
                ownerId:    ownerId,
                group:      group.map { .init(from: $0) }
            )
        })
    }

    // MARK: - Remove

    /// Unlinks one owner from a document.
    /// Returns `(authorized: true, fullyRemoved: true)` when the caller was the last
    /// owner and the document should be physically deleted (file, table, HNSW).
    /// Returns `(authorized: true, fullyRemoved: false)` when other owners remain —
    /// only the caller's personal HNSW entries need to be cleaned up.
    /// Returns `(authorized: false, ...)` when the caller does not own the document.
    func remove(
        documentId: DocumentID,
        group: Database.Group?,
        ownerId: String
    ) async -> (authorized: Bool, fullyRemoved: Bool) {
        var registry = await loadedRegistry()
        let owner = TotemRegistry.Owner(id: ownerId)
        guard registry.documentOwners[documentId]?.contains(owner) == true else {
            return (false, false)
        }
        let fullyRemoved = registry.remove(documentId: documentId, group: group, owner: owner)
        cache.update(registry)
        checkpoint()
        return (true, fullyRemoved)
    }

    // MARK: - Remove All

    /// Unlinks `ownerId` from all their documents.
    /// Returns:
    ///   - `fullyRemoved`: document IDs where this owner was the last — callers should
    ///     delete the physical file and partition table entry.
    ///   - `allOwned`: every document ID the owner had (superset of `fullyRemoved`) —
    ///     callers should remove all from the personal HNSW.
    func removeAll(ownerId: String) async -> (fullyRemoved: [DocumentID], allOwned: [DocumentID]) {
        var registry = await loadedRegistry()
        let owner = TotemRegistry.Owner(id: ownerId)
        let documentIds = registry.ownersDocuments[owner] ?? []
        let ownedGroupIds = (registry.ownersGroups[owner] ?? []).map { $0.id }

        var fullyRemoved: [DocumentID] = []
        for documentId in documentIds {
            let wasLast = registry.remove(documentId: documentId, group: nil, owner: owner)
            if wasLast { fullyRemoved.append(documentId) }
        }

        for groupId in ownedGroupIds {
            registry.groups.removeValue(forKey: groupId)
            registry.groupOwners.removeValue(forKey: groupId)
            registry.groupAccess.removeValue(forKey: groupId)
        }

        registry.ownersDocuments.removeValue(forKey: owner)
        registry.ownersGroups.removeValue(forKey: owner)
        registry.ownerDocumentGroup.removeValue(forKey: ownerId)

        cache.update(registry)
        checkpoint()
        return (fullyRemoved, documentIds)
    }

    /// Unlinks a specific set of (documentId, ownerId) pairs in a single actor
    /// invocation. Returns the IDs that were fully removed (last owner gone).
    @discardableResult
    func removeBatch(
        items: [(documentId: DocumentID, ownerId: String)]
    ) async -> [DocumentID] {
        _ = await loadedRegistry()
        var registry = cache.snapshot ?? TotemRegistry()
        var fullyRemoved: [DocumentID] = []
        for (documentId, ownerId) in items {
            let owner = TotemRegistry.Owner(id: ownerId)
            guard registry.documentOwners[documentId]?.contains(owner) == true else { continue }
            let wasLast = registry.remove(documentId: documentId, group: nil, owner: owner)
            if wasLast { fullyRemoved.append(documentId) }
        }
        cache.update(registry)
        checkpoint()
        return fullyRemoved
    }

    // MARK: - Document Stats

    /// Accumulates credit earnings into the `documentStats` map for each document in
    /// `earnings`. Appends a WAL record so the full registry is not rewritten.
    func accumulateEarnings(_ earnings: [DocumentID: Gita.Credits]) async {
        guard !earnings.isEmpty else { return }
        var registry = await loadedRegistry()
        registry.addEarnings(earnings)
        cache.update(registry)
        appendWAL(.earningsAccumulated(earnings.map { ($0.key, $0.value) }))
    }

    /// Merges per-document performance updates from a `Sinatra.PrepareResult` into
    /// `documentStats`. Appends a WAL record so the full registry is not rewritten.
    func accumulatePerformance(_ updates: [DocumentID: Database.DocumentStats]) async {
        guard !updates.isEmpty else { return }
        var registry = await loadedRegistry()
        registry.addPerformance(updates)
        cache.update(registry)
        appendWAL(.performanceAccumulated(updates.values.map { .init(from: $0) }))
    }

    // MARK: - Access

    @discardableResult
    func updateDocumentAccess(id: String, ownerId: String, access: TotemRegistry.Access) async -> Bool {
        var registry = await loadedRegistry()
        guard registry.documentOwners[id]?.contains(TotemRegistry.Owner(id: ownerId)) == true else { return false }
        registry.updateDocumentAccess(for: id, state: access)
        cache.update(registry)
        checkpoint()
        return true
    }

    @discardableResult
    func updateGroupAccess(id: String, ownerId: String, access: TotemRegistry.Access) async -> Bool {
        var registry = await loadedRegistry()
        guard registry.groupOwners[id]?.id == ownerId else { return false }
        registry.updateGroupAccess(for: id, state: access)
        let documents = registry.groups[id] ?? []
        for documentId in documents {
            registry.updateDocumentAccess(for: documentId, state: access)
        }
        cache.update(registry)
        checkpoint()
        return true
    }

    /// Replaces the entire in-memory cache and checkpoints immediately.
    func replace(with registry: TotemRegistry) {
        cache.update(registry)
        checkpoint()
    }

    @discardableResult
    func updateGroup(_ group: Database.Group, documentId: String, ownerId: String) async -> Bool {
        let id = group.id
        var registry = await loadedRegistry()
        let owner = TotemRegistry.Owner(id: ownerId)
        guard registry.documentOwners[documentId]?.contains(owner) == true else { return false }

        let oldGroupId = registry.ownerDocumentGroup[ownerId]?[documentId]
            ?? registry.documentGroups[documentId]?.first

        registry.ownerDocumentGroup[ownerId, default: [:]][documentId] = id
        if let oldGroupId {
            registry.documentGroups[documentId]?.remove(oldGroupId)
            if registry.documentGroups[documentId]?.isEmpty == true {
                registry.documentGroups.removeValue(forKey: documentId)
            }
        }
        registry.documentGroups[documentId, default: []].insert(id)

        if let oldGroupId {
            var oldGroupHashes = registry.groups[oldGroupId] ?? []
            oldGroupHashes.removeAll(where: { $0 == documentId })
            registry.groups[oldGroupId] = oldGroupHashes
        }
        let isNewGroup = registry.groupOwners[id] == nil
        var newGroupHashes = registry.groups[id] ?? []
        if !newGroupHashes.contains(documentId) { newGroupHashes.append(documentId) }
        registry.groups[id] = newGroupHashes

        if isNewGroup {
            registry.groupOwners[id] = owner
            registry.groupAccess[id] = group.access ?? .restricted
            var ownerGroups = registry.ownersGroups[owner] ?? []
            ownerGroups.append(
                .init(id: group.id, label: group.label, ownerId: group.ownerId, documents: [], metadata: group.metadata)
            )
            registry.ownersGroups[owner] = ownerGroups
            logger.info("Registry", "New group detected, creating records.", service: .database)
        }

        cache.update(registry)
        checkpoint()
        return true
    }

    /// Updates the metadata for a group. Returns `false` if the caller does not own the group.
    @discardableResult
    func updateGroupMetadata(id: String, ownerId: OwnerID, metadata: Database.Group.Metadata) async -> Bool {
        var registry = await loadedRegistry()
        let owner = TotemRegistry.Owner(id: ownerId)
        guard registry.groupOwners[id]?.id == ownerId else { return false }

        var ownerGroups = registry.ownersGroups[owner] ?? []
        guard let idx = ownerGroups.firstIndex(where: { $0.id == id }) else { return false }
        ownerGroups[idx].metadata = metadata
        registry.ownersGroups[owner] = ownerGroups

        cache.update(registry)
        checkpoint()
        return true
    }

    /// Renames a group. Returns `false` if the caller does not own the group.
    @discardableResult
    func renameGroup(id: String, ownerId: String, label: String) async -> Bool {
        var registry = await loadedRegistry()
        let owner = TotemRegistry.Owner(id: ownerId)
        guard registry.groupOwners[id]?.id == ownerId else { return false }

        var ownerGroups = registry.ownersGroups[owner] ?? []
        guard let idx = ownerGroups.firstIndex(where: { $0.id == id }) else { return false }
        ownerGroups[idx].label = label
        registry.ownersGroups[owner] = ownerGroups

        cache.update(registry)
        checkpoint()
        return true
    }

    /// Removes registry metadata for a set of groups that the caller explicitly
    /// deleted. Called by the groups-purge route after document removal completes.
    func removeGroupEntries(_ groupIds: [GroupID], ownerId: String) async {
        guard !groupIds.isEmpty else { return }
        var registry = await loadedRegistry()
        let owner = TotemRegistry.Owner(id: ownerId)
        for groupId in groupIds {
            guard registry.groupOwners[groupId]?.id == ownerId else { continue }
            registry.groups.removeValue(forKey: groupId)
            registry.groupOwners.removeValue(forKey: groupId)
            registry.groupAccess.removeValue(forKey: groupId)
            var ownerGroups = registry.ownersGroups[owner] ?? []
            ownerGroups.removeAll(where: { $0.id == groupId })
            registry.ownersGroups[owner] = ownerGroups
        }
        cache.update(registry)
        checkpoint()
    }
}
