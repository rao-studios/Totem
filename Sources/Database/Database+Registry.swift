import Foundation

extension Database {
    func initializeRegistry() {
        let storage = registryStore
        var registry: TotemRegistry = storage.restore() ?? .init()

        let walURL = FilePersistence.getDefaultURL().appendingPathComponent("registry-wal")
        if let w = try? RegistryWAL(url: walURL),
           let records = try? w.readAll(), !records.isEmpty {
            for record in records { record.apply(to: &registry) }
            logger.info(
                "Registry Init",
                "⚜️ Replayed \(records.count) WAL record(s)",
                service: .database
            )
        }

        if registry.availableDocumentIds.isEmpty && !registry.documentAccess.isEmpty {
            registry.availableDocumentIds = Set(
                registry.documentAccess.compactMap { $0.value == .available ? $0.key : nil }
            )
            storage.save(state: registry)
        }

        if registry.availableGroupIds.isEmpty && !registry.groupAccess.isEmpty {
            registry.availableGroupIds = Set(
                registry.groupAccess.compactMap { $0.value == .available ? $0.key : nil }
            )
            storage.save(state: registry)
        }

        let allDocumentIds = Array(registry.documentOwners.keys)
        var initial: [DocumentID: Database.Document] = [:]
        var orphanedIds: [DocumentID] = []

        for documentId in allDocumentIds {
            let store = documentStore(for: documentId)
            guard FileManager.default.fileExists(atPath: store.url.path()) else {
                orphanedIds.append(documentId)
                continue
            }
            if let document: Database.Document = store.restore() {
                initial[documentId] = document
            } else {
                orphanedIds.append(documentId)
            }
        }

        if !orphanedIds.isEmpty {
            for id in orphanedIds { registry.removeOrphaned(documentId: id) }
            storage.save(state: registry)
            logger.info(
                "Registry Init",
                "⚠️ Removed \(orphanedIds.count) orphaned document(s)",
                service: .database
            )
        }

        var staleGroupCount = 0
        for (owner, ownerGroups) in registry.ownersGroups {
            let cleaned = ownerGroups.filter { registry.groupOwners[$0.id] == owner }
            if cleaned.count != ownerGroups.count {
                staleGroupCount += ownerGroups.count - cleaned.count
                registry.ownersGroups[owner] = cleaned
            }
        }
        if staleGroupCount > 0 {
            storage.save(state: registry)
            logger.warning(
                "⚠️ Removed \(staleGroupCount) stale group reference(s) from ownersGroups",
                service: .database
            )
        }

        registryMutator.seed(registry)
        documentCache.seed(initial)
        logger.debug(
            "Registry Init",
            "⚜️ Database Registry initialized — \(allDocumentIds.count - orphanedIds.count) document(s) pre-cached",
            service: .database
        )
    }

    var registryStore: FilePersistence {
        FilePersistence(key: "registry", kind: .basic, logger: logger.base)
    }
}

extension Database {
    func register(_ document: Database.Document,
                  group: Database.Group?,
                  update: DatabaseUpdate? = nil,
                  ownerId: String) async {
        await registryMutator.register(document, group: group, ownerId: ownerId)

        if let update, update.operation == .remove {
            await remove(documentId: update.documentId, group: group, ownerId: ownerId)
        }
    }

    @discardableResult
    func updateDocumentAccess(_ id: String,
                              ownerId: String,
                              access: TotemRegistry.Access) async -> Bool {
        let updated = await registryMutator.updateDocumentAccess(id: id, ownerId: ownerId, access: access)
        if !updated {
            logger.info("Registry", "Owner does not own this document.", service: .database)
        }
        return updated
    }
}

extension Database {
    nonisolated func groups(for ownerId: OwnerID) -> [Database.Group] {
        guard let registry else { return [] }
        let owner = TotemRegistry.Owner(id: ownerId)
        return (registry.ownersGroups[owner] ?? []).compactMap { entry in
            guard entry.ownerId == owner.id else { return nil }
            return buildGroup(entry: entry, registry: registry)
        }
    }

    nonisolated func availableGroups() -> [Database.Group] {
        guard let registry else { return [] }
        return registry.availableGroupIds
            .compactMap { buildGroup(groupId: $0, registry: registry) }
    }

    func stats(for documentId: DocumentID) -> Database.DocumentStats? {
        registry?.documentStats[documentId]
    }

    @discardableResult
    func updateGroupAccess(_ id: String,
                           ownerId: String,
                           access: TotemRegistry.Access) async -> Bool {
        let updated = await registryMutator.updateGroupAccess(id: id, ownerId: ownerId, access: access)
        if !updated {
            logger.info("Registry", "Owner does not own this group.", service: .database)
        }
        return updated
    }

    @discardableResult
    func renameGroup(id: String, ownerId: String, label: String) async -> Bool {
        await registryMutator.renameGroup(id: id, ownerId: ownerId, label: label)
    }

    @discardableResult
    func updateGroupMetadata(id: String, ownerId: String, metadata: Database.Group.Metadata) async -> Bool {
        await registryMutator.updateGroupMetadata(id: id, ownerId: ownerId, metadata: metadata)
    }

    @discardableResult
    func updateGroup(_ group: Database.Group, documentId: String, ownerId: String) async -> Bool {
        await registryMutator.updateGroup(group, documentId: documentId, ownerId: ownerId)
    }

    nonisolated var registry: TotemRegistry? { registryMutator.snapshot }

    nonisolated func coOwners(for documentIds: some Collection<DocumentID>) -> [DocumentID: Set<OwnerID>] {
        guard let reg = registry else { return [:] }
        return documentIds.reduce(into: [:]) { result, docId in
            var owners = Set(reg.documentOwners[docId]?.map(\.id) ?? [])
            for groupId in reg.documentGroups[docId] ?? [] {
                if let groupOwner = reg.groupOwners[groupId] {
                    owners.insert(groupOwner.id)
                }
            }
            guard owners.count > 1 else { return }
            result[docId] = owners
        }
    }
}
