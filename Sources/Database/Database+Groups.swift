import Foundation

extension Database {
    nonisolated private func groupEntry(groupId: GroupID, in registry: TotemRegistry) -> Database.Group? {
        guard let owner = registry.groupOwners[groupId] else { return nil }
        return registry.ownersGroups[owner]?.first { $0.id == groupId }
    }

    nonisolated func buildGroup(groupId: GroupID, registry: TotemRegistry) -> Database.Group? {
        guard let entry = groupEntry(groupId: groupId, in: registry),
              let owner = registry.groupOwners[groupId]
        else { return nil }

        let docIds = registry.groups[groupId] ?? []
        let documents = docIds.compactMap { document(for: $0) }
        let totalEarnings = registry.totalEarnings(for: groupId)

        return Database.Group(
            id:            entry.id,
            label:         entry.label,
            ownerId:       owner.id,
            documents:     documents,
            access:        registry.groupAccess[groupId],
            totalEarnings: totalEarnings,
            metadata:      entry.metadata
        )
    }
}
