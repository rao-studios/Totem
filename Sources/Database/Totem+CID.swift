import Foundation

extension Database {
    /// Produces a globally unique CID for a partition by prepending this Totem's
    /// persistent node UUID to the local content-addressed hash.
    ///
    /// Format: "{totemUUID}-{localHash}"
    ///
    /// The Totem UUID comes from `nodeId` (loaded from `totem-db/node-id` at startup),
    /// ensuring CIDs are unique across all nodes in the network without coordination.
    nonisolated func totemCID(localId: String) -> String {
        "\(nodeId.uuidString)-\(localId)"
    }
}
