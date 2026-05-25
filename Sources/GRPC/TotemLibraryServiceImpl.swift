import Foundation
import GRPCCore
import GRPCProtobuf

final class TotemLibraryServiceImpl: Totem_V1_TotemLibrary.SimpleServiceProtocol, Sendable {
    let database: Database

    init(database: Database) {
        self.database = database
    }

    func library(
        request: Totem_V1_TotemLibraryRequest,
        context: GRPCCore.ServerContext
    ) async throws -> Totem_V1_TotemLibraryResponse {
        var groups = database.groups(for: request.ownerID)
        if request.includeAvailable {
            let available = database.availableGroups()
            var seen = Set(groups.map { $0.id })
            for g in available where seen.insert(g.id).inserted {
                groups.append(g)
            }
        }

        // Sort by id for stable cursor-based pagination
        groups.sort { $0.id < $1.id }

        // Advance past the anchor cursor
        if !request.afterID.isEmpty {
            groups = groups.filter { $0.id > request.afterID }
        }

        // Slice the page and record whether more follow
        var hasMore = false
        if request.limit > 0 {
            hasMore = groups.count > Int(request.limit)
            groups = Array(groups.prefix(Int(request.limit)))
        }

        var resp = Totem_V1_TotemLibraryResponse()
        resp.hasMore_p = hasMore
        resp.groups = groups.map { g in
            var pg = Totem_V1_TotemGroup()
            pg.id = g.id
            pg.label = g.label
            pg.ownerID = g.ownerId
            pg.access = g.access?.rawValue ?? ""
            pg.totalEarnings = g.totalEarnings ?? 0
            pg.groupDescription = g.metadata?.description ?? ""
            pg.tags = g.metadata?.tags ?? []
            pg.documents = g.documents.map { doc in
                var pd = Totem_V1_TotemDocument()
                pd.id = doc.id
                pd.url = doc.url.absoluteString
                pd.ownerID = doc.ownerId
                pd.name = doc.name ?? ""
                pd.createdAt = Int64(doc.createdAt.timeIntervalSince1970)
                return pd
            }
            return pg
        }
        return resp
    }
}
