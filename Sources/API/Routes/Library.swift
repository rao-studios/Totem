import Vapor

private struct LibraryRequest: Content {
    let ownerId: String
    let includeAvailable: Bool?
    enum CodingKeys: String, CodingKey {
        case ownerId = "owner_id"
        case includeAvailable = "include_available"
    }
}

struct LibraryResponse: Content {
    let groups: [Database.Group]
}

func registerLibraryRoute(_ app: RoutesBuilder, _ database: Database) {
    app.post("v1", "library") { req async throws -> LibraryResponse in
        let body = try req.content.decode(LibraryRequest.self)
        var groups = database.groups(for: body.ownerId)
        if body.includeAvailable == true {
            let available = database.availableGroups()
            var seen = Set(groups.map(\.id))
            for g in available where seen.insert(g.id).inserted {
                groups.append(g)
            }
        }
        return LibraryResponse(groups: groups)
    }
}
