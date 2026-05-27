import Hummingbird

private struct LibraryRequest: Codable {
    let ownerId: String
    let includeAvailable: Bool?
    enum CodingKeys: String, CodingKey {
        case ownerId = "owner_id"
        case includeAvailable = "include_available"
    }
}

private struct LibraryDocumentRequest: Codable {
    let documentId: String
    enum CodingKeys: String, CodingKey {
        case documentId = "document_id"
    }
}

struct LibraryResponse: ResponseCodable {
    let groups: [Database.Group]
}

func registerLibraryRoute(_ app: some RouterMethods<TotemRequestContext>, _ database: Database) {
    app.post("/v1/library") { request, context async throws -> LibraryResponse in
        let body = try await request.decode(as: LibraryRequest.self, context: context)
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

    app.post("/v1/library/document") { request, context async throws -> LibraryResponse in
        let body = try await request.decode(as: LibraryDocumentRequest.self, context: context)
        guard let registry = database.registry else {
            return LibraryResponse(groups: [])
        }
        let groupIds = registry.documentGroups[body.documentId] ?? []
        let groups = groupIds.compactMap { database.buildGroup(groupId: $0, registry: registry) }
        return LibraryResponse(groups: groups)
    }
}
