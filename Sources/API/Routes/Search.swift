import Foundation
import Vapor

func registerSearchRoute(
    _ app: RoutesBuilder,
    _ database: Database,
    embeddingModelProvider: any EmbeddingProviding
) {
    app.post("v1", "search") { req async throws -> SearchResponse in
        let searchRequest = try req.content.decode(SearchRequest.self)
        let searchReqId = "search-\(UUID().uuidString)"

        req.logger.info("Received search request (ID: \(searchReqId)) for model: \(searchRequest.model ?? "Default")")

        let result = try await database.search(
            searchRequest.query,
            request: try searchRequest.totem.from(req),
            embeddingModelProvider: embeddingModelProvider
        )

        return .init(
            texts: result.context,
            references: result.references,
            shardStats: result.shardStats.isEmpty ? nil : result.shardStats
        )
    }
}
