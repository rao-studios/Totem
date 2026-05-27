import Foundation
import Hummingbird

func registerSearchRoute(
    _ app: some RouterMethods<TotemRequestContext>,
    _ database: Database,
    embeddingModelProvider: any EmbeddingProviding
) {
    app.post("/v1/search") { request, context async throws -> SearchResponse in
        let searchRequest = try await request.decode(as: SearchRequest.self, context: context)
        let searchReqId = "search-\(UUID().uuidString)"

        context.logger.info("Received search request (ID: \(searchReqId)) for model: \(searchRequest.model ?? "Default")")

        let result = try await database.search(
            searchRequest.query,
            request: searchRequest.totem.withRequestID(context.id),
            embeddingModelProvider: embeddingModelProvider
        )

        return .init(
            texts: result.context,
            references: result.references,
            shardStats: result.shardStats.isEmpty ? nil : result.shardStats
        )
    }
}
