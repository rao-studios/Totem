import Foundation
import Vapor

struct HealthResponse: Content {
    let status: String
    let timestamp: String
}

func registerHealthRoute(_ app: RoutesBuilder) {
    app.get("health") { req async throws -> HealthResponse in
        return HealthResponse(
            status: "healthy",
            timestamp: ISO8601DateFormatter().string(from: Date())
        )
    }
}