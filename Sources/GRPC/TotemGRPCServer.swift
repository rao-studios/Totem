import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2
import Logging

@available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *)
actor TotemGRPCServer {
    private var serverTask: Task<Void, Error>?

    func start(
        database: Database,
        embeddingProvider: any EmbeddingProviding,
        grpcPort: Int
    ) {
        let query   = TotemQueryServiceImpl(database: database, embeddingProvider: embeddingProvider)
        let library = TotemLibraryServiceImpl(database: database)
        let hnsw    = TotemHNSWServiceImpl(database: database)
        serverTask = Task {
            let transport = HTTP2ServerTransport.Posix(
                address: .ipv4(host: "0.0.0.0", port: grpcPort),
                transportSecurity: .plaintext
            )
            let server = GRPCServer(transport: transport, services: [query, library, hnsw])
            database.logger.info("TotemGRPCServer", "gRPC server listening on port \(grpcPort)", service: .startup)
            try await server.serve()
        }
    }

    func stop() {
        serverTask?.cancel()
        serverTask = nil
    }
}
