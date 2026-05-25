import ArgumentParser
import Foundation
import Logging
import Vapor

func configureRoutes(_ app: Application, _ database: Database, embeddingModelProvider: any EmbeddingProviding) async throws {
    registerHealthRoute(app)
    let protected = app.grouped(Middleware())
    registerSearchRoute(protected, database, embeddingModelProvider: embeddingModelProvider)
    registerBatchEmbeddingsRoute(protected, database, embeddingModelProvider: embeddingModelProvider)
    registerLibraryRoute(protected, database)
    registerHNSWRoutes(protected, database)
}

// Pass-through middleware — no auth in Totem; ownerId comes from the request body.
struct Middleware: AsyncMiddleware {
    func respond(to request: Request, chainingTo next: AsyncResponder) async throws -> Response {
        try await next.respond(to: request)
    }
}

@main
struct TotemServer: AsyncParsableCommand {
    @ArgumentParser.Option(name: .long, help: "Host address.")
    var host: String = AppConstants.defaultHost

    @ArgumentParser.Option(name: .long, help: "Port number.")
    var port: Int = AppConstants.defaultPort

    #if canImport(MLX)
    @ArgumentParser.Flag(name: .long, help: "Use on-device MLX embedding model instead of Mistral API.")
    var useMLX: Bool = false

    @ArgumentParser.Option(name: .long, help: "MLX Hub model ID for on-device embeddings.")
    var mlxModel: String = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    #endif

    @ArgumentParser.Option(name: .long, help: "gRPC server port for Database fan-out calls.")
    var grpcPort: Int = 9090

    @ArgumentParser.Option(name: .long, help: "Mothership (Database) host. Leave empty to run standalone.")
    var mothershipHost: String = "127.0.0.1"

    @ArgumentParser.Option(name: .long, help: "Mothership (Database) gRPC port.")
    var mothershipGrpcPort: Int = 9091

    enum CodingKeys: CodingKey {
        case host, port, grpcPort, mothershipHost, mothershipGrpcPort
        #if canImport(MLX)
        case useMLX, mlxModel
        #endif
    }

    @MainActor
    func run() async throws {
        let app = try await setupApplication()
        app.logger.logLevel = .debug

        let database = Database()
        let embeddingModelProvider: any EmbeddingProviding = makeEmbeddingProvider(logger: app.logger)

        app.routes.defaultMaxBodySize = "100mb"
        configureCORS(app)

        try await configureRoutes(app, database, embeddingModelProvider: embeddingModelProvider)

        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *) {
            let grpcServer = TotemGRPCServer()
            await grpcServer.start(database: database, embeddingProvider: embeddingModelProvider, grpcPort: grpcPort)

            if !mothershipHost.isEmpty {
                let dispatcher = MothershipRequestDispatcher(
                    database: database,
                    embeddingProvider: embeddingModelProvider,
                    logger: app.logger
                )
                let client = MothershipRegistrationClient(
                    mothershipHost: mothershipHost,
                    mothershipGRPCPort: mothershipGrpcPort,
                    totemId: database.nodeId,
                    totemHost: host,
                    totemGRPCPort: grpcPort,
                    totemHTTPPort: port,
                    requestDispatcher: dispatcher,
                    logger: app.logger
                )
                await client.startHeartbeatLoop()
                let protected = app.grouped(Middleware())
                registerAvailabilityRoute(protected, registrationClient: client)
            }
        }

        do {
            try await startServer(app)
        } catch {
            await database.shutdown()
            try? await app.asyncShutdown()
            throw error
        }
        await database.shutdown()
        try? await app.asyncShutdown()
    }

    private func makeEmbeddingProvider(logger: Logger) -> any EmbeddingProviding {
        #if canImport(MLX)
        if useMLX {
            logger.info("Embedding backend: MLX (\(mlxModel))")
            return MLXEmbeddingModelProvider(modelId: mlxModel)
        }
        #endif
        logger.info("Embedding backend: Mistral API (mistral-embed)")
        return EmbeddingModelProvider(logger: logger)
    }

    private func setupApplication() async throws -> Application {
        var env = Environment(name: "production", arguments: ["vapor"])
        try LoggingSystem.bootstrap(from: &env)
        return try await Application.make(env)
    }

    private func configureCORS(_ app: Application) {
        let cors = CORSMiddleware.Configuration(
            allowedOrigin: .all,
            allowedMethods: [.GET, .POST, .OPTIONS],
            allowedHeaders: [.accept, .authorization, .contentType, .origin, .xRequestedWith, .userAgent, .accessControlAllowOrigin]
        )
        app.middleware.use(CORSMiddleware(configuration: cors))
    }

    private func startServer(_ app: Application) async throws {
        app.http.server.configuration.hostname = host
        app.http.server.configuration.port = port
        let logger = TotemLogger(app.logger)
        logger.info("Startup", "Totem starting on http://\(host):\(port)", service: .startup)
        try await app.execute()
    }
}
