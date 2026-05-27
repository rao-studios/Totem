import ArgumentParser
import Foundation
import Logging
import Hummingbird

func configureRoutes(
    _ router: Router<TotemRequestContext>,
    _ database: Database,
    embeddingModelProvider: any EmbeddingProviding
) {
    registerHealthRoute(router)
    registerSearchRoute(router, database, embeddingModelProvider: embeddingModelProvider)
    registerBatchEmbeddingsRoute(router, database, embeddingModelProvider: embeddingModelProvider)
    registerLibraryRoute(router, database)
    registerHNSWRoutes(router, database)
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

    @ArgumentParser.Option(name: .long, help: "Fixed node UUID. Overrides any persisted node-id on disk.")
    var nodeId: String?

    enum CodingKeys: CodingKey {
        case host, port, grpcPort, mothershipHost, mothershipGrpcPort, nodeId
        #if canImport(MLX)
        case useMLX, mlxModel
        #endif
    }

    @MainActor
    func run() async throws {
        let (router, app) = setupApplication()

        let fixedNodeId = nodeId.flatMap { UUID(uuidString: $0) }
        let database = Database(nodeId: fixedNodeId)
        let embeddingModelProvider: any EmbeddingProviding = makeEmbeddingProvider()

        configureRoutes(router, database, embeddingModelProvider: embeddingModelProvider)

        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *) {
            let grpcServer = TotemGRPCServer()
            await grpcServer.start(database: database, embeddingProvider: embeddingModelProvider, grpcPort: grpcPort)

            if !mothershipHost.isEmpty {
                var logger = Logger(label: "totem")
                logger.logLevel = .debug
                let dispatcher = MothershipRequestDispatcher(
                    database: database,
                    embeddingProvider: embeddingModelProvider,
                    logger: logger
                )
                let client = MothershipRegistrationClient(
                    mothershipHost: mothershipHost,
                    mothershipGRPCPort: mothershipGrpcPort,
                    totemId: database.nodeId,
                    totemHost: host,
                    totemGRPCPort: grpcPort,
                    totemHTTPPort: port,
                    requestDispatcher: dispatcher,
                    logger: logger
                )
                await client.startHeartbeatLoop()
                registerAvailabilityRoute(router, registrationClient: client)
            }
        }

        do {
            try await app.runService()
        } catch {
            await database.shutdown()
            throw error
        }
        await database.shutdown()
    }

    private func makeEmbeddingProvider() -> any EmbeddingProviding {
        var logger = Logger(label: "totem")
        logger.logLevel = .debug
        #if canImport(MLX)
        if useMLX {
            logger.info("Embedding backend: MLX (\(mlxModel))")
            return MLXEmbeddingModelProvider(modelId: mlxModel)
        }
        #endif
        logger.info("Embedding backend: Mistral API (mistral-embed)")
        return EmbeddingModelProvider(logger: logger)
    }

    private func setupApplication() -> (Router<TotemRequestContext>, Application<RouterResponder<TotemRequestContext>>) {
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardOutput(label: label)
            handler.logLevel = .debug
            return handler
        }

        var logger = Logger(label: "totem")
        logger.logLevel = .debug

        let router = Router(context: TotemRequestContext.self)

        router.middlewares.add(CORSMiddleware(
            allowOrigin: .all,
            allowHeaders: [.accept, .authorization, .contentType, .origin],
            allowMethods: [.get, .post, .options]
        ))

        let app = Application(
            router: router,
            configuration: .init(
                address: .hostname(host, port: port),
                serverName: "Totem"
            ),
            logger: logger
        )

        let totemLogger = TotemLogger(logger)
        totemLogger.info("Startup", "Totem starting on http://\(host):\(port)", service: .startup)

        return (router, app)
    }
}
