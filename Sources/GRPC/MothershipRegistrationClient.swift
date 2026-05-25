import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2
import Logging

actor MothershipRegistrationClient {
    let mothershipHost: String
    let mothershipGRPCPort: Int
    let totemId: UUID
    let totemHost: String
    let totemGRPCPort: Int
    let totemHTTPPort: Int
    let requestDispatcher: MothershipRequestDispatcher
    private let logger: Logger
    private var sessionTask: Task<Void, Never>?

    init(
        mothershipHost: String,
        mothershipGRPCPort: Int,
        totemId: UUID,
        totemHost: String,
        totemGRPCPort: Int,
        totemHTTPPort: Int,
        requestDispatcher: MothershipRequestDispatcher,
        logger: Logger
    ) {
        self.mothershipHost      = mothershipHost
        self.mothershipGRPCPort  = mothershipGRPCPort
        self.totemId             = totemId
        self.totemHost           = totemHost
        self.totemGRPCPort       = totemGRPCPort
        self.totemHTTPPort       = totemHTTPPort
        self.requestDispatcher   = requestDispatcher
        self.logger              = logger
    }

    // MARK: - Lifecycle

    /// Starts the registration + session stream loop. Reconnects automatically on
    /// connection loss. Call once at startup; registration retries until Seer is reachable.
    func startHeartbeatLoop() {
        sessionTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.runSession()
                guard !Task.isCancelled else { break }
                // Back off 5 s before reconnecting after a dropped session.
                try? await Task.sleep(nanoseconds: 5_000_000_000)
            }
        }
    }

    func stop() {
        sessionTask?.cancel()
        sessionTask = nil
    }

    // MARK: - Availability (one-shot, own connection)

    func sendAvailabilityUpdate(acceptingStorage: Bool) async {
        do {
            try await withGRPCClient(
                transport: .http2NIOPosix(
                    target: .ipv4(host: mothershipHost, port: mothershipGRPCPort),
                    transportSecurity: .plaintext
                )
            ) { client in
                let stub = Totem_V1_TotemRegistration.Client(wrapping: client)
                var req = Totem_V1_AvailabilityUpdateRequest()
                req.totemID = self.totemId.uuidString
                req.acceptingStorage = acceptingStorage
                _ = try await stub.updateAvailability(req)
                self.logger.info("MothershipRegistrationClient: availability updated — accepting_storage=\(acceptingStorage)")
            }
        } catch {
            logger.error("MothershipRegistrationClient: availability update failed — \(error)")
        }
    }

    // MARK: - Session loop

    /// Opens one persistent gRPC connection, registers, then opens the bidirectional
    /// session stream. Seer sends fan-out requests down the stream; this Totem dispatches
    /// them locally and sends responses back. Reconnects automatically on failure.
    private func runSession() async {
        do {
            try await withGRPCClient(
                transport: .http2NIOPosix(
                    target: .ipv4(host: mothershipHost, port: mothershipGRPCPort),
                    transportSecurity: .plaintext
                )
            ) { [self] client in
                let stub = Totem_V1_TotemRegistration.Client(wrapping: client)

                // ── 1. Register — retry until accepted or cancelled ──────────
                var registered = false
                while !Task.isCancelled && !registered {
                    do {
                        var req = Totem_V1_RegisterRequest()
                        req.totemID  = totemId.uuidString
                        req.host     = totemHost
                        req.grpcPort = Int32(totemGRPCPort)
                        req.httpPort = Int32(totemHTTPPort)
                        let resp = try await stub.register(req)
                        if resp.accepted {
                            logger.info("MothershipRegistrationClient: registered with mothership \(resp.mothershipID)")
                            registered = true
                        } else {
                            logger.error("MothershipRegistrationClient: registration rejected (invalid totem ID?)")
                            return
                        }
                    } catch {
                        logger.warning("MothershipRegistrationClient: register failed — \(error), retrying in 5 s")
                        try await Task.sleep(nanoseconds: 5_000_000_000)
                    }
                }
                guard registered else { return }

                // ── 2. Bidirectional session stream ──────────────────────────
                let (outgoing, continuation) = AsyncStream.makeStream(of: Totem_V1_TotemSessionMessage.self)
                let myTotemId  = totemId
                let dispatcher = requestDispatcher

                var sessionOptions = GRPCCore.CallOptions.defaults
                sessionOptions.maxRequestMessageBytes = 100 * 1024 * 1024

                try await stub.session(
                    options: sessionOptions,
                    requestProducer: { [self] writer in
                        var ping = Totem_V1_TotemSessionMessage()
                        ping.totemID = myTotemId.uuidString
                        ping.payload = .ping(Totem_V1_TotemSessionPing())
                        logger.info("MothershipRegistrationClient: session stream opened — sending initial ping")
                        try await writer.write(ping)
                        logger.info("MothershipRegistrationClient: initial ping sent — stream active")

                        do {
                            for await msg in outgoing {
                                logger.info("MothershipRegistrationClient: → Seer \(payloadName(msg.payload)) [\(msg.correlationID.prefix(8))]")
                                try await writer.write(msg)
                                logger.info("MothershipRegistrationClient: → Seer write complete \(payloadName(msg.payload)) [\(msg.correlationID.prefix(8))]")
                            }
                        } catch {
                            logger.warning("MothershipRegistrationClient: requestProducer write error — \(error)")
                            throw error
                        }
                        logger.info("MothershipRegistrationClient: requestProducer outgoing channel finished")
                    },
                    onResponse: { [self] streamingResponse in
                        logger.info("MothershipRegistrationClient: onResponse handler entered")

                        let pingTask = Task {
                            while !Task.isCancelled {
                                try? await Task.sleep(nanoseconds: 30_000_000_000)
                                guard !Task.isCancelled else { break }
                                var ping = Totem_V1_TotemSessionMessage()
                                ping.totemID = myTotemId.uuidString
                                ping.payload = .ping(Totem_V1_TotemSessionPing())
                                continuation.yield(ping)
                            }
                            logger.info("MothershipRegistrationClient: pingTask ended")
                        }
                        defer {
                            pingTask.cancel()
                            continuation.finish()
                            logger.info("MothershipRegistrationClient: onResponse defer — pingTask cancelled, outgoing finished")
                        }

                        do {
                            for try await msg in streamingResponse.messages {
                                switch msg.payload {
                                case .pong:
                                    logger.info("MothershipRegistrationClient: ← pong [\(msg.correlationID.prefix(8))]")
                                case .none:
                                    logger.warning("MothershipRegistrationClient: ← message with no payload [\(msg.correlationID.prefix(8))]")
                                default:
                                    let pname = payloadName(msg.payload)
                                    logger.info("MothershipRegistrationClient: ← Seer request \(pname) [\(msg.correlationID.prefix(8))] — dispatching")
                                    Task {
                                        if let resp = await dispatcher.handle(msg) {
                                            logger.info("MothershipRegistrationClient: dispatch complete \(pname) [\(msg.correlationID.prefix(8))] — queuing response")
                                            continuation.yield(resp)
                                        } else {
                                            logger.warning("MothershipRegistrationClient: dispatch returned nil for \(pname) [\(msg.correlationID.prefix(8))]")
                                        }
                                    }
                                }
                            }
                            logger.info("MothershipRegistrationClient: response stream ended cleanly (Seer closed its send side)")
                        } catch {
                            logger.warning("MothershipRegistrationClient: response stream error — \(error)")
                            throw error
                        }
                        return ()
                    }
                )
            }
        } catch is CancellationError {
            // Normal shutdown — don't log.
        } catch {
            logger.warning("MothershipRegistrationClient: session ended — \(error)")
        }
    }
}

// MARK: - Helpers

private func payloadName(_ payload: Totem_V1_TotemSessionMessage.OneOf_Payload?) -> String {
    switch payload {
    case .ping:                  return "ping"
    case .pong:                  return "pong"
    case .searchRequest:         return "searchRequest"
    case .searchResponse:        return "searchResponse"
    case .indexRequest:          return "indexRequest"
    case .indexResponse:         return "indexResponse"
    case .removeRequest:         return "removeRequest"
    case .removeResponse:        return "removeResponse"
    case .libraryRequest:        return "libraryRequest"
    case .libraryResponse:       return "libraryResponse"
    case .hnswStatsRequest:      return "hnswStatsRequest"
    case .hnswStatsResponse:     return "hnswStatsResponse"
    case .hnswGraphRequest:      return "hnswGraphRequest"
    case .hnswGraphResponse:     return "hnswGraphResponse"
    case .hnswNodeBatchRequest:  return "hnswNodeBatchRequest"
    case .hnswNodeBatchResponse: return "hnswNodeBatchResponse"
    case .hnswNodeRequest:       return "hnswNodeRequest"
    case .hnswNodeResponse:      return "hnswNodeResponse"
    case .hnswDeleteNodeRequest:   return "hnswDeleteNodeRequest"
    case .hnswDeleteNodeResponse:  return "hnswDeleteNodeResponse"
    case .updateGroupRequest:      return "updateGroupRequest"
    case .updateGroupResponse:     return "updateGroupResponse"
    case .updateDocumentRequest:   return "updateDocumentRequest"
    case .updateDocumentResponse:  return "updateDocumentResponse"
    case .statsRequest:            return "statsRequest"
    case .statsResponse:           return "statsResponse"
    case .none:                    return "none"
    }
}
