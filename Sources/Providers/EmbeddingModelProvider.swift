import Foundation
import Logging

// MARK: - Protocol

/// Abstraction over the embedding back-end so tests can substitute a mock
/// without hitting the network.  The three methods below are the only ones
/// called from route handlers; `EmbeddingModelProvider` satisfies this
/// protocol through its normal actor-isolated implementations.
protocol EmbeddingProviding: Actor {
    func acquirePreprocessSlot() async
    func releasePreprocessSlot() async
    func run(
        _ texts: [String],
        logger: Logger,
        priority: Bool
    ) async throws -> (result: [EmbeddingData], usage: Requests.Embedding.Get.Result.Usage)
}

// MARK: - Concrete provider

/// Provider for managing embedding models with caching and reuse.
actor EmbeddingModelProvider: EmbeddingProviding {
    private let logger: Logger
    private let network: NetworkService

    // MARK: - Concurrency control (embedding API slots)

    /// Mistral rejects requests with more than this many inputs.
    static let maxInputsPerBatch = 256

    /// Maximum simultaneous requests to the embedding API.
    /// Keeps the Mistral endpoint from being flooded when many documents
    /// are indexed or multiple chat requests arrive at the same time.
    private let maxConcurrent = 3
    private var activeCount = 0

    /// Priority waiters (search queries) are woken before normal waiters (bulk
    /// indexing). This prevents a surge of indexing calls from delaying search.
    private var priorityWaiters: [CheckedContinuation<Void, Never>] = []
    private var normalWaiters: [CheckedContinuation<Void, Never>] = []

    // MARK: - Concurrency control (preprocessing slots)

    /// Maximum concurrent preprocessing tasks across ALL in-flight batch requests.
    /// A per-request cap does not help when multiple requests run simultaneously —
    /// 5 requests × 20 per-request = 100 concurrent tasks. This shared limit
    /// bounds system-wide preprocessing regardless of how many requests are active.
    private let maxPreprocessConcurrent = 30
    private var preprocessActiveCount = 0
    private var preprocessWaiters: [CheckedContinuation<Void, Never>] = []

    // MARK: - Request coalescing

    /// In-flight tasks keyed by the joined input texts (NUL-separated).
    /// Concurrent calls for identical inputs share one network round-trip.
    private typealias EmbedResult = (result: [EmbeddingData], usage: Requests.Embedding.Get.Result.Usage)
    private var inflightTasks: [String: Task<EmbedResult, Error>] = [:]

    init(logger: Logger) {
        self.logger = logger
        self.network = NetworkService(logger: logger)
    }

    // MARK: - Public API

    /// - Parameter priority: Pass `true` for interactive search queries so they
    ///   are served before any queued bulk-indexing requests.
    func run(
        _ texts: [String],
        logger: Logger,
        priority: Bool = false
    ) async throws -> (result: [EmbeddingData], usage: Requests.Embedding.Get.Result.Usage) {
        logger.debug("Generating embeddings for \(texts.count) text(s)")

        let key = texts.joined(separator: "\u{0000}")

        // Coalesce: if an identical request is already in-flight, share its result.
        if let existing = inflightTasks[key] {
            return try await existing.value
        }

        // Create and register a new task before suspending so any concurrent
        // caller that arrives while we await will find it and join.
        let task = Task<EmbedResult, Error> {
            await self.acquireSlot(priority: priority)
            try Task.checkCancellation()

            let outcome: Result<EmbedResult, Error>
            do {
                let chunks = texts.chunked(by: Self.maxInputsPerBatch)
                var allData: [EmbeddingData] = []
                allData.reserveCapacity(texts.count)
                var totalUsage = Requests.Embedding.Get.Result.Usage.zero

                for (chunkIndex, chunk) in chunks.enumerated() {
                    let response = try await self.network.request(
                        Requests.Embedding.Get(
                            input: chunk,
                            model: "mistral-embed",
                            encodingFormat: "float"
                        )
                    )
                    let offset = chunkIndex * Self.maxInputsPerBatch
                    allData += response.data.map {
                        EmbeddingData(embedding: .floats($0.embedding), index: $0.index + offset)
                    }
                    totalUsage = totalUsage.adding(response.usage)
                }

                logger.debug("Received embeddings successfully (\(chunks.count) batch(es))")
                outcome = .success((allData, totalUsage))
            } catch {
                outcome = .failure(error)
            }

            self.releaseSlot()
            self.removeInflight(key: key)
            return try outcome.get()
        }

        inflightTasks[key] = task
        return try await task.value
    }

    // MARK: - Private helpers

    /// Acquires a concurrency slot, suspending if all slots are in use.
    /// Priority callers are appended to the priority queue and woken before
    /// any normal (indexing) waiters when a slot becomes available.
    private func acquireSlot(priority: Bool) async {
        if activeCount < maxConcurrent {
            activeCount += 1
            return
        }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            if priority {
                priorityWaiters.append(continuation)
            } else {
                normalWaiters.append(continuation)
            }
        }
        // activeCount is unchanged: releaseSlot passes its slot directly to us.
    }

    /// Releases a concurrency slot, waking the next waiter in priority order.
    private func releaseSlot() {
        if let waiter = priorityWaiters.first {
            priorityWaiters.removeFirst()
            waiter.resume()
        } else if let waiter = normalWaiters.first {
            normalWaiters.removeFirst()
            waiter.resume()
        } else {
            activeCount -= 1
        }
    }

    private func removeInflight(key: String) {
        inflightTasks.removeValue(forKey: key)
    }

    // MARK: - Preprocessing slot API

    /// Acquires a server-level preprocessing slot, suspending if all slots are taken.
    /// Call before any per-input work (text splitting, sanitization, hash computation)
    /// in a batch preprocessing task group.
    func acquirePreprocessSlot() async {
        if preprocessActiveCount < maxPreprocessConcurrent {
            preprocessActiveCount += 1
            return
        }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            preprocessWaiters.append(continuation)
        }
        // preprocessActiveCount is unchanged: releasePreprocessSlot passes its slot directly.
    }

    /// Releases a preprocessing slot, waking the next waiter if one is queued.
    func releasePreprocessSlot() {
        if let waiter = preprocessWaiters.first {
            preprocessWaiters.removeFirst()
            waiter.resume()
        } else {
            preprocessActiveCount -= 1
        }
    }
}

private extension Array {
    func chunked(by size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

private extension Requests.Embedding.Get.Result.Usage {
    static var zero: Self {
        .init(
            promptAudioSeconds: nil,
            promptTokens: 0,
            totalTokens: 0,
            completionTokens: 0,
            requestCount: nil,
            promptTokenDetails: nil
        )
    }

    func adding(_ other: Self) -> Self {
        .init(
            promptAudioSeconds: nil,
            promptTokens: promptTokens + other.promptTokens,
            totalTokens: totalTokens + other.totalTokens,
            completionTokens: completionTokens + other.completionTokens,
            requestCount: nil,
            promptTokenDetails: nil
        )
    }
}
