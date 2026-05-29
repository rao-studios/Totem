#if canImport(MLX)
import Foundation
import Logging
import MLX
import mlx_embeddings

/// On-device embedding provider backed by an MLX model loaded via the Hub.
///
/// Activated with `--use-mlx` at server startup. Falls back to `EmbeddingModelProvider`
/// (Mistral API) when the flag is absent.
actor MLXEmbeddingModelProvider: EmbeddingProviding {
    private let modelId: String
    private var loadedContainer: ModelContainer?
    private var loadingTask: Task<ModelContainer, Error>?

    static let maxInputsPerBatch = 8
    // Limits attention matrix size: O(seq²). 512 → 4× less memory than 1024.
    static let maxTokensPerSequence = 512

    // MARK: - Preprocessing slots (mirrors EmbeddingModelProvider)

    private let maxPreprocessConcurrent = 30
    private var preprocessActiveCount = 0
    private var preprocessWaiters: [CheckedContinuation<Void, Never>] = []

    init(modelId: String = "mlx-community/snowflake-arctic-embed-m-v1.5") {
        // Enlarge the SDPA kernel cache so varying sequence lengths across
        // sub-batches don't thrash the 256-slot default and trigger a fatal error.
        setenv("MLX_CUDA_SDPA_CACHE_SIZE", "2048", 0)
        self.modelId = modelId
    }

    // MARK: - EmbeddingProviding

    func run(
        _ texts: [String],
        logger: Logger,
        priority: Bool = false
    ) async throws -> (result: [EmbeddingData], usage: Requests.Embedding.Get.Result.Usage) {
        let container = try await loadedModel(logger: logger)

        // container.perform runs on MLX's evaluation thread. Do NOT call any
        // MLX.Memory.* or Stream.* APIs from inside this closure — they
        // re-enter the CUDA allocator or CommandEncoder while it is still
        // active and cause SIGSEGV (null-pointer-offset crashes at tiny
        // addresses like 0x6529). All cleanup is done after the closure returns.
        let result = await container.perform { model, tokenizer in
            var allData: [EmbeddingData] = []
            var promptTokens = 0
            var index = 0

            for batchStart in stride(from: 0, to: texts.count, by: Self.maxInputsPerBatch) {
                let batchEnd = min(batchStart + Self.maxInputsPerBatch, texts.count)
                let batch = Array(texts[batchStart..<batchEnd])

                let tokenized = batch.map {
                    Array(tokenizer.encode(text: $0, addSpecialTokens: true).prefix(Self.maxTokensPerSequence))
                }
                promptTokens += tokenized.reduce(0) { $0 + $1.count }

                let rawMax = tokenized.map { $0.count }.max() ?? 16
                // Round up to power-of-2 so SDPA sees a small set of distinct shapes,
                // reducing cache thrash on MLX_CUDA_SDPA_CACHE_SIZE.
                var maxLen = 1
                while maxLen < rawMax { maxLen <<= 1 }
                let padId  = tokenizer.eosTokenId ?? 0

                let paddedArrays = tokenized.map { tokens in
                    MLXArray(tokens + Array(repeating: padId, count: maxLen - tokens.count))
                }
                guard !paddedArrays.isEmpty else { continue }

                let padded        = MLX.stacked(paddedArrays)
                let attentionMask = padded .!= MLXArray(padId)
                let tokenTypeIds  = MLXArray.zeros(like: padded)

                let output     = model(padded, positionIds: nil, tokenTypeIds: tokenTypeIds, attentionMask: attentionMask)
                let embeddings = output.textEmbeds
                MLX.eval(embeddings)

                for i in 0..<embeddings.shape[0] {
                    allData.append(EmbeddingData(embedding: .floats(embeddings[i].asArray(Float.self)), index: index))
                    index += 1
                }
            }

            let usage = Requests.Embedding.Get.Result.Usage(
                promptAudioSeconds: nil,
                promptTokens: promptTokens,
                totalTokens: promptTokens,
                completionTokens: 0,
                requestCount: nil,
                promptTokenDetails: nil
            )
            return (allData, usage)
        }

        // CUDA context is idle here — safe to touch the allocator.
        // Drop cache to zero, flush everything, then restore the 20 MB reserve
        // so the next request starts with a clean pool.
        MLX.Memory.cacheLimit = 0
        MLX.Memory.clearCache()
        MLX.Memory.cacheLimit = 20 * 1_024 * 1_024

        return result
    }

    func acquirePreprocessSlot() async {
        if preprocessActiveCount < maxPreprocessConcurrent {
            preprocessActiveCount += 1
            return
        }
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            preprocessWaiters.append(c)
        }
    }

    func releasePreprocessSlot() {
        if let waiter = preprocessWaiters.first {
            preprocessWaiters.removeFirst()
            waiter.resume()
        } else {
            preprocessActiveCount -= 1
        }
    }

    // MARK: - Private

    private func loadedModel(logger: Logger) async throws -> ModelContainer {
        if let container = loadedContainer { return container }
        if let task = loadingTask { return try await task.value }

        logger.info("Loading MLX embedding model: \(modelId)")
        let modelId = self.modelId
        let task = Task<ModelContainer, Error> {
            let config = ModelConfiguration(id: modelId)
            let container = try await loadModelContainer(configuration: config)
            return container
        }
        loadingTask = task

        do {
            let container = try await task.value
            loadedContainer = container
            loadingTask = nil
            MLX.Memory.cacheLimit = 20 * 1_024 * 1_024
            logger.info("MLX embedding model ready: \(modelId)")
            return container
        } catch {
            loadingTask = nil
            throw error
        }
    }
}
#endif
