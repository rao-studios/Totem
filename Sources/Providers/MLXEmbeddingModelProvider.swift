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
    private let batchSize: Int

    // MARK: - Preprocessing slots (mirrors EmbeddingModelProvider)

    private let maxPreprocessConcurrent = 30
    private var preprocessActiveCount = 0
    private var preprocessWaiters: [CheckedContinuation<Void, Never>] = []

    init(
        modelId: String = "mlx-community/snowflake-arctic-embed-m-v1.5",
        batchSize: Int = 32
    ) {
        self.modelId = modelId
        self.batchSize = batchSize
    }

    // MARK: - EmbeddingProviding

    func run(
        _ texts: [String],
        logger: Logger,
        priority: Bool = false
    ) async throws -> (result: [EmbeddingData], usage: Requests.Embedding.Get.Result.Usage) {
        let container = try await loadedModel(logger: logger)
        let localBatchSize = batchSize

        return await container.perform { model, tokenizer in
            var allData: [EmbeddingData] = []
            var promptTokens = 0
            var index = 0

            for batchStart in stride(from: 0, to: texts.count, by: localBatchSize) {
                let batchEnd = min(batchStart + localBatchSize, texts.count)
                let batch = Array(texts[batchStart..<batchEnd])

                let tokenized = batch.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
                promptTokens += tokenized.reduce(0) { $0 + $1.count }

                let maxLen = tokenized.map { $0.count }.max() ?? 16
                let padId  = tokenizer.eosTokenId ?? 0

                let paddedArrays = tokenized.map { tokens in
                    MLXArray(tokens + Array(repeating: padId, count: maxLen - tokens.count))
                }
                guard !paddedArrays.isEmpty else { continue }

                let padded          = MLX.stacked(paddedArrays)
                let attentionMask   = padded .!= MLXArray(padId)
                let tokenTypeIds    = MLXArray.zeros(like: padded)

                let output     = model(padded, positionIds: nil, tokenTypeIds: tokenTypeIds, attentionMask: attentionMask)
                let embeddings = output.textEmbeds

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
        logger.info("Loading MLX embedding model: \(modelId)")
        let config    = ModelConfiguration(id: modelId)
        let container = try await loadModelContainer(configuration: config)
        MLX.GPU.set(cacheLimit: 20 * 1_024 * 1_024)
        loadedContainer = container
        logger.info("MLX embedding model ready: \(modelId)")
        return container
    }
}
#endif
