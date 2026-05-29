import Foundation

extension Database {
    nonisolated func search(_ queryData: [EmbeddingData],
                queryTagEmbedding: [Float]? = nil,
                database: DatabaseRequest) -> SearchResult {
        guard var table = self.table else {
            logger.debug("Search", "Index could not be retrieved to search for partitions.", service: .database, request: database)
            return .init(data: [], adjustments: [], shardStats: [])
        }

        let compiled = queryData.map {
            if case let .floats(array) = $0.embedding {
                return array
            } else {
                return []
            }
        }

        var data: [PartitionSearchResult] = []
        var adjustments: [SinatraAdjustment] = []
        var shardStats: [SearchShardStat] = []

        if let registry = self.registry {
            let loader: PartitionDataLoader = { [self] docId, partId in
                self.partitionData(documentId: docId, partitionId: partId)
            }
            for embedding in compiled {
                let result = table.search(embedding: embedding,
                                          queryTagEmbedding: queryTagEmbedding,
                                          sinatra: sinatra,
                                          registry: registry,
                                          request: database,
                                          metadataLoader: loader,
                                          logger: logger)
                data.append(contentsOf: result.partitions)
                adjustments.append(contentsOf: result.adjustments)
                shardStats = result.shardStats
            }
        }

        let globalEf = table.activeShard.efSearch
        let globalEma = table.activeShard.emaExplored
        Task { [tableMutator] in
            await tableMutator.syncEf(efSearch: globalEf, emaExplored: globalEma)
        }

        return SearchResult(data: data, adjustments: adjustments, shardStats: shardStats)
    }

    nonisolated func search(_ query: String?,
                request: DatabaseRequest,
                embeddingModelProvider: (any EmbeddingProviding)?,
                topK: Int = 3) async throws -> SearchChatResult {
        guard let query else {
            logger.info("Search", "No query to search.", service: .database, request: request, flow: .chat)
            return SearchChatResult(context: [], adjustments: [], references: [])
        }

        let requestTags = request.tags ?? []
        let queryEmbedding: [EmbeddingData]
        let queryTagEmbedding: [Float]?

        if requestTags.isEmpty {
            if let embeddingModelProvider {
                queryEmbedding = try await StandaloneGeneration
                    .runEmbedding([query], modelProvider: embeddingModelProvider,
                                  logger: logger.base, priority: true)
            } else {
                queryEmbedding = try await StandaloneGeneration
                    .runAPIEmbedding([query], logger: logger.base)
            }
            queryTagEmbedding = nil
        } else {
            let queryTagString = requestTags.sorted().joined(separator: " ")
            let allQueryData: [EmbeddingData]
            if let embeddingModelProvider {
                allQueryData = try await StandaloneGeneration
                    .runEmbedding([query, queryTagString],
                                  modelProvider: embeddingModelProvider,
                                  logger: logger.base, priority: true)
            } else {
                allQueryData = try await StandaloneGeneration
                    .runAPIEmbedding([query, queryTagString], logger: logger.base)
            }
            queryEmbedding = Array(allQueryData.prefix(1))
            queryTagEmbedding = allQueryData.dropFirst().first.flatMap {
                if case .floats(let v) = $0.embedding { return v }
                return nil
            }
        }

        await nonisolatedTableMutator.waitForIndices()

        let result = await withCheckedContinuation { (continuation: CheckedContinuation<SearchResult, Never>) in
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                let r = self.search(queryEmbedding, queryTagEmbedding: queryTagEmbedding, database: request)
                continuation.resume(returning: r)
            }
        }
        let partitions = result.partitions

        logger.info(
            "Search",
            "Retrieved \(partitions.count) partition(s) for query",
            service: .database,
            request: request,
            flow: .chat
        )

        return SearchChatResult(
            context: partitions.map { $0.text },
            adjustments: result.adjustments,
            references: result.asDocumentReference,
            partitions: partitions,
            shardStats: result.shardStats
        )
    }
}
