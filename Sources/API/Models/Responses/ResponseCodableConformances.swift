// Hummingbird ResponseCodable conformances for all shared response types.
// Keeps model files framework-agnostic while wiring up automatic JSON encoding
// via context.responseEncoder (JSONEncoder with ISO-8601 dates).
import Hummingbird

// MARK: - Search
extension SearchShardStat: ResponseCodable {}
extension SearchResponse: ResponseCodable {}

// MARK: - HNSW
extension HNSWGraphStats: ResponseCodable {}
extension HNSWStatsResponse: ResponseCodable {}
extension HNSWNodeResponse: ResponseCodable {}
extension HNSWGraphResponse: ResponseCodable {}
extension HNSWNodeBatchResponse: ResponseCodable {}
extension HNSWNodeDeleteResponse: ResponseCodable {}

// MARK: - Embeddings
extension EmbeddingResponse: ResponseCodable {}
extension EmbeddingBatchResponse: ResponseCodable {}
