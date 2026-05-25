//
//  ProductQuantizer.swift
//  database-server
//
//  Created by Ritesh Pakala on 11/13/25.
//

// IMPORTANT: Needs refinement towards a standardization into a semantic codec.
// A real write-up behind the embedding compression process.
// Since it is lossy, we need to monitor what gets lost in the compression
// process and how it affects search recall. The winning codec drives the
// industry.

// Split 1024-dim vector into 16 chunks of 64 dims
// Create codebook for each chunk (size scaled dynamically at train time)
// Result: 16 small LUTs instead of one impossible LUT

import Foundation

/*
 // Option 1: Finer granularity (better accuracy, slower)  — 1024-dim input
 let numSubvectors = 32  // 32 × 32 = 1024
 let codebookSize = 256  // 64 bytes per document (2 bytes × 32 with UInt16)

 // Option 2: Coarser (faster, less accurate)  — 1024-dim input
 let numSubvectors = 8   // 8 × 128 = 1024
 let codebookSize = 256  // 16 bytes per document (2 bytes × 8 with UInt16)

 // Option 3: Default (calibrated)  — 1024-dim input
 let numSubvectors = 16  // 16 × 64 = 1024
 let codebookSize = 65536  // 32 bytes per document (2 bytes × 16 with UInt16)
*/
struct PartitionQuantizer: Codable {
    var numSubvectors = 16  // Split 1024-dim vector into 16 × 64-dim chunks
    var codebookSize = 65536 // Actual value set dynamically during train(); UInt16 ceiling
    var codebooks: [[[Float]]] = []  // numSubvectors codebooks, codebookSize entries each

    /// Hard ceiling for codebook size. UInt16 encodes up to 65536 centroid indices.
    static let maxCodebookSize: Int = 65536

    /// Minimum codebook size. Below 2 all documents share one centroid — no discrimination.
    static let minCodebookSize: Int = 2

    /// Faiss rule of thumb: at least this many training vectors per centroid for
    /// stable k-means convergence. Used by `scaledCodebookSize(for:)`.
    static let vectorsPerCentroid: Int = 39

    /// Computes the largest power-of-two codebook size that satisfies the
    /// `vectorsPerCentroid` density requirement, clamped to [minCodebookSize, maxCodebookSize].
    /// This lets small per-document indices use k=2–16 while a large global index
    /// can grow all the way to k=65536 as the corpus expands.
    static func scaledCodebookSize(for vectorCount: Int) -> Int {
        let ideal = vectorCount / vectorsPerCentroid
        guard ideal >= minCodebookSize else { return minCodebookSize }
        // Round down to nearest power of two so codebook entries stay aligned.
        var k = 1
        while k * 2 <= ideal && k * 2 <= maxCodebookSize { k *= 2 }
        return k
    }

    /// Calibrated per-subvector distance baseline derived from empirical benchmarks.
    /// Current reference codec: numSubvectors=16, 1024-dim input, codebookSize=65536, UInt16.
    /// When switching to a new codec, re-run held-out retrieval benchmarks and update
    /// this value — both `distanceThreshold` and `defaultDistanceThreshold` will
    /// then reflect the new calibration everywhere (PartitionIndex and Oracle).
    /// Calibrated for 1024-dim Mistral embeddings, 16 subvectors × 64-dim each, codebookSize=65536.
    /// `adaptiveThreshold` (computed per-partition from reconstruction errors during train())
    /// overrides this for trained indices — this is the static fallback only.
    static let calibratedThresholdPerSubvector: Float = 8.0 / 16  // ≈ 0.5 — 1024-dim Mistral

    /// Default numSubvectors used by this codec. Mirrors the stored-property default
    /// so that `defaultDistanceThreshold` can be computed without an instance.
    static let defaultNumSubvectors: Int = 16

    /// Canonical distance threshold for the default codec configuration.
    /// Use this wherever a compile-time constant is needed (e.g. Oracle defaults).
    /// Per-instance threshold is `distanceThreshold`, which adapts when `numSubvectors` changes.
    static let defaultDistanceThreshold: Float = calibratedThresholdPerSubvector * Float(defaultNumSubvectors)

    /// Aggregate PQ distance threshold for this codec configuration.
    /// Scales linearly with `numSubvectors` so the cutoff stays valid when the codec
    /// is re-parameterised. Not persisted — always derived from the current codec config.
    var distanceThreshold: Float {
        Float(numSubvectors) * Self.calibratedThresholdPerSubvector
    }

    /// Per-partition threshold derived from reconstruction errors of the training vectors.
    /// Nil for indexes trained before adaptive calibration was introduced — use `effectiveThreshold`.
    var adaptiveThreshold: Float? = nil

    /// The threshold to use during search. Prefers the adaptive per-partition value;
    /// falls back to the global codec calibration for legacy indexes.
    var effectiveThreshold: Float {
        adaptiveThreshold ?? distanceThreshold
    }

    /// Builds LUTs from document/partition vectors.
    /// - Parameter vectors: The partition embedding vector.
    mutating func train(vectors: [[Float]]) {
        guard !vectors.isEmpty else { return }
        let subvectorDim = vectors[0].count / numSubvectors

        // Scale codebook size to the training corpus so k-means is always well-populated.
        // Small per-doc indices (e.g. a social post with 3 partitions) get k=2;
        // a global index with 100k partitions grows to k=2048 or higher.
        codebookSize = Self.scaledCodebookSize(for: vectors.count)

        for i in 0..<numSubvectors {
            // Extract this chunk from all vectors
            let subvectors = vectors.map { vector in
                Array(vector[(i*subvectorDim)..<((i+1)*subvectorDim)])
            }

            // K-means to create `codebookSize` reference points (the LUT)
            let codebook = kmeans(subvectors, k: codebookSize)

            codebooks.append(codebook)
        }

        // Calibrate an adaptive threshold from the reconstruction errors of the
        // training vectors. Each vector is encoded and the distance back to its
        // centroid representation is its quantization error. The distribution of
        // these errors reflects the geometry of this specific partition's codebook:
        // tight semantic clusters → low errors → tight threshold;
        // broad/noisy content → high errors → looser threshold.
        let errors = vectors.map { computeDistance(queryVector: $0, documentCodes: encode(vector: $0)) }
        let mean = errors.reduce(0, +) / Float(errors.count)
        let variance = errors.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(errors.count)
        adaptiveThreshold = max(mean + 1.5 * sqrt(variance), distanceThreshold)
    }
    
    /// Compressing a vector to `UInt16` codes (one per subvector).
    /// UInt16 supports codebook sizes up to 65536, matching `codebookSize`.
    /// - Parameter vector: The embedding vector.
    /// - Returns: The compressed vector as codebook indices.
    func encode(vector: [Float]) -> [UInt16] {
        var codes: [UInt16] = []
        codes.reserveCapacity(numSubvectors)
        let subvectorDim = vector.count / numSubvectors

        for i in 0..<numSubvectors {
            let start = i * subvectorDim
            let end = start + subvectorDim

            // Find nearest centroid using index arithmetic — no Array allocation.
            var minDist = Float.infinity
            var bestCode = 0
            for j in 0..<codebooks[i].count {
                let centroid = codebooks[i][j]
                var d: Float = 0
                for k in start..<end {
                    let diff = vector[k] - centroid[k - start]
                    d += diff * diff
                }
                if d < minDist {
                    minDist = d
                    bestCode = j
                }
            }
            codes.append(UInt16(bestCode))
        }

        return codes
    }

    /// Build a per-query distance table for Asymmetric Distance Computation (ADC).
    ///
    /// `table[i][j]` = distance from the query's i-th subvector to codebook[i]'s j-th centroid.
    /// Call once per query; pass the result to `computeDistance(table:documentCodes:)`.
    /// Cost: O(numSubvectors × codebookSize × subvectorDim) — amortised across all partitions.
    func buildDistanceTable(queryVector: [Float]) -> [[Float]] {
        let subvectorDim = queryVector.count / numSubvectors
        var table: [[Float]] = []
        table.reserveCapacity(numSubvectors)

        for i in 0..<numSubvectors {
            let start = i * subvectorDim
            let numCentroids = codebooks[i].count
            var row = [Float](repeating: 0, count: numCentroids)
            for j in 0..<numCentroids {
                let centroid = codebooks[i][j]
                var d: Float = 0
                for k in 0..<subvectorDim {
                    let diff = queryVector[start + k] - centroid[k]
                    d += diff * diff
                }
                row[j] = sqrt(d)
            }
            table.append(row)
        }

        return table
    }

    /// O(numSubvectors) distance lookup using a precomputed ADC distance table.
    /// Each partition costs exactly `numSubvectors` array reads and additions.
    /// - Parameters:
    ///   - table: Distance table built by `buildDistanceTable(queryVector:)`.
    ///   - documentCodes: The compressed partition codes.
    /// - Returns: The approximate distance to the query.
    func computeDistance(table: [[Float]], documentCodes: [UInt16]) -> Float {
        var dist: Float = 0
        for i in 0..<numSubvectors {
            dist += table[i][Int(documentCodes[i])]
        }
        return dist
    }

    /// Compute distance between a query vector and compressed document codes.
    /// Used during training for adaptive threshold calibration.
    /// For search, prefer `buildDistanceTable` + `computeDistance(table:documentCodes:)`.
    func computeDistance(queryVector: [Float], documentCodes: [UInt16]) -> Float {
        let table = buildDistanceTable(queryVector: queryVector)
        return computeDistance(table: table, documentCodes: documentCodes)
    }
}

// MARK: - Helpers

private extension PartitionQuantizer {
    // K-means clustering to create codebook (LUT)
    func kmeans(_ vectors: [[Float]], k: Int, maxIterations: Int = 20) -> [[Float]] {
        guard !vectors.isEmpty else { return [] }
        guard vectors.count >= k else {
            // If fewer vectors than k, just return the vectors themselves
            return vectors
        }
        
        let dimension = vectors[0].count
        
        // Initialize centroids using k-means++ for better convergence
        var centroids = kMeansPlusPlusInit(vectors, k: k)
        
        for _ in 0..<maxIterations {
            // Assign each vector to nearest centroid
            var assignments = [Int](repeating: 0, count: vectors.count)
            for (idx, vector) in vectors.enumerated() {
                var minDist = Float.infinity
                var bestCentroid = 0
                
                for (centroidIdx, centroid) in centroids.enumerated() {
                    let dist = distance(vector, centroid)
                    if dist < minDist {
                        minDist = dist
                        bestCentroid = centroidIdx
                    }
                }
                assignments[idx] = bestCentroid
            }
            
            // Update centroids as mean of assigned vectors
            var newCentroids = Array(repeating: [Float](repeating: 0, count: dimension), count: k)
            var counts = [Int](repeating: 0, count: k)
            
            for (idx, assignment) in assignments.enumerated() {
                counts[assignment] += 1
                for d in 0..<dimension {
                    newCentroids[assignment][d] += vectors[idx][d]
                }
            }
            
            // Compute averages
            for i in 0..<k {
                if counts[i] > 0 {
                    for d in 0..<dimension {
                        newCentroids[i][d] /= Float(counts[i])
                    }
                } else {
                    // Empty cluster - reinitialize with random vector
                    newCentroids[i] = vectors.randomElement()!
                }
            }
            
            // Check convergence (if centroids barely moved)
            var maxShift: Float = 0
            for i in 0..<k {
                let shift = distance(centroids[i], newCentroids[i])
                maxShift = max(maxShift, shift)
            }
            
            centroids = newCentroids
            
            if maxShift < 0.001 {
                break  // Converged <3
            }
        }
        
        return centroids
    }
    
    // K-means++ initialization for better clustering
    func kMeansPlusPlusInit(_ vectors: [[Float]], k: Int) -> [[Float]] {
        var centroids: [[Float]] = []
        
        // First centroid: random vector
        centroids.append(vectors.randomElement()!)
        
        // Remaining centroids: choose proportional to squared distance from nearest existing centroid
        for _ in 1..<k {
            var distances = [Float](repeating: Float.infinity, count: vectors.count)
            
            // Find minimum distance to any existing centroid for each vector
            for (idx, vector) in vectors.enumerated() {
                for centroid in centroids {
                    let dist = distance(vector, centroid)
                    distances[idx] = min(distances[idx], dist)
                }
            }
            
            // Square distances for probability distribution
            let squaredDistances = distances.map { $0 * $0 }
            let totalDist = squaredDistances.reduce(0, +)
            
            // Choose next centroid with probability proportional to squared distance
            if totalDist > 0 {
                let target = Float.random(in: 0..<totalDist)
                var cumulative: Float = 0
                
                for (idx, sqDist) in squaredDistances.enumerated() {
                    cumulative += sqDist
                    if cumulative >= target {
                        centroids.append(vectors[idx])
                        break
                    }
                }
            } else {
                // Fallback: just pick a random vector
                centroids.append(vectors.randomElement()!)
            }
        }
        
        return centroids
    }
    
    // Euclidean distance between two vectors
    func distance(_ a: [Float], _ b: [Float]) -> Float {
        assert(a.count == b.count, "Vectors must have same dimension")
        
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        
        return sqrt(sum)
    }
    
    // Faster squared Euclidean distance (if you don't need actual distance value)
    func squaredDistance(_ a: [Float], _ b: [Float]) -> Float {
        assert(a.count == b.count, "Vectors must have same dimension")
        
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        
        return sum
    }
}
