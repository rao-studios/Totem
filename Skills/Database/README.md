# Database

The `Database` actor is Totem's storage and search core. It owns every piece of mutable state: the partition table, the registry, and all HNSW shards. All reads and writes are serialized through this actor.

---

## Component Map

```
Database (actor)
├── TotemRegistry          — ownership, deduplication, access control
│   └── RegistryWAL        — append-only mutation log
├── TableMutator           — serialized writes to PartitionTable
│   └── HNSWTopologyWAL    — append-only graph topology log
└── PartitionTable         — multi-shard search index
    ├── HNSWShard × N      — proximity graph shards
    │   ├── HNSWGraph      — multi-layer approximate nearest-neighbor graph
    │   └── HNSWVectorStore — memory-mapped float32 embeddings
    └── PartitionIndex × M — per-document index
        └── PartitionQuantizer — PQ codebooks + ADC
```

---

## Registry

`TotemRegistry` is the ownership and access-control layer. Every document must be registered before it can be indexed or searched.

| Concept | Description |
|---|---|
| `ownerId` | Arbitrary string identifying the caller — passed in the request, no auth |
| `DocumentID` | SHA-256 hash of the document's text chunks — content-addressed |
| `GroupID` | Owner-defined namespace bundling documents together |
| `Access` | `.available` (globally searchable) or `.restricted` (owner-only, default) |

**Deduplication**: if two owners submit identical text, only one copy of the vectors is stored. The second owner is linked to the existing document via `linkOwner`.

### Registry mutations (all WAL-backed)

- `register(document:owner:group:)` — create entry, assign to owner
- `linkOwner(_:document:)` — attach a second owner to an existing document
- `updateAccess(_:document:owner:)` — change `.restricted` ↔ `.available`

---

## PartitionTable

`PartitionTable` is a multi-shard container. Each document maps to one `PartitionIndex`. The HNSW graph lives in one or more `HNSWShard` instances that span all documents.

### PartitionIndex

Holds everything for one document:

- Raw text chunks
- PQ-compressed embeddings (`PartitionQuantizer`)
- Tags embedding (for tag pre-filtering during search)
- Learned codebooks (trained from the document's own partitions)
- Optional metadata blob

### HNSWShard / HNSWGraph

Approximate nearest-neighbor proximity graph. Each shard auto-spawns when the active shard crosses the configured node-count threshold (`AppConstants.hnswShardLimit`).

**Insertion**: new nodes enter at a random max level drawn from a geometric distribution. The node is linked into each layer from the top down using a greedy beam search (`ef` candidates) to find the best neighbors. Layer 0 gets the full neighbor list; upper layers are pruned to `M` connections.

**Search**: queries descend greedily through upper layers, then run a bounded beam search at layer 0. Returns the top-K nearest partition IDs by squared-L2 distance.

**Shard fan-out**: `Database+Search` fans the query to all shards and merges results. Seer fans the same query to all registered Totem nodes and merges those.

---

## Product Quantization (PQ)

`PartitionQuantizer` compresses 1024-float embeddings into compact `UInt16` code sequences.

- The embedding is split into `subvectorCount` equal chunks.
- Each chunk is encoded against an independent k-means codebook trained on the document's own partitions.
- Codebook size scales with training corpus: small documents use k=2; large shared indices may reach k=2048.
- Search uses **Asymmetric Distance Computation (ADC)**: pre-compute distance tables from the query to all centroids, then scan compressed codes in tight loops — no full vector loads needed.
- `adaptiveThreshold` is calibrated per-document from reconstruction errors and overrides the static per-codec fallback at search time.

---

## Search Flow

1. `Database+Search.search(request:)` receives a query text or precomputed embedding.
2. If text: embed via `EmbeddingModelProvider`.
3. Fan-out to all `HNSWShard` instances; merge candidate partition IDs.
4. For each candidate: load full text from `PartitionIndex`, apply access filter via `TotemRegistry`.
5. Return top-K `(text, partitionId, distance)` tuples.

---

## Index Flow

1. `Database+Put.index(request:)` receives text chunks and owner metadata.
2. Embed all chunks via `EmbeddingModelProvider`.
3. Register document with `TotemRegistry` (or link existing if deduplicated).
4. `TableMutator` writes to `PartitionTable`: create/update `PartitionIndex`, insert nodes into active `HNSWShard`.
5. WAL entries appended for both topology and registry mutations.

---

## Key Files

| File | Purpose |
|---|---|
| [Database.swift](../../Sources/Database/Database.swift) | Actor declaration, startup, WAL replay |
| [Database+Index.swift](../../Sources/Database/Database+Index.swift) | Index orchestration |
| [Database+Search.swift](../../Sources/Database/Commands/Database+Search.swift) | Search fan-out and merge |
| [Database+Registry.swift](../../Sources/Database/Database+Registry.swift) | Registry delegation |
| [PartitionTable.swift](../../Sources/Database/PartitionTable/PartitionTable.swift) | Multi-shard container |
| [PartitionIndex.swift](../../Sources/Database/PartitionTable/PartitionIndex.swift) | Per-document index |
| [PartitionQuantizer.swift](../../Sources/Database/PartitionTable/PartitionQuantizer.swift) | PQ codebooks and ADC |
| [HNSWGraph.swift](../../Sources/Database/PartitionTable/HNSW/HNSWGraph.swift) | Graph insert and beam search |
| [HNSWShard.swift](../../Sources/Database/PartitionTable/HNSW/HNSWShard.swift) | Shard lifecycle and auto-spawn |
