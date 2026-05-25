# Persistence

Totem's on-disk state is split across two write-ahead logs and one memory-mapped binary store. All three survive process restarts and are replayed before any request is served.

---

## HNSWTopologyWAL

Append-only log of graph topology mutations: node insertions and edge updates for every `HNSWShard`.

- Each entry records the partition ID, its level, and all neighbor IDs per layer.
- Written by `TableMutator` before the in-memory `HNSWGraph` is updated.
- On startup, `Database` replays this log to reconstruct the full graph state.

**Tombstoning**: nodes present in the WAL but missing a matching `PartitionIndex` (e.g. after a crash mid-index) are tombstoned by `markIndicesReady` to prevent stale entries from surfacing in search.

File: [HNSWTopologyWAL.swift](../../Sources/Utilities/Persistence/HNSWTopologyWAL.swift)

---

## RegistryWAL

Append-only log of registry mutations: register, linkOwner, updateAccess, earnings updates.

- Each entry captures the full mutation payload needed to reconstruct registry state.
- Written by `RegistryMutator` before the in-memory `TotemRegistry` is updated.
- On startup, `Database` replays this log before `HNSWTopologyWAL` so owner references are valid before graph nodes are inserted.

File: [RegistryWAL.swift](../../Sources/Utilities/Persistence/RegistryWAL.swift)

---

## HNSWVectorStore

Memory-mapped binary file for raw `float32` embeddings.

- Each shard has its own `.mmap` file.
- Embeddings are written once at index time and never mutated in place.
- The vector store is read via `mmap` so the OS page cache handles hot-partition access efficiently.
- Partitions deleted from the graph are soft-deleted in-memory; their mmap slots are reclaimed during compaction.

File: [HNSWVectorStore.swift](../../Sources/Utilities/Persistence/HNSWVectorStore.swift)

---

## WAL Replay Order

1. **RegistryWAL** — owner entries must exist before graph nodes reference them.
2. **HNSWTopologyWAL** — graph nodes are inserted referencing already-restored owner IDs.
3. **`markIndicesReady`** — scans for WAL nodes whose `PartitionIndex` is absent and tombstones them.

Do not reorder these steps. A topology replay before a registry replay will create dangling owner references.

---

## Data Directory

WAL files and mmap stores are created in the process working directory by default. The location is controlled by `AppConstants.dataDirectory`. Do not delete or move these files while the server is running — both the WAL and mmap store are held open by the process.

---

## Key Files

| File | Purpose |
|---|---|
| [HNSWTopologyWAL.swift](../../Sources/Utilities/Persistence/HNSWTopologyWAL.swift) | Graph topology WAL — append and replay |
| [RegistryWAL.swift](../../Sources/Utilities/Persistence/RegistryWAL.swift) | Registry mutation WAL — append and replay |
| [HNSWVectorStore.swift](../../Sources/Utilities/Persistence/HNSWVectorStore.swift) | Memory-mapped float32 vector storage |
| [Persistence.swift](../../Sources/Utilities/Persistence/Persistence.swift) | Shared persistence types and helpers |
| [PersistenceActor.swift](../../Sources/Utilities/Persistence/PersistenceActor.swift) | Actor wrapper for file I/O serialization |
| [FilePersistence.swift](../../Sources/Utilities/Persistence/FilePersistence.swift) | Low-level file read/write helpers |
| [NodeIdentity.swift](../../Sources/Utilities/Persistence/NodeIdentity.swift) | Stable per-process node UUID (used in Seer registration) |
