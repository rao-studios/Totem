# Concurrency

Totem uses Swift actors throughout. All mutable state lives inside the `Database` actor. Writes to the partition table and registry are further serialized through dedicated mutators.

---

## Database Actor

`Database` is the root actor. It:

- Owns `TotemRegistry`, `PartitionTable`, and all HNSW shards.
- Serializes all reads and writes through its executor.
- Replays both WALs on startup before serving any requests.
- Exposes async methods used by gRPC service impls and HTTP route handlers.

---

## RegistryMutator

`RegistryMutator` serializes all writes to `TotemRegistry` and the `RegistryWAL`.

Every registry mutation (register, linkOwner, updateAccess) goes through this mutator. It appends to the WAL before updating in-memory state, ensuring that a crash after the append but before the in-memory update is recoverable on replay.

File: [RegistryMutator.swift](../../Sources/Database/Mutators/RegistryMutator.swift)

---

## TableMutator

`TableMutator` serializes all writes to `PartitionTable` and the `HNSWTopologyWAL`.

Every HNSW insertion and partition index update goes through this mutator. It appends graph topology changes to the WAL before updating in-memory state.

File: [TableMutator.swift](../../Sources/Database/Mutators/TableMutator.swift)

---

## Caching — ReadWriteValue and LockedValue

`ReadWriteValue<T>` — reader/writer lock for values that are read-heavy and written rarely (e.g. cached search results, group metadata). Multiple readers can proceed concurrently; a write excludes all readers.

`LockedValue<T>` — simple mutex wrapper for values that need mutual exclusion without reader/writer distinction.

Both are defined in [Utilities/Database/](../../Sources/Utilities/Database/).

---

## TotemCache / DocumentCache

`TotemCache` — an LRU-evicting in-memory cache for recently accessed search results and intermediate data.

`DocumentCache` — per-document cache layer that sits in front of `PartitionIndex` reads. Avoids redundant disk access for hot documents.

Files: [TotemCache.swift](../../Sources/Utilities/Database/TotemCache.swift), [DocumentCache.swift](../../Sources/Utilities/Database/DocumentCache.swift)

---

## Indexing — Background Task Pattern

`Database+Put.index(request:)` returns immediately to the caller after validating the request and submitting a detached background task. The background task embeds text, updates the registry, and drains writes through `TableMutator`. This keeps the gRPC call latency low even for large batches.

---

## Rules

- Never mutate `PartitionTable` or `TotemRegistry` directly — always go through the matching mutator.
- Never skip WAL writes. In-memory updates without WAL entries are lost on restart.
- Use `ReadWriteValue` for cache-like state; use `LockedValue` for small critical sections; use actor isolation for everything owned by `Database`.
