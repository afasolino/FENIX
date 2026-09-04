# Decision 0007: Externalized state uses raw, versioned storage banks

## Status

Accepted for experiment preparation. Runtime consumption remains separately qualified.

## Context

The decisive FENIX A/B requires `ple_externalized` to mean that the full PLE table is not retained as managed host-DRAM residency. Building that representation by first loading the complete PLE would defeat offline preparation and create unnecessary transient memory demand.

The Qwen3.8-Flash-Next checkpoint already stores the PLE as split safetensors tensors. Safetensors is a simple header plus contiguous tensor-byte ranges, so the PLE bank can be constructed directly from checkpoint bytes without CUDA, torch, or materializing the table in DRAM.

## Decision

FENIX cold/externalized state is stored as raw contiguous byte banks accompanied by JSON manifests. A bank manifest records exact source tensor names, source checkpoint files and byte ranges, tensor geometry, destination offsets, total byte length, checkpoint-index SHA-256 and data SHA-256.

The PLE builder:

1. reads `model.safetensors.index.json`;
2. discovers tensors ending in `ple_embedding.ngram_embedding.shard_<N>.weight`;
3. requires one unambiguous logical PLE prefix unless explicitly selected;
4. requires contiguous numeric shard indices, one dtype and one embedding width;
5. copies only the referenced byte ranges in shard-index order;
6. fsyncs and atomically promotes the data file;
7. hashes the completed bank and writes the manifest last;
8. refuses to overwrite any existing bank.

The resulting `ple.bin` is mmap-ready. This decision does **not** claim that the runtime mmap path is performance-qualified; runtime attachment is a separate implementation and evidence gate.

## Consequences

PLE-bank construction is CPU/storage-only and can run while the A6000 is occupied. The resulting large artifact remains outside Git and must be retained under the experiment storage root with its manifest. Full SHA-256 verification is required before it is admitted to a measured run.
