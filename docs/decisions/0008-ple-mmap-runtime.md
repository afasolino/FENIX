# Decision 0008: Externalized PLE uses the existing CPU-offload worker with a file-backed bank

## Status

Accepted for implementation; requires A6000 functional qualification before measured evidence.

## Context

FENIX already runs the Qwen3.8-Flash-Next PLE embedding in a dedicated CPU-offload subprocess. The GPU worker retains the checkpoint's global FP8 scale and consumes the raw embedding output through the existing shared host-to-device buffer. The pinned checkpoint scan establishes the actual PLE bank geometry as 128 shards, `F8_E4M3`, 320,001,536 rows, 160 bytes per row, and 51,200,245,760 bytes total.

The motivation experiment requires two PLE placements in the same runtime image:

- `resident`: the qualified existing path, where the complete PLE embedding is materialized in host DRAM;
- `mmap`: the same CPU worker computes identical n-gram row IDs, but the embedding rows are gathered from the versioned FENIX PLE storage bank through a file-backed mapping.

Changing serving engines or moving PLE lookup into the GPU model runner would add unnecessary experimental confounders.

## Decision

The externalized path is implemented only inside `Qwen3_8FlashNextNGramEmbedding` when both conditions hold:

1. execution is inside the existing PLE CPU-offload subprocess; and
2. `FENIX_PLE_STORAGE_MODE=mmap`.

The regular GPU-worker placeholder and the resident CPU path remain unchanged and selectable in the same image.

The runtime helper validates, before inference:

- bank schema and artifact kind;
- data-file existence and byte length;
- exact row count and row width against runtime geometry;
- `F8_E4M3` storage;
- SHA-256 equality between the bank's source checkpoint index and the checkpoint mounted at `/model`.

The 51.2-GB bank is deliberately not re-hashed on every server start because doing so would stream the entire bank into the page cache before the experiment. Full bank SHA-256 remains a build/verification gate in `scripts.build_ple_bank`.

Row gathering is byte-exact. The bank is mapped as `uint8`; selected rows are copied directly into the existing FP8 shared output buffer. The GPU worker still owns the original `weight_scale` and performs the same downstream dequantization as the resident path.

The launcher exposes the placement explicitly:

```text
--ple-storage-mode resident
--ple-storage-mode mmap --ple-bank-manifest <path>
```

For mmap mode the bank directory is mounted read-only and the launch preamble records `FENIX_PLE_STORAGE_MODE`, `FENIX_PLE_BANK_MANIFEST`, and the mounted model-index path.

## Evidence implications

This implementation does not itself establish the motivation claim. Before any A/B measurements, the mmap path must pass a real A6000 sentinel demonstrating:

- server startup;
- checkpoint/bank provenance match;
- exact endpoint token counts;
- no runtime/traceback/OOM contamination;
- output sanity against the resident path;
- host-memory behavior consistent with removal of the full resident PLE allocation.

Only after that qualification can the placement be used by the measured capacity A/B campaign.
