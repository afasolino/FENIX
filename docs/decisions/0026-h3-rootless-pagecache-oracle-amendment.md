# ADR 0026: Rootless page-cache dominance oracle amendment

## Context

The frozen H3 v6 measurement campaign at `e08bc8d32b85ef3fe929d350ca122159fcee4073` completed the exact 84-point residency/fio/actual-H3-Ramulator matrix.  The remaining Linux page-cache control was designed around cgroup v2 `MemoryMax`, `MemorySwapMax`, memory-pressure, and per-device I/O accounting.

The Rocky Linux 8.10 execution host is booted with cgroup v1.  An unprivileged `systemd-run --user --scope` probe showed that the requested memory limit was not delegated/enforced and blkio remained at the root controller.  No page-cache performance result was observed before this limitation was discovered.

Changing the host boot hierarchy or creating privileged controller groups would be operationally invasive and is not necessary for a one-directional falsification test.

## Decision

Preserve every v6 measurement artifact unchanged and add an analysis-only hostile oracle for the two page-cache strata (`session`, `long_context_8k`).

The oracle is intentionally stronger than the intended 7-GiB Linux page cache:

1. all PLE lower-tier storage traffic is free and consumes no cache capacity;
2. the partial terminal page of every expert transfer is free;
3. the complete 7-GiB capacity is dedicated to equal-size full expert pages;
4. replacement uses exact future knowledge (Belady/MIN on the expert full-page reference stream);
5. metadata and process-memory overhead are zero;
6. storage latency for remaining expert misses is perfectly hidden;
7. only unavoidable expert-capacity-miss LPDDR fill/shared-bus traffic is retained.

Because every relaxation favors the conventional hierarchy, a gap that survives this oracle also survives the weaker measured Linux page-cache implementation targeted by v6.  Conversely, the oracle is not realizable evidence and cannot establish conventional-memory sufficiency.

## Provenance

The original H3 contract remains unchanged, so all existing artifact contract SHAs stay valid.  `configs/h3/pagecache_oracle_amendment_v1.json` is a separate SHA-bound analysis amendment.

Amended decisions record two identities:

- `measurement_execution_repository`: the frozen v6 implementation that produced residency/fio/Ramulator/prerequisite evidence;
- top-level `execution_repository`: the analysis implementation that generated oracle-aware decisions and the final gate.

Prerequisite evidence must match the frozen measurement identity.  Oracle artifacts additionally bind the exact source page-cache window, iolog, trace manifest, contract, and frozen measurement identity.

## Gate semantics

For `H3_MEMORY_GAP_SUPPORTED`, each required page-cache stratum may be covered by either qualified measured buffered+mmap controls or the hostile oracle.

For `CONVENTIONAL_MEMORY_SUFFICIENT`, measured realizable page-cache controls remain mandatory.  If the realizable bounds appear sufficient but only the oracle is available, the result is `INCONCLUSIVE_MEASURED_PAGECACHE_REQUIRED_FOR_SUFFICIENCY`.

If the oracle reduces any required gap bound below the predeclared threshold, H3 is not promoted.  The next step is another rootless/isolated experiment design, rerunning affected experiments if needed.  System-level host modification is reserved for last resort.
