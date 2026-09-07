# ADR 0022 — Persistent LFU victim index for workstation-scale replay

ADR 0021 removed repeated full-cache sorting within one LFU epoch, but the first
real H3 canary exposed a second scale effect: its replacement heap was rebuilt
from the complete resident cache whenever an epoch first needed an eviction.
The authoritative trace contains many small timestamp epochs, so a cache with a
large PLE resident set can still pay an O(resident_objects) heap construction
across a very large number of epochs and appear stalled.

The LFU scientific policy remains unchanged. The implementation now maintains a
persistent lazy min-heap of resident replacement scores. Frequency and recency
are still learned atomically before admission. When a resident object's score
changes, a new versioned heap record is pushed and old records become stale;
stale records are discarded only when they reach the heap front. Successful and
failed admissions retain the exact ADR-0021 victim ordering and no-eviction-on-
failure behavior. The heap is compacted only when stale records materially
outnumber live state, avoiding a full-cache rebuild at every trace epoch.

The existing randomized differential test continues to require exact equality
with the original repeated-sort LFU for hits, entries, resident bytes,
frequencies, recency and epoch count. A scale regression additionally verifies
that many small eviction epochs do not trigger resident-set-sized score work per
epoch.

This changes only execution complexity and internal indexing. It does not change
capacity, granularity, replacement score, tie-breaking, same-timestamp atomicity,
admission order, victim eligibility, or any H3 claim threshold.
