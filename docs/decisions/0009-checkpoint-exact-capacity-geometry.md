# Decision 0009: Capacity accounting uses checkpoint-exact PLE geometry

## Status

Accepted before any measured capacity A/B experiment.

## Context

The original trace projection reconstructed the PLE row population as
`(ngram_size - 1) * heads_per_ngram * ngram_vocab_size_base`, yielding
320,000,000 rows. That expression describes the base hash-table geometry but
not the exact stored checkpoint table: Qwen3.8-Flash-Next uses per-head prime
sizes and then pads the aggregate embedding vocabulary to the configured
multiple.

The pinned checkpoint was scanned byte-for-byte while building the versioned
PLE bank. It contains 128 `F8_E4M3` shards, 320,001,536 rows, 160 bytes per row,
and 51,200,245,760 bytes of PLE data.

The difference is small, but allowing projection and runtime to disagree by
1,536 rows would make host-cache capacity depend on which implementation did
the accounting.

## Decision

`configs/campaign.json` records `model.ple_addressable_rows = 320001536`.
Capacity analysis treats this versioned checkpoint-exact value as authoritative
when present. The older base-vocabulary formula remains only as a compatibility
fallback for synthetic/unit-test configurations that do not declare an exact
row population.

Measured row width still comes from trace evidence. Therefore the campaign PLE
host charge is:

```text
320001536 rows * 160 bytes/row = 51200245760 bytes
```

At the predeclared budgets and the observed 2,534,400-byte complete expert
slot, this changes no conclusion at 64 GiB (6,912 slots = 144 experts/layer)
or 112 GiB (full saturation). It makes the exact 96-GiB baseline capacity
20,469 slots, distributed as 427 experts in 21 layers and 426 experts in 27
layers.

## Evidence implications

This is a geometry correction made before any measured capacity A/B result.
It does not change the evidence class of the trace projection and cannot
establish the motivation claim.
