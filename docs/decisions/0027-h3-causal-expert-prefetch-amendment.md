# ADR 0027: Rootless causal expert-prefetch qualification

## Context

The H3 v6 physical residency/fio/Ramulator campaign is frozen at
`e08bc8d32b85ef3fe929d350ca122159fcee4073`.

The subsequent rootless page-cache dominance oracle deliberately gave the
conventional hierarchy exact future knowledge, free PLE storage traffic, free
expert terminal-page traffic, the complete 7-GiB budget for expert pages, the
most favorable measured LPDDR bandwidth, and completely hidden lower-tier
storage service.  On `session / useful_object_lru / full`, that deliberately
unphysical envelope reduced the global gap-falsification lower bound to zero.
Therefore H3 cannot be promoted by claiming that the gap survives arbitrary
perfect expert prefetch.

That failure does not establish conventional-memory sufficiency.  It identifies
the unresolved question: whether exact cold-expert storage service can actually
be issued early enough to be hidden in the pinned host-driven MoE runtime.
Unlike PLE addresses, routed expert IDs are data-dependent outputs of the MoE
router.

## Decision

Predeclare a rootless causal experiment on the pinned Qwen3.8-Flash-Next runtime
revision `7b5f0465db90fc49d6324904f48ad995ebdcb62f`.

The staged runtime already observes final `topk_ids` immediately before
`quant_method.apply()` dispatch.  Add two low-overhead monotonic timestamps:

1. `fenix_h3_host_ids_ready_ns`: after the exact routed IDs have been
   materialized on the host;
2. `fenix_h3_dispatch_call_ready_ns`: immediately afterward and before trace
   emission, so trace-file I/O does not manufacture artificial prefetch slack.

The causal qualification compares the maximum observed native interval against
the fastest lower 95% confidence endpoint of the QD32 expert-storage service
coefficient observed anywhere in the frozen e08 fio campaign.  The transfer
size is one physical expert stride.

The gate passes only when:

- both prefill and decode are represented;
- all 48 MoE layers are represented in each phase;
- each phase has at least 48 router events; and
- even the fastest measured one-expert transfer does not fit inside the maximum
  native host-ID-ready to dispatch-ready interval.

This is intentionally favorable to conventional storage: it uses the fastest
measured QD32 expert service from the entire frozen campaign, not the local or
median value.

## Decision semantics after a causal pass

A causal pass removes only the sensitivity
`PLE_and_expert_perfect_prefetch_sensitivity`, because that sensitivity assumes
exact future expert knowledge and fully hidden expert storage service.

The deterministic PLE perfect-prefetch sensitivity remains.  PLE addresses are
not treated like router-selected experts.

For the two full-run page-cache strata (`session` and `long_context_8k`), the
future-aware Belady cache remains as the replacement-policy upper bound.  It
continues to receive free PLE traffic, free expert terminal pages, all 7 GiB for
expert pages, and the most favorable point-local LPDDR bandwidth.  The only
restored physical constraint is unavoidable expert lower-tier bandwidth:
Belady-minimized expert miss bytes are projected through the measured QD32 fio
service, with perfect overlap between LPDDR and storage resources.  Thus the
new bound remains hostile to H3 while no longer granting an infinite-bandwidth
lower tier.

## Claim boundary

This amendment applies only to the measured conventional host-driven runtime
path.  It does not prove that learned expert prediction, speculative expert
prefetch, GPUDirect Storage, or another redesigned scheduler cannot obtain
additional lookahead.  No such mechanism is measured here.

The causal path is gap-falsification only.  It cannot establish
`CONVENTIONAL_MEMORY_SUFFICIENT`; that still requires realizable measured
page-cache/scheduler evidence.

No boot, cgroup, kernel, GRUB, swap, filesystem, or privileged host change is
required.  If this rootless causal experiment fails, the next escalation is a
redesigned user-space scheduler/prefetch experiment before any system-level
modification is considered.
