# ADR 0002: Qualify a pinned runtime lane before model acquisition

- Status: Accepted
- Date: 2026-09-01
- Base campaign commit: `9d436e409c809ade645d12e4b3826a9ce1c73fcb`

## Context

The FENIX motivation experiment requires one Qwen3.8-Flash-Next runtime that can
expose both PLE activity and routed-expert residency behavior on the target
single RTX A6000 system.

The primary candidate remains the published community runtime
`DominikBucko/qwen38-flash-next-2x3090` at revision
`7b5f0465db90fc49d6324904f48ad995ebdcb62f`. It is useful because it combines:

- the pinned W4A16 target checkpoint with FP8 PLE;
- PLE CPU offload;
- host/UVA expert offload;
- static and dynamic expert caches;
- measured evidence that expert residency materially affects decode
  performance.

However, its published validation endpoint is 2x RTX 3090 with Docker and the
NVIDIA Container Toolkit, not a single RTX A6000.

A second complication is the TP=1 PLE path in the underlying preview vLLM
image. vLLM issue #53960 documents a startup hang with
`VLLM_PLE_CPU_OFFLOAD=1` and TP=1. The diagnosed cause for the original hang is
that the preview image selects the uniprocess executor at TP=1 while PLE worker
startup was only wired through the multiprocess executor. The practical
workaround for that image is:

`--distributed-executor-backend mp`

The missing uniprocess initialization was subsequently fixed by upstream commit
`95dc96d1d012a25ff5c3823a1e77197c8dae4654` in the still-open PLE-offload PR
#53899. The pinned image reports vLLM `0.1.dev20073+g8e685d198` and predates
that fix.

Issue #53960 also contains later reports of additional TP=1 stalls after the
worker is successfully spawned on some hardware. Therefore the multiprocess
workaround removes one known initialization defect; it is not evidence that
TP=1 is fully qualified on the A6000.

The direct-UVA PLE implementation in vLLM PR #54371 is still draft as of this
decision and is not promoted into the primary FENIX lane.

## Decision

Runtime acquisition and model acquisition are separate gates.

FENIX will:

1. fetch only the pinned runtime source;
2. verify the exact runtime revision and required PLE/expert-cache source
   structures without modifying the third-party checkout;
3. record target-host prerequisites and known TP=1 conditions;
4. force the multiprocess executor in the FENIX launcher when TP=1 is used with
   the pinned preview image;
5. permit the large model download only after the source and execution
   prerequisites are reported as ready.

Source qualification is intentionally weaker than runtime qualification.
Passing source checks means that the lane is structurally suitable for an
A6000 experiment. It does not claim that the model has booted or served on the
A6000.

## Lane status vocabulary

- `SOURCE_INCOMPATIBLE` — required pinned source semantics are absent.
- `ENVIRONMENT_BLOCKED` — source is suitable, but the selected execution path
  is unavailable on the host.
- `READY_FOR_MODEL_FETCH` — source and host execution prerequisites are
  satisfied; acquiring the pinned checkpoint is justified.
- `RUNTIME_QUALIFIED` — reserved for a later measured boot/serve smoke test
  using the actual checkpoint.

The source-only qualification script cannot emit `RUNTIME_QUALIFIED`.

## No-reinvention rule

FENIX does not repair or redesign vLLM PLE offload during this gate. If the
pinned lane fails after the documented upstream workaround is applied, the next
action is to reassess a mature upstream lane before considering any local
runtime patch.

Any local runtime modification would require a new ADR and evidence that a
mature upstream implementation cannot satisfy the experiment.

## Model-download gate

The approximately 116 GiB target checkpoint is not downloaded as a side effect
of runtime acquisition.

`scripts/fetch_model.py` requires a qualification report whose status is
`READY_FOR_MODEL_FETCH`, and additionally requires an explicit `--execute`
flag. This makes the large network/storage action deliberate and auditable.

## Consequences

- source inspection is cheap and reproducible;
- known TP=1 risk is explicit rather than rediscovered experimentally;
- the FENIX repository never silently modifies the pinned third-party runtime;
- a missing Docker/NVIDIA-container execution path is exposed before model
  acquisition;
- the eventual model download occurs once, only after a viable execution lane
  exists.

## Evidence references

- Pinned runtime:
  https://github.com/DominikBucko/qwen38-flash-next-2x3090
- vLLM TP=1 PLE issue:
  https://github.com/vllm-project/vllm/issues/53960
- vLLM PLE-offload PR:
  https://github.com/vllm-project/vllm/pull/53899
- vLLM direct-UVA PLE PR:
  https://github.com/vllm-project/vllm/pull/54371

## CLI packaging

FENIX command modules support both repository-root invocation styles:

- `python -m scripts.fetch_runtime`
- `python scripts/fetch_runtime.py`

The module form is preferred in documentation because it follows normal Python
package resolution. Direct-script execution explicitly inserts the repository
root before importing sibling FENIX packages. Both forms are covered by
regression tests.

## Acquisition-state correction

The first Milestone-1 implementation checked third-party worktree cleanliness
immediately after `git clone --no-checkout`. That Git state intentionally has an
empty worktree and reports tracked paths as deletions, so the safety check
mistook initialization for user modification.

The corrected acquisition state machine checks cleanliness before changing an
existing checkout, but initializes a newly created no-checkout clone before
evaluating its final cleanliness.

An explicit `--repair-incomplete-clone` option recognizes only the exact
empty-worktree/staged-deletion state produced by the old bug. It refuses any
checkout containing ordinary edits or non-Git worktree files.
