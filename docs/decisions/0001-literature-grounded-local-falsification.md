# ADR 0001: Reuse established systems evidence and measure the FENIX-specific coupling

- Status: Accepted
- Date: 2026-09-01

## Context

FENIX investigates a host-centric Qwen3.8-Flash-Next deployment in which scarce
GPU VRAM holds the active compute working set, host DRAM holds additional MoE
experts, lower-tier storage provides cold capacity, and the model's large PLE
table competes with experts for host-memory capacity.

Several adjacent questions are already addressed by prior systems work:

- Engram establishes deterministic conditional-memory lookup and host-memory
  prefetchability.
- CXL-Engram studies disaggregated memory for Engram-style tables.
- TF-Engram studies a GPU/DRAM/SSD hierarchy for conditional memory.
- PowerInfer establishes locality-aware heterogeneous placement under limited
  GPU memory.
- Fiddler studies CPU/GPU orchestration for MoE inference.
- KTransformers demonstrates high-performance CPU/GPU hybrid MoE inference with
  optimized CPU kernels and asynchronous scheduling.
- Current Qwen/SGLang/vLLM implementations provide concrete PLE offload paths.

Reproducing all of those results on the FENIX workstation would consume
substantial engineering and measurement effort without answering the new
scientific question.

The missing question is the interaction between the two sparse capacity
mechanisms in the same model:

> How much does Qwen3.8-Flash-Next lose when its PLE consumes host DRAM that
> could otherwise increase routed-expert residency?

## Decision

FENIX will use a literature-grounded, local-falsification methodology.

Established prior-work findings are reused at the level supported by their
primary sources. FENIX will not reproduce them solely to create another
endpoint-specific copy of an already established result.

Local experiments are restricted to evidence that is specific to the FENIX
hypothesis:

1. qualify one mature Qwen3.8-Flash-Next runtime path on the target machine;
2. collect and independently verify the exact PLE row-access stream;
3. collect the exact routed-expert access stream;
4. quantify the host-memory capacity tradeoff between PLE residency and expert
   residency;
5. measure the same-endpoint baseline and counterfactual at informative memory
   budgets.

A trace-derived model may be used to select informative measurement points. It
cannot establish the motivation claim.

The final motivation claim requires measured results with the same model
revision, runtime revision, hardware, workload, and quality settings.

## Numerical-comparison rule

Published throughput and latency values from endpoint-incompatible systems are
context, not FENIX datapoints.

FENIX may reuse a prior result to support a mechanism-level statement, but it
must not place an incompatible numerical value on a FENIX curve or use it to
compute a FENIX speedup.

## Consequences

### Positive

- avoids reimplementing mature systems;
- concentrates experiments on the novel resource-coupling question;
- makes the provenance of every claim explicit;
- reduces the chance that runtime engineering obscures the hardware research
  question;
- creates a clean falsification boundary before SRAM/ferroelectric-memory
  architecture work begins.

### Negative

- some external-memory baselines will remain cross-endpoint literature
  references rather than locally reproduced systems;
- any claim that depends on Qwen3.8-specific access locality still requires
  local traces;
- a positive trace projection is insufficient; the measured capacity tradeoff
  remains mandatory.

## Reconsideration triggers

Revisit this decision if any of the following occurs:

- a mature prior work directly measures the same Qwen3.8-Flash-Next
  PLE-versus-expert host-DRAM tradeoff;
- the selected runtime changes the PLE or expert access semantics in a way that
  invalidates the trace methodology;
- the local capacity tradeoff is too small to pass the predeclared motivation
  gate;
- the eventual FENIX architecture requires a latency or bandwidth regime not
  covered by existing conditional-memory evidence.
