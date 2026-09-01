# ADR 0003: Use project-local Podman/CDI for the A6000 runtime lane

- Status: Accepted
- Date: 2026-09-01
- Parent runtime decision: ADR 0002
- Repository base: `24c93fa299230d28aba3678a340167d2f9e266a3`

## Context

Milestone 1 source qualification established that the pinned Qwen3.8 runtime is
structurally suitable, but the original lane assumed Docker. The target Rocky
Linux 8.10 workstation already provides Podman 4.9 and `fuse-overlayfs`.

Local qualification established that NVIDIA CDI exposes the RTX A6000 through
Podman and that the exact pinned vLLM base image executes a real CUDA tensor
operation on the A6000. The observed base image reported PyTorch
`2.13.0+cu130` / CUDA 13.0 while the host driver remained `550.135`.
No driver change is justified by this observed endpoint.

This evidence is runtime-environment qualification only. It is not a FENIX
performance result.

## Decision

FENIX uses rootless Podman + NVIDIA CDI rather than installing a second Docker
daemon.

All project-owned OCI state is stored under `.runtime/podman/`.

The pinned third-party runtime checkout under `external/runtime/qwen38` remains
immutable. Instrumentation is applied to a generated copy under
`.runtime/instrumented/qwen38`.

The approximately 116-GiB model download remains blocked until the actual built
image `fenix-qwen38:locked`, not merely its base image, passes the FENIX
CUDA/runtime smoke gate.

## PLE timing identity

PLE address and GPU-consumption events originate in different processes. Their
counters are local sequence identifiers, not a shared request/step identity.
FENIX must not claim exact cross-process address-to-consumption latency from
those counters alone.
