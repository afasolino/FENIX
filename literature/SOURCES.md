# Primary-source ledger

The source ledger records publication status separately from evidence role.

## Conditional memory

- **Engram** — *Conditional Memory via Scalable Lookup: A New Axis of Sparsity
  for Large Language Models*. arXiv:2601.07372, 2026. Measured preprint.
  Reuse role: deterministic conditional-memory addressing and host-prefetch
  feasibility.
  https://arxiv.org/abs/2601.07372

- **CXL-Engram** — *Pooling Engram Conditional Memory in Large Language Models
  using CXL*. arXiv:2603.10087, 2026. Measured preprint / EuroMLSys 2026 work.
  Reuse role: lower-tier/disaggregated-memory feasibility for Engram-style
  tables.
  https://arxiv.org/abs/2603.10087

- **TF-Engram** — *TF-Engram: A Train-Free Engram with SSD-Backed Memory for
  Large Language Models*. arXiv:2607.07388, 2026. Measured preprint.
  Reuse role: GPU/DRAM/SSD conditional-memory hierarchy and predictive
  prefetching.
  https://arxiv.org/abs/2607.07388

## Heterogeneous and MoE inference

- **PowerInfer** — *PowerInfer: Fast Large Language Model Serving with a
  Consumer-grade GPU*. SOSP 2024, DOI 10.1145/3694715.3695964.
  Peer-reviewed measured evidence.
  Reuse role: locality-aware heterogeneous placement under constrained VRAM.
  https://doi.org/10.1145/3694715.3695964

- **Fiddler** — *Fiddler: CPU-GPU Orchestration for Fast Inference of
  Mixture-of-Experts Models*. ICLR 2025.
  Peer-reviewed measured evidence.
  Reuse role: CPU/GPU execution and transfer tradeoffs for MoE inference.
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/8cd1ce03ea58b3d7dfd809e4d42f08ea-Abstract-Conference.html

- **KTransformers** — *KTransformers: Unleashing the Full Potential of CPU/GPU
  Hybrid Inference for MoE Models*. SOSP 2025,
  DOI 10.1145/3731569.3764843.
  Peer-reviewed measured evidence.
  Reuse role: optimized CPU expert execution and asynchronous CPU/GPU
  scheduling for large MoE models.
  https://doi.org/10.1145/3731569.3764843

## Qwen3.8-Flash-Next implementations

- Official model:
  https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- vLLM model support:
  https://github.com/vllm-project/vllm/pull/53896
- vLLM PLE CPU offload:
  https://github.com/vllm-project/vllm/pull/53899
- vLLM direct-UVA PLE:
  https://github.com/vllm-project/vllm/pull/54371
- SGLang Qwen support:
  https://github.com/sgl-project/sglang/pull/36497
- SGLang NVMe PLE:
  https://github.com/sgl-project/sglang/pull/36567
- Pinned Ampere-oriented runtime candidate:
  https://github.com/DominikBucko/qwen38-flash-next-2x3090

Upstream implementation evidence establishes that an implementation path exists;
it does not make endpoint-incompatible performance numbers FENIX measurements.
