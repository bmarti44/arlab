# Spark notes (measured 2026-09-26, M0)

| Item | Measured |
|---|---|
| GPU | NVIDIA GB10, compute capability (12, 1), driver 580.159.03 |
| Memory | MemTotal 119.7 GiB unified; MemAvailable ~112-118 GB idle (page cache large) |
| Disk | `/` 3.7T, 1.9T available |
| Docker images | `nvcr.io/nvidia/pytorch:25.10-py3` (torch 2.9.0a0+145a3a7bda.nv25.10, CUDA 13.0), `nvcr.io/nvidia/vllm:26.04-py3`, `nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04`, `node:22-bookworm` |
| Tools | uv 0.11.7, Python 3.12.3, codex-cli 0.157.1, systemd 255 (user units OK) |

## Verified behaviour
- pytorch:25.10 as `--user 1000:1000`: bf16 4096² matmul and `F.scaled_dot_product_attention` fwd+bwd (causal, bf16) run.
- vLLM 26.04 serves `Qwen/Qwen3.5-4B` offline from the HF cache at `--gpu-memory-utilization 0.35 --max-model-len 8192`
  (41 GB GPU used, KV cache 243k tokens, ~110 s cold start incl. graph capture; MemAvailable fell 118 → 70 GB).
  Qwen3 chat template: pass `chat_template_kwargs: {enable_thinking: false}` for short answers.
- **HF cache reality:** Qwen3-4B, Qwen3-30B-A3B, Qwen3.8-27B-FP8/NVFP4 are refs-only stubs (no weights).
  Actually present: Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-2B, Qwen3.5-4B, SmolLM2-1.7B, gpt2, bge-small-en-v1.5, Ouro-2.6B, huginn-0125.
- GPU PIDs of containers: `/proc/<pid>/cgroup` = `0::/system.slice/docker-<full container id>.scope`.
- **GPU memory is NOT charged to the container memcg** (10 GB cudaMalloc → memory.current ≈ 0).
  arlab's `peak_mem_gb` = max over 5 s samples of (memcg `memory.current` + nvidia-smi `used_memory` of the container's PIDs).
- `nvidia-smi --query-gpu=memory.*` is `[N/A]`; `utilization.gpu` is unreliable — never used.
- arlab-agent:0.157.1 builds from node:22-bookworm (+python3, git, ripgrep) in < 1 min.
- **Heterogeneous CPU:** cpus 5-9,15-19 = Cortex-X925 @3.9 GHz, cpus 0-4,10-14 = Cortex-A725 @2.8 GHz. Single-threaded
  wall time differs by up to ~40% depending on placement; arlab pins RUN/EVALUATE to the X925 cores (`--cpuset-cpus`).
- nanochat-lite baseline (62.9M tokens): RUN 228–229 s across seeds (GPU-bound, ~285k tok/s), EVAL 12 s, peak ~9.5 GB, GPU 82–83 °C.
