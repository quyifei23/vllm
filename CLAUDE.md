# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **vLLM** (vllm-project/vllm), a high-throughput and memory-efficient LLM inference and serving engine. The codebase is built on **vLLM 0.19.1**.

### Active Work: Bidaw Implementation

A major implementation task is underway: reproducing the **Bidaw** paper's three mechanisms on top of vLLM 0.19.1 with TP/PP multi-GPU support. The full implementation plan is at `../Bidaw-Inplement-Task.md`. Key constraint: **implement incrementally and push to GitHub for review after each small module**.

## Development Workflow

### Environment Setup

```bash
# Always use `uv` for Python environment management — NEVER system python3 or bare pip
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate

# Install pre-commit hooks
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing Dependencies

```bash
# Python-only changes (use precompiled wheels for speed):
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# C/C++ changes too (full rebuild):
uv pip install -e . --torch-backend=auto
```

### Running Tests

```bash
# Install test deps (unresolved for current platform):
uv pip install -r requirements/test.in
# Or on x86_64 (pinned):
uv pip install -r requirements/test.txt

# Run a specific test:
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

### Running Linters

```bash
pre-commit run              # staged files
pre-commit run --all-files  # all files
pre-commit run ruff-check --all-files  # specific hook
pre-commit run mypy-3.10 --all-files --hook-stage manual  # mypy
```

### Contribution Policy (MANDATORY for vllm-project/vllm)

- **Check for duplicates** before opening PRs: use `gh issue view`, `gh pr list` to verify no existing PR addresses the same change.
- **No low-value busywork PRs** — bundle mechanical cleanups with substantive work.
- **AI-assisted PRs require human submitter** who understands and reviews every changed line.

## Code Architecture

### Top-Level Structure

```
vllm/                    ← Main Python package
├── v1/                  ← Next-gen engine (v1 executor, worker, attention, sampling)
├── engine/              ← Core engine: LLMEngine, AsyncLLMEngine, arg utils
├── core/                ← Scheduler, block manager, core data structures
├── worker/              ← Worker processes, cache engine (KV swap)
├── executor/            ← Execution backends: GPU, Ray, multiproc
├── model_executor/      ← Model implementations, layers, CUDA kernels
│   └── models/          ← Per-model implementations (llama, opt, qwen, etc.)
├── entrypoints/         ← CLI, API server (OpenAI-compatible), LLM class
├── config/              ← VllmConfig, model/cache/scheduler configs
├── distributed/         ← Parallel group management, communication
├── attention/           ← Attention backends (FlashAttention, etc.)
├── kernels/             ← Custom CUDA/triton kernels
├── multimodal/          ← Multi-modal (vision, audio) support
├── lora/                ← LoRA serving
└── tokenizers/          ← Tokenizer utilities
```

### Key Architectural Concepts

1. **Engine → Scheduler → Worker pipeline**: `LLMEngine` coordinates request lifecycle. The `Scheduler` (in `vllm/core/`) decides which requests run each step. `Worker` processes execute the actual computation.

2. **KV Cache management**: `BlockManager` and `CacheEngine` handle GPU/CPU KV cache allocation and swap operations. This is the primary area for Bidaw modifications.

3. **Model Registry**: Models are registered via `ModelRegistry` and implement a standard interface. Attention patterns (MHA vs GQA) affect caching strategy.

4. **Executor abstraction**: `gpu_executor` (single GPU), `multiproc_gpu_executor` (multi-process TP), `ray_gpu_executor` (Ray-based distributed TP/PP).

5. **v1 vs legacy engine**: `vllm/v1/` contains the next-generation engine with reorganized attention, sampling, and worker code. Check which engine path is relevant to your changes.

### Important Files for Bidaw Work

| Area | Key Files |
|------|-----------|
| Scheduling | `vllm/core/scheduler.py`, `vllm/core/block_manager.py` |
| KV Cache | `vllm/worker/cache_engine.py` |
| Config | `vllm/config.py`, `vllm/config/*.py` |
| Engine | `vllm/engine/llm_engine.py`, `vllm/engine/async_llm_engine.py` |
| Executors | `vllm/executor/gpu_executor.py`, `vllm/executor/ray_gpu_executor.py` |
| Models | `vllm/model_executor/models/*.py` |
| Attention | `vllm/attention/` directory |

## AI Agent Instructions (from AGENTS.md)

Full instructions at `AGENTS.md`. Key highlights:
- Always use `uv` and `.venv/bin/python` — never `pip` or `python3` directly
- Include `Co-authored-by:` trailers in commits for AI assistance
- Run relevant tests before submitting changes
- Check for duplicate PRs before opening new ones
