# Environment & scripts

What you need to build and run the project, and what every script under `scripts/` does.
For the short quick start see the [README](../README.md); this page is the reference.

## Environment

Reference dev box: RTX 4060 Laptop 8G + i7-14650HX, Ubuntu 24.04 (WSL2), GCC 13.3,
CMake 3.28, Python 3.12.

- Linux x86_64; a C++20 compiler (tested: GCC 13.3 and Clang 18); CMake ≥ 3.16
- GPU is optional — only the LLM branch needs one
- Python is only for the helper scripts; the server itself has no Python dependency

Everything else arrives one of three ways:

| Dependency | How it arrives | Used by |
|---|---|---|
| cpp-httplib v0.58.0 | `scripts/fetch_deps.sh` (pinned + sha256) | HTTP server/client, both branches |
| ONNX Runtime 1.30.0 (CPU tarball) | `scripts/fetch_deps.sh` (pinned + sha256) | v1 ONNX engine |
| `wrk` | system package (`sudo apt install wrk`) | the benchmark scripts |
| `clang-format` / `clang-tidy` | system package | `scripts/check.sh` |
| PyTorch + ONNX Runtime wheels | a Python env of your choice | `scripts/export_model.py` (one-time) |
| Python `requests` | pip | `scripts/bench_llm.py`, `scripts/quant_probe.py` |
| A vLLM server | separate install — see [vllm-setup.md](vllm-setup.md) | v2 gateway benchmarks |

## Build & run

```bash
scripts/fetch_deps.sh              # pinned third-party deps into third_party/
cmake -B build && cmake --build build -j
python3 scripts/export_model.py    # once: exports models/tiny_mlp.onnx
./build/mini-infer 8080            # server; /health /echo /version /predict
```

Build outputs: `build/mini-infer` — one binary for both branches (ONNX engine + vLLM gateway;
see the README for the env vars) — and `build/kv_cache_demo` (KV-cache microbenchmark).

## Scripts

All scripts resolve paths themselves; examples assume the repo root. `bench.sh` and
`sweep_batch.sh` need `wrk`; the Python scripts need a matching interpreter (see above).

### `fetch_deps.sh`

Downloads cpp-httplib and the ONNX Runtime CPU tarball into `third_party/`, both pinned and
sha256-verified (GitHub fallback via `gh-proxy.com`). Re-runnable; skips what already exists.

### `check.sh`

Style + static analysis over `src/` and `tests/`: `clang-format --dry-run --Werror` plus
`clang-tidy -p build`. Needs a configure pass first (`cmake -B build`). Prints
`check passed` / `check failed`.

### `bench.sh`

Quick wrk benchmark of any endpoint (defaults: `http://localhost:8080/echo`, 5 s):

```bash
scripts/bench.sh
scripts/bench.sh http://localhost:8080/predict 10s
```

### `sweep_batch.sh`

Batch-parameter sweep for `/predict`: for each `(window µs, max batch)` config it starts
`./build/mini-infer`, runs wrk (POST `1,2,3,4`), and reports medians of QPS / P50 / P99
across rounds.

```bash
scripts/sweep_batch.sh                                  # tiny_mlp, c=10, 3 rounds, 3 s each
scripts/sweep_batch.sh models/wide_mlp.onnx 50 3 3s
BATCH_CONFIGS="0 1;200 16" scripts/sweep_batch.sh       # custom grid
```

Args: `[model] [conns] [rounds] [duration]`. Env: `BATCH_CONFIGS` (default
`0 1;0 16;200 16;2000 16;10000 32`), `PORT` (default 8080).

### `export_model.py`

Exports the demo MLP to ONNX with a dynamic batch axis (`--hidden 512` gives the `wide_mlp`
used in the batching sweep):

```bash
python3 scripts/export_model.py                              # -> models/tiny_mlp.onnx
python3 scripts/export_model.py --hidden 512 --out models/wide_mlp.onnx
```

Args: `--hidden` (default 8), `--out`.

### `bench_llm.py`

TTFT / TPOT / throughput over an OpenAI-compatible `/v1/chat/completions` (streaming;
`--no-stream` for latency/throughput only). Works against vLLM directly or through the gateway:

```bash
python3 scripts/bench_llm.py --base http://127.0.0.1:8001 --concurrency 1,2,4,8 --rounds 3
python3 scripts/bench_llm.py --base http://127.0.0.1:8080 --no-stream
```

Args: `--base`, `--model`, `--prompt`, `--max-tokens`, `--concurrency`, `--rounds`,
`--no-stream`, `--timeout`.

### `quant_probe.py`

Quality probe for quantized models: 12 closed-form questions + 5 open prompts, greedy
(`temperature=0`). Progress goes to stderr; the full JSON result goes to stdout.

```bash
python3 scripts/quant_probe.py --base http://127.0.0.1:8001 \
    --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8 > int8.json
```

The server must be running (the script fails fast with a hint if it is not).
Args: `--base`, `--model`, `--timeout`.

Experiment data and methodology live in [benchmarks.md](benchmarks.md).
