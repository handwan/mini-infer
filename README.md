# mini-infer

A minimal C++ inference server — dynamic batching over ONNX Runtime.

## Requirements

- Linux x86_64; a C++20 compiler (tested: GCC 13.3 and Clang 18 on Ubuntu 24.04); CMake ≥ 3.16 (tested 3.28)
- Dependencies are fetched by script (pinned + sha256) — no extra system packages beyond a compiler and CMake
- Python with PyTorch + ONNX Runtime — **only** to export the demo model
- Optional: `wrk` (benchmarks), `clang-format` + `clang-tidy` (`scripts/check.sh`)

## Quick start

```bash
scripts/fetch_deps.sh      # fetch dependencies (pinned + sha256)
cmake -B build && cmake --build build -j
python3 scripts/export_model.py            # -> models/tiny_mlp.onnx (once; export only)
./build/mini-infer 8080                    # serve /health /echo /version /predict

# another terminal:
scripts/bench.sh                           # quick /echo benchmark
scripts/sweep_batch.sh                     # batch parameter sweep for /predict
```

## API

| endpoint | method | notes |
|----------|--------|-------|
| `/health` | GET | liveness |
| `/echo` | GET/POST | HTTP-layer baseline (no inference) |
| `/version` | GET | `{"name":"mini-infer","version":"0.1.0"}` |
| `/predict` | POST | body: comma-separated floats (`1,2,3,4`) → `{"output":[...]}` |

Environment knobs (no rebuild to sweep): `PORT`, `MODEL_PATH`, `BATCH_MAX`, `BATCH_WINDOW_US`.

## Design

```
client ──► HTTP server (thread pool)
             └─ /predict: parse + validate
                  └─ DynamicBatcher: queue ─► single worker thread
                       ├─ gathers a micro-batch: up to BATCH_MAX,
                       │  window cap (BATCH_WINDOW_US), or on 200 µs idle
                       └─ one ONNX Runtime Run() for the whole batch
                  └─ results fanned back to each waiting request
```

The batcher dispatches when the batch fills, when the window cap expires, **or when the
queue stays idle for 200 µs** — the idle rule avoids waiting the full window whenever
concurrency is below `BATCH_MAX`.
