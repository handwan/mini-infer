# mini-infer

A minimal C++ inference server with two faces:

- **ONNX engine** — dynamic batching over ONNX Runtime for small models (`/predict`)
- **LLM gateway** — a C++ proxy in front of a vLLM backend (`/v1/chat/completions`)

One binary serves both; the paths share only the HTTP layer.

## Requirements

- Linux x86_64; a C++20 compiler (tested: GCC 13.3 and Clang 18 on Ubuntu 24.04); CMake ≥ 3.16 (tested 3.28)
- Dependencies are fetched by script (pinned + sha256) — no extra system packages beyond a compiler and CMake
- Python with PyTorch + ONNX Runtime — **only** to export the demo model
- Optional: `wrk` (benchmarks), `clang-format` + `clang-tidy` (`scripts/check.sh`)
- LLM gateway only: a vLLM server, installed separately (tested 0.30.0); Python `requests` for `scripts/bench_llm.py`

## Quick start

```bash
scripts/fetch_deps.sh      # fetch dependencies (pinned + sha256)
cmake -B build && cmake --build build -j
python3 scripts/export_model.py            # -> models/tiny_mlp.onnx (once; export only)
./build/mini-infer 8080                    # port 8080; serves /health /echo /version /predict

# another terminal:
scripts/bench.sh                           # quick /echo benchmark
scripts/sweep_batch.sh                     # batch parameter sweep for /predict
```

The gateway needs a running vLLM (any OpenAI-compatible server works). The same binary loads the
ONNX model at startup even in gateway mode — run the setup above once first:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001 --gpu-memory-utilization 0.80
./build/mini-infer 8080                    # same binary; gateway → 127.0.0.1:8001

python3 scripts/bench_llm.py --base http://127.0.0.1:8080 --concurrency 1,2,4,8 --rounds 3   # streaming, through the gateway
python3 scripts/bench_llm.py --base http://127.0.0.1:8001 --concurrency 1,2,4,8 --rounds 3   # direct to vLLM
```

## API

| endpoint | method | notes |
|----------|--------|-------|
| `/health` | GET | liveness |
| `/echo` | GET/POST | HTTP-layer baseline (no inference) |
| `/version` | GET | `{"name":"mini-infer","version":"0.2.0"}` |
| `/predict` | POST | body: comma-separated floats (`1,2,3,4`) → `{"output":[...]}` |
| `/v1/chat/completions` | POST | OpenAI-compatible chat; proxied to vLLM. `stream: true` responses are forwarded incrementally (SSE passthrough) |

Environment knobs (no rebuild needed):

- ONNX engine: `PORT`, `MODEL_PATH`, `BATCH_MAX`, `BATCH_WINDOW_US`
- LLM gateway: `VLLM_HOST`, `VLLM_PORT` (default `127.0.0.1:8001`); `GATEWAY_CONNECT_TIMEOUT_MS` (2000), `GATEWAY_READ_TIMEOUT_MS` (60000), `GATEWAY_MAX_RETRIES` (1), `GATEWAY_RATE_LIMIT_QPS` (0 = off)

## Design

### ONNX engine — dynamic batching

```
client ──► HTTP server (thread pool)
             └─ /predict: parse + validate
                  └─ DynamicBatcher: queue ─► single worker thread
                       ├─ gathers a micro-batch: up to BATCH_MAX,
                       │  window cap (BATCH_WINDOW_US), or on 200 µs idle
                       └─ one ONNX Runtime Run() for the whole batch
                  └─ results fanned back to each waiting request
```

The **idle rule** — dispatch after 200 µs of queue silence — avoids waiting out the full
window whenever concurrency is below `BATCH_MAX`.

### LLM gateway — proxying to vLLM

```
client ──► POST /v1/chat/completions
             ├─ per-IP token bucket → 429 + Retry-After when over
             └─ forward ─► vLLM :8001
                  ├─ connect timeout 2 s — fail fast (the 300 s default is unusable
                  │  against a backend that goes silent)
                  ├─ retry only connect failures (the request never reached the model);
                  │  read timeouts are not retried — the model may already be generating
                  └─ error mapping: 502 = unreachable / peer closed, 504 = timed out
                     while waiting upstream
```

- **No connection pool** (measured): keep-alive saves 269 µs/request (480 → 211 µs) —
  0.02–0.09 % of an LLM call. Pools are for TIME_WAIT churn at high sustained RPS,
  not LLM latency; per-request clients stay lock-free.
- **Graceful shutdown**: the first SIGINT/SIGTERM drains in-flight requests (bounded by
  their timeouts); a second signal force-exits via `std::_Exit`.
- Thread-per-connection, no shared mutable state along the request path.

## Numbers

Qwen2.5-1.5B-Instruct on vLLM 0.30.0, single RTX 4060 Laptop (8 GiB), Ubuntu 24.04.

- `/predict` (CPU): up to **53.6k rps @ c=32** (tiny MLP); micro-batching only pays off
  for compute-heavy models, not sub-100 µs ones — `BATCH_MAX=1` wins there
- vLLM concurrency: 4 concurrent 64-token generations finish in **0.96 s vs 6.04 s serial
  (~6×)** — the gateway does not serialize; continuous batching does the work
- Streaming metrics (128 output tokens, direct to vLLM): **TTFT 32 → 50 ms** and
  **TPOT ≈ 13–14 ms flat** from 1 to 8 concurrent; system throughput **73.8 → 560.6 tok/s
  (7.6× at 8×)**
- Gateway overhead is below measurement noise next to generation (~1.7 s per request)
- KV cache: 28 KB/token (GQA); measured linear at 0.40 / 2.57 / 9.82 % of a 99,616-token
  pool for 0.4k / 2.6k / 9.8k prompt tokens
