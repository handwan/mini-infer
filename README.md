# mini-infer

A minimal C++ inference server with two faces:

- **ONNX engine** — dynamic batching over ONNX Runtime for small models (`/predict`)
- **LLM gateway** — a transparent C++ proxy in front of a vLLM backend
  (`/v1/chat/completions`, `/v1/responses`: create, retrieve, cancel)

One binary serves both; the two paths share only the HTTP layer.

This is a hands-on project: the goal was to build the boring-but-real parts from scratch —
HTTP concurrency in C++, micro-batching, timeouts and retries, streaming passthrough, and
quantization measured end to end on a single laptop GPU.

## Requirements

- Linux x86_64; a C++20 compiler (tested: GCC 13.3 and Clang 18 on Ubuntu 24.04); CMake ≥ 3.16 (tested 3.28)
- Dependencies are fetched by script (pinned + sha256) — no extra system packages beyond a compiler and CMake
- Python with PyTorch + ONNX Runtime — **only** to export the demo model (pinned in `requirements.txt`)
- Optional: `wrk` (benchmarks), `clang-format` + `clang-tidy` (`scripts/check.sh`)
- LLM gateway only: a vLLM server, installed separately (tested 0.30.0; setup notes in [docs/vllm-setup.md](docs/vllm-setup.md)); Python `requests` for the benchmark scripts (pinned in `requirements.txt`)

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

The gateway needs a running vLLM (any OpenAI-compatible server works). The binary starts even
without the ONNX model — `/predict` answers 503 until `scripts/export_model.py` has been run once:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001 --gpu-memory-utilization 0.80
./build/mini-infer 8080                    # same binary; gateway → 127.0.0.1:8001

# one call each way, through the gateway:
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"hi"}]}'
curl -sN http://127.0.0.1:8080/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","input":"count to three","stream":true}'

python3 scripts/bench_llm.py --base http://127.0.0.1:8080 --concurrency 1,2,4,8 --rounds 3   # streaming, through the gateway
python3 scripts/bench_llm.py --base http://127.0.0.1:8001 --concurrency 1,2,4,8 --rounds 3   # direct to vLLM
```

### Container

`docker build -t mini-infer .` builds a two-stage image with the server only (no compiler,
no sources). vLLM stays a separate service and the ONNX model is not baked in — mount
`models/` for `/predict`, or run gateway-only:

```bash
docker run --rm -p 8080:8080 -v "$PWD/models:/app/models" mini-infer
# reaching a vLLM on the host: add --network host (Linux), then the defaults just work
```

## API

| endpoint | method | notes |
|----------|--------|-------|
| `/health` | GET | liveness |
| `/echo` | GET/POST | HTTP-layer baseline (no inference) |
| `/version` | GET | `{"name":"mini-infer","version":"0.3.0"}` |
| `/predict` | POST | body: comma-separated floats (`1,2,3,4`) → `{"output":[...]}` |
| `/v1/chat/completions` | POST | OpenAI-compatible chat; proxied to vLLM. `stream: true` responses are forwarded incrementally (SSE passthrough) |
| `/v1/responses` | POST | OpenAI Responses API; proxied to the same path on vLLM. `stream: true` uses typed SSE events — forwarded incrementally, same as chat |
| `/v1/responses/{id}` | GET | retrieve a stored response |
| `/v1/responses/{id}/cancel` | POST | cancel an in-progress (background) response |

Responses store (vLLM 0.30) is off by default — `store`, `background`, retrieve and cancel
need `VLLM_ENABLE_RESPONSES_API_STORE=1` at launch (without it, `store: true` answers, but
retrieve → 404).

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
window whenever concurrency is below `BATCH_MAX`. (Whether batching *pays off* is a separate
question — see Numbers and Trade-offs.)

### LLM gateway — transparent proxy

```
client ──► POST /v1/chat/completions (or /v1/responses)
             ├─ per-IP token bucket → 429 + Retry-After when over
             └─ forward ─► vLLM :8001
                  ├─ connect timeout 2 s — fail fast (the 300 s default is unusable
                  │  against a backend that goes silent)
                  ├─ retry only connect failures (the request never reached the model);
                  │  read timeouts are not retried — the model may already be generating
                  └─ error mapping: 502 = unreachable / peer closed, 504 = timed out
                     while waiting upstream
```

The gateway waits for the upstream response head before touching the body — upstream errors
keep their real status codes — then relays by shape:

- **SSE bodies** (`stream: true`) are piped chunk-by-chunk as they arrive; a client disconnect
  stops reading upstream. Chat completion frames and Responses typed events travel the same
  relay unchanged: it forwards bytes, not protocol.
- **Everything else** is read whole and forwarded with the original status code.

More decisions:

- **No connection pool** (measured): keep-alive saves 269 µs/request (480 → 211 µs) —
  0.02–0.09 % of an LLM call. Pools are for TIME_WAIT churn at high sustained RPS,
  not LLM latency; per-request clients stay lock-free.
- **Graceful shutdown**: the first SIGINT/SIGTERM drains in-flight requests (bounded by
  their timeouts); a second signal force-exits via `std::_Exit`.
- **Thread-per-connection, no shared mutable state** along the request path.
- **The Responses API came almost free**: `POST /v1/responses`, `GET /v1/responses/{id}` and
  `POST /v1/responses/{id}/cancel` are three routes over the same relay — the only change was
  forwarding `method + path` instead of a hard-coded `POST /v1/chat/completions`. State
  (`store`, `background`, `previous_response_id`) lives in vLLM; the gateway stays stateless.

## Numbers

Qwen2.5-1.5B-Instruct on vLLM 0.30.0, single RTX 4060 Laptop (8 GiB). Run-to-run variance
±10–15 %; method and raw tables per block in [docs/benchmarks.md](docs/benchmarks.md).

- `/predict` (CPU): up to **53.6k rps @ c=32** (tiny MLP); micro-batching only pays off for
  compute-heavy models, not sub-100 µs ones — `BATCH_MAX=1` wins there
- vLLM concurrency: 4 concurrent 64-token generations finish in **0.96 s vs 6.04 s serial
  (~6×)** — the gateway does not serialize; continuous batching does the work
- Streaming (128 output tokens): **TTFT 32 → 50 ms** and **TPOT ≈ 13–14 ms flat** from 1 to
  8 concurrent; system throughput **73.8 → 560.6 tok/s (7.6× at 8×)**. Through the gateway
  (SSE passthrough) TPOT and throughput match direct runs within noise; TTFT +1–7 ms
- Gateway overhead is below measurement noise next to generation (~1.7 s per request); a fresh
  upstream TCP + HTTP connection costs ~480 µs
- KV cache: 28 KB/token (GQA); vLLM usage is linear at 0.40 / 2.57 / 9.82 % of a 99,616-token
  pool for 0.4k / 2.6k / 9.8k prompt tokens; the toy demo shows caching is ~30–80× cheaper
  than no-cache but still O(N²)
- Quantization (GPTQ, official checkpoints): weights **2.98 → 1.74 → 1.10 GiB**, reinvested by
  vLLM into KV capacity (**99,616 → 133,472 → 142,992 tokens**); TPOT **14.6 → 8.8 → 6.4 ms**;
  8-concurrency throughput **579 → 845 → 1126 tok/s** — with Marlin kernels, quantization is
  also *faster* on this card. Quality smoke probe: 10/12, 10/12, 9/12 — INT8 ≈ FP16, INT4
  shows the first cracks

## Trade-offs & lessons

- **Batching is not a default win.** With sub-100 µs models, HTTP and scheduling overhead
  dominate — `BATCH_MAX=1` beats micro-batching (53.6k vs 43.1k rps @ c=32). The batching
  machinery earns its keep on the compute-heavy path, not the tiny MLP.
- **Measure before optimizing.** The connection pool looked obviously right; the measured
  saving (see Design) doesn't justify it. Pooling exists for TIME_WAIT churn at sustained
  RPS — not for LLM latency.
- **Fail fast or hang forever.** The dev box silently drops SYNs to unbound ports (no RST), so
  a 300 s default connect timeout was unusable — every outbound client sets explicit timeouts.
- **httplib collapses read-timeout and peer-close into one error code.** The gateway
  distinguishes 504 from 502 with a read-deadline check (`elapsed >= read_timeout`) — a
  documented heuristic; a vendor patch or another client library would make it exact.
- **The gateway stays protocol-agnostic.** Bytes in, bytes out — no schema parsing, no
  server-side state. Every new endpoint is routes, not an implementation.
- **Backend flags are the backend's job.** vLLM's Responses store is opt-in
  (`VLLM_ENABLE_RESPONSES_API_STORE=1`); without it `store: true` answers normally but nothing
  is retained. A proxy cannot — and should not — paper over that.
- **A probe is not an evaluation.** "INT8 ≈ FP16" means "no visible damage on 12 + 5 smoke
  questions", not a quality verdict; INT4's drift is a signal to build a real eval set. And
  INT4 being faster here is a Marlin-on-Ada result, not a universal law.
- **Scope calls, explicitly.** Docker packaging was on the cut list and eventually landed
  anyway (a server-only image — see Container). Decide in the open instead of half-finishing.

## Reproducing the numbers

Dependencies are pinned; every block in [docs/benchmarks.md](docs/benchmarks.md) lists its
exact command and settings. The short version:

- CPU sweep: `BATCH_CONFIGS="0 1;200 16" scripts/sweep_batch.sh models/tiny_mlp.onnx 32 5 3s` (needs `wrk`)
- LLM metrics (gateway or direct): `python3 scripts/bench_llm.py --base <url> --concurrency 1,2,4,8 --rounds 3`
- Quantization: serve a `…-GPTQ-Int8` / `…-GPTQ-Int4` checkpoint of the same model, then rerun
  `scripts/bench_llm.py` and `scripts/quant_probe.py --model <checkpoint>`
- KV-cache toy: `./build/kv_cache_demo 128 256 512`
- Style + static analysis: `scripts/check.sh`

Reference box: Ubuntu 24.04 (WSL2), i7-14650HX, RTX 4060 Laptop 8G, GCC 13.3, CMake 3.28;
vLLM 0.30.0 in a venv per [docs/vllm-setup.md](docs/vllm-setup.md) — machine-specific quirks
(HF mirror, `--gpu-memory-utilization 0.80`) are recorded there too.

## Project layout

```
Dockerfile          two-stage container build (server only; see Container)
src/                the server — HTTP + routing (main.cc), ONNX engine + batcher, vLLM gateway
src/experiments/    standalone benchmarks (kv_cache_demo)
scripts/            fetch deps, export model, benchmarks, style check, sweep
docs/               vLLM setup, full benchmark data, dev guide
third_party/        pinned dependencies (fetched by scripts/fetch_deps.sh)
```

## History

- **v0.1** (tag `v0.1`) — HTTP server, ONNX Runtime engine, dynamic batching, parameter sweeps
- **v0.2** (tag `v0.2`) — vLLM backend, C++ gateway, KV-cache study, LLM metrics
- **v0.3** (tag `v0.3`) — quantization comparison, SSE passthrough, Responses API, fail-fast
  startup, Docker packaging, documentation pass
