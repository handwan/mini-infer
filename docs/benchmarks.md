# Benchmarks

Experiment notes and raw numbers for the project — the data behind the README highlights,
with the method and reproduction commands per block.

Tested on the dev box: Ubuntu 24.04 / GCC 13.3 (Intel i7-14650HX) for the CPU path,
RTX 4060 Laptop 8G for the GPU path. `wrk -t2` runs: 3 s per run, medians (2–5 runs);
run-to-run variance is ±10–15%.

HTTP ceiling (`/echo`, no inference): **37.1k rps @ c=10, 61.9k rps @ c=32**

`POST /predict` (QPS medians):

| model | config | c=10 | c=32 |
|-------|--------|-----:|-----:|
| tiny_mlp (4→8→2) | `BATCH_MAX=1` | 24.9k | 53.6k |
| tiny_mlp | `200 µs / 16` | 17.1k | 43.1k |
| wide_mlp (4→512→2) | `BATCH_MAX=1` | 23.7k | 48.9k |
| wide_mlp | `200 µs / 16` | 16.5k | 40.5k |

Reproduce: `BATCH_CONFIGS="0 1;200 16" scripts/sweep_batch.sh models/tiny_mlp.onnx 32 5 3s`

## Findings

- The sweep caught a dispatch-policy bug: when `BATCH_MAX` exceeds the connection count,
  batches never fill and every request waits the full window (throughput ≈ connections /
  window; at `5000 µs / 16` with 10 connections: **1.6k rps**). Dispatching on idle fixed
  it — the same config now sustains **13.2k rps**.
- With these sub-100 µs CPU models, micro-batching does **not** beat `BATCH_MAX=1`:
  inference is cheap compared to HTTP/scheduling overhead, and ONNX Runtime batch time
  scales roughly linearly at this size. Micro-batching is infrastructure for
  compute-heavy models — that is what the v2 (LLM) branch is for.
- No `/predict` configuration beat the `/echo` ceiling; the gap is per-request inference
  plus queueing.

## Connection overhead

How much does the gateway's per-request TCP connection cost? Probe: httplib client against a
local mini-infer `/health` target (loopback), 2000 requests per mode.

| Mode | Per request |
|------|-------------|
| Fresh `Client` per request (current gateway design) | **480 µs** |
| Reused `Client`, keep-alive | **211 µs** |
| Pure TCP connect/close, no HTTP | 133 µs |

Decision: **no connection pool.** The ~269 µs saved per request is 0.02–0.09% of a vLLM request
(300 ms–1.5 s) on localhost; the per-request `Client` keeps the code lock-free and simple.
Notes for a future revisit:

- httplib's client keep-alive defaults to **off** (`keep_alive_ = false`); reuse requires an
  explicit `set_keep_alive(true)`. Fresh connections also pile up TIME_WAIT sockets (first run:
  9 → 1892) — at high sustained RPS that churn, not the 269 µs, is the real reason pools exist.
- `httplib::Client` is not thread-safe, so reuse would need per-thread clients or a pool.
- Remote/TLS backends would widen the gap, but LLM latency still dominates.

## Gateway concurrency

vLLM 0.30.0 + Qwen2.5-1.5B-Instruct on :8001, gateway in front, single RTX 4060 Laptop 8G.

| Test | Config | Result |
|------|--------|--------|
| 10 concurrent requests, unreachable backend | `GATEWAY_MAX_RETRIES=0`, connect timeout 2 s | all 504, wall **2.01 s** (serial ≈ 20 s) |
| 4 concurrent chat requests, distinct prompts | arithmetic prompts, `temperature=0` | all answers correct (2/4/6/8), wall **0.37 s** |
| 4 × 64-token generations | `ignore_eos=true` | serial **6.04 s** vs concurrent **0.96 s** (≈6×) |

Findings:

- The gateway does not serialize: httplib runs a thread per connection, and each request builds
  its own outbound `Client` — no shared state, no locking needed.
- Concurrency = throughput: vLLM continuously batches in-flight requests (4×64 tokens in
  0.96 s vs 6.04 s serial).
- Test-signal caveat: a "reply with one word" fruit prompt produced odd outputs (`红itu`, `美фи`);
  direct-to-vLLM runs reproduced them exactly, so it was the 1.5B model, not the gateway or a
  request mix-up. Use signals the model is reliable at (arithmetic) when verifying fan-out.

## KV cache demo

Standalone toy (`src/experiments/kv_cache_demo.cc`): single-head/single-layer attention over
random data (d=64), one token per step. "no cache" re-projects the whole prefix every step;
"cache" appends only the current token's K/V. The *attention* work is identical on both sides
by design (O(t) reads per step).

`./build/kv_cache_demo [N ...]` (Release build):

| N | no-cache (ms) | cache (ms) | ratio | cache KB |
|--:|--:|--:|--:|--:|
| 128 | 12.79 | 0.41 | 31.3 | 64 |
| 256 | 52.49 | 0.98 | 53.8 | 128 |
| 512 | 208.7 | 3.0 | 67.8 | 256 |
| 1024 | 836.7 | 11.0 | 76.1 | 512 |
| 2048 | 3326.1 | 41.5 | 80.2 | 1024 |

Findings:

- no-cache is clean O(N²): ×3.94–3.97 per doubling.
- cache is ~30–80× cheaper, but **not linear**: once N > d the dominant remaining cost is
  attention over the cached rows (O(t) per step → O(N²) total), so it also nearly quadruples
  per doubling. The ratio grows sub-linearly, not without bound.
- Lesson: KV cache removes the *redundant projection* work (a factor ≈ d), not attention's
  quadratic cost — that is what FlashAttention (compute) and PagedAttention (memory) attack.
- Real model reference (Qwen2.5-1.5B): KV cache ≈ 28 KB/token = 2(K/V) × 28 layers × 2 GQA
  KV heads × 128 dim × 2 bytes; 32k context ≈ 0.9 GB (≈ 5.4 GB without GQA).

## vLLM KV cache usage

Qwen2.5-1.5B on vLLM (0.80 GPU util). Startup log: `GPU KV cache size: 99,616 tokens`
(≈ 2.86 GB at 28 KB/token). Metric `vllm:kv_cache_usage_perc` (1.0 = 100%) polled while each
request ran (`max_tokens=128`, `ignore_eos=true`):

| prompt chars | prompt tokens | +128 gen | peak KV usage | theoretical (tokens / 99,616) |
|--:|--:|--:|--:|--:|
| 400 | 275 | 403 | 0.40% | 0.40% |
| 4,000 | 2,435 | 2,563 | 2.57% | 2.57% |
| 16,000 | 9,635 | 9,763 | 9.82% | 9.80% |

- Usage is dead linear in sequence length and matches the 28 KB/token estimate to within
  0.02 pp — at long contexts the cache is the dominant memory variable.
- vLLM's own startup line: "Maximum concurrency for 32,768 tokens per request: 3.04x" —
  the capacity/batch-size trade-off in one number.

## LLM metrics

Streaming via `/v1/chat/completions` (`stream=true`, `stream_options.include_usage`), fixed
English prompt, `temperature=0`, `max_tokens=128`, 3 rounds, warm-up excluded.
Script: `scripts/bench_llm.py` — TTFT = first non-empty content delta; TPOT = decode time /
(tokens − 1); system tok/s = total completion tokens / wall.

Direct to vLLM (:8001):

| conc | TTFT | TPOT | decode tok/s (per req) | system tok/s | wall |
|--:|--:|--:|--:|--:|--:|
| 1 | 32.4 ms | 13.4 ms | 74.6 | 73.8 | 1.73 s |
| 2 | 34.9 ms | 13.8 ms | 72.5 | 142.8 | 1.79 s |
| 4 | 39.2 ms | 14.3 ms | 69.8 | 274.8 | 1.86 s |
| 8 | 49.6 ms | 14.0 ms | 71.6 | 560.6 | 1.83 s |

Through the gateway (:8080, non-streaming):

| conc | mean latency | system tok/s |
|--:|--:|--:|
| 1 | 1731.9 ms | 73.9 |
| 4 | 1797.8 ms | 284.2 |

Direct non-stream control: 1791.0 ms / 71.6 tok/s (c=1), 1766.3 ms / 289.3 tok/s (c=4).

Findings:

- TTFT 32 → 50 ms from c=1 to c=8 (prefill + queueing) — two orders of magnitude below the
  ~1.7 s full generation. First token is cheap; the rest is the token treadmill.
- TPOT ≈ 13–14 ms (≈ 70–75 tok/s per request) and **flat across concurrency**: per-request
  decode speed does not degrade; more users are absorbed by continuous batching.
- System throughput scales ~linearly: 73.8 → 560.6 tok/s (7.6× at 8×). The laptop GPU is the
  wall; batching is how you spend it.
- Gateway overhead is unmeasurable at this scale (Δ within run-to-run noise on ~1.7 s
  requests; raw forwarding cost was hundreds of µs in the connection-overhead probe) — non-streaming
  forwarding is effectively free next to generation time.
- The gateway has no SSE passthrough yet, so TTFT/TPOT can only be measured direct to vLLM.
  Streaming proxy is on the backlog.

## Quantization — FP16 vs GPTQ INT8/INT4

Same box and server flags as the LLM metrics section (fresh FP16 baseline rerun for
apples-to-apples; reproduces the earlier numbers within noise). Checkpoints: official `Qwen2.5-1.5B-Instruct-GPTQ-Int8` /
`-GPTQ-Int4`; vLLM 0.30 picks **Marlin** kernels for both on this Ada GPU (sm89). Weights/KV
numbers come from the startup log; perf from `scripts/bench_llm.py` (conc 1/2/4/8, 2 rounds,
warm-up excluded); quality from `scripts/quant_probe.py` (12 closed-form questions + 5 open
prompts, greedy, `temperature=0`).

| condition | weights | KV available | KV capacity | max conc @32k | conc | TTFT | TPOT | decode tok/s | system tok/s |
|-----------|--------:|-------------:|------------:|--------------:|-----:|-----:|-----:|-------------:|-------------:|
| FP16 | 2.98 GiB | 2.66 GiB | 99,616 tok | 3.04× | 1 | 36.8 ms | 14.6 ms | 68.4 | 67.6 |
| FP16 | | | | | 2 | 37.1 ms | 14.1 ms | 71.1 | 139.9 |
| FP16 | | | | | 4 | 38.8 ms | 14.1 ms | 70.9 | 279.1 |
| FP16 | | | | | 8 | 44.4 ms | 13.5 ms | 73.8 | 579.0 |
| GPTQ-Int8 | 1.74 GiB | 3.56 GiB | 133,472 tok | 4.07× | 1 | 24.0 ms | 8.8 ms | 113.0 | 111.5 |
| GPTQ-Int8 | | | | | 2 | 26.7 ms | 9.1 ms | 109.9 | 215.8 |
| GPTQ-Int8 | | | | | 4 | 28.6 ms | 9.2 ms | 109.2 | 428.7 |
| GPTQ-Int8 | | | | | 8 | 34.9 ms | 9.2 ms | 108.2 | 845.0 |
| GPTQ-Int4 | 1.10 GiB | 3.82 GiB | 142,992 tok | 4.36× | 1 | 19.5 ms | 6.4 ms | 157.0 | 154.5 |
| GPTQ-Int4 | | | | | 2 | 22.1 ms | 6.5 ms | 153.1 | 299.6 |
| GPTQ-Int4 | | | | | 4 | 23.9 ms | 6.5 ms | 153.3 | 599.4 |
| GPTQ-Int4 | | | | | 8 | 32.8 ms | 6.9 ms | 145.3 | 1126.1 |

Findings:

- **Freed weight memory becomes KV capacity, not slack**: weights 2.98 → 1.74 → 1.10 GiB, and
  vLLM re-invests it — capacity 99,616 → 133,472 → 142,992 tokens (max concurrency @32k:
  3.04× → 4.07× → 4.36×). Total GPU use stays ~6.3 GB on the 8 GB card in all three runs; the
  0.80 budget is the cap, not the weights.
- **On this GPU the quantized paths are also faster** — Marlin kernels beat the FP16 path
  outright: TPOT 14.6 → 8.8 → 6.4 ms (×1.7 / ×2.3), system throughput @conc 8: 579 → 845 →
  1126 tok/s (×1.46 / ×1.94), and TTFT shrinks with prefill included (36.8 → 24.0 → 19.5 ms).
  Kernel-defined, not universal — on other hardware INT8 can easily come out slower.
- **Quality (smoke level)**: closed set 10/12 (FP16), 10/12 (INT8), 9/12 (INT4). INT8 is
  byte-identical to FP16 on 4/5 open prompts; INT4 shows the first cracks — a wrong product in
  `23×17−56` (`209`) and a summary that flips the source's "延迟低" into "较高的延迟". Both
  weak spots of the FP16/INT8 runs are shared and pre-existing (all three answer `10` to the
  3-consecutive-integers prompt; the clock-angle question is wrong in all three, differently).
- Caveats: 2 rounds (±10–15%); single laptop GPU; the probe is a smoke test, not an evaluation
  — read "INT8 ≈ FP16" as "no visible damage on this set", and INT4's drift as a signal that
  a real eval set is needed to quantify, not as a measured quality loss.

Reproduce:

```bash
# huggingface.co is unreachable here (poisoned DNS) — always set the mirror
export HF_ENDPOINT=https://hf-mirror.com
~/.venvs/vllm/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8 \
    --port 8001 --gpu-memory-utilization 0.80
~/.venvs/vllm/bin/python scripts/bench_llm.py --base http://127.0.0.1:8001 \
    --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8 --concurrency 1,2,4,8 --rounds 2
~/.venvs/vllm/bin/python scripts/quant_probe.py --base http://127.0.0.1:8001 \
    --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8
```
