#!/usr/bin/env python3
"""LLM 基准：TTFT / TPOT / tokens/s（OpenAI 兼容 /v1/chat/completions）。

TTFT、TPOT 需流式（stream=true）才可测；固定 temperature=0 与 prompt 保证可复现。

用法：
  python3 scripts/bench_llm.py --base http://127.0.0.1:8001 --concurrency 1,2,4,8 --rounds 3
  python3 scripts/bench_llm.py --base http://127.0.0.1:8080 --no-stream   # 经网关（非流式）
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_DEFAULT = (
    "Explain in detail what a KV cache is in transformer inference, "
    "why it speeds up decoding, and what it costs in memory."
)


def one_request(base, model, prompt, max_tokens, stream, timeout):
    t0 = time.perf_counter()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    r = requests.post(base.rstrip("/") + "/v1/chat/completions",
                      json=body, stream=stream, timeout=timeout)
    r.raise_for_status()

    t_first = None
    tokens = None
    if stream:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                tokens = obj["usage"].get("completion_tokens")
            choices = obj.get("choices") or []
            if choices and t_first is None:
                if (choices[0].get("delta") or {}).get("content"):
                    t_first = time.perf_counter()
        t_end = time.perf_counter()
    else:
        obj = r.json()
        t_first = time.perf_counter()
        t_end = t_first
        tokens = (obj.get("usage") or {}).get("completion_tokens")

    return {"t0": t0, "t_first": t_first, "t_end": t_end, "tokens": tokens}


def run_level(base, model, prompt, max_tokens, stream, timeout, concurrency):
    barrier = threading.Barrier(concurrency)

    def worker():
        barrier.wait()
        return one_request(base, model, prompt, max_tokens, stream, timeout)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda _: worker(), range(concurrency)))


def summarize(results, stream):
    t0 = min(r["t0"] for r in results)
    t_end = max(r["t_end"] for r in results)
    wall = t_end - t0
    total_tokens = sum(r["tokens"] or 0 for r in results)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    lat_ms = mean([(r["t_end"] - r["t0"]) * 1e3 for r in results])
    ttft_ms = mean([(r["t_first"] - r["t0"]) * 1e3
                    for r in results if r["t_first"]])
    tpots, decode_tps = [], []
    for r in results:
        n = r["tokens"] or 0
        dec = r["t_end"] - (r["t_first"] or r["t_end"])
        if n > 1 and dec > 0:
            tpots.append(dec / (n - 1) * 1e3)
            decode_tps.append((n - 1) / dec)

    return {
        "wall_s": wall,
        "tokens": total_tokens,
        "lat_ms": lat_ms,
        "ttft_ms": ttft_ms if stream else float("nan"),
        "tpot_ms": mean(tpots),
        "decode_tps": mean(decode_tps),
        "system_tps": total_tokens / wall if wall > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8001",
                    help="服务地址（vLLM 或网关）")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--prompt", default=PROMPT_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", default="1",
                    help="并发档位，逗号分隔，如 1,2,4,8")
    ap.add_argument("--rounds", type=int, default=1, help="每档并发重复轮数")
    ap.add_argument("--no-stream", action="store_true",
                    help="非流式（只能测总延迟/吞吐）")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    args.stream = not args.no_stream

    levels = [int(x) for x in args.concurrency.split(",")]
    print(f"base={args.base} model={args.model} stream={args.stream} "
          f"max_tokens={args.max_tokens} rounds={args.rounds}")

    # 预热，不计入
    one_request(args.base, args.model, args.prompt, 8, args.stream, args.timeout)

    for c in levels:
        agg = [summarize(
            run_level(args.base, args.model, args.prompt, args.max_tokens,
                      args.stream, args.timeout, c), args.stream)
            for _ in range(args.rounds)]
        m = {k: sum(a[k] for a in agg) / len(agg) for k in agg[0]}
        if args.stream:
            print(f"conc={c:>2}  TTFT {m['ttft_ms']:7.1f} ms  "
                  f"TPOT {m['tpot_ms']:5.1f} ms  "
                  f"decode {m['decode_tps']:6.1f} tok/s  "
                  f"system {m['system_tps']:7.1f} tok/s  "
                  f"({int(m['tokens'])} tok / {m['wall_s']:.2f} s)")
        else:
            print(f"conc={c:>2}  延迟 {m['lat_ms']:7.1f} ms  "
                  f"system {m['system_tps']:7.1f} tok/s  "
                  f"({int(m['tokens'])} tok / {m['wall_s']:.2f} s)")


if __name__ == "__main__":
    main()
