#!/usr/bin/env python3
"""量化质量探针：固定题集 + temperature=0，输出 JSON，供不同精度（FP16/INT8/INT4）对照。

客户端脚本——先起服务，再跑本脚本（服务没起会在 3 秒内快速报错）。

两类题：
  closed —— 有确定答案，算命中率（数字/公式/单词题为主）
  open   —— 无标准答案，用于两种精度的输出并排看差异

用法：
  # 1) 起服务（三种精度换模型 ID 即可；HF_ENDPOINT 见 docs/vllm-setup.md）
  export HF_ENDPOINT=https://hf-mirror.com
  ~/.venvs/vllm/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct \
      --port 8001 --gpu-memory-utilization 0.80

  # 2) 跑探针（stdout=JSON，stderr=进度）
  ~/.venvs/vllm/bin/python scripts/quant_probe.py \
      --base http://127.0.0.1:8001 \
      --model Qwen/Qwen2.5-1.5B-Instruct > fp16.json
"""
import argparse
import json
import re
import sys

import requests

# (id, prompt, expected) —— expected 会被规范化后做“包含”匹配
CLOSED = [
    ("arith1", "只输出数字：23 × 17 − 56 = ?", "335"),
    ("arith2", "只输出数字：3 个连续整数之和是 33，中间那个是几？", "11"),
    ("arith3", "只输出数字：2^10 = ?", "1024"),
    ("arith4", "只输出数字：200 的 15% 是多少？", "30"),
    ("arith5", "只输出数字：1 + 2 + 3 + … + 100 = ?", "5050"),
    ("phys1", "水在标准大气压下的沸点（摄氏度）？只输出数字。", "100"),
    ("chem1", "What is the chemical formula of water? Formula only.", "H2O"),
    ("trans1", "Translate 苹果 to English. One word only.", "apple"),
    ("cal1", "How many days are there in a leap year? Number only.", "366"),
    ("phys2", "A train covers 60 km in 1.5 hours. Average speed in km/h? Number only.", "40"),
    ("geo1", "北京是哪个国家的首都？只输出国家名。", "中国"),
    ("clock1", "A clock shows exactly 3:15. What is the angle between the hour and "
              "minute hands, in degrees? Number only.", "7.5"),
]

# (id, prompt) —— 无标准答案
OPEN = [
    ("kv", "用一句话解释什么是 KV cache。"),
    ("summary", "用一句话总结下面这段话：TCP 通过三次握手建立连接，用序号和确认保证"
                "可靠传输，并通过滑动窗口做流量控制；UDP 无连接、不保证可靠，但开销小、延迟低。"),
    ("code", "写一个 Python 函数 is_prime(n)，只输出代码。"),
    ("translate", "把这句话翻译成英文：推理服务的延迟主要取决于显存带宽。"),
    ("json", "把“上海人口约 2490 万”转成 JSON：字段 city 和 population_millions（数字）。只输出 JSON。"),
]


def clean(s):
    return re.sub(r"[\s，。,.：:；;！!？?\"'“”‘’]", "", s).lower()


def ask(base, model, prompt, max_tokens, timeout, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = requests.post(
        base.rstrip("/") + "/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    # 健康检查要快速失败：本机对未监听端口静默丢 SYN（无 RST），不设短超时会干等
    try:
        requests.get(args.base.rstrip("/") + "/health", timeout=3)
    except requests.exceptions.RequestException as e:
        print(f"!! 连不上 {args.base}（{e.__class__.__name__}）——服务没起、或还在加载模型",
              file=sys.stderr)
        print("   探针是客户端脚本，需要背后有服务在跑，例如：", file=sys.stderr)
        print("   HF_ENDPOINT=https://hf-mirror.com "
              "~/.venvs/vllm/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct "
              "--port 8001 --gpu-memory-utilization 0.80", file=sys.stderr)
        sys.exit(2)

    print(f"probe: {args.base} model={args.model} "
          f"({len(CLOSED)} closed + {len(OPEN)} open)", file=sys.stderr)
    out = {"model": args.model, "closed": [], "open": []}
    hits = 0
    for n, (i, prompt, expect) in enumerate(CLOSED, 1):
        ans = ask(args.base, args.model, prompt, 128, args.timeout,
                  system="只给最终答案，不要过程，不要多余文字。")
        hit = clean(expect) in clean(ans)
        hits += hit
        print(f"  [{n:>2}/{len(CLOSED)}] closed:{i}: {'hit' if hit else 'MISS'}",
              file=sys.stderr)
        out["closed"].append({"id": i, "prompt": prompt, "expect": expect,
                              "answer": ans.strip(), "hit": hit})
    for n, (i, prompt) in enumerate(OPEN, 1):
        ans = ask(args.base, args.model, prompt, 256, args.timeout)
        print(f"  [{n}/{len(OPEN)}] open:{i}", file=sys.stderr)
        out["open"].append({"id": i, "prompt": prompt, "answer": ans.strip()})

    out["closed_hits"] = hits
    out["closed_total"] = len(CLOSED)
    print(f"done: closed {hits}/{len(CLOSED)}（JSON 在 stdout）", file=sys.stderr)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
