# vLLM Setup

Backend engine for the LLM branch: serves an OpenAI-compatible API on `:8001`, proxied by the C++ gateway. (For the v1 ONNX engine, see the README quick start.)

Recorded 2026-10-03 · Machine: RTX 4060 Laptop (8188 MiB, driver 617.14), Ubuntu 24.04.5, Python 3.12.3.

## Install

```bash
python3 -m venv ~/.venvs/vllm
~/.venvs/vllm/bin/pip install vllm   # ~4 min; pip mirror: Tsinghua (~/.config/pip/pip.conf)
```

Result: vllm 0.30.0, 199 packages, ~8.1G. Key versions (pinned by vllm's own metadata, installed automatically):

- torch 2.13.0+cu130, torchvision 0.28.0, torchaudio 2.11.0
- transformers 5.18.0, flashinfer-python 0.6.18.post1

CUDA 13.0 runtime ships as `nvidia-*` pip wheels — no system CUDA toolkit needed.

### CLI entry points (live in the venv, not on PATH)

The dependency packages also generated commands under `~/.venvs/vllm/bin/`:

| Command | From | Note |
|---|---|---|
| `vllm` | vllm | serve / bench / chat |
| `hf`, `huggingface-cli` | huggingface_hub | model download & cache management |
| `flashinfer` | flashinfer-python | inspect / build config |
| `uvicorn`, `fastapi` | server stack | run the API server manually |
| `torchrun` | torch | distributed launcher (unused) |
| `ninja` | vllm, flashinfer | JIT build helper |
| `supervisord`, `supervisorctl`, ... | model-hosting-container-standards (vllm dep) | process control (unused) |

Call them as `~/.venvs/vllm/bin/<cmd>` (or activate the venv first).

## Model

`Qwen/Qwen2.5-1.5B-Instruct` (~2.9G), auto-downloaded on the first `serve`. Use the HF mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Cache: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct`.

## Run

```bash
export HF_ENDPOINT=https://hf-mirror.com
~/.venvs/vllm/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8001 --gpu-memory-utilization 0.80
```

- `--gpu-memory-utilization 0.80` is required on this 8G card; the default 0.92 fails at startup with:
  `Free memory ... (6.93/8.0 GiB) is less than desired (0.92, 7.36 GiB)`.
- Forgot the `export`? `serve` then hangs with no GPU activity and no error (it stalls connecting to
  huggingface.co, whose DNS is poisoned here to unreachable `2a03:2880:...`). Ctrl-C and restart
  with the env var — the model is cached, nothing re-downloads.
- First start takes ~2 min (weight load); VRAM after startup: ~6.3G / 8G.
- Run it in its own terminal/tmux — the process dies with the shell that started it.

## Verify

```bash
# 1. is it up?
curl -s http://127.0.0.1:8001/v1/models

# 2. end-to-end chat completion
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"你好"}]}'
```

## Disk footprint / cleanup

| Path | Size | Note |
|------|------|------|
| `~/.venvs/vllm` | 8.1G | venv (nvidia wheels 3.0G, torch 1.2G, vllm 0.8G) |
| `~/.cache/huggingface` | 2.9G | model weights |
| `~/.cache/pip` | 5.7G | wheel download cache; safe to delete |

Full cleanup: `rm -rf ~/.venvs/vllm ~/.cache/huggingface ~/.cache/pip`.
