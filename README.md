# mini-infer

A minimal C++ inference server — dynamic batching, ONNX Runtime + vLLM backends, quantization benchmarks.

## Quick start

```bash
scripts/fetch_deps.sh     # fetch dependencies (pinned + sha256)
cmake -B build && cmake --build build -j
scripts/check.sh          # format + static checks
./build/mini-infer 8080   # run server (/health, /echo, /version); blocks the terminal
# then in another terminal:
scripts/bench.sh          # benchmark (server must be running)
```
