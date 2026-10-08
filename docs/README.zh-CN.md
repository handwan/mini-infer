# mini-infer

一个极简的 C++ 推理服务器，两张面孔：

- **ONNX 引擎** —— 面向小模型的 ONNX Runtime 动态批处理（`/predict`）
- **LLM 网关** —— vLLM 前面的透明 C++ 代理（`/v1/chat/completions`、`/v1/responses`：创建、取回、取消）

一个二进制同时提供两者；两条路径只共享 HTTP 层。

这是一个动手项目：把"无聊但真实"的部分从零做一遍——C++ 里的 HTTP 并发、微批处理、超时与重试、流式透传，以及在一台笔记本 GPU 上端到端实测的量化。

## 环境要求

- Linux x86_64；C++20 编译器（实测：Ubuntu 24.04 上的 GCC 13.3 与 Clang 18）；CMake ≥ 3.16（实测 3.28）
- 依赖由脚本获取（固定版本 + sha256）——除编译器和 CMake 外无需额外系统包
- Python + PyTorch + ONNX Runtime —— **仅**用于导出演示模型（版本固定在 `requirements.txt`）
- 可选：`wrk`（基准测试）、`clang-format` + `clang-tidy`（`scripts/check.sh`）
- 仅 LLM 网关需要：单独安装的 vLLM 服务器（实测 0.30.0；安装记录见 [vllm-setup.md](vllm-setup.md)）；基准脚本需要 Python `requests`（固定在 `requirements.txt`）

## 快速开始

```bash
scripts/fetch_deps.sh      # 获取依赖（固定版本 + sha256）
cmake -B build && cmake --build build -j
python3 scripts/export_model.py            # 生成 models/tiny_mlp.onnx（只需一次；仅导出）
./build/mini-infer 8080                    # 端口 8080；提供 /health /echo /version /predict

# 另一个终端：
scripts/bench.sh                           # /echo 快速基准
scripts/sweep_batch.sh                     # /predict 批处理参数扫描
```

网关需要先跑一个 vLLM（任何 OpenAI 兼容服务器都行）。即使没有 ONNX 模型，二进制也能启动——`/predict` 会返回 503，直到跑过一次 `scripts/export_model.py`：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001 --gpu-memory-utilization 0.80
./build/mini-infer 8080                    # 同一个二进制；网关 → 127.0.0.1:8001

# 通过网关各来一发：
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"hi"}]}'
curl -sN http://127.0.0.1:8080/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","input":"count to three","stream":true}'

python3 scripts/bench_llm.py --base http://127.0.0.1:8080 --concurrency 1,2,4,8 --rounds 3   # 流式，经网关
python3 scripts/bench_llm.py --base http://127.0.0.1:8001 --concurrency 1,2,4,8 --rounds 3   # 直连 vLLM
```

### 容器

`docker build -t mini-infer .` 得到两阶段构建的服务器镜像（不带编译器、不带源码）。vLLM 仍是
外部服务，ONNX 模型不打包进镜像——要 `/predict` 就把 `models/` 挂进去，否则以 gateway-only 运行：

```bash
docker run --rm -p 8080:8080 -v "$PWD/models:/app/models" mini-infer
# 容器要连宿主机的 vLLM：加 --network host（Linux），默认地址直接可用
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 存活检查 |
| `/echo` | GET/POST | HTTP 层基线（不做推理） |
| `/version` | GET | `{"name":"mini-infer","version":"0.3.0"}` |
| `/predict` | POST | body：逗号分隔的浮点数（`1,2,3,4`）→ `{"output":[...]}` |
| `/v1/chat/completions` | POST | OpenAI 兼容对话；转发到 vLLM。`stream: true` 增量转发（SSE 透传） |
| `/v1/responses` | POST | OpenAI Responses API；转发到 vLLM 的同名路径。`stream: true` 是带类型的 SSE 事件——与 chat 同样增量转发 |
| `/v1/responses/{id}` | GET | 取回已保存的响应 |
| `/v1/responses/{id}/cancel` | POST | 取消进行中（后台）的响应 |

Responses 的 store（vLLM 0.30）默认关闭——`store`、`background`、取回、取消都需要启动时带 `VLLM_ENABLE_RESPONSES_API_STORE=1`（不带的话，`store: true` 正常返回，但取回会 404）。

环境变量（无需重新编译）：

- ONNX 引擎：`PORT`、`MODEL_PATH`、`BATCH_MAX`、`BATCH_WINDOW_US`
- LLM 网关：`VLLM_HOST`、`VLLM_PORT`（默认 `127.0.0.1:8001`）；`GATEWAY_CONNECT_TIMEOUT_MS`（2000）、`GATEWAY_READ_TIMEOUT_MS`（60000）、`GATEWAY_MAX_RETRIES`（1）、`GATEWAY_RATE_LIMIT_QPS`（0 = 关闭）

## 设计

### ONNX 引擎 —— 动态批处理

```
客户端 ──► HTTP 服务器（线程池）
  └─ /predict：解析 + 校验
      └─ DynamicBatcher：队列 ─► 单个工作线程
          ├─ 攒一个微批：最多 BATCH_MAX、窗口上限（BATCH_WINDOW_US）、或 200 µs 空闲
          └─ 整个批次一次 ONNX Runtime Run()
      └─ 结果发回每个等待中的请求
```

**空闲规则**——队列安静 200 µs 就派发——在并发低于 `BATCH_MAX` 时不必等满窗口。（至于批处理是否*划算*，是另一个问题——见「实测数据」与「取舍与教训」。）

### LLM 网关 —— 透明代理

```
客户端 ──► POST /v1/chat/completions（或 /v1/responses）
  ├─ 每 IP 令牌桶 → 超限 429 + Retry-After
  └─ 转发 ─► vLLM :8001
      ├─ 连接超时 2 s —— 快速失败（默认 300 s，面对"沉默的后端"不可用）
      ├─ 只重试连接失败（请求还没到达模型）；读超时不重试——模型可能已经在生成
      └─ 错误映射：502 = 连不上 / 对端断开，504 = 等上游读超时
```

网关先等上游响应头、再碰 body——上游错误因此保留真实状态码——然后按响应形状分流：

- **SSE body**（`stream: true`）边到边发；客户端断开就停止读上游。chat 的 data 帧和 Responses 的带名事件走同一条通道、原样通过：它转发的是字节，不是协议。
- **其余响应**整收后按原状态码转发。

更多决定：

- **不做连接池**（实测）：keep-alive 每请求省 269 µs（480 → 211 µs）——一次 LLM 调用耗时的 0.02–0.09%。连接池是为高持续 RPS 下的 TIME_WAIT 回收存在的，不是为 LLM 延迟；每请求新建 client 让请求路径保持无锁。
- **优雅退出**：第一次 SIGINT/SIGTERM 等在途请求排空（以其超时为界）；第二次直接 `std::_Exit`。
- **每连接一线程、请求路径上无共享可变状态**。
- **Responses API 的接入成本几乎为零**：`POST /v1/responses`、`GET /v1/responses/{id}`、`POST /v1/responses/{id}/cancel` 是同一套转发上的三条路由——唯一的改动是把写死的 `POST /v1/chat/completions` 换成转发 `method + path`。状态（`store`、`background`、`previous_response_id`）都在 vLLM 里；网关保持无状态。

## 实测数据

Qwen2.5-1.5B-Instruct + vLLM 0.30.0，单张 RTX 4060 Laptop（8 GiB）。批次间波动 ±10–15%；每块的方法与原始表格在 [benchmarks.md](benchmarks.md)。

- `/predict`（CPU）：最高 **53.6k rps @ c=32**（tiny MLP）；微批只对计算密集的模型划算，对亚 100 µs 的模型不划算——`BATCH_MAX=1` 反而赢
- vLLM 并发：4 个 64-token 的生成，并发 **0.96 s vs 串行 6.04 s（约 6×）**——网关不串行化；干活的是 continuous batching
- 流式（128 输出 token）：**TTFT 32 → 50 ms**，**TPOT ≈ 13–14 ms，从 1 到 8 并发保持平坦**；系统吞吐 **73.8 → 560.6 tok/s（8 并发时 7.6×）**。经网关（SSE 透传）TPOT 与吞吐和直连在噪声内一致；TTFT +1–7 ms
- 网关转发开销在生成时间（约 1.7 s/请求）面前低于测量噪声；上游一条新建 TCP + HTTP 连接约 480 µs
- KV cache：28 KB/token（GQA）；vLLM 用量随长度线性增长：0.4k / 2.6k / 9.8k prompt token 时占 99,616-token 池的 0.40 / 2.57 / 9.82%；玩具 demo 显示带 cache 比不带便宜约 30–80×，但仍然近似 O(N²)
- 量化（GPTQ 官方 checkpoint）：权重 **2.98 → 1.74 → 1.10 GiB**，被 vLLM 再投资成 KV 容量（**99,616 → 133,472 → 142,992 token**）；TPOT **14.6 → 8.8 → 6.4 ms**；8 并发吞吐 **579 → 845 → 1126 tok/s**——在 Marlin 内核加持下，这张卡上量化反而更快。质量冒烟测试：10/12、10/12、9/12——INT8 ≈ FP16，INT4 开始出现裂缝

## 取舍与教训

- **批处理不是默认的赢家。** 亚 100 µs 的模型上，HTTP 与调度开销占主导——`BATCH_MAX=1` 打败微批（c=32 时 53.6k vs 43.1k rps）。批处理机制在计算密集的路径上才回本，而不是那个小 MLP。
- **先测量，再优化。** 连接池看起来是显然正确的事；实测的数字（见「设计」）说不值得。连接池是为高持续 RPS 下的 TIME_WAIT 堆积准备的——不是为 LLM 延迟。
- **要么快速失败，要么永久挂起。** 开发机对未监听端口静默丢 SYN（不回 RST），默认 300 s 连接超时没法用——所有对外 client 都显式设置超时。
- **httplib 把"读超时"和"对端断开"塞进了同一个错误码。** 网关用读截止时间判断（`elapsed >= read_timeout`）来区分 504 和 502——一个记录在案的启发式；改 vendored 库或换客户端才能做到精确。
- **网关保持协议无关。** 字节进、字节出——不解析 schema、不保存服务端状态。每加一个端点，多的是路由，不是实现。
- **后端的开关归后端管。** vLLM 的 Responses store 需要显式开启（`VLLM_ENABLE_RESPONSES_API_STORE=1`）；不开启时 `store: true` 照常返回，但不落档。代理没法——也不该——替它粉饰。
- **探针不是评测。**"INT8 ≈ FP16"的意思是"12+5 道冒烟题上看不出损伤"，不是质量结论；INT4 的漂移是"该建一个真正的评测集"的信号。另外，这里 INT4 更快是 Marlin-on-Ada 的结果，不是普适规律。
- **范围决策要摆在明面上。** Docker 打包曾在可砍清单上，最后又做了回来（只含服务器的镜像——见「容器」）。与其两头都做成半成品，不如把取舍说清楚。

## 复现这些数字

依赖全部钉版本；[benchmarks.md](benchmarks.md) 里每一块都写了确切的命令与设置。简版：

- CPU 扫描：`BATCH_CONFIGS="0 1;200 16" scripts/sweep_batch.sh models/tiny_mlp.onnx 32 5 3s`（需要 `wrk`）
- LLM 指标（网关或直连）：`python3 scripts/bench_llm.py --base <url> --concurrency 1,2,4,8 --rounds 3`
- 量化：起同一个模型的 `…-GPTQ-Int8` / `…-GPTQ-Int4` checkpoint，再跑 `scripts/bench_llm.py` 和 `scripts/quant_probe.py --model <checkpoint>`
- KV cache 玩具：`./build/kv_cache_demo 128 256 512`
- 风格 + 静态检查：`scripts/check.sh`

参考机器：Ubuntu 24.04（WSL2）、i7-14650HX、RTX 4060 Laptop 8G、GCC 13.3、CMake 3.28；vLLM 0.30.0 按 [vllm-setup.md](vllm-setup.md) 装进 venv——机器相关的坑（HF 镜像、`--gpu-memory-utilization 0.80`）也记在那里。

## 项目结构

```
Dockerfile          两阶段容器构建（只含服务器；见「容器」）
src/                服务器本体——HTTP + 路由（main.cc）、ONNX 引擎 + 批处理、vLLM 网关
src/experiments/    独立基准（kv_cache_demo）
scripts/            依赖获取、模型导出、基准、风格检查、参数扫描
docs/               vLLM 安装、完整基准数据、开发指南
third_party/        固定版本依赖（由 scripts/fetch_deps.sh 获取）
```

## 版本历史

- **v0.1**（tag `v0.1`）——HTTP 服务器、ONNX Runtime 引擎、动态批处理、参数扫描
- **v0.2**（tag `v0.2`）——vLLM 后端、C++ 网关、KV cache 研究、LLM 指标
- **v0.3**（tag `v0.3`）——量化对比、SSE 透传、Responses API、fail-fast 启动、Docker 打包、文档整理
