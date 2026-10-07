#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "batch/dynamic_batcher.h"
#include "engine/onnx_engine.h"
#include "gateway/gateway.h"
#include "gateway/rate_limiter.h"
#include "httplib.h"

namespace {

std::atomic<bool> g_stop{false};

void onSignal(int /*signo*/) {
  // 第一次信号：置标志，由 watcher 优雅收尾（等在途请求结束）；
  // 第二次信号：跳过等待，直接退出（_Exit 在信号处理里是安全的）。
  if (g_stop.exchange(true)) {
    std::_Exit(0);
  }
}

int64_t nowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// 打印一次处理的耗时（毫秒）
void logCost(const char* name, int64_t start_ms) {
  printf("%s took %" PRId64 " ms\n", name, nowMs() - start_ms);
}

// 网关配置：VLLM_HOST / VLLM_PORT / GATEWAY_CONNECT_TIMEOUT_MS /
// GATEWAY_READ_TIMEOUT_MS / GATEWAY_MAX_RETRIES（默认值见 gateway.h）
Gateway::Config gatewayConfigFromEnv() {
  Gateway::Config config;
  if (const char* env = std::getenv("VLLM_HOST")) {
    config.host = env;
  }
  if (const char* env = std::getenv("VLLM_PORT")) {
    config.port = std::atoi(env);
  }
  if (const char* env = std::getenv("GATEWAY_CONNECT_TIMEOUT_MS")) {
    config.connect_timeout = std::chrono::milliseconds(std::atoll(env));
  }
  if (const char* env = std::getenv("GATEWAY_READ_TIMEOUT_MS")) {
    config.read_timeout = std::chrono::milliseconds(std::atoll(env));
  }
  if (const char* env = std::getenv("GATEWAY_MAX_RETRIES")) {
    config.max_retries = std::atoi(env);
  }
  return config;
}

// 限流配置：GATEWAY_RATE_LIMIT_QPS（0 = 关闭）
double rateLimitQpsFromEnv() {
  if (const char* env = std::getenv("GATEWAY_RATE_LIMIT_QPS")) {
    return std::atof(env);
  }
  return 0.0;
}

void handleChatCompletion(const httplib::Request& req, httplib::Response& res,
                          Gateway& gateway, RateLimiter& limiter) {
  const int64_t t0 = nowMs();
  if (!limiter.Allow(req.remote_addr)) {
    res.status = 429;
    res.set_header("Retry-After", "1");
    res.set_content(R"({"error":"rate limit exceeded"})", "application/json");
    logCost("gateway", t0);
    return;
  }

  Gateway::Stream stream = gateway.OpenChatStream(req.body);
  if (!stream.Valid()) {
    // 超时类失败 → 504（等上游等超了）；连不上/被断开 → 502
    res.status = stream.TimedOut() ? 504 : 502;
    res.set_content(R"({"error":")" + stream.Error() + R"("})",
                    "application/json");
    logCost("gateway", t0);
    return;
  }

  std::string content_type = stream.ContentType();
  if (content_type.empty()) {
    content_type = "application/json";
  }
  if (!stream.IsSse()) {
    // 非流式（或上游错误）：整收后按原状态码转回
    res.status = stream.Status();
    res.set_content(stream.ReadAll(), content_type);
    logCost("gateway", t0);
    return;
  }

  // SSE：边收边透传；客户端断开 → 停止读上游（句柄析构时关连接）
  res.status = stream.Status();
  auto session = std::make_shared<Gateway::Stream>(std::move(stream));
  res.set_chunked_content_provider(
      content_type, [session, t0](size_t /*offset*/, httplib::DataSink& sink) {
        std::array<char, 16384> buf{};
        bool client_ok = true;
        for (;;) {
          const std::ptrdiff_t n = session->Read(buf.data(), buf.size());
          if (n == 0) {
            break;  // 上游正常收尾
          }
          if (n < 0) {
            printf("gateway: upstream stream interrupted: %s\n",
                   session->ReadError().c_str());
            break;
          }
          if (!sink.write(buf.data(), static_cast<size_t>(n))) {
            client_ok = false;  // 客户端断开
            break;
          }
        }
        sink.done();
        logCost("gateway stream", t0);
        return client_ok;
      });
  logCost("gateway", t0);
}

// /predict 的处理：解析逗号分隔输入 → 走动态批处理 → 返回 JSON。
void handlePredict(const httplib::Request& req, httplib::Response& res,
                   OnnxEngine* engine, DynamicBatcher* batcher) {
  if (batcher == nullptr || engine == nullptr) {
    res.status = 503;
    res.set_content(
        R"({"error":"ONNX model not loaded; run scripts/export_model.py"})",
        "application/json");
    return;
  }
  std::vector<float> xs;
  std::stringstream ss(req.body);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    try {
      xs.push_back(std::stof(tok));
    } catch (const std::exception&) {
      res.status = 400;
      res.set_content(R"({"error":"invalid number"})", "application/json");
      return;
    }
  }

  const int64_t dim = engine->input_dim();
  if (xs.empty() || xs.size() % static_cast<size_t>(dim) != 0) {
    res.status = 400;
    const std::string err = R"({"error":"input size must be a multiple of )" +
                            std::to_string(dim) + R"("})";
    res.set_content(err, "application/json");
    return;
  }

  const int64_t t0 = nowMs();
  std::vector<float> out;
  for (size_t off = 0; off < xs.size(); off += static_cast<size_t>(dim)) {
    const std::vector<float> sample(
        xs.begin() + static_cast<ptrdiff_t>(off),
        xs.begin() + static_cast<ptrdiff_t>(off + dim));
    const std::vector<float> r = batcher->Submit(sample);
    out.insert(out.end(), r.begin(), r.end());
  }
  logCost("predict", t0);

  std::string body = R"({"output":[)";
  for (size_t i = 0; i < out.size(); ++i) {
    std::array<char, 32> buf{};
    snprintf(buf.data(), buf.size(), "%.6g", out[i]);
    if (i > 0) {
      body += ",";
    }
    body += buf.data();
  }
  body += "]}";
  res.set_content(body, "application/json");
}

}  // namespace

int main(int argc, char** argv) {
  int port = 8080;
  if (argc > 1) {
    port = std::atoi(argv[1]);
  }
  if (const char* env = std::getenv("PORT")) {
    port = std::atoi(env);
  }

  // 模型路径可用环境变量换（做对照实验）：MODEL_PATH=models/wide_mlp.onnx
  std::string model_path = "models/tiny_mlp.onnx";
  if (const char* env = std::getenv("MODEL_PATH")) {
    model_path = env;
  }

  // 加载模型：缺失/损坏不致命——网关面不依赖它（Ort 异常也属 std::exception）
  std::unique_ptr<OnnxEngine> engine;
  try {
    engine = std::make_unique<OnnxEngine>(model_path);
  } catch (const std::exception& e) {
    fprintf(stderr, "warning: ONNX model unavailable (%s): %s\n",
            model_path.c_str(), e.what());
    fprintf(stderr,
            "         /predict disabled; run scripts/export_model.py to "
            "enable it\n");
  }

  // 动态批处理：多个请求攒成一批一起算
  // 参数支持环境变量（做扫描实验时不用重新编译）：BATCH_MAX、BATCH_WINDOW_US
  DynamicBatcher::Config batch_cfg;
  batch_cfg.max_batch = 8;
  batch_cfg.window = std::chrono::microseconds(2000);
  if (const char* env = std::getenv("BATCH_MAX")) {
    batch_cfg.max_batch = std::atoll(env);
  }
  if (const char* env = std::getenv("BATCH_WINDOW_US")) {
    batch_cfg.window = std::chrono::microseconds(std::atoll(env));
  }
  // 没模型就没有批处理（尺寸也无从谈起）
  std::unique_ptr<DynamicBatcher> batcher;
  if (engine) {
    batch_cfg.input_dim = engine->input_dim();
    batch_cfg.output_dim = engine->output_dim();
    batcher = std::make_unique<DynamicBatcher>(
        batch_cfg, [&engine](const std::vector<float>& x, int64_t /*batch*/) {
          return engine->Run(x);
        });
  }

  // 网关：把 OpenAI 兼容请求转发给 vLLM
  // 后端地址可换：VLLM_HOST / VLLM_PORT（默认 127.0.0.1:8001）
  const Gateway::Config gateway_cfg = gatewayConfigFromEnv();
  Gateway gateway(gateway_cfg);

  const double rate_qps = rateLimitQpsFromEnv();
  RateLimiter rate_limiter(rate_qps);

  httplib::Server svr;

  // 访问日志：时间戳 方法 路径 → 状态码
  svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
    printf("[%" PRId64 "] %s %s -> %d\n", nowMs(), req.method.c_str(),
           req.path.c_str(), res.status);
    fflush(stdout);
  });

  svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("ok\n", "text/plain");
  });

  // 以后 version 要出现在别处，再抽成常量
  svr.Get("/version", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"name":"mini-infer","version":"0.2.0"})",
                    "application/json");
  });

  // /echo：压测用
  svr.Post("/echo", [](const httplib::Request& req, httplib::Response& res) {
    const int64_t t0 = nowMs();
    res.set_content(req.body, "text/plain");
    logCost("echo", t0);
  });
  svr.Get("/echo", [](const httplib::Request& req, httplib::Response& res) {
    const int64_t t0 = nowMs();
    res.set_content(req.target, "text/plain");  // 回显路径+query
    logCost("echo", t0);
  });

  // /v1/chat/completions：转发给 vLLM（OpenAI 兼容接口）
  svr.Post("/v1/chat/completions",
           [&gateway, &rate_limiter](const httplib::Request& req,
                                     httplib::Response& res) {
             handleChatCompletion(req, res, gateway, rate_limiter);
           });

  // /predict：body 是逗号分隔的输入（每 4 个数一条样本），返回 JSON
  svr.Post("/predict", [&batcher, &engine](const httplib::Request& req,
                                           httplib::Response& res) {
    handlePredict(req, res, engine.get(), batcher.get());
  });

  // 优雅退出：收到信号 → 另起线程调 stop()（listen 会返回）
  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);
  std::thread watcher([&svr] {
    while (!g_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    printf("\nshutdown signal received, stopping...\n");
    svr.stop();
  });

  printf(
      "mini-infer listening on http://0.0.0.0:%d (/health, /echo, /version, "
      "/predict, /v1/chat/completions)\n",
      port);
  if (batcher) {
    printf("batching: max_batch=%lld window=%lldus\n",
           static_cast<long long>(batch_cfg.max_batch),
           static_cast<long long>(batch_cfg.window.count()));
  } else {
    printf("ONNX engine: disabled (no model; /predict -> 503)\n");
  }
  printf("gateway: vllm backend http://%s:%d\n", gateway_cfg.host.c_str(),
         gateway_cfg.port);
  printf("rate limit: %.1f qps/client (0 = off)\n", rate_qps);
  if (!svr.listen("0.0.0.0", port)) {
    fprintf(stderr, "failed to start: port %d may be in use\n", port);
    g_stop.store(true);
    watcher.join();
    return 1;
  }

  watcher.join();
  if (batcher) {
    const auto st = batcher->stats();
    const double avg = st.batches > 0 ? static_cast<double>(st.requests) /
                                            static_cast<double>(st.batches)
                                      : 0.0;
    printf("batcher stats: requests=%llu batches=%llu avg=%.2f max=%llu\n",
           static_cast<unsigned long long>(st.requests),
           static_cast<unsigned long long>(st.batches), avg,
           static_cast<unsigned long long>(st.max_seen));
  }
  printf("exited\n");
  return 0;
}
