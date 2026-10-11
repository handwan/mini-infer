#include "gateway/gateway.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

#include "httplib.h"

namespace {

// 测试用上游：进程内起一个真的 httplib server（路由先注册、再开始监听），
// 析构时把服务器停掉。这样测的是 Gateway 的真实网络路径，不是 mock 打桩。
class MockUpstream {
 public:
  using SetupFn = std::function<void(httplib::Server&)>;

  explicit MockUpstream(const SetupFn& setup) {
    setup(server_);
    port_ = server_.bind_to_any_port("127.0.0.1");
    thread_ = std::thread([this] { server_.listen_after_bind(); });
    WaitUntilListening();
  }

  ~MockUpstream() {
    server_.stop();
    thread_.join();
  }

  MockUpstream(const MockUpstream&) = delete;
  MockUpstream& operator=(const MockUpstream&) = delete;

  [[nodiscard]] int port() const { return port_; }

 private:
  // listen 线程刚起时 socket 未必已 listen()：用探测请求等到真的能连上
  void WaitUntilListening() const {
    for (int i = 0; i < 200; ++i) {
      httplib::Client probe("127.0.0.1", port_);
      probe.set_connection_timeout(std::chrono::milliseconds(100));
      if (probe.Get("/__ping__") != nullptr) {
        return;  // 有响应（哪怕是 404）就说明在听了
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ADD_FAILURE() << "mock upstream 没能在预期时间内开始监听";
  }

  httplib::Server server_;
  int port_ = -1;
  std::thread thread_;
};

Gateway::Config GatewayConfig(int port) {
  Gateway::Config cfg;
  cfg.host = "127.0.0.1";
  cfg.port = port;
  cfg.max_retries = 0;  // 测试里不重试，失败要立刻可见
  return cfg;
}

// SSE 分两段发送的状态
struct SseState {
  std::atomic<bool> first_chunk_read{false};
  int step = 0;  // 只有服务端线程碰
};

// SSE 响应体提供者：先发第一段，等客户端消费后再发第二段
bool SendSseChunks(SseState& state, httplib::DataSink& sink) {
  static constexpr std::string_view kFirst = "data: one\n\n";
  static constexpr std::string_view kSecond = "data: two\n\n";
  if (state.step == 0) {
    state.step = 1;
    sink.write(kFirst.data(), kFirst.size());
    return true;
  }
  if (state.step != 1) {
    return false;
  }
  // 等客户端消费第一段（最多 5 秒，避免卡死）
  for (int i = 0; i < 5000 && !state.first_chunk_read.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  state.step = 2;
  sink.write(kSecond.data(), kSecond.size());
  sink.done();
  return false;
}

// 读到 needle 出现（或流结束）为止，返回已读到的内容
std::string ReadUntil(Gateway::Stream& stream, const std::string& needle) {
  std::string got;
  std::array<char, 64> buf{};
  while (got.find(needle) == std::string::npos) {
    const std::ptrdiff_t n = stream.Read(buf.data(), buf.size());
    if (n <= 0) {
      break;
    }
    got.append(buf.data(), static_cast<size_t>(n));
  }
  return got;
}

}  // namespace

// 正常响应：状态码 / Content-Type / body 原样透传（回显 body 顺带验证转发内容）
TEST(GatewayTest, PassesThroughStatusAndBody) {
  MockUpstream up([](httplib::Server& sv) {
    sv.Post("/v1/chat/completions",
            [](const httplib::Request& req, httplib::Response& res) {
              res.set_content(req.body, "application/json");
            });
  });

  Gateway gw(GatewayConfig(up.port()));
  Gateway::Stream s =
      gw.OpenStream("POST", "/v1/chat/completions", R"({"model":"m"})");
  ASSERT_TRUE(s.Valid());
  EXPECT_EQ(s.Status(), 200);
  EXPECT_EQ(s.ContentType(), "application/json");
  EXPECT_FALSE(s.IsSse());
  EXPECT_EQ(s.ReadAll(), R"({"model":"m"})");
}

// 上游错误码原样转回：网关不把 5xx 吞成 502
TEST(GatewayTest, PreservesUpstreamErrorStatus) {
  MockUpstream up([](httplib::Server& sv) {
    sv.Post("/v1/responses",
            [](const httplib::Request&, httplib::Response& res) {
              res.status = 503;
              res.set_content("overloaded", "text/plain");
            });
  });

  Gateway gw(GatewayConfig(up.port()));
  Gateway::Stream s = gw.OpenStream("POST", "/v1/responses", "{}");
  ASSERT_TRUE(s.Valid());
  EXPECT_EQ(s.Status(), 503);
  EXPECT_EQ(s.ReadAll(), "overloaded");
}

// SSE：识别 Content-Type + 增量透传（第二段要等客户端读走第一段才发，
// 排除了"攒完再发"——那种实现下这个测试会读不到第一段）
TEST(GatewayTest, StreamsSseIncrementally) {
  auto state = std::make_shared<SseState>();
  MockUpstream up([state](httplib::Server& sv) {
    sv.Post("/v1/chat/completions", [state](const httplib::Request&,
                                            httplib::Response& res) {
      res.set_chunked_content_provider(
          "text/event-stream", [state](size_t, httplib::DataSink& sink) {
            return SendSseChunks(*state, sink);
          });
    });
  });

  Gateway gw(GatewayConfig(up.port()));
  Gateway::Stream s =
      gw.OpenStream("POST", "/v1/chat/completions", R"({"stream":true})");
  ASSERT_TRUE(s.Valid());
  EXPECT_TRUE(s.IsSse());

  // 先把第一段读出来，再放行第二段
  const std::string first = ReadUntil(s, "one");
  state->first_chunk_read.store(true);
  EXPECT_EQ(first, "data: one\n\n");
  EXPECT_EQ(s.ReadAll(), "data: two\n\n");
}

// 上游没起来：Valid()=false 且不算超时（网关据此回 502）
TEST(GatewayTest, RefusedConnectionIsInvalidNotTimeout) {
  int port = 0;
  {
    MockUpstream up([](httplib::Server&) {});
    port = up.port();
  }  // 析构后端口没人监听了

  Gateway gw(GatewayConfig(port));
  Gateway::Stream s = gw.OpenStream("POST", "/v1/chat/completions", "{}");
  EXPECT_FALSE(s.Valid());
  EXPECT_FALSE(s.TimedOut());
  EXPECT_FALSE(s.Error().empty());
}

// 上游收下请求但迟迟不回响应头：报读超时（网关据此回 504）
TEST(GatewayTest, SlowHeadersBecomeReadTimeout) {
  MockUpstream up([](httplib::Server& sv) {
    sv.Post("/v1/chat/completions",
            [](const httplib::Request&, httplib::Response& res) {
              std::this_thread::sleep_for(std::chrono::milliseconds(400));
              res.set_content("late", "application/json");
            });
  });

  Gateway::Config cfg = GatewayConfig(up.port());
  cfg.read_timeout = std::chrono::milliseconds(100);
  Gateway gw(cfg);
  Gateway::Stream s = gw.OpenStream("POST", "/v1/chat/completions", "{}");
  EXPECT_FALSE(s.Valid());
  EXPECT_TRUE(s.TimedOut());
}
