#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

// 推理网关：把 OpenAI 兼容的请求转发给后端推理引擎（vLLM）
//
//   客户端 → mini-infer /v1/chat/completions → vLLM /v1/chat/completions
//
// 一次请求只连一次上游：先等上游响应头（此时能如实回 502/504/上游状态码），
// 然后两条路——上游回 SSE 就边收边透传；否则整收后按原状态码转回。
class Gateway {
 public:
  struct Config {
    std::string host = "127.0.0.1";
    int port = 8001;
    int max_retries = 1;  // 只重试"请求没送达"的失败（连接阶段）
    std::chrono::milliseconds connect_timeout{2000};
    std::chrono::milliseconds read_timeout{60000};
  };

  // 一次转发会话：Open 后上游响应头已就绪，body 用 Read 增量取。
  class Stream {
   public:
    Stream() = default;
    ~Stream();
    Stream(Stream&&) noexcept;
    Stream& operator=(Stream&&) noexcept;
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] bool Valid() const;  // true = 连上且收到响应头
    [[nodiscard]] int Status() const;  // 上游状态码（Valid 时有效）
    [[nodiscard]] std::string ContentType() const;
    [[nodiscard]] bool IsSse() const;  // Content-Type 含 text/event-stream
    [[nodiscard]] std::string Error() const;  // !Valid 时的传输层描述
    [[nodiscard]] bool TimedOut() const;  // !Valid 时：超时类失败 → 504
    [[nodiscard]] std::string ReadError() const;  // Read <0 后的错误描述

    // 增量读上游 body：>0 = 读到的字节数；0 = 正常结束；<0 = 读失败
    std::ptrdiff_t Read(char* buf, std::size_t len);
    std::string ReadAll();  // 整收（非 SSE 的响应）

   private:
    friend class Gateway;
    struct Impl;
    explicit Stream(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

  explicit Gateway(Config config);

  // 发起转发并等上游响应头；没等到（连不上/超时）时 Valid() 为 false。
  [[nodiscard]] Stream OpenChatStream(const std::string& body) const;

 private:
  Config config_;
};
