#pragma once

#include <chrono>
#include <string>

// 推理网关：把 OpenAI 兼容的请求转发给后端推理引擎（vLLM）
//
//   客户端 → mini-infer /v1/chat/completions → vLLM /v1/chat/completions
//
// 目前只做非流式转发。
class Gateway {
   public:
    struct Config {
        std::string host = "127.0.0.1";
        int port = 8001;
        std::chrono::milliseconds connect_timeout{2000};
        std::chrono::milliseconds read_timeout{60000};
    };

    struct Result {
        int status = 0;
        std::string body;   // 后端响应体（原样透传）
        std::string error;  // 传输层失败描述，如 "Connection timed out"
        bool timed_out = false;  // 超时类失败 → 504，其余传输失败 → 502
    };

    explicit Gateway(Config config);

    // 转发一次 chat completion 请求；body 是客户端原始 JSON。
    // 返回 true = 拿到后端响应（看 status/body）；false = 没连上（看 error）
    bool ForwardChat(const std::string& body, Result& out) const;

   private:
    Config config_;
};
