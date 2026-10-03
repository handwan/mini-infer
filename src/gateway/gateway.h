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
        std::chrono::milliseconds connect_timeout{2000};  // 连不上快速失败
        std::chrono::milliseconds read_timeout{60000};  // 等后端生成的最长时间
    };

    // 一次转发的结果
    struct Result {
        int status = 0;     // 后端 HTTP 状态码
        std::string body;   // 后端响应体（原样透传）
        std::string error;  // 连接失败/超时时填，如 "connect failed"
    };

    explicit Gateway(Config config);

    // 转发一次 chat completion 请求；body 是客户端原始 JSON。
    // 返回 true = 拿到后端响应（看 status/body）；false = 没连上（看 error）
    bool ForwardChat(const std::string& body, Result& out) const;

   private:
    Config config_;
};
