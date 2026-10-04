#include "gateway/gateway.h"

#include <chrono>
#include <cstdio>
#include <thread>
#include <utility>

#include "httplib.h"

Gateway::Gateway(Config config) : config_(std::move(config)) {}

bool Gateway::ForwardChat(const std::string& body, Result& out) const {
    for (int attempt = 0; attempt <= config_.max_retries; ++attempt) {
        httplib::Client cli(config_.host, config_.port);
        cli.set_connection_timeout(config_.connect_timeout);
        cli.set_read_timeout(config_.read_timeout);

        const auto start = std::chrono::steady_clock::now();
        auto res = cli.Post("/v1/chat/completions", body, "application/json");

        if (res) {
            out.status = res->status;
            out.body = res->body;
            return true;
        }

        const httplib::Error err = res.error();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        // 连接超时能直接分辨；读超时会被 httplib 归成 Error::Read（和对端
        // 断开同码），因此补一个 deadline 判断：用满读预算的失败算读超时。
        const bool read_timeout =
            err == httplib::Error::Read && elapsed >= config_.read_timeout;
        out.timed_out = err == httplib::Error::ConnectionTimeout ||
                        err == httplib::Error::Timeout || read_timeout;
        // 读超时时给个准确的描述（库原文是 "Failed to read connection"）
        out.error = read_timeout ? "Read timeout" : httplib::to_string(err);

        // 只重试"请求没送达"的失败（连接阶段）；读超时说明后端可能已经在
        // 生成，重试等于让它白算一遍，所以不重试。
        const bool retryable = err == httplib::Error::Connection ||
                               err == httplib::Error::ConnectionTimeout;
        if (!retryable || attempt == config_.max_retries) {
            return false;
        }
        printf("gateway: %s, retrying (%d/%d)\n", out.error.c_str(),
               attempt + 1, config_.max_retries);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;  // 兜底（max_retries < 0 时才会走到）
}
