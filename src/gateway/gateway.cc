#include "gateway/gateway.h"

#include <chrono>
#include <utility>

#include "httplib.h"

Gateway::Gateway(Config config) : config_(std::move(config)) {}

bool Gateway::ForwardChat(const std::string& body, Result& out) const {
    httplib::Client cli(config_.host, config_.port);
    cli.set_connection_timeout(config_.connect_timeout);
    cli.set_read_timeout(config_.read_timeout);

    const auto start = std::chrono::steady_clock::now();
    auto res = cli.Post("/v1/chat/completions", body, "application/json");

    // 没打通（后端没起/端口错/超时）：res 是空的
    if (!res) {
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
        return false;
    }

    out.status = res->status;
    out.body = res->body;
    return true;
}
