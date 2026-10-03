#include "gateway/gateway.h"

#include <utility>

#include "httplib.h"

Gateway::Gateway(Config config) : config_(std::move(config)) {}

bool Gateway::ForwardChat(const std::string& body, Result& out) const {
    // 拨号：准备一个指向 vLLM 的客户端
    httplib::Client cli(config_.host, config_.port);
    cli.set_connection_timeout(config_.connect_timeout);  // 连不上快速失败
    cli.set_read_timeout(config_.read_timeout);  // 连上后最多等 60 秒

    // 说事：把客户端原始 JSON 原样 POST 过去
    auto res = cli.Post("/v1/chat/completions", body, "application/json");

    // 没打通（后端没起/端口错/超时）：res 是空的
    if (!res) {
        out.error = httplib::to_string(res.error());  // 错误码 → 文字
        return false;
    }

    // 听到了回话：状态码和响应体原样带回
    out.status = res->status;
    out.body = res->body;
    return true;
}
