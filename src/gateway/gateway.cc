#include "gateway/gateway.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <utility>

#include "httplib.h"

namespace {

// 传输层失败归类：读超时会被 httplib 归成 Error::Read（和对端断开同码），
// 因此用满读预算的失败才算读超时；其余超时类失败直接看错误码。
struct Failure {
    bool timed_out = false;
    std::string message;
};

Failure classifyFailure(httplib::Error err,
                        std::chrono::steady_clock::duration elapsed,
                        std::chrono::milliseconds read_timeout) {
    const bool is_read_timeout =
        err == httplib::Error::Read && elapsed >= read_timeout;
    Failure out;
    out.timed_out = err == httplib::Error::ConnectionTimeout ||
                    err == httplib::Error::Timeout || is_read_timeout;
    out.message = is_read_timeout ? "Read timeout" : httplib::to_string(err);
    return out;
}

bool isRetryable(httplib::Error err) {
    return err == httplib::Error::Connection ||
           err == httplib::Error::ConnectionTimeout;
}

}  // namespace

struct Gateway::Stream::Impl {
    httplib::ClientImpl::StreamHandle handle;
    std::string error;
    bool timed_out = false;
};

Gateway::Stream::Stream(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Gateway::Stream::~Stream() = default;
Gateway::Stream::Stream(Stream&&) noexcept = default;
Gateway::Stream& Gateway::Stream::operator=(Stream&&) noexcept = default;

bool Gateway::Stream::Valid() const {
    return impl_ && impl_->handle.is_valid();
}

int Gateway::Stream::Status() const {
    return (impl_ && impl_->handle.response) ? impl_->handle.response->status
                                             : 0;
}

std::string Gateway::Stream::ContentType() const {
    if (!impl_ || !impl_->handle.response) {
        return "";
    }
    return impl_->handle.response->get_header_value("Content-Type");
}

bool Gateway::Stream::IsSse() const {
    return ContentType().find("text/event-stream") != std::string::npos;
}

std::string Gateway::Stream::Error() const { return impl_ ? impl_->error : ""; }

bool Gateway::Stream::TimedOut() const { return impl_ && impl_->timed_out; }

std::string Gateway::Stream::ReadError() const {
    if (!impl_ || !impl_->handle.has_read_error()) {
        return "";
    }
    return httplib::to_string(impl_->handle.get_read_error());
}

std::ptrdiff_t Gateway::Stream::Read(char* buf, std::size_t len) {
    if (!impl_ || !impl_->handle.is_valid()) {
        return -1;
    }
    return static_cast<std::ptrdiff_t>(impl_->handle.read(buf, len));
}

std::string Gateway::Stream::ReadAll() {
    std::string out;
    std::array<char, 16384> buf{};
    for (;;) {
        const std::ptrdiff_t n = Read(buf.data(), buf.size());
        if (n <= 0) {
            break;
        }
        out.append(buf.data(), static_cast<std::size_t>(n));
    }
    return out;
}

Gateway::Gateway(Config config) : config_(std::move(config)) {}

Gateway::Stream Gateway::OpenChatStream(const std::string& body) const {
    for (int attempt = 0; attempt <= config_.max_retries; ++attempt) {
        httplib::Client cli(config_.host, config_.port);
        cli.set_connection_timeout(config_.connect_timeout);
        cli.set_read_timeout(config_.read_timeout);

        const auto start = std::chrono::steady_clock::now();
        auto handle = cli.open_stream("POST", "/v1/chat/completions", {}, {},
                                      body, "application/json");
        if (handle.is_valid()) {
            auto impl = std::make_unique<Stream::Impl>();
            impl->handle = std::move(handle);
            return Stream(std::move(impl));
        }

        const auto elapsed = std::chrono::steady_clock::now() - start;
        const Failure fail =
            classifyFailure(handle.error, elapsed, config_.read_timeout);
        if (!isRetryable(handle.error) || attempt == config_.max_retries) {
            auto impl = std::make_unique<Stream::Impl>();
            impl->error = fail.message;
            impl->timed_out = fail.timed_out;
            return Stream(std::move(impl));
        }
        printf("gateway: %s, retrying (%d/%d)\n", fail.message.c_str(),
               attempt + 1, config_.max_retries);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return Stream(nullptr);  // 兜底（max_retries < 0 时才会走到）
}
