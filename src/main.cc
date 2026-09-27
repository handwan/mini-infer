#include <atomic>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "httplib.h"

namespace {

std::atomic<bool> g_stop{false};

void onSignal(int /*signo*/) {
    g_stop.store(true);
}  // 信号处理里只置标志（别的动作不安全）

int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// 打印一次 /echo 的处理耗时（毫秒）
void logEchoCost(int64_t start_ms) {
    printf("echo took %" PRId64 " ms\n", nowMs() - start_ms);
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

    httplib::Server svr;

    // 访问日志：时间戳 方法 路径 → 状态码
    svr.set_logger(
        [](const httplib::Request& req, const httplib::Response& res) {
            printf("[%" PRId64 "] %s %s -> %d\n", nowMs(), req.method.c_str(),
                   req.path.c_str(), res.status);
            fflush(stdout);
        });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok\n", "text/plain");
    });

    // 以后 version 要出现在别处，再抽成常量
    svr.Get("/version", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"name":"mini-infer","version":"0.1.0"})",
                        "application/json");
    });

    // /echo：压测用
    svr.Post("/echo", [](const httplib::Request& req, httplib::Response& res) {
        const int64_t t0 = nowMs();
        res.set_content(req.body, "text/plain");
        logEchoCost(t0);
    });
    svr.Get("/echo", [](const httplib::Request& req, httplib::Response& res) {
        const int64_t t0 = nowMs();
        res.set_content(req.target, "text/plain");  // 回显路径+query
        logEchoCost(t0);
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
        "mini-infer listening on http://0.0.0.0:%d (/health, /echo, "
        "/version)\n",
        port);
    if (!svr.listen("0.0.0.0", port)) {
        fprintf(stderr, "failed to start: port %d may be in use\n", port);
        g_stop.store(true);
        watcher.join();
        return 1;
    }

    watcher.join();
    printf("exited\n");
    return 0;
}
