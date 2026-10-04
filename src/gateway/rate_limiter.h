#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

// 按 key 限流（令牌桶），key 一般是客户端 IP。
//
// 桶容量 = qps（允许 1 秒的突发），按经过的时间连续补充令牌；
// qps <= 0 表示关闭（全部放行）。线程安全。
//
// 桶表不淘汰（规模够用）；生产环境应按 API key 限流，按 IP 在反向代理
// 后面会失真。
class RateLimiter {
   public:
    explicit RateLimiter(double qps);

    // 取 1 个令牌：true = 放行；false = 超限
    bool Allow(const std::string& key);

   private:
    struct Bucket {
        double tokens = 0;
        std::chrono::steady_clock::time_point last;
    };

    const double qps_;
    std::mutex mutex_;
    std::unordered_map<std::string, Bucket> buckets_;
};
