#include "gateway/rate_limiter.h"

#include <algorithm>

RateLimiter::RateLimiter(double qps) : qps_(qps) {}

bool RateLimiter::Allow(const std::string& key) {
  if (qps_ <= 0) {
    return true;
  }
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> guard(mutex_);
  Bucket& bucket = buckets_[key];
  if (bucket.last.time_since_epoch().count() == 0) {
    bucket.tokens = qps_;  // 新客户端：满桶起步（允许一次突发）
    bucket.last = now;
  }
  const double elapsed =
      std::chrono::duration<double>(now - bucket.last).count();
  bucket.tokens = std::min(qps_, bucket.tokens + elapsed * qps_);
  bucket.last = now;
  if (bucket.tokens < 1.0) {
    return false;
  }
  bucket.tokens -= 1.0;
  return true;
}
