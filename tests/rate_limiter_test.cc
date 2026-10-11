#include "gateway/rate_limiter.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

// 新客户端满桶起步：突发上限 = qps
TEST(RateLimiterTest, BurstUpToCapacity) {
  RateLimiter limiter(5.0);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(limiter.Allow("a")) << "第 " << (i + 1) << " 个应放行";
  }
  EXPECT_FALSE(limiter.Allow("a"));  // 桶已空
}

// 令牌按时间补充：5 qps → 每 200 ms 补一个
TEST(RateLimiterTest, RefillsOverTime) {
  RateLimiter limiter(5.0);
  for (int i = 0; i < 5; ++i) {
    limiter.Allow("a");
  }
  EXPECT_FALSE(limiter.Allow("a"));
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  EXPECT_TRUE(limiter.Allow("a"));
}

// 不同 key 的桶互相独立
TEST(RateLimiterTest, KeysAreIsolated) {
  RateLimiter limiter(1.0);
  EXPECT_TRUE(limiter.Allow("a"));
  EXPECT_FALSE(limiter.Allow("a"));
  EXPECT_TRUE(limiter.Allow("b"));  // b 的桶还是满的
}

// qps <= 0 关闭限流：全部放行
TEST(RateLimiterTest, DisabledWhenQpsIsNonPositive) {
  RateLimiter off(0.0);
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(off.Allow("a"));
  }
}
