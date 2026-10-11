#include "batch/dynamic_batcher.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

// 并发提交 kRequests 条样本，返回每条请求各自拿到的结果
std::vector<int> ConcurrentSubmits(DynamicBatcher& batcher, int kRequests) {
  std::vector<int> results(static_cast<size_t>(kRequests), -1);
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(kRequests));
  for (int i = 0; i < kRequests; ++i) {
    threads.emplace_back([&batcher, &results, i] {
      const std::vector<float> r = batcher.Submit({static_cast<float>(i)});
      if (!r.empty()) {
        results[static_cast<size_t>(i)] = static_cast<int>(r[0]);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  return results;
}

}  // namespace

// 单条提交：返回值按 output_dim 切好、内容正确
TEST(DynamicBatcherTest, SingleSubmitReturnsResult) {
  DynamicBatcher batcher({.input_dim = 2,
                          .output_dim = 1,
                          .max_batch = 4,
                          .window = std::chrono::microseconds(2000)},
                         [](const std::vector<float>& in, int64_t n) {
                           std::vector<float> out(static_cast<size_t>(n));
                           for (int64_t i = 0; i < n; ++i) {
                             out[static_cast<size_t>(i)] =
                                 in[static_cast<size_t>(i * 2)] +
                                 in[static_cast<size_t>(i * 2 + 1)];
                           }
                           return out;
                         });
  const std::vector<float> r = batcher.Submit({1.0F, 2.0F});
  ASSERT_EQ(r.size(), 1U);
  EXPECT_FLOAT_EQ(r[0], 3.0F);
}

// 输入长度不对：直接抛 invalid_argument
TEST(DynamicBatcherTest, WrongInputSizeThrows) {
  DynamicBatcher batcher(
      {.input_dim = 2,
       .output_dim = 1,
       .max_batch = 4,
       .window = std::chrono::microseconds(2000)},
      [](const std::vector<float>&, int64_t) { return std::vector<float>{}; });
  EXPECT_THROW(batcher.Submit({1.0F, 2.0F, 3.0F}), std::invalid_argument);
}

// 引擎（run）抛异常：Submit 把它传播给调用者
TEST(DynamicBatcherTest, RunErrorPropagatesToCaller) {
  DynamicBatcher batcher(
      {.input_dim = 1,
       .output_dim = 1,
       .max_batch = 4,
       .window = std::chrono::microseconds(2000)},
      [](const std::vector<float>&, int64_t) -> std::vector<float> {
        throw std::runtime_error("boom");
      });
  EXPECT_THROW(batcher.Submit({1.0F}), std::runtime_error);
}

// 静默期提前开跑：单条请求配超长 window 也能很快返回（不用硬等满 window）
TEST(DynamicBatcherTest, QuietPeriodFiresWithoutWaitingFullWindow) {
  DynamicBatcher batcher(
      {.input_dim = 1,
       .output_dim = 1,
       .max_batch = 4,
       .window = std::chrono::seconds(10)},
      [](const std::vector<float>& in, int64_t) { return in; });
  const auto start = std::chrono::steady_clock::now();
  const std::vector<float> r = batcher.Submit({7.0F});
  const auto elapsed = std::chrono::steady_clock::now() - start;
  ASSERT_EQ(r.size(), 1U);
  EXPECT_FLOAT_EQ(r[0], 7.0F);
  EXPECT_LT(elapsed, std::chrono::seconds(2));
}

// 并发提交会被攒批，且每条请求拿回自己的结果
TEST(DynamicBatcherTest, ConcurrentSubmitsAreBatched) {
  constexpr int kRequests = 8;
  DynamicBatcher batcher({.input_dim = 1,
                          .output_dim = 1,
                          .max_batch = kRequests,
                          .window = std::chrono::microseconds(200000)},
                         [](const std::vector<float>& in, int64_t) {
                           std::this_thread::sleep_for(
                               std::chrono::milliseconds(5));  // 模拟推理耗时
                           return in;
                         });
  const std::vector<int> results = ConcurrentSubmits(batcher, kRequests);

  for (int i = 0; i < kRequests; ++i) {
    EXPECT_EQ(results[static_cast<size_t>(i)], i);
  }
  const DynamicBatcher::Stats st = batcher.stats();
  EXPECT_EQ(st.requests, static_cast<uint64_t>(kRequests));
  EXPECT_GE(st.max_seen, 2U);  // 确实攒在一起过
  EXPECT_LE(st.max_seen, static_cast<uint64_t>(kRequests));
  EXPECT_LT(st.batches, static_cast<uint64_t>(kRequests));
}

// max_batch 是硬上限：超出的并发请求拆批，每批不超过上限
TEST(DynamicBatcherTest, MaxBatchSplitsConcurrentRequests) {
  constexpr int kRequests = 8;
  constexpr int kMaxBatch = 4;
  DynamicBatcher batcher(
      {.input_dim = 1,
       .output_dim = 1,
       .max_batch = kMaxBatch,
       .window = std::chrono::microseconds(200000)},
      [](const std::vector<float>& in, int64_t) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return in;
      });
  const std::vector<int> results = ConcurrentSubmits(batcher, kRequests);

  for (int i = 0; i < kRequests; ++i) {
    EXPECT_EQ(results[static_cast<size_t>(i)], i);
  }
  const DynamicBatcher::Stats st = batcher.stats();
  EXPECT_EQ(st.requests, static_cast<uint64_t>(kRequests));
  EXPECT_GE(st.batches, 2U);  // 8 条 ÷ 每批最多 4 条 → 至少两批
  EXPECT_LE(st.max_seen, static_cast<uint64_t>(kMaxBatch));
}
