#include "batch/dynamic_batcher.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {
// 静默期：这段时间没有新请求到来，就认为这批凑完了、立即开跑
constexpr auto kQuiet = std::chrono::microseconds(200);
}  // namespace

DynamicBatcher::DynamicBatcher(const Config& config, RunFn run)
    : input_dim_(config.input_dim),
      output_dim_(config.output_dim),
      max_batch_(config.max_batch),
      window_(config.window),
      run_(std::move(run)),
      worker_([this] { WorkerLoop(); }) {}

DynamicBatcher::~DynamicBatcher() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();  // 等当前批跑完
  }
}

DynamicBatcher::Stats DynamicBatcher::stats() const {
  std::lock_guard<std::mutex> lk(stats_mu_);
  return stats_;
}

// 请求侧：入队 → 睡在自己的条件变量上 → 被 worker 叫醒
std::vector<float> DynamicBatcher::Submit(const std::vector<float>& sample) {
  if (static_cast<int64_t>(sample.size()) != input_dim_) {
    throw std::invalid_argument("sample size must equal input_dim");
  }

  auto item = std::make_shared<Pending>();
  item->input = sample;
  {
    std::lock_guard<std::mutex> lk(mu_);
    queue_.push_back(item);
  }
  cv_.notify_one();

  std::unique_lock<std::mutex> lk(item->mu);
  item->cv.wait(lk, [&] { return item->done; });
  if (item->error) {
    std::rethrow_exception(item->error);
  }
  return item->output;
}

// 后台线程：不断把队列里的请求攒成一批，跑完分发结果
void DynamicBatcher::WorkerLoop() {
  std::vector<float> flat;  // 拼好的输入（循环外复用，少分配）
  std::vector<std::shared_ptr<Pending>> batch;

  while (true) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return !queue_.empty() || stop_; });
    if (queue_.empty() && stop_) {
      return;
    }

    // 凑批：window 是等待上限；静默 kQuiet（没有新请求）或凑满 max_batch
    // 就立即开跑——否则 max_batch 大于并发时，每条请求都要硬等满 window
    const auto deadline = std::chrono::steady_clock::now() + window_;
    auto quiet = std::chrono::steady_clock::now() + kQuiet;
    while (static_cast<int64_t>(queue_.size()) < max_batch_ && !stop_) {
      if (cv_.wait_until(lk, std::min(deadline, quiet)) ==
          std::cv_status::timeout) {
        break;
      }
      quiet =
          std::chrono::steady_clock::now() + kQuiet;  // 新请求到 → 顺延静默期
    }

    batch.clear();
    const int64_t n = std::min(static_cast<int64_t>(queue_.size()), max_batch_);
    for (int64_t i = 0; i < n; ++i) {
      batch.push_back(std::move(queue_.front()));
      queue_.pop_front();
    }
    lk.unlock();

    // 拼大张量：[n, input_dim]
    flat.clear();
    flat.reserve(static_cast<size_t>(n * input_dim_));
    for (const auto& item : batch) {
      flat.insert(flat.end(), item->input.begin(), item->input.end());
    }

    // 跑一批；成功或失败都要把 n 个请求叫醒
    try {
      const std::vector<float> out = run_(flat, n);
      for (int64_t i = 0; i < n; ++i) {
        Pending& item = *batch[static_cast<size_t>(i)];
        {
          std::lock_guard<std::mutex> lk_item(item.mu);
          item.output.assign(out.begin() + i * output_dim_,
                             out.begin() + (i + 1) * output_dim_);
          item.done = true;
        }
        item.cv.notify_one();
      }
    } catch (...) {
      for (int64_t i = 0; i < n; ++i) {
        Pending& item = *batch[static_cast<size_t>(i)];
        {
          std::lock_guard<std::mutex> lk_item(item.mu);
          item.error = std::current_exception();
          item.done = true;
        }
        item.cv.notify_one();
      }
    }

    {
      std::lock_guard<std::mutex> lk(stats_mu_);
      stats_.requests += static_cast<uint64_t>(n);
      stats_.batches += 1;
      stats_.max_seen = std::max(stats_.max_seen, static_cast<uint64_t>(n));
    }
  }
}
