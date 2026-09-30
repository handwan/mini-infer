#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// 动态批处理：把多个请求攒成一批，一次推理
//
//   请求 → Submit（阻塞等自己的结果）
//          ↓ 进队列
//   后台线程：等 window 或凑够 max_batch → 一次 Run → 分发结果
//
// 取舍：等待窗口越大 → 批越大（吞吐高），但每条请求的延迟也越大。
class DynamicBatcher {
   public:
    // 参数打包（不然 3 个相邻 int64_t 很容易传反）
    struct Config {
        int64_t input_dim = 0;                   // 每条样本的输入长度
        int64_t output_dim = 0;                  // 每条样本的输出长度
        int64_t max_batch = 1;                   // 一批最多几条
        std::chrono::microseconds window{1000};  // 凑批等待窗口
    };

    // 跑一批：输入是 batch × input_dim 的扁平数据，返回 batch × output_dim
    using RunFn = std::function<std::vector<float>(const std::vector<float>&,
                                                   int64_t batch)>;

    DynamicBatcher(const Config& config, RunFn run);
    ~DynamicBatcher();

    DynamicBatcher(const DynamicBatcher&) = delete;
    DynamicBatcher& operator=(const DynamicBatcher&) = delete;

    // 提交一条样本（长度必须 = input_dim），阻塞直到本批算完
    std::vector<float> Submit(const std::vector<float>& sample);

    // 统计（给压测对比用）
    struct Stats {
        uint64_t requests = 0;  // 总请求数
        uint64_t batches = 0;   // 总批数
        uint64_t max_seen = 0;  // 见过的最大批
    };
    Stats stats() const;

   private:
    struct Pending {  // 一条在等结果的请求
        std::vector<float> input;
        std::vector<float> output;
        std::exception_ptr error;

        std::mutex mu;
        std::condition_variable cv;
        bool done = false;
    };

    void WorkerLoop();

    const int64_t input_dim_;
    const int64_t output_dim_;
    const int64_t max_batch_;
    const std::chrono::microseconds window_;
    RunFn run_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Pending>> queue_;  // 等待凑批的请求
    bool stop_ = false;

    mutable std::mutex stats_mu_;
    Stats stats_;

    std::thread worker_;  // 放最后：它一构造就开始跑，前面成员必须已就绪
};
