// KV cache 实验：逐 token 解码时，"每步重算全部 K/V" vs "K/V 增量缓存"。
//
// 数据随机生成（固定种子，可复现）：x_t 是第 t 个 token 的输入向量（kDim
// 维），Wk / Wv 是投影矩阵（kDim×kDim）。两种实现只有 K/V 的获取方式不同。
// 结论：no-cache 的投影是 O(N²)；cache 把投影降到 O(1)/步，但注意力每步
// 仍要扫过历史缓存（O(t)/步）——总量依然是 O(N²)，只是常数小几十倍。这份
// 注意力平方开销也是真实长上下文仍然贵的根因。
//
// 性能实验请用 Release 构建（本项目默认即是）。
// 用法：./build/kv_cache_demo [N ...]   默认 128 256 512

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kDim = 64;

// 可复现的伪随机数（xorshift32），值域约 [-0.5, 0.5)
class Rng {
 public:
  explicit Rng(uint32_t seed) : state_(seed) {}
  float Next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ % 1000) / 1000.0F - 0.5F;
  }

 private:
  uint32_t state_;
};

// 模型参数 + 输入（打包传参，避免一长串同类型 vector）
struct Params {
  std::vector<float> wk;
  std::vector<float> wv;
  std::vector<float> xs;  // max_n × kDim
  std::vector<float> qs;  // max_n × kDim
};

// K/V 缓存：count 行 × kDim，行主序
struct KvCache {
  std::vector<float> k;
  std::vector<float> v;
};

// 单 token 投影：out = W·x（W 为 kDim×kDim 行主序，out 至少 kDim 个 float）
void Project(const std::vector<float>& w, const float* x, float* out) {
  for (int i = 0; i < kDim; ++i) {
    const float* row = &w[static_cast<size_t>(i) * kDim];
    float sum = 0.0F;
    for (int j = 0; j < kDim; ++j) {
      sum += row[j] * x[j];
    }
    out[i] = sum;
  }
}

// 单头注意力：q 对缓存（前 count 行）→ out（kDim 个 float）
//    scores[i] = q·kv.k[i]，softmax 后 out = Σ scores[i]·kv.v[i]
void Attend(const float* q, const KvCache& kv, int count, float* out) {
  std::vector<float> scores(static_cast<size_t>(count));
  float max_score = 0.0F;
  for (int i = 0; i < count; ++i) {
    const float* krow = &kv.k[static_cast<size_t>(i) * kDim];
    float dot = 0.0F;
    for (int j = 0; j < kDim; ++j) {
      dot += q[j] * krow[j];
    }
    scores[i] = dot;
    if (i == 0 || dot > max_score) {
      max_score = dot;
    }
  }

  float total = 0.0F;
  for (int i = 0; i < count; ++i) {
    scores[i] = std::exp(scores[i] - max_score);
    total += scores[i];
  }

  for (int j = 0; j < kDim; ++j) {
    out[j] = 0.0F;
  }
  for (int i = 0; i < count; ++i) {
    const float weight = scores[i] / total;
    const float* vrow = &kv.v[static_cast<size_t>(i) * kDim];
    for (int j = 0; j < kDim; ++j) {
      out[j] += weight * vrow[j];
    }
  }
}

// no-cache：每步把历史整段重新投影
double RunNoCache(const Params& p, int n) {
  KvCache kv;
  std::vector<float> out(kDim);
  const auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < n; ++t) {
    const int count = t + 1;
    kv.k.resize(static_cast<size_t>(count) * kDim);
    kv.v.resize(static_cast<size_t>(count) * kDim);
    for (int r = 0; r < count; ++r) {
      Project(p.wk, &p.xs[static_cast<size_t>(r) * kDim],
              &kv.k[static_cast<size_t>(r) * kDim]);
      Project(p.wv, &p.xs[static_cast<size_t>(r) * kDim],
              &kv.v[static_cast<size_t>(r) * kDim]);
    }
    Attend(&p.qs[static_cast<size_t>(t) * kDim], kv, count, out.data());
  }
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

// cache：K/V 只增不重算——每个 t 只投影当前 token 追加到缓存尾部，
// Attend 读 [0, t+1)。
double RunCache(const Params& p, int n) {
  KvCache kv;
  kv.k.resize(static_cast<size_t>(n) * kDim);
  kv.v.resize(static_cast<size_t>(n) * kDim);
  std::vector<float> out(kDim);
  const auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < n; ++t) {
    Project(p.wk, &p.xs[static_cast<size_t>(t) * kDim],
            &kv.k[static_cast<size_t>(t) * kDim]);
    Project(p.wv, &p.xs[static_cast<size_t>(t) * kDim],
            &kv.v[static_cast<size_t>(t) * kDim]);
    Attend(&p.qs[static_cast<size_t>(t) * kDim], kv, t + 1, out.data());
  }
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> ns;
  for (int i = 1; i < argc; ++i) {
    ns.push_back(std::atoi(argv[i]));
  }
  if (ns.empty()) {
    ns = {128, 256, 512};
  }
  int max_n = 0;
  for (int n : ns) {
    max_n = std::max(max_n, n);
  }

  Rng rng(42);
  Params p;
  p.wk.resize(static_cast<size_t>(kDim) * kDim);
  p.wv.resize(static_cast<size_t>(kDim) * kDim);
  p.xs.resize(static_cast<size_t>(max_n) * kDim);
  p.qs.resize(static_cast<size_t>(max_n) * kDim);
  for (float& w : p.wk) {
    w = rng.Next();
  }
  for (float& w : p.wv) {
    w = rng.Next();
  }
  for (float& x : p.xs) {
    x = rng.Next();
  }
  for (float& q : p.qs) {
    q = rng.Next();
  }

  printf("%6s %14s %12s %8s %10s\n", "N", "no-cache(ms)", "cache(ms)", "ratio",
         "cache(KB)");
  for (int n : ns) {
    const double no_cache_ms = RunNoCache(p, n);
    const double cache_ms = RunCache(p, n);
    const double ratio = cache_ms > 0 ? no_cache_ms / cache_ms : 0.0;
    const double cache_kb = 2.0 * n * kDim * 4 / 1024.0;
    printf("%6d %14.2f %12.2f %8.1f %10.1f\n", n, no_cache_ms, cache_ms, ratio,
           cache_kb);
  }
  return 0;
}
