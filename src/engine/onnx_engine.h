#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <string>
#include <vector>

// ONNX Runtime 的 RAII 封装：构造 = 加载模型，Run() = 跑一次推理
class OnnxEngine {
 public:
  explicit OnnxEngine(const std::string& model_path);

  // input 是 batch × input_dim 的扁平数据；返回 batch × output_dim 的扁平数据
  // 注：ORT 的 Session::Run 不是 const 成员，所以这里也不能标 const
  std::vector<float> Run(const std::vector<float>& input);

  [[nodiscard]] int64_t input_dim() const { return input_dim_; }
  [[nodiscard]] int64_t output_dim() const { return output_dim_; }

 private:
  Ort::Env env_;  // 先于 session_ 构造（session 要用它）
  Ort::Session session_;
  std::string input_name_;
  std::string output_name_;
  int64_t input_dim_ = 0;
  int64_t output_dim_ = 0;
};
