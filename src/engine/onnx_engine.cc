#include "engine/onnx_engine.h"

#include <array>
#include <stdexcept>

namespace {

// 打开模型：线程数暂定 4
Ort::Session OpenSession(Ort::Env& env, const std::string& model_path) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(4);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return {env, model_path.c_str(), options};
}

}  // namespace

OnnxEngine::OnnxEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "mini-infer"),
      session_(OpenSession(env_, model_path)) {
  // 输入/输出的名字要记下来：Run 的时候按名字传
  Ort::AllocatorWithDefaultOptions allocator;
  input_name_ = session_.GetInputNameAllocated(0, allocator).get();
  output_name_ = session_.GetOutputNameAllocated(0, allocator).get();

  // 形状是 [batch, dim]：第 0 维是动态 batch，最后一维是特征维
  input_dim_ = session_.GetInputTypeInfo(0)
                   .GetTensorTypeAndShapeInfo()
                   .GetShape()
                   .back();
  output_dim_ = session_.GetOutputTypeInfo(0)
                    .GetTensorTypeAndShapeInfo()
                    .GetShape()
                    .back();
}

std::vector<float> OnnxEngine::Run(const std::vector<float>& input) {
  if (input_dim_ <= 0 || input.size() % static_cast<size_t>(input_dim_) != 0) {
    throw std::invalid_argument("input size must be a multiple of input_dim");
  }
  const int64_t batch = static_cast<int64_t>(input.size()) / input_dim_;
  const std::array<int64_t, 2> input_shape{batch, input_dim_};

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float*>(input.data()), input.size(),
      input_shape.data(), input_shape.size());

  const std::array<const char*, 1> input_names{input_name_.c_str()};
  const std::array<const char*, 1> output_names{output_name_.c_str()};
  auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names.data(),
                              &input_tensor, 1, output_names.data(), 1);

  const float* data = outputs[0].GetTensorMutableData<float>();
  const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  return {data, data + shape[0] * shape[1]};
}
