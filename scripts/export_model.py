#!/usr/bin/env python3
"""导出一个小模型给 C++ 侧用（需要 ~/.venvs/ai：torch + onnxruntime）

用法：
    ~/.venvs/ai/bin/python scripts/export_model.py

产物：
    models/tiny_mlp.onnx —— 输入 [batch, 4] float32，输出 [batch, 2] float32
    batch 轴是**动态**的，服务端做批处理时直接能用
"""
import pathlib

import torch

OUT = pathlib.Path(__file__).resolve().parent.parent / "models" / "tiny_mlp.onnx"


class TinyMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    torch.manual_seed(0)  # 固定权重，保证可复现
    model = TinyMlp().eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 4),
        str(OUT),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=18,  # torch 的导出器要求 >= 18（低于它会自动转换并警告）
    )
    print(f"exported: {OUT}  ({OUT.stat().st_size} bytes)")

    # torch 的导出器偶尔留一个 0 字节的 .data 文件（权重都已内嵌），删掉
    data_file = OUT.with_name(OUT.name + ".data")
    if data_file.exists() and data_file.stat().st_size == 0:
        data_file.unlink()

    # 用 onnxruntime 自检：名字、形状、数值都对得上
    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    print("inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

    x = torch.randn(3, 4)  # 故意用 batch=3 验证动态轴
    got = sess.run(None, {"input": x.numpy()})[0]
    want = model(x).detach().numpy()
    print(f"batch=3 -> shape {got.shape}  allclose: {bool((abs(got - want) < 1e-5).all())}")


if __name__ == "__main__":
    main()
