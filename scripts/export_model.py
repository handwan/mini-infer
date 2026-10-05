#!/usr/bin/env python3
"""导出一个小 MLP 给 C++ 侧用（需要 ~/.venvs/ai：torch + onnxruntime）

用法：
    ~/.venvs/ai/bin/python scripts/export_model.py                    # 默认：隐藏层 8
    ~/.venvs/ai/bin/python scripts/export_model.py --hidden 512       # 大一点的（做批处理对照实验）
    ~/.venvs/ai/bin/python scripts/export_model.py --hidden 512 --out models/wide_mlp.onnx

产物：
    models/tiny_mlp.onnx —— 输入 [batch, 4] float32，输出 [batch, 2] float32
    batch 轴是**动态**的，服务端做批处理时直接能用
"""
import argparse
import logging
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent


class Mlp(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, hidden)
        self.fc2 = torch.nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    parser = argparse.ArgumentParser(description="导出 MLP 到 ONNX")
    parser.add_argument("--hidden", type=int, default=8, help="隐藏层宽度（默认 8）")
    parser.add_argument("--out", default="models/tiny_mlp.onnx", help="输出路径（相对项目根）")
    args = parser.parse_args()

    out = ROOT / args.out
    torch.manual_seed(0)  # 固定权重，保证可复现
    model = Mlp(args.hidden).eval()

    out.parent.mkdir(parents=True, exist_ok=True)

    # 导出器会为未安装的 torchvision 打无关警告（视觉算子），只过滤这一条
    logging.getLogger("torch.onnx._internal.exporter._registration").addFilter(
        lambda r: "torchvision is not installed" not in r.getMessage())

    torch.onnx.export(
        model,
        torch.randn(1, 4),
        str(out),
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"x": {0: "batch"}},
        opset_version=18,  # torch 的导出器要求 >= 18（低于它会自动转换并警告）
    )
    print(f"exported: {out}  ({out.stat().st_size} bytes, hidden={args.hidden})")

    # torch 的导出器偶尔留一个 0 字节的 .data 文件（权重都已内嵌），删掉
    data_file = out.with_name(out.name + ".data")
    if data_file.exists() and data_file.stat().st_size == 0:
        data_file.unlink()

    # 用 onnxruntime 自检：名字、形状、数值都对得上
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    print("inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

    x = torch.randn(3, 4)  # 故意用 batch=3 验证动态轴
    got = sess.run(None, {"input": x.numpy()})[0]
    want = model(x).detach().numpy()
    print(f"batch=3 -> shape {got.shape}  allclose: {bool((abs(got - want) < 1e-5).all())}")


if __name__ == "__main__":
    main()
