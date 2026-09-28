#!/usr/bin/env bash
# 拉取第三方依赖
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

HTTPLIB_VER="v0.58.0"
HTTPLIB_SHA256="aa14e7e7bd2703694e0a6b6855af3b8c406102ab1fc56ac905fe33619b31faa5"
if [ ! -f third_party/httplib.h ]; then
  echo "downloading cpp-httplib $HTTPLIB_VER ..."
  URL="https://raw.githubusercontent.com/yhirose/cpp-httplib/$HTTPLIB_VER/httplib.h"
  curl -fsSL --max-time 60  -o third_party/httplib.h "$URL" \
    || curl -fsSL --max-time 120 -o third_party/httplib.h "https://gh-proxy.com/$URL"
fi
if ! echo "$HTTPLIB_SHA256  third_party/httplib.h" | sha256sum -c - >/dev/null 2>&1; then
  echo "checksum mismatch: file does not match $HTTPLIB_VER"
  exit 1
fi
echo "  cpp-httplib $HTTPLIB_VER (sha256 ok)"

# ONNX Runtime（CPU 预编译包；GPU 版是另一个包，见 third_party/README.md）
ORT_VER="1.30.0"
ORT_SHA256="a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd"
ORT_TGZ="third_party/onnxruntime-${ORT_VER}-linux-x64.tgz"
ORT_DIR="third_party/onnxruntime"
if [ ! -f "$ORT_TGZ" ]; then
  echo "downloading onnxruntime $ORT_VER ..."
  URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz"
  curl -fsSL --max-time 30 -o "$ORT_TGZ" "$URL" \
    || curl -fsSL --max-time 300 -o "$ORT_TGZ" "https://gh-proxy.com/$URL"
fi
if ! echo "$ORT_SHA256  $ORT_TGZ" | sha256sum -c - >/dev/null 2>&1; then
  echo "checksum mismatch: file does not match onnxruntime $ORT_VER"
  exit 1
fi
if [ ! -d "$ORT_DIR/include" ]; then
  echo "extracting onnxruntime $ORT_VER ..."
  mkdir -p "$ORT_DIR"
  tar -xzf "$ORT_TGZ" -C "$ORT_DIR" --strip-components=1
fi
echo "  onnxruntime $ORT_VER (sha256 ok)"

echo "deps ready"
