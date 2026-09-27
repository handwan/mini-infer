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

echo "deps ready"
