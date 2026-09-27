#!/usr/bin/env bash
# 压测脚本：用法 scripts/bench.sh [url] [duration]
set -u
URL="${1:-http://localhost:8080/echo}"
DUR="${2:-5s}"

if ! command -v wrk >/dev/null; then
  echo "wrk not found: install it first (e.g. sudo apt install wrk)"
  exit 1
fi

echo "wrk benchmark: $URL  duration $DUR"
echo "----------------------------------------"
wrk -t2 -c10 -d"$DUR" --latency "$URL"
