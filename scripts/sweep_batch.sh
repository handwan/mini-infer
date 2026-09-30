#!/usr/bin/env bash
# 批处理参数扫描：不同 (window, max_batch) 下的吞吐/延迟，每档多轮取中位
#   用法：scripts/sweep_batch.sh [model] [conns] [rounds] [duration]
#   例：  scripts/sweep_batch.sh                                  # 默认 tiny_mlp / c=10 / 3 轮 / 3s
#         scripts/sweep_batch.sh models/wide_mlp.onnx 50 3 3s
set -u
cd "$(dirname "$0")/.."

MODEL="${1:-models/tiny_mlp.onnx}"
CONNS="${2:-10}"
ROUNDS="${3:-3}"
DUR="${4:-3s}"
PORT="${PORT:-8080}"

CONFIGS_STR="${BATCH_CONFIGS:-0 1;2000 8;5000 16;10000 32}"  # "窗口us 最大批"，分号分隔
IFS=';' read -ra CONFIGS <<<"$CONFIGS_STR"

command -v wrk >/dev/null 2>&1 || { echo "wrk not found (sudo apt install wrk)"; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }

LUA=$(mktemp)
TMPD=$(mktemp -d)
trap 'rm -rf "$LUA" "$TMPD"' EXIT

cat > "$LUA" <<'EOF'
wrk.method = "POST"
wrk.body   = "1,2,3,4"
wrk.headers["Content-Type"] = "text/plain"
EOF

median() { sort -n | awk '{a[NR]=$1} END {print (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}'; }
us() { awk '{v=$NF; if (v ~ /ms$/) {sub(/ms$/,"",v); v*=1000} else {sub(/us$/,"",v)}; print v}'; }

printf 'model=%s  conns=%s  rounds=%s  duration=%s\n' "$MODEL" "$CONNS" "$ROUNDS" "$DUR"
printf '%-14s %-10s %-12s %-11s %-11s\n' "window/max" "avg-batch" "QPS(median)" "P50us(med)" "P99us(med)"

for cfg in "${CONFIGS[@]}"; do
  read -r win max <<<"$cfg"
  : >"$TMPD/qps"
  : >"$TMPD/p50"
  : >"$TMPD/p99"
  for _ in $(seq 1 "$ROUNDS"); do
    MODEL_PATH="$MODEL" BATCH_WINDOW_US="$win" BATCH_MAX="$max" ./build/mini-infer "$PORT" >"$TMPD/run.log" 2>&1 &
    srv=$!
    sleep 0.8
    out=$(wrk -t2 -c"$CONNS" -d"$DUR" --latency -s "$LUA" "http://localhost:$PORT/predict")
    printf '%s\n' "$(printf '%s' "$out" | awk '/Requests\/sec/{print $2}')" >>"$TMPD/qps"
    printf '%s\n' "$(printf '%s' "$out" | awk '/^ *50%/{print $2}' | us)" >>"$TMPD/p50"
    printf '%s\n' "$(printf '%s' "$out" | awk '/^ *99%/{print $2}' | us)" >>"$TMPD/p99"
    kill -INT "$srv" 2>/dev/null
    wait "$srv" 2>/dev/null
  done
  avg=$(grep -o 'avg=[0-9.]*' "$TMPD/run.log" | tail -1 | cut -d= -f2)
  printf '%-14s %-10s %-12s %-11s %-11s\n' "$win/$max" "${avg:-?}" \
    "$(median <"$TMPD/qps")" "$(median <"$TMPD/p50")" "$(median <"$TMPD/p99")"
done
