#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

shopt -s nullglob globstar extglob
FILES=(src/**/*.@(cc|cpp|cxx|h|hpp|hxx) tests/**/*.@(cc|cpp|cxx|h|hpp|hxx))
if [ ${#FILES[@]} -eq 0 ]; then
  echo "no source files found (src/, tests/)"
  exit 1
fi

FAIL=0

echo "== 1. clang-format =="
if ! clang-format --dry-run --Werror "${FILES[@]}" 2>&1 | sed 's/^/  /'; then
  echo "  -> fix: clang-format -i ${FILES[*]}"
  FAIL=1
fi

echo
echo "== 2. clang-tidy =="
if [ ! -f build/compile_commands.json ]; then
  echo "  -> run first: cmake -B build && cmake --build build -j"
  exit 1
fi
for f in "${FILES[@]}"; do
  case "$f" in *.@(cc|cpp|cxx)) ;; *) continue ;; esac
  OUT=$(clang-tidy -p build "$f" 2>&1 \
    | grep -vE "^[0-9]+ warnings? generated|^Suppressed|^Use -header-filter")
  if [ -n "$OUT" ]; then
    echo "$OUT" | sed 's/^/  /'
  fi
  if echo "$OUT" | grep -qE ': (warning|error):'; then
    FAIL=1
  fi
done

echo
if [ "$FAIL" -eq 0 ]; then
  echo "check passed"
else
  echo "check failed"
fi
exit $FAIL
