#!/usr/bin/env bash
# Compile each firmware sketch against the Arduino stand-in and run every scenario in its own process.
set -euo pipefail
cd "$(dirname "$0")"
pass=0; fail=0
run_suite() {                      # $1 = test source, $2 = label
  local bin="${TMPDIR:-/tmp}/securebot_$2_test"
  g++ -std=c++17 -Wall -Wextra -Wno-unused-variable -x c++ -Ishim "$1" -o "$bin"
  for t in $("$bin"); do
    if "$bin" "$t" >/tmp/fwtest.out 2>&1; then echo "PASS $2/$t"; pass=$((pass+1))
    else echo "FAIL $2/$t"; cat /tmp/fwtest.out; fail=$((fail+1)); fi
  done
}
run_suite firmware_test.cpp firmware
run_suite diag_test.cpp diag
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
