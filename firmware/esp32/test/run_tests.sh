#!/usr/bin/env bash
# Compile the firmware against the Arduino stand-in and run each scenario in its own process.
set -euo pipefail
cd "$(dirname "$0")"
BIN="${TMPDIR:-/tmp}/securebot_fw_test"
g++ -std=c++17 -Wall -Wextra -Wno-unused-variable -x c++ -Ishim firmware_test.cpp -o "$BIN"
pass=0; fail=0
for t in $("$BIN"); do
  if "$BIN" "$t" >/tmp/fwtest.out 2>&1; then echo "PASS $t"; pass=$((pass+1))
  else echo "FAIL $t"; cat /tmp/fwtest.out; fail=$((fail+1)); fi
done
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
