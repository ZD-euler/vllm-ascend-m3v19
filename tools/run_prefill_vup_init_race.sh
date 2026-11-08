#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON:-python3}"
device="${1:-0}"
result_dir="${2:-vup-init-check-$(date +%Y%m%d-%H%M%S)}"
command -v "$python_bin" >/dev/null
command -v timeout >/dev/null
mkdir -p -- "$result_dir"
result_dir="$(cd -- "$result_dir" && pwd)"

export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
overall=0
for mode in cold sync hold warm; do
    status=0
    timeout --kill-after=5s 90s "$python_bin" tools/prefill_vup_init_race.py \
        --device "$device" --mode "$mode" --output "$result_dir/$mode.json" \
        >"$result_dir/$mode.log" 2>&1 || status=$?
    printf '\n%s: exit=%s\n' "$mode" "$status"
    if ! grep -E '^\[VUP_INIT\].*"event": "result"' "$result_dir/$mode.log"; then
        tail -n 20 "$result_dir/$mode.log"
        overall=2
    fi
    if [[ "$status" != 0 && !( "$mode" == cold && "$status" == 1 ) ]]; then
        overall=2
    fi
done
printf '\nresult_dir: %s\n' "$result_dir"
exit "$overall"
