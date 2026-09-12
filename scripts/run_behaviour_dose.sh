#!/usr/bin/env bash
# Drive eval_behaviour_dose.py in slices of --stop-after prompts, resuming between slices, so the
# MPS graph cache (one compiled graph per new tensor shape, never freed) cannot grow past a slice.
# Usage: scripts/run_behaviour_dose.sh <model> <seed> [extra harness args...]
set -u
cd "$(dirname "$0")/.."
MODEL="$1"; SEED="$2"; shift 2
TAG="${MODEL//\//__}"; [ "$SEED" != "0" ] && TAG="${TAG}__seed${SEED}"
OUT="scripts/behaviour_dose_results__${TAG}.json"
SLICE="${SLICE:-30}"
while [ ! -f "$OUT" ]; do
  echo "=== slice start $(date '+%H:%M:%S') ==="
  uv run --with datasets python scripts/eval_behaviour_dose.py --models "$MODEL" --seed "$SEED" \
    --resume --stop-after "$SLICE" "$@"
  rc=$?
  if [ "$rc" -eq 3 ]; then fails=0; continue; fi
  if [ "$rc" -eq 2 ]; then                       # uv could not resolve deps (transient DNS/network)
    fails=$(( ${fails:-0} + 1 ))
    if [ "$fails" -le 5 ]; then echo "uv resolution failed (rc 2), retry $fails/5 in 60s"; sleep 60; continue; fi
  fi
  if [ "$rc" -ne 0 ]; then echo "harness exited with $rc; stopping"; exit "$rc"; fi
done
echo "=== done $(date '+%H:%M:%S') -> $OUT ==="
