#!/usr/bin/env bash
# Run the W_p/W_n sweeps recorded in sim/results/. Run inside WSL from the repo root.
set -e
mkdir -p sim/results
for pf in pfet_01v8_hvt pfet_01v8; do
  for wn in 0.42 0.65; do
    python3 sim/inv_ratio_sweep.py --pfet "$pf" --wn "$wn" --ratios 1.0:8.0:0.25 \
      --csv "sim/results/${pf}_wn${wn}_tt.csv"
    echo
  done
done
