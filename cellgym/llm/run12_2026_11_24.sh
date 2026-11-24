#!/bin/bash
# Larger k with router-free selection (local, after run11): sample k=32 per cell from the SL model and
# the given RL models, then score the first 8 / 16 / 32 samples routing only the fewest-column placements.
#   bash cellgym/llm/run12_2026_11_24.sh [MODEL_TAG=DIR ...]     default: step05v2 and rl8/rl9 best
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=v2
D=out/llm; P=$D/progress
say() { echo "$* $(date)" >> $P; }
M=${*:-step05v2=$D/step05v2/model rl8=$D/rl8/best rl9=$D/rl9/best}
say "run12 start: $M"
for x in $M; do
  tag=${x%%=*}; m=${x#*=}; o=$D/k32/$tag; mkdir -p $o
  [ -d $m ] || { say "$tag skipped (no model $m)"; continue; }
  [ -f $o/samples.jsonl ] || python -m cellgym.llm.infer --model $m --mode step --k 32 --out $o/samples.jsonl > $o/infer.log 2>&1 \
    || { say "$tag k32 sampling FAILED"; continue; }
  say "$tag k32 sampled"
  ( for k in 8 16 32; do
      python -m cellgym.llm.score $o/samples.jsonl --jobs 5 --tracks area --select fewcol --k $k --tag fewcol_k$k > $o/score_fewcol_k$k.txt 2>&1
    done; say "$tag k32 scored" ) &
done
wait
echo "RUN12_DONE $(date)" >> $P
