#!/bin/bash
# Large probes (8/10/12/16 inputs at 14/16/20/24 devices per side) at k = 32. They have no Euler reference
# (the Euler search does not finish there), so they were only ever sampled at k = 4 and reported as pass
# rates. Here: k = 32, fewest-column selection routed at the area-optimal track count (pass rates), and the
# column count against the search-free lower bound (cellgym.llm.bound).
#   [CELLGYM_STATE=v3] bash cellgym/llm/run21_2026_11_26.sh TAG=MODELDIR ...     last progress line RUN21_DONE
# Outputs: out/llm/probe32/<tag>/samples.jsonl, score_fewcol_k{8,32}.txt, bound_k{4,8,32}.txt
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=${CELLGYM_STATE:-v2}
D=out/llm; P=$D/progress
say() { echo "$* $(date)" >> $P; }
X=""; for s in i8d14 i10d16 i12d20 i16d24; do X="$X $s=out/probe/$s:all"; done
say "run21 start: $* state=$CELLGYM_STATE"
for x in "$@"; do
  tag=${x%%=*}; m=${x#*=}; o=$D/probe32/$tag; mkdir -p $o
  [ -f $o/samples.jsonl ] || python -m cellgym.llm.infer --model $m --mode step --k 32 --extra $X --out $o/samples.jsonl > $o/infer.log 2>&1 \
    || { say "$tag probes k32 sampling FAILED"; continue; }
  say "$tag probes k32 sampled"
  for k in 4 8 32; do python -m cellgym.llm.bound $o/samples.jsonl --k $k --extra $X --out $o/bound_k$k.json > $o/bound_k$k.txt 2>&1; done
  ( for k in 8 32; do
      python -m cellgym.llm.score $o/samples.jsonl --jobs 4 --tracks area --select fewcol --k $k --tag fewcol_k$k --extra $X > $o/score_fewcol_k$k.txt 2>&1
    done; say "$tag probes k32 scored" ) &
done
wait
echo "RUN21_DONE $(date)" >> $P
