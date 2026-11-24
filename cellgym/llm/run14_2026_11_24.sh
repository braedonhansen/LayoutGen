#!/bin/bash
# Composition (1 -> 2-3 stages), local, no training: the SL model step05v2 sampled at k=32 on the stage
# probes under three spellings of the stage nets (x1.. as generated; letters; n31..), see renameprobe.py.
# Scored with every sample routed at k=4 and k=32, and fewest-column selection at k=8/16/32.
#   [VARIANTS=base] bash cellgym/llm/run14_2026_11_24.sh [MODEL_DIR] [TAG]      last progress line RUN14_DONE
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=v2
D=out/llm; P=$D/progress; M=${1:-$D/step05v2/model}; T=${2:-step05v2}; O=$D/stage/$T; mkdir -p $O
say() { echo "$* $(date)" >> $P; }
say "run14 start model=$M"
for s in stage2 stage3; do for v in letter net; do
  [ -d out/probe/${s}_$v ] || python -m cellgym.llm.renameprobe --src out/probe/$s --style $v --out out/probe/${s}_$v >> $O/rename.log 2>&1
done; done
for v in ${VARIANTS:-base letter net}; do
  X=""; for s in stage2 stage3; do n=$s; [ $v = base ] || n=${s}_$v; X="$X $n=out/probe/$n:all"; done
  f=$O/$v.jsonl
  [ -f $f ] || python -m cellgym.llm.infer --model $M --mode step --k 32 --extra $X --out $f > $O/${v}_infer.log 2>&1 \
    || { say "run14 $v sampling FAILED"; continue; }
  say "run14 $v sampled"
  ( for k in 4 32; do python -m cellgym.llm.score $f --jobs 5 --tracks area --k $k --tag all_k$k --extra $X > $O/${v}_all_k$k.txt 2>&1; done
    for k in 8 16 32; do python -m cellgym.llm.score $f --jobs 5 --tracks area --select fewcol --k $k --tag fewcol_k$k --extra $X > $O/${v}_fewcol_k$k.txt 2>&1; done
    say "run14 $v scored" ) &
done
wait
echo "RUN14_DONE $(date)" >> $P
