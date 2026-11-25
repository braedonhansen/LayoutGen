#!/bin/bash
# Mean (single-shot) gap, first lever: sampling temperature. Every llm run so far sampled at the infer default
# T = 0.7. Sample step05v2s (both seeds) greedy (T = 0, k = 1) and at T = 0.3 / 0.5 (k = 8) on the single-stage
# sets and the stage probes, and score area at the area-optimal track count. No training.
#   bash cellgym/llm/run17_2026_11_25.sh [TAG=MODELDIR ...]      last progress line RUN17_DONE
# Outputs: out/llm/temp/<tag>/t<T>.jsonl, st_t<T>.jsonl (stage probes), score_t<T>.txt, score_st_t<T>.txt
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=v2
D=out/llm; P=$D/progress
say() { echo "$* $(date)" >> $P; }
M=${*:-step05v2s=$D/step05v2s/model step05v2s_s2=$D/step05v2s_s2/model}
X="stage2=out/probe/stage2:all stage3=out/probe/stage3:all"
say "run17 start: $M"
for x in $M; do
  tag=${x%%=*}; m=${x#*=}; o=$D/temp/$tag; mkdir -p $o
  for t in ${TEMPS:-0 0.3 0.5}; do
    k=8; [ $t = 0 ] && k=1
    [ -f $o/t$t.jsonl ] || python -m cellgym.llm.infer --model $m --mode step --k $k --temperature $t --out $o/t$t.jsonl > $o/t${t}_infer.log 2>&1 \
      || { say "$tag T=$t sampling FAILED"; continue; }
    [ -f $o/st_t$t.jsonl ] || python -m cellgym.llm.infer --model $m --mode step --k $k --temperature $t --extra $X --out $o/st_t$t.jsonl > $o/st_t${t}_infer.log 2>&1 \
      || { say "$tag T=$t stage sampling FAILED"; continue; }
    say "$tag T=$t sampled"
    ( python -m cellgym.llm.score $o/t$t.jsonl --jobs 3 --tracks area --k $k --tag area > $o/score_t$t.txt 2>&1
      python -m cellgym.llm.score $o/st_t$t.jsonl --jobs 2 --tracks area --k $k --tag area --extra $X > $o/score_st_t$t.txt 2>&1
      say "$tag T=$t scored" ) &
  done
done
wait
echo "RUN17_DONE $(date)" >> $P
