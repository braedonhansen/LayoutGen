#!/bin/bash
# Mean (single-shot) gap, third lever: is it plain underfitting? step05v2s saw its corpus once (1 epoch) and
# reaches its own labels' column count on 42% of training cells greedily (run18). Same base, data
# (out/llm/data_v2s), recipe and seed, but 3 epochs -> step05v2s_e3.
# Trained axes unchanged: <= 5 inputs, <= 9 per side, stack <= 4, 1-2 stages, 1 output.
# Measured: greedy label-match on the 5,000 training cells of out/distill1 (in-sample for both models), then
# greedy and T=0.7 k=8 on all single-stage sets and the stage probes (run17), then k=32 (run12).
#   [EPOCHS=3 TAG=step05v2s_e3] bash cellgym/llm/run19_2026_11_26.sh        last progress line RUN19_DONE
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=v2
D=out/llm; P=$D/progress; E=${EPOCHS:-3}; N=${TAG:-step05v2s_e$E}; O=$D/$N
say() { echo "$* $(date)" >> $P; }
run() { local name=$1 log=$2; shift 2; "$@" > $log 2>&1; local rc=$?
        if [ $rc -eq 0 ]; then say "$name ok"; else say "$name FAILED rc=$rc"; fi; return $rc; }
say "run19 start $N epochs=$E"
[ -d $O/model ] || run "$N trained" $D/${N}_train.log python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B \
  --data $D/data_v2s/step_train.jsonl --val $D/data_v2s/step_val.jsonl \
  --out $O --epochs $E --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 --seed 1 || exit 1
G=out/distill1_greedy/$N
[ -f $G/samples.jsonl ] || run "$N greedy on training cells" $G.log python -m cellgym.llm.distill sample --model $O/model \
  --plans out/synth5/plans.jsonl out/synth2s/plans.jsonl --cells 5000 --k 1 --temperature 0 --seed 0 --out $G
python -m cellgym.llm.distill select --out $G --keep 1 > $G.select.txt 2>&1
TEMPS="0 0.7" bash cellgym/llm/run17_2026_11_25.sh $N=$O/model
bash cellgym/llm/run12_2026_11_24.sh $N=$O/model
echo "RUN19_DONE $(date)" >> $P
