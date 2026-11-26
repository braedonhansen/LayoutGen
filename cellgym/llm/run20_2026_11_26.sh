#!/bin/bash
# Mean (single-shot) gap, fourth lever: a state line for the lookahead the policy lacks. cellgym.llm.regret
# shows where greedy placements on training cells lose columns (see bench/llm_runs.md, run20 section): the
# largest single share is the first column. State v3 adds `odd: n=.. p=..`, the nets with an odd number of
# unplaced device ends per row (a parity count, not a search). Same base, corpora, recipe and seed as
# step05v2s (1 epoch) -> step05v3s. Trained axes unchanged.
#   bash cellgym/llm/run20_2026_11_26.sh        last progress line RUN20_DONE
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=v3
D=out/llm; P=$D/progress; N=${TAG:-step05v3s}; O=$D/$N
say() { echo "$* $(date)" >> $P; }
run() { local name=$1 log=$2; shift 2; "$@" > $log 2>&1; local rc=$?
        if [ $rc -eq 0 ]; then say "$name ok"; else say "$name FAILED rc=$rc"; fi; return $rc; }
say "run20 start $N state=$CELLGYM_STATE"
[ -f $D/data_v3s/step_train.jsonl ] || run "data_v3s built" $D/build_v3s.log python -m cellgym.llm.build \
  --plans out/synth5/plans.jsonl out/synth2s/plans.jsonl --out $D/data_v3s --steps-per-plan 0 || exit 1
[ -n "$BUILD_ONLY" ] && exit 0
[ -d $O/model ] || run "$N trained" $D/${N}_train.log python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B \
  --data $D/data_v3s/step_train.jsonl --val $D/data_v3s/step_val.jsonl \
  --out $O --epochs ${EPOCHS:-1} --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 --seed 1 || exit 1
G=out/distill1_greedy/$N
[ -f $G/samples.jsonl ] || run "$N greedy on training cells" $G.log python -m cellgym.llm.distill sample --model $O/model \
  --plans out/synth5/plans.jsonl out/synth2s/plans.jsonl --cells 5000 --k 1 --temperature 0 --seed 0 --out $G
python -m cellgym.llm.distill select --out $G --keep 1 > $G.select.txt 2>&1
TEMPS="${EVAL_TEMPS:-0 0.7}" bash cellgym/llm/run17_2026_11_25.sh $N=$O/model
[ -n "$SKIP_K32" ] || bash cellgym/llm/run12_2026_11_24.sh $N=$O/model
echo "RUN20_DONE $(date)" >> $P
