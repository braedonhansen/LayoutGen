#!/bin/bash
# Local run (RTX 3090, WSL; env/LOCAL.md). HANDOFF step 2 without a pod, weights kept under out/llm:
#   SL step05v2 (seed 1, v2 state; the run10 recipe)
#   rl8 = SL + 60 RL steps, area reward, FIXED entropy coefficient (no adaptive controller)
#   rl9 = the same from the same SL model, plus per-step break shaping (--break-shape 1)
# rl8 against rl9 isolates the shaping; rl8 against run10's rl6 is fixed against adaptive entropy.
# Each model is sampled on every set (k=8) and every probe (k=4), then scored at --tracks area.
# Stages already done (model dir / samples file present) are skipped, so the script can be rerun.
#   bash cellgym/llm/run11_2026_11_24.sh        progress: out/llm/progress, last line RUN11_DONE
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=v2
D=out/llm; P=$D/progress; mkdir -p $D
J=${JOBS:-14}; H=${RL_STEPS:-60}; EC=${ENT_COEF:-0.05}
say() { echo "$* $(date)" >> $P; }
run() { local name=$1 log=$2; shift 2; "$@" > $log 2>&1; local rc=$?
        if [ $rc -eq 0 ]; then say "$name ok"; else say "$name FAILED rc=$rc"; fi; return $rc; }
say "run11 start state=$CELLGYM_STATE ent_coef=$EC steps=$H"

[ -f $D/data_v2/step_train.jsonl ] || run "data built" $D/build_v2.log python -m cellgym.llm.build --out $D/data_v2 --steps-per-plan 0

[ -d $D/step05v2/model ] || run "step05v2 trained" $D/step05v2_train.log python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B \
  --data $D/data_v2/step_train.jsonl --val $D/data_v2/step_val.jsonl \
  --out $D/step05v2 --epochs 1 --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 --seed 1

X=""; for s in i8d14 i10d16 i12d20 i16d24 stage2 stage3; do
  [ -f out/probe/$s/index.json ] && X="$X $s=out/probe/$s:all"; done

sample() {   # sample MODEL OUTDIR TAG, then score both files at --tracks area (CPU, in the background)
  local m=$1 o=$2 tag=$3
  [ -d $m ] || { say "$tag sampling skipped (no model $m)"; return 1; }
  [ -f $o/samples.jsonl ] || run "$tag sampled" $D/${tag}_infer.log python -m cellgym.llm.infer --model $m --mode step --k 8 --out $o/samples.jsonl
  if [ ! -f $o/probe_samples.jsonl ]; then
    for x in $X; do
      local s=${x%%=*}
      run "$tag probe $s sampled" $D/${tag}_probe_${s}_infer.log timeout ${PROBE_TIMEOUT:-2400} \
        python -m cellgym.llm.infer --model $m --mode step --k 4 --extra $x --out $o/probe_$s.jsonl \
        && cat $o/probe_$s.jsonl >> $o/probe_samples.tmp
    done
    [ -f $o/probe_samples.tmp ] && mv $o/probe_samples.tmp $o/probe_samples.jsonl
  fi
  ( run "$tag scored area" $D/${tag}_score_area.txt python -m cellgym.llm.score $o/samples.jsonl --jobs 6 --tracks area --tag area
    [ -f $o/probe_samples.jsonl ] && run "$tag probes scored area" $D/${tag}_probe_score_area.txt \
      python -m cellgym.llm.score $o/probe_samples.jsonl --jobs 6 --tracks area --tag area --extra $X ) &
}

rl() {   # rl NAME [extra flags]: 60 steps from the SL model
  local name=$1; shift
  [ -d $D/$name/last ] && return 0
  run "$name trained" $D/$name.log python -m cellgym.llm.rl --init $D/step05v2/model --out $D/$name \
    --steps $H --cells 32 --group 8 --lr 4e-6 --beta 0.02 --reward area \
    --ent-coef $EC --select both --keep-ckpts --eval-every 20 --eval-cells 96 --jobs 8 "$@"
}

sample $D/step05v2/model $D/step05v2 step05v2
if [ -d $D/step05v2/model ]; then
  rl rl8
  sample $D/rl8/best $D/rl8 rl8
  rl rl9 --break-shape 1
  sample $D/rl9/best $D/rl9 rl9
else
  say "RL skipped (no SL model)"
fi
wait
echo "RUN11_DONE $(date)" >> $P
