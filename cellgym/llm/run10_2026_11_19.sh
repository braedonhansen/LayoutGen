#!/bin/bash
# Channel-aware step state (v2: an "open:" line = nets crossing the frontier) + RL on area that keeps
# sampling diversity (entropy bonus with an adaptive coefficient holding ENT_TARGET of the starting
# entropy; checkpoint selected on (mean area + best-of-k area) / 2).
#   SL step05v2 (seed 1, as step05s1 but on the v2 state) -> RL rl6 (first half) -> RL rl7 (second
#   half, continued from rl6/last with rl6's entropy state).
# NO scoring here (done locally afterwards from the samples). After each of the three sampling stages
# a cumulative results bundle without weights is written, so a pod that dies later loses nothing
# already sampled:  out/llm/results_run10_{a,b,c}.tgz  (progress lines STAGE_A_DONE, STAGE_B_DONE,
# RUN10_DONE; the last line of out/llm/progress is always RUN10_DONE).
# A failed stage is logged and later stages that do not need it still run; without an SL model the RL
# and its sampling are skipped.
# Needs on the pod: cellgym/, bench/{splits_v0,euler_widths_v0}.json, out/synth5/plans.jsonl (or
# .xz/.gz, unpacked here), out/cells, out/plans, the eval set dirs (cells, plans, index.json),
# out/probe, out/arearef (shipped; step b is then a no-op). 9 vCPUs, one A40.
cd /workspace/lg
export HF_HOME=/root/hf PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CELLGYM_STATE=v2      # every stage (build, infer, rl) takes the state text from cellgym.llm.fmt
D=out/llm; P=$D/progress; mkdir -p $D
J=${JOBS:-9}
H=${RL_HALF:-60}
say() { echo "$* $(date)" >> $P; }
run() {   # run NAME LOGFILE CMD...: run a stage, record its exit code
  local name=$1 log=$2; shift 2
  "$@" > $log 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then say "$name ok"; else say "$name FAILED rc=$rc"; fi
  return $rc
}
bundle() {   # bundle LETTER: everything so far except weights (logs, progress, samples, RL logs)
  local f=$D/results_run10_$1.tgz
  tar czf $f $D/*.log $D/progress $D/data_v2/manifest.json out/arearef/*.json \
    $D/step05v2/*.jsonl $D/rl6/*.jsonl $D/rl6/ent_state.json $D/rl7/*.jsonl $D/rl7/ent_state.json 2>/dev/null
  say "bundle $1 $(ls -la $f 2>/dev/null | awk '{print $5}') bytes"
}
say "start state=$CELLGYM_STATE"

# a. data with the v2 state
S=out/synth5/plans.jsonl
[ -f $S ] || { [ -f $S.xz ] && xz -dk $S.xz; }
[ -f $S ] || { [ -f $S.gz ] && gunzip -k $S.gz; }
run "data built" $D/build_v2.log python -m cellgym.llm.build --out $D/data_v2 --steps-per-plan 0

# b. Euler area references (shipped: a no-op), in the background while SL trains
(python -m cellgym.llm.arearef --all --jobs ${AREF_JOBS:-6} > $D/arearef.log 2>&1
 echo "arearef rc=$? $(date)" >> $P) &
AREF=$!

# c. SL on the v2 state, hyperparameters of step05s1 (seed 1)
run "step05v2 trained" $D/step05v2_train.log python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B \
  --data $D/data_v2/step_train.jsonl --val $D/data_v2/step_val.jsonl \
  --out $D/step05v2 --epochs 1 --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 --seed 1

# probe sets present (shipped, not built here)
X=""; for s in i8d14 i10d16 i12d20 i16d24 stage2 stage3; do
  [ -f out/probe/$s/index.json ] && X="$X $s=out/probe/$s:all"; done
say "probes:${X:- none}"

sample() {   # sample MODEL OUTDIR TAG: all sets k=8, then each probe set k=4 (one call per set)
  local m=$1 o=$2 tag=$3
  [ -d $m ] || { say "$tag sampling skipped (no model $m)"; return 1; }
  run "$tag sampled" $D/${tag}_infer.log python -m cellgym.llm.infer --model $m --mode step --k 8 --out $o/samples.jsonl
  rm -f $o/probe_samples.jsonl
  for x in $X; do
    local s=${x%%=*}
    run "$tag probe $s sampled" $D/${tag}_probe_${s}_infer.log timeout ${PROBE_TIMEOUT:-2400} \
      python -m cellgym.llm.infer --model $m --mode step --k 4 --extra $x --out $o/probe_$s.jsonl \
      && cat $o/probe_$s.jsonl >> $o/probe_samples.jsonl
  done
}

rl() {   # rl NAME INIT [extra flags]: one half of the RL run
  local name=$1 init=$2; shift 2
  run "$name trained" $D/$name.log python -m cellgym.llm.rl --init $init --out $D/$name \
    --steps $H --cells 32 --group 8 --lr ${RL_LR:-4e-6} --beta 0.02 --reward area \
    --ent-target ${ENT_TARGET:-0.7} --ent-coef 0.02 --ent-lr ${ENT_LR:-0.3} --select both --keep-ckpts \
    --eval-every 20 --eval-cells 96 --jobs $J "$@"
}

# d. sample the SL model; first bundle
sample $D/step05v2/model $D/step05v2 step05v2
bundle a
say "STAGE_A_DONE"

# e. RL on area, first half (the corpus references are shipped; rl computes them itself if missing)
wait $AREF
if [ -d $D/step05v2/model ]; then
  rl rl6 $D/step05v2/model
else
  say "rl6 skipped (no SL model)"
fi

# f. sample rl6's selected checkpoint; second bundle
sample $D/rl6/best $D/rl6 rl6
bundle b
say "STAGE_B_DONE"

# g. RL second half: continue from rl6/last (rl6/best if the first half died before writing last) with
# rl6's entropy state (same H0 and target, current coefficient). The KL anchor's reference is the
# --init model, so in this half it is rl6/last and not the SL model: accepted (the anchor then limits
# the drift per half, not over the whole run). The optimizer state is not carried over either.
I=$D/rl6/last; [ -d $I ] || I=$D/rl6/best
if [ -d $I ]; then
  E=""; [ -f $D/rl6/ent_state.json ] && E="--ent-state $D/rl6/ent_state.json"
  say "rl7 init $I ${E:-(no entropy state)}"
  rl rl7 $I $E --step-offset $H
else
  say "rl7 skipped (no rl6 model)"
fi

# h. sample rl7's selected checkpoint; final bundle
sample $D/rl7/best $D/rl7 rl7
bundle c
echo "RUN10_DONE $(date)" >> $P
