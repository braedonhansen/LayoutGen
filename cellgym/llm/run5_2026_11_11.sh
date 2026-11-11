#!/bin/bash
# RL (width reward) on the step policy, in-distribution cells only; then full OOD eval of best.pt
cd /workspace/lg
export HF_HOME=/root/hf PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=out/llm; P=$D/progress
while ! grep -q ALL_DONE $P; do sleep 30; done
echo "rl start $(date)" >> $P
python -m cellgym.llm.rl --init $D/step05s1/model --out $D/rl1 --steps 120 --cells 32 --group 8 --lr 2e-6 --beta 0.02 \
  --eval-every 20 --eval-cells 96 --jobs 32 > $D/rl1.log 2>&1
echo "rl trained $(date)" >> $P
python -m cellgym.llm.infer --model $D/rl1/best --mode step --k 8 --out $D/rl1/samples.jsonl > $D/rl1_infer.log 2>&1
python -m cellgym.llm.score $D/rl1/samples.jsonl --jobs 32 > $D/rl1_score.txt 2>&1
X=""; for s in i8d14 i10d16 i8d18; do [ -f out/probe/$s/index.json ] && X="$X $s=out/probe/$s:all"; done
if [ -n "$X" ]; then
  python -m cellgym.llm.infer --model $D/rl1/best --mode step --k 4 --extra $X --out $D/rl1/probe_samples.jsonl > $D/rl1_probe_infer.log 2>&1
  python -m cellgym.llm.score $D/rl1/probe_samples.jsonl --jobs 32 --extra $X > $D/rl1_probe_score.txt 2>&1
fi
tar czf $D/results_rl.tgz $D/*.txt $D/*.log $D/progress $D/*/samples.summary.json $D/*/probe_samples.summary.json $D/*/log.jsonl 2>/dev/null
echo "RL_DONE $(date)" >> $P
