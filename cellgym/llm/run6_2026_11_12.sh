#!/bin/bash
# RL for routability: SL step policy (seed 1, as step05s1) -> RL with width reward minus a channel-density
# penalty (rl3), same RL settings as rl2 (80 steps, 32 cells x 8, lr 2e-6, beta 0.02), in-distribution
# cells only. Then full OOD eval (+ i8d14 probe). Progress -> out/llm/progress
cd /workspace/lg
export HF_HOME=/root/hf PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=out/llm; P=$D/progress; mkdir -p $D
echo "start $(date)" >> $P
python -m cellgym.llm.build --out $D/data --steps-per-plan 0 > $D/build.log 2>&1
echo "data built $(date)" >> $P
(timeout 3600 python -m cellgym.llm.probe --out out/probe --spec 8:14 --count 20 > $D/probe_8_14.log 2>&1 &)
python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B --data $D/data/step_train.jsonl --val $D/data/step_val.jsonl \
  --out $D/step05s1 --epochs 1 --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 --seed 1 > $D/step05s1_train.log 2>&1
echo "step05s1 trained $(date)" >> $P
python -m cellgym.llm.rl --init $D/step05s1/model --out $D/rl3 --steps 80 --cells 32 --group 8 --lr 2e-6 --beta 0.02 \
  --lam-dens 0.15 --eval-every 20 --eval-cells 96 --jobs 32 > $D/rl3.log 2>&1
echo "rl3 trained $(date)" >> $P
python -m cellgym.llm.infer --model $D/rl3/best --mode step --k 8 --out $D/rl3/samples.jsonl > $D/rl3_infer.log 2>&1
echo "rl3 sampled $(date)" >> $P
python -m cellgym.llm.score $D/rl3/samples.jsonl --jobs 32 > $D/rl3_score.txt 2>&1
X=""; [ -f out/probe/i8d14/index.json ] && X="i8d14=out/probe/i8d14:all"
if [ -n "$X" ]; then
  python -m cellgym.llm.infer --model $D/rl3/best --mode step --k 4 --extra $X --out $D/rl3/probe_samples.jsonl > $D/rl3_probe_infer.log 2>&1
  python -m cellgym.llm.score $D/rl3/probe_samples.jsonl --jobs 32 --extra $X > $D/rl3_probe_score.txt 2>&1
fi
tar czf $D/results_rl3.tgz $D/*.txt $D/*.log $D/progress $D/rl3/*.jsonl $D/rl3/*.summary.json 2>/dev/null
echo "RL3_DONE $(date)" >> $P
