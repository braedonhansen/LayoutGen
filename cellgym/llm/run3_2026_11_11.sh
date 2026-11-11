#!/bin/bash
# probes far outside training (no Euler labels): step05, k=4
cd /workspace/lg
export HF_HOME=/root/hf PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=out/llm; P=$D/progress
while ! grep -q CONTROL_DONE $P; do sleep 30; done
X=""
for s in i8d14 i10d16 i12d20; do [ -f out/probe/$s/index.json ] && X="$X $s=out/probe/$s:all"; done
echo "probes:$X $(date)" >> $P
python -m cellgym.llm.infer --model $D/step05/model --mode step --k 4 --extra $X --out $D/step05/probe_samples.jsonl > $D/probe_infer.log 2>&1
python -m cellgym.llm.score $D/step05/probe_samples.jsonl --jobs 14 --extra $X > $D/probe_score.txt 2>&1
python -m cellgym.llm.infer --mode random --k 4 --extra $X --out $D/probe_random.jsonl > $D/probe_random_infer.log 2>&1
python -m cellgym.llm.score $D/probe_random.jsonl --jobs 14 --extra $X > $D/probe_random_score.txt 2>&1
echo "PROBE_DONE $(date)" >> $P
