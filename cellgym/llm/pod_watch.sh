#!/bin/bash
# Watch an unattended pod run from the PC and TERMINATE THE POD when it ends, without needing a chat turn.
#
#   bash cellgym/llm/pod_watch.sh POD_ID IP PORT DONE_MARKER LOCAL_DIR [MAX_HOURS] [BUNDLE_GLOB]
#
#   POD_ID       RunPod pod id (terminated with `runpodctl pod delete`)
#   IP PORT      direct SSH endpoint of the pod (key: ~/.ssh/runpod_layoutgen)
#   DONE_MARKER  text that appears in /workspace/lg/out/llm/progress when the run is over (e.g. RUN10_DONE)
#   LOCAL_DIR    where progress and result bundles are copied
#   MAX_HOURS    hard deadline from now (default 8): the pod is terminated even if the run is not done
#   BUNDLE_GLOB  remote result bundles to copy as they appear (default out/llm/results_*.tgz)
#
# The pod is terminated when: the done marker appears (after the bundles are copied), the host stops answering
# SSH for ~10 minutes, or the deadline passes. Bundles are copied every poll, so a dead host loses at most the
# stage in progress. Needs a runpodctl that can delete pods: the script checks that BEFORE watching and refuses
# to start otherwise, so a run is never left depending on a chat turn to stop it.
set -u
# This task has its own key: RUNPOD_API_KEY_LAYOUTGEN (pod read/write). It is used for the runpodctl calls
# in this script only; the machine's general RUNPOD_API_KEY (meant for something else) is left untouched.
if [ -n "${RUNPOD_API_KEY_LAYOUTGEN:-}" ]; then export RUNPOD_API_KEY="$RUNPOD_API_KEY_LAYOUTGEN"; fi
POD=$1; IP=$2; PORT=$3; MARK=$4; OUT=$5; MAXH=${6:-8}; GLOB=${7:-out/llm/results_*.tgz}
POLL=${POLL_SECONDS:-180}
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -i $HOME/.ssh/runpod_layoutgen -p $PORT root@$IP"
mkdir -p "$OUT"
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$OUT/watch.log"; }

# Refuse to watch if we could not terminate: the pod must be visible to this runpodctl.
if ! runpodctl pod get "$POD" 2>/dev/null | grep -q "\"$POD\""; then
  log "ABORT: runpodctl cannot read pod $POD (set RUNPOD_API_KEY_LAYOUTGEN to a key with pod read/write). Not watching."
  exit 2
fi

terminate() {
  log "terminating $POD ($1)"
  for i in 1 2 3 4 5; do
    runpodctl pod delete "$POD" >> "$OUT/watch.log" 2>&1
    sleep 5
    if ! runpodctl pod get "$POD" 2>/dev/null | grep -q "\"$POD\""; then log "TERMINATED $POD"; return 0; fi
  done
  log "TERMINATE FAILED for $POD: stop it by hand"
  return 1
}

pull() {   # copy any bundle whose remote size differs from the local copy
  $SSH "cd /workspace/lg && ls -l $GLOB 2>/dev/null | awk '{print \$5, \$9}'" 2>/dev/null | while read -r size path; do
    [ -n "${path:-}" ] || continue
    f="$OUT/$(basename "$path")"
    if [ "$(stat -c %s "$f" 2>/dev/null)" != "$size" ]; then
      $SSH "cat /workspace/lg/$path" > "$f.part" 2>/dev/null && mv "$f.part" "$f" && log "pulled $(basename "$path") ($size bytes)"
    fi
  done
}

deadline=$(( $(date +%s) + ${MAXH%.*} * 3600 ))
dead=0
log "watching $POD at $IP:$PORT for '$MARK', deadline ${MAXH}h, poll ${POLL}s"
while true; do
  P=$($SSH "cat /workspace/lg/out/llm/progress" 2>/dev/null)
  if [ -z "$P" ]; then
    dead=$((dead + 1))
    log "no answer from host ($dead)"
    if [ $((dead * POLL)) -ge 600 ]; then terminate "host dead"; exit 3; fi
  else
    dead=0
    echo "$P" > "$OUT/progress.txt"
    pull
    if echo "$P" | grep -q "$MARK"; then pull; terminate "run finished"; exit 0; fi
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then pull; terminate "deadline ${MAXH}h"; exit 4; fi
  sleep "$POLL"
done
