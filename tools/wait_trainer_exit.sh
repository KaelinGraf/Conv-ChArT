#!/bin/bash
# Block until the set of running ablation trainers CHANGES, then report what changed
# and exit. Run in the background so the session is re-invoked on any completion.
#
# argv[0]/argv[1] exact match, never `pgrep -f` -- the substring form matches this
# script's own argv and has previously deadlocked a wait loop for 47 minutes.
PY=/home/kaelin/anaconda3/envs/MLWS/bin/python
cd "/home/kaelin/p4p/dense deep charuco" || exit 1

running() {
  ps -eo args= | awk -v p="$PY" '$1==p && $2=="tools/train_detector.py"{print $4}' | sort
}

BEFORE=$(running)
echo "[watch $(date -Is)] waiting on:"; echo "$BEFORE" | sed 's/^/    /'
while :; do
  sleep 60
  NOW=$(running)
  [ "$NOW" = "$BEFORE" ] && continue
  echo "[watch $(date -Is)] TRAINER SET CHANGED"
  echo "  finished: $(comm -23 <(echo "$BEFORE") <(echo "$NOW") | tr '\n' ' ')"
  echo "  started : $(comm -13 <(echo "$BEFORE") <(echo "$NOW") | tr '\n' ' ')"
  echo "  now running: $(echo "$NOW" | tr '\n' ' ')"
  exit 0
done
