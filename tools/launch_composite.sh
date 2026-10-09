#!/bin/bash
# Wait for A-SIGMA05 to finish, pick the winning sigma, launch the COMPOSITE into the
# slot it frees -- ahead of anything tools/queue_ablations.sh would otherwise start.
#
# Kaelin 2026-07-29: "only wait for sigma=0.5 to finish. We are only running xsa+nodilate
# to get a baseline to compare to the composite and motivate it academically, so we don't
# have to wait for it to complete to start training the composite."
#
# THE RACE THIS CLOSES: A-SIGMA05 exits ~40 min before A-XSA+NODILATE. Without this, the
# queue takes that slot for the A-XSA resume and the composite waits an extra ~4 h behind
# it. queue_ablations.sh therefore blocks on the sentinel this script writes.
#
# SIGMA SELECTION is mechanical, and was committed in writing BEFORE A-SIGMA05 reported
# (paper/results_rev6/08_ablations/00_QUEUE_AND_DECISION_RULE.md): take the sigma arm
# with the better final m01 p95, provided its M-04 is not worse by more than the
# +/-0.15 pp inter-validation noise band. p95 because M-04 has saturated at 99.0-99.2%
# across every arm and can no longer discriminate, while A-SIGMA1 held a -0.076 to
# -0.081 px p95 advantage across nine consecutive validations without ever reversing.
PY=/home/kaelin/anaconda3/envs/MLWS/bin/python
cd "/home/kaelin/p4p/dense deep charuco" || exit 1
SENTINEL=runs/.composite_launched
LOG=runs/abl_sigma05_50k_rev6.log

running() { ps -eo args= | awk -v p="$PY" '$1==p && $2=="tools/train_detector.py" && $4=="configs/abl_sigma05.yaml"' | wc -l; }

echo "[composite $(date -Is)] waiting for A-SIGMA05 to finish"
while :; do
  if grep -q "train_detector\] done" "$LOG" 2>/dev/null; then
    echo "[composite $(date -Is)] A-SIGMA05 completed gracefully"; break
  fi
  if [ "$(running)" -eq 0 ]; then
    # Process gone with no done-line = killed or crashed. Do NOT pick a sigma from a
    # truncated run; leave the sentinel unwritten so the queue stays blocked and the
    # main session investigates rather than silently composing on bad numbers.
    echo "[composite $(date -Is)] ABORT: A-SIGMA05 process gone with no done-line -- killed or crashed. Not launching."
    exit 1
  fi
  sleep 60
done

PICK=$(env PYTHONPATH= "$PY" - <<'PYEOF'
import json
def fin(r):
    L=[json.loads(l) for l in open('runs/%s/metrics.jsonl'%r)]
    v=[d for d in L if 'val' in d][-1]['val']
    return v['m01']['p95'], v['m04']['accuracy']
p5,m5 = fin('abl_sigma05_50k_rev6')
p1,m1 = fin('abl_sigma1_50k_rev6')
# better p95 wins, unless its M-04 is worse by more than the noise band
if p5 < p1 and (m5 - m1) > -0.0015:   win = 's05'
elif p1 < p5 and (m1 - m5) > -0.0015: win = 's1'
else:                                  win = 's05' if m5 >= m1 else 's1'
print("%s %.4f %.4f %.4f %.4f" % (win, p5, m5, p1, m1))
PYEOF
)
set -- $PICK
WIN=$1
echo "[composite $(date -Is)] sigma05 p95=$2 M04=$3 | sigma1 p95=$4 M04=$5 -> WINNER $WIN"

while [ "$(ps -eo args= | awk -v p="$PY" '$1==p && $2=="tools/train_detector.py"' | wc -l)" -ge 2 ]; do sleep 60; done
NAME=abl_composite_${WIN}_50k_rev6
echo "[composite $(date -Is)] launching $NAME (configs/abl_composite_${WIN}.yaml)"
env PYTHONPATH= "$PY" tools/train_detector.py --config "configs/abl_composite_${WIN}.yaml" \
    --name "$NAME" --steps 50000 --workers 6 > "runs/$NAME.log" 2>&1 &
echo "[composite $(date -Is)]   pid $!"
# the sentinel CONTENT is the winning sigma tag: downstream arms
# (A-NODILATE+WIDTH_HALF) carry the same sigma, and re-deriving the
# comparison in a second place would risk the two disagreeing.
echo "$WIN" > "$SENTINEL"
echo "[composite $(date -Is)] sentinel written; queue released"
