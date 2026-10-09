#!/bin/bash
# Self-healing training resume. Finds every ablation run that is INCOMPLETE and has a
# resumable checkpoint, and relaunches it from that checkpoint -- up to 2 concurrent.
#
# Written after the 2026-07-30 03:23 kswapd lockup cost 7 hours of idle GPU: the machine
# was wedged from 03:27 until a manual power cycle at 10:31. Checkpoints were fine (only
# 150 and 122 steps lost); the entire cost was nobody being there to restart it.
#
# NO LOGIN IS REQUIRED. This runs as a systemd SYSTEM service, so it starts at boot with
# no user session, no desktop, no terminal. "Log the computer back in" is not a step.
#
# "Incomplete" is decided by the graceful-completion marker, NOT by the process being
# absent -- `[train_detector] done:` appears only on a clean finish. That distinction is
# load-bearing: on 2026-07-29 two arms were pre-empted and their truncated results were
# nearly reported as converged.
PY=/home/kaelin/anaconda3/envs/MLWS/bin/python
REPO="/home/kaelin/p4p/dense deep charuco"
STEPS=50000       # DEFAULT ONLY -- steps_for() below overrides per run, and must, because
                  # cosine_lr anchors the LR anneal to --steps (dcc/trainutil.py:53).
                  # Resuming a 30k-budget arm on a 50k schedule would silently un-anneal it:
                  # lr jumps back to ~1.09e-4 (36% of peak) instead of the 3.0e-6 floor, so
                  # the run would end WORSE than if it had never been resumed.
MAX_CONCURRENT=2
WORKERS=10       # 6 during the 2026-07-30 freeze hunt; back to 10 once PL1 250W->125W fixed
                 # the actual cause (thermal), since shedding workers was only masking it.
cd "$REPO" || exit 1

n_trainers() {
  ps -eo args= | awk -v p="$PY" '$1==p && $2=="tools/train_detector.py"' | wc -l
}

# Step budget from the run-dir name: every arm is named ..._<N>k_rev6, and that N IS the
# budget it was scheduled with. Reading it back rather than hardcoding keeps a resume on the
# same LR schedule as the original launch. Falls back to $STEPS if the name has no _Nk_.
steps_for() {
  case "$1" in
    *_[0-9]*k_*) echo $(( $(echo "$1" | sed 's/.*_\([0-9]\+\)k_.*/\1/') * 1000 )) ;;
    *)           echo "$STEPS" ;;
  esac
}

# CONFIG RESOLUTION -- checkpoint first, case list only as a fallback.
#
# 2026-08-04: the hand-maintained cfg_for() case list below had gone stale and this script was
# protecting almost nothing. It covered only the 2026-07-30-era abl_* arms; every arm since (the
# seed replicates, the lambda/LR sweeps, C1-C5, and ALL THREE release models) was unmapped, and
# the run loop's `runs/abl_*/` glob skipped the rel_* dirs outright. A reboot would have silently
# dropped a 100k release run 5 h from finishing -- exactly the 7-hour-idle-GPU failure this
# script was written for, reintroduced by a maintenance trap.
#
# save_ckpt (dcc/trainutil.py) stores the FULL config dict in every checkpoint, so the run can
# always describe itself. Dumping that to a sidecar is self-maintaining -- it needs no edit when
# a new arm is added -- and is strictly MORE correct than the case list, which names a config
# FILE that may have been edited since the run launched.
cfg_from_ckpt() {
  local ck="$1" out="$2"
  env PYTHONPATH= "$PY" - "$ck" "$out" <<'PYEOF' 2>/dev/null
import sys, torch, yaml
cfg = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("cfg")
if not cfg: raise SystemExit(1)
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
PYEOF
}

# run-dir basename -> config. FALLBACK ONLY, for checkpoints predating the embedded cfg.
cfg_for() {
  case "$1" in
    abl_sigma05_50k_rev6)        echo configs/abl_sigma05.yaml ;;
    abl_sigma1_50k_rev6)         echo configs/abl_sigma1.yaml ;;
    abl_xsa_nodilate_50k_rev6)   echo configs/abl_xsa_nodilate.yaml ;;
    abl_xsa_50k_rev6)            echo configs/abl_xsa.yaml ;;
    abl_ce_50k_rev6)             echo configs/abl_ce.yaml ;;
    abl_nodilate_50k_rev6)       echo configs/abl_nodilate.yaml ;;
    abl_beta0_50k_rev6)          echo configs/abl_beta0.yaml ;;
    abl_gates_off_50k_rev6)      echo configs/abl_gates_off.yaml ;;
    abl_width_half_50k_rev6)     echo configs/abl_width_half.yaml ;;
    abl_nodilate_width_quarter_s05_50k_rev6) echo configs/abl_nodilate_width_quarter_s05.yaml ;;
    # 30k-budget rungs (tiered-model queue, 2026-07-30): the run dir carries the budget in
    # its name, so these are distinct entries from the 50k ones above -- a missing mapping
    # here means a reboot SKIPS the arm silently, which is why they are added with the queue.
    abl_nodilate_width_quarter_s05_30k_rev6) echo configs/abl_nodilate_width_quarter_s05.yaml ;;
    abl_nodilate_width_quarter_xsa_s05_30k_rev6) echo configs/abl_nodilate_width_quarter_xsa_s05.yaml ;;
    abl_nodilate_width_quarter_xsa_s05_50k_rev6) echo configs/abl_nodilate_width_quarter_xsa_s05.yaml ;;
    abl_nodilate_width_half_xsa_s05_50k_rev6) echo configs/abl_nodilate_width_half_xsa_s05.yaml ;;
    abl_nodilate_width_half_s05_50k_rev6) echo configs/abl_nodilate_width_half_s05.yaml ;;
    abl_nodilate_width_half_s1_50k_rev6)  echo configs/abl_nodilate_width_half_s1.yaml ;;
    abl_composite_s05_50k_rev6)  echo configs/abl_composite_s05.yaml ;;
    abl_composite_s1_50k_rev6)   echo configs/abl_composite_s1.yaml ;;
    *)                           echo "" ;;
  esac
}

# MUTUAL EXCLUSION -- caught by a dry run on 2026-07-30. This script resumes ANY
# incomplete arm, and tools/queue_ablations.sh also launches queued arms (including the
# A-XSA and A-CE resumes). Run both at once and they race: each waits for the same free
# slot and both launch into it, giving two trainers on one run dir writing one
# metrics.jsonl. The dry run sat in the slot-wait loop for exactly this reason.
#
# At boot the queue supervisor is never alive (it is not a service), so this guard costs
# nothing in the case the script exists for -- it only stops a manual invocation from
# fighting a live queue.
for s in queue_ablations.sh launch_composite.sh; do
  if ps -eo args= | awk -v s="$s" '$1=="bash" && $2 ~ s' | grep -q .; then
    echo "[resume $(date -Is)] ABORT: $s is running -- it owns launch decisions. Not racing it."
    exit 0
  fi
done

# DELIBERATELY STOPPED ARMS -- must not be resumed. These have NO `done:` marker because
# they were early-exited on Kaelin's call (converged: the per-validation gain flattened),
# not because they crashed. Without this list a reboot would resume them from their
# checkpoints, consume both slots, and evict the arms actually queued. Caught 2026-07-30
# while raising the worker count, before any reboot exercised it.
#
# Add a name here whenever an arm is stopped short on purpose, and say why.
STOPPED="
abl_convonly_50k_rev6           converged at 41,658 (five vals within +/-0.06 pp M-04)
abl_sigma05_50k_rev6            early-exit 36,487, m01+M-04 converged, p95 creeping 0.0006/1k
abl_xsa_nodilate_50k_rev6       early-exit 32,051, m01+M-04 converged, XSA verdict already clear
abl_nodilate_width_half_s05_50k_rev6  early-exit 30,657 (ckpt at 30,000): p95 deltas -0.0012 and
                                -0.0012, M-04 98.97 -> 98.94 inside noise. Stopped ON PURPOSE to
                                free the slot for the S rung (Kaelin: coverage of the ladder beats
                                annealing an also-ran). NOT annealed -- if this tier wins the
                                size/performance tradeoff, resume with --steps 35000 for a
                                compressed anneal. See paper/conference/TIERED_MODELS.md.
abl_nodilate_width_half_xsa_s05_50k_rev6  DROPPED at ~23,500 of 50k on evidence, not a crash:
                                behind plain M on M-04 at SIX consecutive matched vals (-0.07,
                                -0.14, -0.13, -0.34, -0.39, -0.20 pp) and on p95 at 3 of the last
                                4. Verdict: parameter-free XSA neither helps nor hurts at 882k
                                params -- it only pays where capacity binds (it IS winning at
                                222k, see the S+XSA arm). Its GPU slot went to M's anneal.
                                Full val history and a step-23,000 ckpt are banked, so the
                                matched-50k comparison can be finished later if the paper wants it.
abl_composite_s05_50k_rev6      early-exit ~31,600: p95 deltas -0.0001/-0.0009 then +0.0025 (two
                                consecutive sub-0.002 then a reversal), M-04 flat inside 0.06 pp
                                across five vals. NOT annealed (lr ~9.5e-5, 32% of peak) -- the
                                L-tier number is converged-but-unannealed and must be quoted as such.
"

echo "[resume $(date -Is)] scanning for incomplete runs"
# Newest checkpoint first, so the arm closest to finishing gets a slot first.
# abl_* AND rel_*: release models are resumable arms too. Deliberately NOT `runs/*/`, which
# would sweep in the Deep ChArUco baselines (charuconet_*) and the rev640_160k_* fine-tunes --
# those were stopped on purpose and have no `done:` marker, so a blind glob would resume them
# and evict the arms that matter.
for d in $(ls -dt runs/abl_*/ runs/rel_*/ 2>/dev/null); do
  name=$(basename "$d")
  ck="$d/ckpt_latest.pt"
  [ -f "$ck" ] || continue
  grep -q "train_detector\] done" "runs/$name.log" 2>/dev/null && continue      # finished cleanly
  if echo "$STOPPED" | awk '{print $1}' | grep -qx "$name"; then
    echo "  $name stopped on purpose, not resuming"; continue
  fi
  # "Already running" must also catch tools/train_pair.py, which trains several arms in one
  # process and names them as --arm NAME:CONFIG:STEPS rather than --name NAME. Matching the
  # bare name anywhere in a pair process's argv is sufficient and errs toward NOT relaunching.
  # Without this, a manual invocation while a pair is live would start a second writer on the
  # same run dir and one metrics.jsonl would get two authors.
  ps -eo args= | awk -v p="$PY" -v n="$name" \
      '$1==p && $2=="tools/train_detector.py" && $6==n {found=1}
       $1==p && $2=="tools/train_pair.py" && index($0, n":")>0 {found=1}
       END{exit !found}' && { echo "  $name already running"; continue; }
  cfg="$d/resume_cfg.yaml"
  if cfg_from_ckpt "$ck" "$cfg" && [ -s "$cfg" ]; then
    :                                   # self-describing checkpoint -- the normal path
  else
    cfg=$(cfg_for "$name")
    [ -n "$cfg" ] || { echo "  SKIP $name: checkpoint carries no cfg and no fallback mapping"; continue; }
    [ -f "$cfg" ] || { echo "  SKIP $name: fallback config $cfg missing"; continue; }
  fi
  step=$(env PYTHONPATH= "$PY" -c "import torch;print(torch.load('$ck',map_location='cpu',weights_only=False)['step'])" 2>/dev/null)
  [ -n "$step" ] || { echo "  SKIP $name: checkpoint unreadable"; continue; }
  budget=$(steps_for "$name")
  [ "$step" -ge "$budget" ] && { echo "  $name at $step/$budget, nothing to do"; continue; }

  while [ "$(n_trainers)" -ge "$MAX_CONCURRENT" ]; do sleep 60; done
  echo "[resume $(date -Is)] RESUMING $name from step $step/$budget ($cfg)"
  # Off cpu19 -- see the identical pin in tools/queue_ablations.sh for the fault history.
  env PYTHONPATH= taskset -c 0-18,20-23 "$PY" tools/train_detector.py --config "$cfg" --name "$name" \
      --steps "$budget" --workers "$WORKERS" --resume "$ck" >> "runs/$name.log" 2>&1 &
  sleep 120        # let it claim VRAM before the next slot check
done
echo "[resume $(date -Is)] done; $(n_trainers) trainer(s) running"
