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
STEPS=50000
MAX_CONCURRENT=2
WORKERS=10
cd "$REPO" || exit 1

n_trainers() {
  ps -eo args= | awk -v p="$PY" '$1==p && $2=="tools/train_detector.py"' | wc -l
}

steps_for() {
  case "$1" in
    *_[0-9]*k_*) echo $(( $(echo "$1" | sed 's/.*_\([0-9]\+\)k_.*/\1/') * 1000 )) ;;
    *)           echo "$STEPS" ;;
  esac
}

cfg_from_ckpt() {
  local ck="$1" out="$2"
  env PYTHONPATH= "$PY" - "$ck" "$out" <<'PYEOF' 2>/dev/null
import sys, torch, yaml
cfg = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("cfg")
if not cfg: raise SystemExit(1)
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
PYEOF
}

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

for s in queue_ablations.sh launch_composite.sh; do
  if ps -eo args= | awk -v s="$s" '$1=="bash" && $2 ~ s' | grep -q .; then
    echo "[resume $(date -Is)] ABORT: $s is running -- it owns launch decisions. Not racing it."
    exit 0
  fi
done

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
for d in $(ls -dt runs/abl_*/ runs/rel_*/ 2>/dev/null); do
  name=$(basename "$d")
  ck="$d/ckpt_latest.pt"
  [ -f "$ck" ] || continue
  grep -q "train_detector\] done" "runs/$name.log" 2>/dev/null && continue
  if echo "$STOPPED" | awk '{print $1}' | grep -qx "$name"; then
    echo "  $name stopped on purpose, not resuming"; continue
  fi
  ps -eo args= | awk -v p="$PY" -v n="$name" \
      '$1==p && $2=="tools/train_detector.py" && $6==n {found=1}
       $1==p && $2=="tools/train_pair.py" && index($0, n":")>0 {found=1}
       END{exit !found}' && { echo "  $name already running"; continue; }
  cfg="$d/resume_cfg.yaml"
  if cfg_from_ckpt "$ck" "$cfg" && [ -s "$cfg" ]; then
    :
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
  env PYTHONPATH= taskset -c 0-18,20-23 "$PY" tools/train_detector.py --config "$cfg" --name "$name" \
      --steps "$budget" --workers "$WORKERS" --resume "$ck" >> "runs/$name.log" 2>&1 &
  sleep 120
done
echo "[resume $(date -Is)] done; $(n_trainers) trainer(s) running"
