#!/usr/bin/env bash
# tools/run_finetune_arms.sh -- run a LANE of the minimal-fine-tuning ladder sequentially (lanes run side by
# side; isolated per-step GPU cost comes from tools/finetune_cost.py, logged wall time is contended), each arm: train -> 10k full val -> 1000-frame pose set, all on
# the transfer board (the 10k full val is the trainer's own, logged at step 5000). Per-arm start/end epochs go to paper/results_rev6/28_finetune_minimal/timing.jsonl.
# Usage: tools/run_finetune_arms.sh ft01_head ft02_... (arm = config stem = run name)
set -u
cd "$(dirname "$0")/.."
PY="/home/kaelin/anaconda3/envs/MLWS/bin/python"
D=paper/results_rev6/28_finetune_minimal; RF=runs/ref_s15_10k_rev6/ckpt_0010000.pt
mkdir -p "$D/logs"
for ARM in "$@"; do
  CFG=configs/$ARM.yaml
  STEPS=$($PY -c "import yaml; print(yaml.safe_load(open('$CFG'))['train']['steps'])")
  CK=$(printf 'runs/%s/ckpt_%07d.pt' "$ARM" "$STEPS")
  t0=$(date +%s.%N)
  env PYTHONPATH= DCC_TRAINER_METRICS="" $PY tools/train_detector.py --config "$CFG" --name "$ARM" --workers "${WORKERS:-12}" > "$D/logs/train_$ARM.log" 2>&1
  rc=$?; t1=$(date +%s.%N)
  rc2=0; t2=$t1
  env PYTHONPATH= DCC_TRAINER_METRICS="" $PY tools/eval_pose_ours.py --pose-set eval_pose_rev6_b1_dict6x6 --ckpt "$CK" --refiner-ckpt $RF --out "$D/pose_$ARM.json" --per-image "$D/pose_${ARM}_per_image.jsonl" > "$D/logs/pose_$ARM.log" 2>&1
  rc3=$?; t3=$(date +%s.%N)
  printf '{"arm":"%s","train_s":%.1f,"fullval_s":%.1f,"pose_s":%.1f,"rc":[%d,%d,%d],"start":%.0f,"end":%.0f}\n' \
    "$ARM" "$(echo "$t1-$t0"|bc)" "$(echo "$t2-$t1"|bc)" "$(echo "$t3-$t2"|bc)" $rc $rc2 $rc3 "$t0" "$t3" >> "$D/timing.jsonl"
  echo "[$(date +%H:%M:%S)] $ARM done rc=$rc/$rc2/$rc3 train=$(echo "($t1-$t0)/60"|bc)min"
done
echo "ALL DONE"
