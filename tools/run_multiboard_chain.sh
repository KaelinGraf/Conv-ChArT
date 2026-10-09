#!/usr/bin/env bash
# tools/run_multiboard_chain.sh NAME STEPS -- the multi-board pretraining chain, parameterised (supersedes
# run_multiboard.sh, which was hardwired to the 15-board base): train configs/<NAME>_base.yaml to STEPS, score it
# per training board (2k each), on the original board (10k + pose set), zero-shot on the held-out DICT_6X6_250,
# then run the minimal fine-tune arms configs/<NAME>_ft*.yaml from it. Results: paper/results_rev6/29_multiboard_base/<NAME>/.
set -u
cd "$(dirname "$0")/.."
NAME=$1; STEPS=$2; RUN=${NAME}_base_$((STEPS/1000))k
PY="/home/kaelin/anaconda3/envs/MLWS/bin/python"
D=paper/results_rev6/29_multiboard_base/$NAME; RF=runs/ref_s15_10k_rev6/ckpt_0010000.pt; CK=$(printf 'runs/%s/ckpt_%07d.pt' "$RUN" "$STEPS")
mkdir -p "$D/logs"
stamp() { printf '{"stage":"%s","rc":%d,"s":%d,"end":%d}\n' "$1" "$2" "$3" "$(date +%s)" >> "$D/timing.jsonl"; }
t0=$(date +%s)
env PYTHONPATH= DCC_TRAINER_METRICS="" $PY tools/train_detector.py --config "configs/${NAME}_base.yaml" --name "$RUN" --workers "${WORKERS:-8}" > "$D/logs/train_$RUN.log" 2>&1
stamp train $? $(( $(date +%s) - t0 ))
t0=$(date +%s)
for c in configs/${NAME}_eval_*.yaml; do
  n=$(basename "$c" .yaml); n=${n#${NAME}_eval_}
  env PYTHONPATH= DCC_TRAINER_METRICS="" $PY tools/eval_checkpoint.py --ckpt "$CK" --config "$c" --n 2000 --out "$D/perboard_$n.json" > "$D/logs/perboard_$n.log" 2>&1
done
stamp perboard_2k $? $(( $(date +%s) - t0 ))
t0=$(date +%s)
env PYTHONPATH= DCC_TRAINER_METRICS="" $PY tools/eval_checkpoint.py --ckpt "$CK" --config configs/abl_c2_wh_clsfocal_lam2.yaml --out "$D/fullval_dict5x5_10k.json" > "$D/logs/fullval_dict5x5_10k.log" 2>&1
env PYTHONPATH= DCC_TRAINER_METRICS="" $PY tools/eval_pose_ours.py --pose-set eval_pose_rev6_b1 --ckpt "$CK" --refiner-ckpt $RF --allow-board-mismatch --out "$D/pose_dict5x5_B1.json" --per-image "$D/pose_dict5x5_B1_per_image.jsonl" > "$D/logs/pose_dict5x5.log" 2>&1
env PYTHONPATH= DCC_TRAINER_METRICS="" $PY tools/eval_checkpoint.py --ckpt "$CK" --config configs/transfer_dict6x6.yaml --out "$D/zeroshot_fullval_dict6x6_10k.json" > "$D/logs/zeroshot_fullval_dict6x6.log" 2>&1
env PYTHONPATH= DCC_TRAINER_METRICS="" $PY tools/eval_pose_ours.py --pose-set eval_pose_rev6_b1_dict6x6 --ckpt "$CK" --refiner-ckpt $RF --allow-board-mismatch --out "$D/zeroshot_pose_dict6x6_B1.json" --per-image "$D/zeroshot_pose_dict6x6_B1_per_image.jsonl" > "$D/logs/zeroshot_pose_dict6x6.log" 2>&1
stamp evals $? $(( $(date +%s) - t0 ))
WORKERS="${WORKERS:-8}" bash tools/run_finetune_arms.sh ${NAME}_ft01_head ${NAME}_ft02_head_bneck_hm_lr0p1 ${NAME}_ft14_head_bneck_hm_lr1
echo "ALL DONE"
