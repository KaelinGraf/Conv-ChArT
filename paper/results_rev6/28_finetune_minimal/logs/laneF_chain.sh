#!/usr/bin/env bash
# lane F: wait for the orphaned arm-15 trainer, finish its evaluation by hand (its driver was killed to fix the
# checkpoint-name bug), then run the post-bottleneck arms Kaelin asked for, the mb15 head-only extra, and the
# two remaining head-only variants.
cd "/home/kaelin/p4p/dense deep charuco"
PY=/home/kaelin/anaconda3/envs/MLWS/bin/python; D=paper/results_rev6/28_finetune_minimal; A=ft15_head_bneck_lr0p3_hm_lr0p1
until grep -q '"done"' runs/$A/metrics.jsonl 2>/dev/null; do sleep 60; done
sleep 30
env PYTHONPATH= DCC_TRAINER_METRICS="" $PY tools/eval_pose_ours.py --pose-set eval_pose_rev6_b1_dict6x6 --ckpt runs/$A/ckpt_0005000.pt --refiner-ckpt runs/ref_s15_10k_rev6/ckpt_0010000.pt --out $D/pose_$A.json --per-image $D/pose_${A}_per_image.jsonl > $D/logs/pose_$A.log 2>&1
printf '{"arm":"%s","train_s":null,"fullval_s":0.0,"pose_s":null,"rc":[0,0,%d],"note":"driver killed mid-arm; pose eval run by lane F"}\n' "$A" $? >> $D/timing.jsonl
WORKERS=8 bash tools/run_finetune_arms.sh mb15_ft17_post_bneck ft17_post_bneck mb15_ft01_head ft11_head_lr3x ft12_head_20k
