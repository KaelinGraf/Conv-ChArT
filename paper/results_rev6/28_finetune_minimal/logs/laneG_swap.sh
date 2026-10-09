#!/usr/bin/env bash
# lane G: when lane F stamps arm 11, retire the 20k head-only arm from the FRONT of the queue (it restarts at
# the end), and run the two top-of-ladder rungs first.
cd "/home/kaelin/p4p/dense deep charuco"; D=paper/results_rev6/28_finetune_minimal
until grep -q '"arm":"ft11_head_lr3x"' $D/timing.jsonl 2>/dev/null; do sleep 20; done
pkill -f "^bash tools/run_finetune_arms.sh mb15_ft17_post_bneck"; sleep 2
pkill -f "^/home/kaelin/anaconda3/envs/MLWS/bin/python tools/train_detector.py --config configs/ft12_head_20k.yaml"; sleep 8
rm -rf runs/ft12_head_20k $D/logs/train_ft12_head_20k.log
WORKERS=8 bash tools/run_finetune_arms.sh ft18_head_bneck_lr3x_hm_lr0p1 ft19_head_bneck_e4_hm_lr1 ft12_head_20k
