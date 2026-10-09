#!/bin/bash
# Regenerate every multi-panel report figure with --separate into scratch, then copy ONLY the panels into
# 30_REPORT_transfer_finetune/figures_separate/ (composites are cmp'd against the report copies as a determinism check).
set -u
cd "/home/kaelin/p4p/dense deep charuco"
PY="env PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python"
R=paper/results_rev6/30_REPORT_transfer_finetune
D=paper/results_rev6/28_finetune_minimal
Z=paper/results_rev6/27_transfer_zeroshot
V=paper/results_rev6/26_pose_error_variance
S=${SCRATCH:-/tmp/dcc_regen_separate}   # throwaway: composites + per-factor sets are rendered here, only what the report needs is copied out
mkdir -p $S $R/figures_separate; rm -rf $R/figures_separate/*.png $S/fw_*
(
$PY tools/plot_pose_error_range.py --out $S/fig0_kalman_error_vs_range.png --separate
$PY tools/plot_transfer.py --arm "trained board (release)=$Z/fullval_control_dict5x5.json:$V/pose_REL882_B1.json" --arm "new board, zero-shot=$Z/fullval_transfer_dict6x6.json:$Z/pose_zeroshot_dict6x6_B1.json" --arm "fine-tuned 5k (head+bneck 3x)=$R/fullval_ft18_head_bneck_lr3x_hm_lr0p1.json:$D/pose_ft18_head_bneck_lr3x_hm_lr0p1.json" --arm "fine-tuned 20k (head+bneck 3x)=$R/fullval_ft21_head_bneck_lr3x_hm_lr0p1_20k.json:$D/pose_ft21_head_bneck_lr3x_hm_lr0p1_20k.json" --arm "fine-tuned 5k (whole model 3x)=$R/fullval_ft20_full_lr3x.json:$D/pose_ft20_full_lr3x.json" --out $S/fig1_transfer_end_to_end.png --title "Conv-ChArT 882k on a new board (DICT_6X6_250): zero-shot vs minimal fine-tuning vs the trained board" --separate
$PY tools/plot_transfer.py --arm "trained board DICT_5X5_50=$Z/fullval_control_dict5x5.json:$V/pose_REL882_B1.json" --arm "zero-shot DICT_6X6_250=$Z/fullval_transfer_dict6x6.json:$Z/pose_zeroshot_dict6x6_B1.json" --out $S/fig1b_transfer_zeroshot_only.png --title "Conv-ChArT 882k, no retraining: trained board vs. a board with a different marker dictionary" --separate
$PY tools/plot_finetune.py --arms ft01_head ft02_head_bneck_hm_lr0p1 ft03_head_bneck_lr0p01_hm_lr0p1 ft04_full_llrd0p3 ft05_head_gate3 ft06_head_d3_lr0p1 ft07_head_reinit ft08_bitfit ft09_full_lr0p1 ft10_full_lr1 --labels "1 head" "2 head+bneck+hm 0.1x" "3 head+bneck 0.01x+hm 0.1x" "4 full, LLRD 0.3" "5 head+gate3" "6 head+d3 0.1x" "7 head re-init" "8 BitFit" "9 full 0.1x" "10 full 1x" --out $S/fig3_finetune_ladder_round1.png --table $S/fig3.md --separate | tail -1
$PY tools/plot_finetune.py --arms ft03_head_bneck_lr0p01_hm_lr0p1 ft02_head_bneck_hm_lr0p1 ft15_head_bneck_lr0p3_hm_lr0p1 ft16_head_bneck_lr1_hm_lr0p1 ft18_head_bneck_lr3x_hm_lr0p1 ft19_head_bneck_e4_hm_lr1 ft10_full_lr1 ft20_full_lr3x ft13_head_bneck_hm_lr0p1_20k ft21_head_bneck_lr3x_hm_lr0p1_20k --labels "bneck 0.01x" "bneck 0.1x" "bneck 0.3x" "bneck 1x" "bneck 3x" "bneck+e4 1x" "full 1x" "full 3x" "bneck 0.1x, 20k" "bneck 3x, 20k" --targets 80 90 95 98 99 --title "Bottleneck learning rate and budget: head + bottleneck (+ corner head) vs the whole model, 5k and 20k steps, onto DICT_6X6_250" --out $S/fig4_finetune_ladder_rate_budget.png --table $S/fig4.md --separate | tail -1
$PY tools/plot_finetune.py --arms ft01_head mb3_ft01_head ft02_head_bneck_hm_lr0p1 mb3_ft02_head_bneck_hm_lr0p1 ft14_head_bneck_hm_lr1 mb3_ft14_head_bneck_hm_lr1 ft17_post_bneck mb3_ft17_post_bneck mb15_ft17_post_bneck --labels "head | 882k" "head | 3-board" "bneck 0.1x | 882k" "bneck 0.1x | 3-board" "bneck 1x | 882k" "bneck 1x | 3-board" "post-bneck | 882k" "post-bneck | 3-board" "post-bneck | 15-board" --title "Fine-tuning to DICT_6X6_250 from three base checkpoints: single-board 882k release vs 3-family and 15-family multi-board pretrains" --out $S/fig5_finetune_from_multiboard_bases.png --table $S/fig5.md --separate | tail -1
) > $S/log_a.txt 2>&1 &
for m in id_acc recall err_median; do
  $PY tools/plot_four_way.py --ours paper/results_rev6/20_robustness_REL882 --fast $Z/robustness_dict6x6 --dc $Z/_no_baselines --classical $Z/_no_baselines --out-dir $S/fw_transfer_$m --ours-label "882k, trained board DICT_5X5_50" --fast-label "882k, zero-shot DICT_6X6_250" --fast-refined --panel-metric $m --panel-tag transfer_$m --separate > $S/log_transfer_$m.txt 2>&1 &
done
for m in recall id_acc err_median; do
  $PY tools/plot_four_way.py --ours paper/results_rev6/20_robustness_REL882 --fast $R/robustness_finetuned_dict6x6 --dc $Z/_no_baselines --classical $Z/_no_baselines --out-dir $S/fw_ft_$m --ours-label "882k, trained board DICT_5X5_50" --fast-label "882k fine-tuned 20k, DICT_6X6_250" --fast-refined --panel-metric $m --panel-tag ft_$m --separate > $S/log_ft_$m.txt 2>&1 &
done
wait
echo "=== composites identical to the report copies? ==="
for n in fig0_kalman_error_vs_range fig1_transfer_end_to_end fig1b_transfer_zeroshot_only fig3_finetune_ladder_round1 fig4_finetune_ladder_rate_budget fig5_finetune_from_multiboard_bases; do cmp -s $S/$n.png $R/figures/$n.png && echo "  same: $n" || echo "  DIFFER: $n"; done
cmp -s $S/fw_transfer_id_acc/fourway_PANEL_id_acctransfer_id_acc.png $R/figures/fig2a_zeroshot_robustness_id.png && echo "  same: fig2a" || echo "  DIFFER: fig2a"
cmp -s $S/fw_transfer_recall/fourway_PANEL_recalltransfer_recall.png $R/figures/fig2b_zeroshot_robustness_recall.png && echo "  same: fig2b" || echo "  DIFFER: fig2b"
cmp -s $S/fw_ft_recall/fourway_PANEL_recallft_recall.png $R/figures/fig6a_finetuned_vs_trained_robustness_recall.png && echo "  same: fig6a" || echo "  DIFFER: fig6a"
cmp -s $S/fw_ft_id_acc/fourway_PANEL_id_accft_id_acc.png $R/figures/fig6b_finetuned_vs_trained_robustness_id.png && echo "  same: fig6b" || echo "  DIFFER: fig6b"
cmp -s $S/fw_ft_err_median/fourway_PANEL_err_medianft_err_median.png $R/figures/fig6c_finetuned_vs_trained_robustness_loc.png && echo "  same: fig6c" || echo "  DIFFER: fig6c"
echo "=== PANEL composites: the darkness-override fix changes the id_acc / err_median grids, so refresh the copies of record ==="
cp $S/fw_transfer_id_acc/fourway_PANEL_id_acctransfer_id_acc.png $Z/figures_vs_control/fourway_PANEL_id_acctransfer_id.png
cp $S/fw_transfer_id_acc/fourway_PANEL_id_acctransfer_id_acc.png $R/figures/fig2a_zeroshot_robustness_id.png
cp $S/fw_transfer_err_median/fourway_PANEL_err_mediantransfer_err_median.png $Z/figures_vs_control/fourway_PANEL_err_mediantransfer_loc.png
cp $S/fw_ft_id_acc/fourway_PANEL_id_accft_id_acc.png $R/robustness_figs_finetuned_vs_trained/
cp $S/fw_ft_id_acc/fourway_PANEL_id_accft_id_acc.png $R/figures/fig6b_finetuned_vs_trained_robustness_id.png
cp $S/fw_ft_err_median/fourway_PANEL_err_medianft_err_median.png $R/robustness_figs_finetuned_vs_trained/
cp $S/fw_ft_err_median/fourway_PANEL_err_medianft_err_median.png $R/figures/fig6c_finetuned_vs_trained_robustness_loc.png
echo "=== copy panels into figures_separate ==="
cp $S/fig0_*_p*.png $S/fig1_*_p*.png $S/fig1b_*_p*.png $S/fig3_*_p*.png $S/fig4_*_p*.png $S/fig5_*_p*.png $R/figures_separate/
for f in $S/fw_transfer_id_acc/fourway_PANEL_id_acctransfer_id_acc_p*.png; do cp "$f" "$R/figures_separate/fig2a_zeroshot_robustness_id_${f##*transfer_id_acc_}"; done
for f in $S/fw_transfer_recall/fourway_PANEL_recalltransfer_recall_p*.png; do cp "$f" "$R/figures_separate/fig2b_zeroshot_robustness_recall_${f##*transfer_recall_}"; done
for f in $S/fw_ft_recall/fourway_PANEL_recallft_recall_p*.png; do cp "$f" "$R/figures_separate/fig6a_finetuned_vs_trained_robustness_recall_${f##*ft_recall_}"; done
for f in $S/fw_ft_id_acc/fourway_PANEL_id_accft_id_acc_p*.png; do cp "$f" "$R/figures_separate/fig6b_finetuned_vs_trained_robustness_id_${f##*ft_id_acc_}"; done
for f in $S/fw_ft_err_median/fourway_PANEL_err_medianft_err_median_p*.png; do cp "$f" "$R/figures_separate/fig6c_finetuned_vs_trained_robustness_loc_${f##*ft_err_median_}"; done
$PY -I -c "
import cv2; R='$R'; im = cv2.imread(f'{R}/figures/figA_boards_trained_vs_new.png', 0); W = (im.shape[1] - 16) // 2
cv2.imwrite(f'{R}/figures_separate/figA_p1_board_trained_DICT_5X5_50.png', im[:, :W]); cv2.imwrite(f'{R}/figures_separate/figA_p2_board_new_DICT_6X6_250.png', im[:, W + 16:]); print('figA split', im.shape, '->', W)"
echo "=== logs ==="; cat $S/log_a.txt; grep -h PANEL $S/log_transfer_*.txt $S/log_ft_*.txt; grep -il "error\|traceback" $S/log_*.txt || echo "no errors in logs"
echo "=== figures_separate ==="; ls $R/figures_separate | wc -l; ls $R/figures_separate
