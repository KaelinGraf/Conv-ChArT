# Diagnostic at step 14,000 of the 15-board base (2026-10-08 14:50)

Rolling checkpoint `ckpt_latest.pt` (step 14,000) scored per training board, 200 samples each
(`perboard_*.json`, tools/eval_checkpoint.py). Every board: ID 24.9-26.4%, recall 91.8-92.9%, p95 0.72-0.77 px.

25% is one in four: the corner's lattice position learned from the board OUTLINE, which fixes the index only up to
the lattice's 90-degree symmetry. It is uniform across families and across apparent-scale octaves (the mixed val
shows 25.0% even at s=64-128 px, where every marker is legible), so marker reading has not begun on any family.
Compare the single-board release run at the same steps: 50.7% at 5k, 96.4% at 7.5k, 98.3% at 10k. Train loss:
2.59 (15-board) vs 1.50 (single) at 15k, creeping rather than dropping.

Decision rule set in advance: at the 25k full validation, if ID is still below 40% the run is stopped as a negative
result (lattice-symmetric identity is learned from geometry alone; marker reading diluted over 15 families did not
start within 25k steps of the release recipe) and lane A runs the reduced 3-family base (`configs/mb3_base.yaml`,
30k steps) to test whether multi-board identity is learnable at all before scaling the board count.

## Outcome (16:01): STOPPED at 25k by the rule above

Full 10k validation at step 25,000: ID 25.9% (by octave 26.0 / 26.3 / 25.9 / 25.3), corner median 0.4081 px,
p95 0.6926 px, recall 92.0-95.0% by octave. In-loop 1k vals from 5k to 22.5k: 24.1, 25.3, 24.8, 25.2, 25.7, 25.9,
25.5 ... -- flat at the one-in-four symmetric solution for 20,000 steps with the LR still above half its peak.
Localisation and recall are as good as the single-board run's at the same step. The checkpoint is kept as the record:
`runs/mb15_base_50k/ckpt_0025000.pt` (run name says 50k; it was stopped at 25k). The chain's later stages
(per-board eval, zero-shot, fine-tunes from it) were NOT run -- a 25%-ID base has nothing to transfer.

Finding for the report: with the release recipe, 15 marker families at once do not get past the geometry-only
solution within 25k steps, whereas one family is read at 96% by 7.5k. Reduced test launched next: 3 families
(`configs/mb3_base.yaml`, 30k steps) -- if identity is learnable there, board count (gradient dilution over
families) is the lever; if not, the recipe (class-loss weight, bottleneck width) is.
