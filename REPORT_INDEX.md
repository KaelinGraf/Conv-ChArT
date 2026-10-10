# Conv-ChArT -- report branch index

`main` carries the method only (library, entry points, conventions tests, deployment docs; see `.gitignore`).
This branch adds everything the thesis cites: the paper, the results of record with their provenance, every
config that was trained, the tooling that produced each number and figure, and the conference deliverables.

| where | what |
|---|---|
| `paper/conv_chart.tex`, `paper/conv_chart.pdf`, `paper/figs/` | the paper source, its last build, and its figures |
| `paper/conference/` | the conference deck, notes and figures |
| `figures/` | the conference deliverables (headline panel, pose and cost tables, architecture) |
| `paper/results_rev6/00_REPORT_REFERENCE/CONV_CHART_REFERENCE.md` | **the single complete reference**: the facts document plus every figure (253, also in `figures/`), table and record of the results tree, all ablation arms with learning curves, historical material labelled; built by `tools/build_report_reference.py` |
| `paper/results_rev6/00_FACTS_FOR_REPORT.md` | **start here**: every fact about the board, models, pipeline, training, data, evaluation, results, ablations, deployment, transfer and testing, each with its source file, plus the list of places where older documents and the paper disagree with the artefacts |
| `paper/results_rev6/` | **results of record** (rev-6 generator; release models `rel_w25_lam2_100k_rev6` 222k, `rel_w375_lam15_100k_rev6` 502k, `rel_w882_c2_100k_rev6` 882k) -- one directory per study, each with a README stating provenance for every number; `PLAN_autonomous_campaign.md` is the dated campaign log |
| `paper/results_rev5/` | the superseded rev-5 generator revision (`00_SUPERSEDED.md`, `00_README_START_HERE.md`); kept for the record, never for headline numbers |
| `configs/` | every config: the release tiers, the ablation arms (`abl_*`), fine-tune arms (`ft*`), multi-board (`mb*`) |
| `tools/` | evaluation, sweeps, figure and table production (`tools/*_pdf.py`, `plot_*.py`), fine-tune and multi-board cutters |
| `docs/PROJECT_KNOWLEDGE.md` | the consolidated project-knowledge document; `ARCHITECTURE.md`, `TOOLING.md` |
| `tests/` | conventions tests, including the tooling tests |

Not on this branch: model weights and ONNX exports (`runs/`, `*.pt`, `*.onnx`, including
`paper/results_rev6/29_multiboard_base/diag_step14k/ckpt_latest_copy.pt` and the ONNX files in `24_export_parity/`),
generated image sets (`eval_pose*/`, `sheets/`, `audit*`), `tools/charuconet.py` (a byte-for-byte copy of
JunkyByte/deepcharuco's `dcModel`, MIT; the vendored upstream lives outside this repo), LaTeX build files, and the
eight 12 MB rev-5 generator-audit overlay sheets.

## `paper/results_rev6/` directories

| directory | README heading |
|---|---|
| `02_refiner/` | (no README) |
| `03_robustness/` | (no README) |
| `03_robustness_ConvChArT/` | (no README) |
| `03_robustness_FAST/` | (no README) |
| `03_robustness_L/` | (no README) |
| `03_robustness_L_tau015/` | (no README) |
| `03_robustness_PREGUARD/` | (no README) |
| `05_comparison/` | (no README) |
| `06_tables/` | (no README) |
| `07_introspection/` | (no README) |
| `07_introspection_L/` | (no README) |
| `08_ablations/` | (no README) |
| `08_ablations_NOTE_sigma_justification.md` | |
| `08_ablations_peak_sharpness.json` | |
| `10_receptive_field/` | (no README) |
| `11_attention_gate/` | (no README) |
| `12_loss/` | (no README) |
| `13_pose/` | (no README) |
| `14_data/` | (no README) |
| `15_cost/` | (no README) |
| `16_future_work/` | (no README) |
| `17_range/` | (no README) |
| `20_robustness_REL222/` | (no README) |
| `20_robustness_REL502/` | (no README) |
| `20_robustness_REL882/` | (no README) |
| `21_pose_REL222.json` | |
| `21_pose_REL222_B1.json` | |
| `21_pose_REL502.json` | |
| `21_pose_REL502_B1.json` | |
| `21_pose_REL882.json` | |
| `21_pose_REL882_B1.json` | |
| `22_fourway_RELEASE/` | (no README) |
| `22_fourway_RELEASE_882_vs_222/` | (no README) |
| `23_code_audit_FINDINGS.md` | |
| `24_export_parity/` | (no README) |
| `24_export_parity_detfp16/` | (no README) |
| `25_runtime_latency/` | (no README) |
| `26_pose_error_variance/` | 26 — Pose error variance of the 882k release model (Kalman measurement noise R) | `README.md`
| `27_transfer_zeroshot/` | 27 — Zero-shot board transfer: the 882k release model on a board it was never trained on | `README.md`
| `28_finetune_minimal/` | 28 — Minimal fine-tuning: what has to move to put Conv-ChArT 882k on a new board, and what it costs | `README.md`
| `29_multiboard_base/` | 29 — Multi-board pretraining: does a trunk trained on many marker families transfer better? | `README.md`
| `30_REPORT_transfer_finetune/` | Conv-ChArT 882k: pose measurement noise, board transfer and minimal fine-tuning — headline results for the report | `REPORT.md`
| `31_print_board/` | 31 — Print-ready release board | `README.md`
| `PLAN_autonomous_campaign.md` | |
