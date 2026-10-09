# abl_gates_off_50k_rev6

Trained 50,000 steps, 20 validations. Checkpoints: 3 in `runs/abl_gates_off_50k_rev6/`.

| | step | m01 mean | m01 median | m01 p95 | tail>4px | M-04 | val loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| best M-04 | 45,000 | 0.4572 | 0.4233 | 0.8192 | 0.129% | 99.07% | 0.1491 |
| final | 50,000 | 0.4582 | 0.4233 | 0.8216 | 0.117% | 99.03% | 0.1471 |

Full curve and per-octave breakdown in `train_metrics.json`.
