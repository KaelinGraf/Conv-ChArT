# Hardest sweep step (lowest control recall), refined arm: control (trained board DICT_5X5_50) vs zero-shot (DICT_6X6_250), identical frames, n=60/step

Source: paper/results_rev6/20_robustness_REL882 (control) and 27_transfer_zeroshot/robustness_dict6x6 (zero-shot).

| factor | hardest step (min control recall) | recall ctrl / zero-shot (%) | ID ctrl / zero-shot (%) | loc median refined ctrl / zero-shot (px) |
|---|---|---|---|---|
| board_occlusion | 50 | 85.9 / 87.1 | 71.2 / 2.9 | 0.092 / 0.090 |
| brightness | 0.35 | 85.9 / 85.9 | 97.8 / 0.7 | 0.118 / 0.119 |
| contrast | 0.8 | 92.8 / 93.0 | 100.0 / 4.9 | 0.074 / 0.076 |
| darkness | 0.01 | 45.8 / 44.7 | 69.1 / 4.0 | 0.463 / 0.452 |
| defocus_blur | 3 | 91.5 / 90.9 | 99.3 / 3.5 | 0.070 / 0.073 |
| diff_ambient | 0.95 | 92.0 / 91.6 | 96.9 / 10.6 | 0.340 / 0.337 |
| diff_ghosting | 0.5 | 93.5 / 94.0 | 98.9 / 8.6 | 0.255 / 0.255 |
| diff_ratio | 0.15 | 90.1 / 87.7 | 91.7 / 9.1 | 0.428 / 0.419 |
| distance | 64 | 90.7 / 91.4 | 98.2 / 3.2 | 0.103 / 0.110 |
| distance_extrap | 6 | 0.4 / 0.2 | 0.0 / 0.0 | 0.222 / 0.244 |
| droplets | 0 | 92.9 / 93.0 | 98.6 / 2.8 | 0.087 / 0.088 |
| ink_contrast | 1.6 | 70.3 / 69.2 | 80.8 / 6.9 | 0.232 / 0.246 |
| motion_blur | 9 | 72.8 / 74.2 | 98.0 / 7.9 | 0.333 / 0.336 |
| object_occlusion | 1 | 89.0 / 89.3 | 97.2 / 1.1 | 0.090 / 0.094 |
| occlusion | 0 | 91.8 / 92.1 | 99.0 / 5.3 | 0.081 / 0.080 |
| rotation | 150 | 93.7 / 94.2 | 99.9 / 8.3 | 0.106 / 0.100 |
| sensor_noise_K | 0.02 | 16.6 / 16.3 | 83.2 / 0.0 | 0.441 / 0.384 |
| specular | 220 | 90.6 / 91.3 | 99.2 / 3.9 | 0.111 / 0.116 |
| tilt | 50 | 92.4 / 92.0 | 99.6 / 2.5 | 0.090 / 0.088 |
| vignette | 0.05 | 93.9 / 94.1 | 98.6 / 6.2 | 0.086 / 0.086 |
