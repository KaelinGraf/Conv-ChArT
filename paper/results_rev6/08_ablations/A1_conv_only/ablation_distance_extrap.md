# A1: bottleneck attention (reference) vs conv-only (attn_blocks=0) — distance_extrap (board square s (px), TRAINED 12-128)

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| board square s (px), TRAINED 12-128 | reference           | conv-only            |
|-------------------------------------|---------------------|----------------------|
| 6                                   | 0.6 / 0.0 / 0.732   | 0.1 / 0.0 / 0.461    |
| 8                                   | 34.6 / 14.3 / 0.451 | 34.0 / 7.2 / 0.440   |
| 10                                  | 96.3 / 81.0 / 0.454 | 93.7 / 80.3 / 0.453  |
| 12                                  | 98.9 / 86.9 / 0.442 | 98.1 / 88.2 / 0.448  |
| 16                                  | 99.0 / 95.2 / 0.422 | 99.0 / 95.2 / 0.420  |
| 64                                  | 97.8 / 99.9 / 0.425 | 97.6 / 100.0 / 0.427 |
| 128                                 | 96.0 / 99.3 / 0.417 | 96.0 / 96.7 / 0.419  |
| 160                                 | 95.9 / 82.9 / 0.432 | 95.4 / 52.3 / 0.432  |
| 192                                 | 97.7 / 33.6 / 0.414 | 97.6 / 20.0 / 0.418  |
| 256                                 | 92.7 / 1.2 / 0.416  | 95.2 / 2.6 / 0.420   |
