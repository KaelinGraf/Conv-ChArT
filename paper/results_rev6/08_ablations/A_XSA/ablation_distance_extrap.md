# A-XSA: standard MHSA (reference) vs exclusive self-attention — distance_extrap (board square s (px), TRAINED 12-128)

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| board square s (px), TRAINED 12-128 | reference           | XSA                 |
|-------------------------------------|---------------------|---------------------|
| 6                                   | 0.6 / 0.0 / 0.732   | 2.5 / 0.0 / 0.588   |
| 8                                   | 34.6 / 14.3 / 0.451 | 44.0 / 7.2 / 0.463  |
| 10                                  | 96.3 / 81.0 / 0.454 | 95.2 / 77.1 / 0.453 |
| 12                                  | 98.9 / 86.9 / 0.442 | 98.1 / 88.4 / 0.450 |
| 16                                  | 99.0 / 95.2 / 0.422 | 98.9 / 94.6 / 0.422 |
| 64                                  | 97.8 / 99.9 / 0.425 | 97.7 / 99.9 / 0.424 |
| 128                                 | 96.0 / 99.3 / 0.417 | 96.1 / 99.2 / 0.416 |
| 160                                 | 95.9 / 82.9 / 0.432 | 95.4 / 76.2 / 0.431 |
| 192                                 | 97.7 / 33.6 / 0.414 | 97.1 / 28.1 / 0.415 |
| 256                                 | 92.7 / 1.2 / 0.416  | 92.0 / 1.0 / 0.420  |
