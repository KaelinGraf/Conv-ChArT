# A-CE: Gaussian target (reference) vs strict one-hot BCE — distance_extrap (board square s (px), TRAINED 12-128)

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| board square s (px), TRAINED 12-128 | reference           | A-CE                |
|-------------------------------------|---------------------|---------------------|
| 6                                   | 0.6 / 0.0 / 0.732   | 0.0 / 0.0 / 0.000   |
| 8                                   | 34.6 / 14.3 / 0.451 | 39.1 / 9.1 / 0.388  |
| 10                                  | 96.3 / 81.0 / 0.454 | 84.3 / 86.1 / 0.412 |
| 12                                  | 98.9 / 86.9 / 0.442 | 89.1 / 94.7 / 0.409 |
| 16                                  | 99.0 / 95.2 / 0.422 | 94.8 / 96.5 / 0.407 |
| 64                                  | 97.8 / 99.9 / 0.425 | 96.2 / 99.3 / 0.415 |
| 128                                 | 96.0 / 99.3 / 0.417 | 95.1 / 94.6 / 0.406 |
| 160                                 | 95.9 / 82.9 / 0.432 | 93.7 / 69.9 / 0.416 |
| 192                                 | 97.7 / 33.6 / 0.414 | 95.0 / 38.4 / 0.405 |
| 256                                 | 92.7 / 1.2 / 0.416  | 85.6 / 1.3 / 0.383  |
