# A-CE: Gaussian target (reference) vs strict one-hot BCE — distance (board square s (px))

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| board square s (px) | reference           | A-CE                |
|---------------------|---------------------|---------------------|
| 12                  | 99.7 / 96.2 / 0.429 | 96.5 / 96.9 / 0.410 |
| 16                  | 99.5 / 94.0 / 0.429 | 91.7 / 98.7 / 0.407 |
| 24                  | 99.7 / 97.7 / 0.415 | 95.6 / 97.3 / 0.400 |
| 32                  | 95.4 / 97.6 / 0.430 | 90.1 / 99.5 / 0.409 |
| 48                  | 97.6 / 99.9 / 0.423 | 96.3 / 99.5 / 0.409 |
| 64                  | 97.8 / 99.9 / 0.425 | 96.2 / 99.3 / 0.415 |
| 96                  | 96.8 / 99.7 / 0.415 | 95.8 / 99.2 / 0.412 |
| 128                 | 95.7 / 99.2 / 0.403 | 94.0 / 95.1 / 0.397 |
