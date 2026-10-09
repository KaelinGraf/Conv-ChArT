# A-XSA: standard MHSA (reference) vs exclusive self-attention — distance (board square s (px))

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| board square s (px) | reference           | XSA                 |
|---------------------|---------------------|---------------------|
| 12                  | 99.7 / 96.2 / 0.429 | 98.8 / 96.3 / 0.420 |
| 16                  | 99.5 / 94.0 / 0.429 | 99.2 / 94.3 / 0.429 |
| 24                  | 99.7 / 97.7 / 0.415 | 99.4 / 97.9 / 0.413 |
| 32                  | 95.4 / 97.6 / 0.430 | 92.6 / 99.0 / 0.425 |
| 48                  | 97.6 / 99.9 / 0.423 | 97.6 / 99.9 / 0.422 |
| 64                  | 97.8 / 99.9 / 0.425 | 97.7 / 99.9 / 0.424 |
| 96                  | 96.8 / 99.7 / 0.415 | 96.7 / 99.9 / 0.424 |
| 128                 | 95.7 / 99.2 / 0.403 | 95.9 / 98.8 / 0.401 |
