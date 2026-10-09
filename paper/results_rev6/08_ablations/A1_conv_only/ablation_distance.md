# A1: bottleneck attention (reference) vs conv-only (attn_blocks=0) — distance (board square s (px))

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| board square s (px) | reference           | conv-only            |
|---------------------|---------------------|----------------------|
| 12                  | 99.7 / 96.2 / 0.429 | 98.9 / 97.5 / 0.429  |
| 16                  | 99.5 / 94.0 / 0.429 | 99.3 / 94.4 / 0.430  |
| 24                  | 99.7 / 97.7 / 0.415 | 98.4 / 98.7 / 0.415  |
| 32                  | 95.4 / 97.6 / 0.430 | 91.8 / 99.6 / 0.419  |
| 48                  | 97.6 / 99.9 / 0.423 | 97.4 / 99.9 / 0.420  |
| 64                  | 97.8 / 99.9 / 0.425 | 97.6 / 100.0 / 0.427 |
| 96                  | 96.8 / 99.7 / 0.415 | 96.7 / 98.9 / 0.418  |
| 128                 | 95.7 / 99.2 / 0.403 | 95.7 / 91.8 / 0.402  |
