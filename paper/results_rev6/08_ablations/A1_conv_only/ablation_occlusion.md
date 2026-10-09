# A1: bottleneck attention (reference) vs conv-only (attn_blocks=0) — occlusion (rectangular occluders (n))

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| rectangular occluders (n) | reference           | conv-only           |
|---------------------------|---------------------|---------------------|
| 0                         | 98.1 / 99.9 / 0.419 | 97.9 / 99.0 / 0.415 |
| 2                         | 97.2 / 96.6 / 0.417 | 96.7 / 96.3 / 0.422 |
| 4                         | 96.0 / 98.5 / 0.420 | 95.9 / 98.0 / 0.425 |
| 6                         | 97.2 / 97.3 / 0.416 | 97.0 / 96.8 / 0.419 |
| 8                         | 99.8 / 99.5 / 0.422 | 99.6 / 98.3 / 0.427 |
| 10                        | 95.7 / 99.3 / 0.421 | 95.5 / 97.6 / 0.424 |
