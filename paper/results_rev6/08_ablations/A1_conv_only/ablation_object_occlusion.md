# A1: bottleneck attention (reference) vs conv-only (attn_blocks=0) — object_occlusion (real object occluders (n, SAM2 bank))

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| real object occluders (n, SAM2 bank) | reference           | conv-only            |
|--------------------------------------|---------------------|----------------------|
| 0                                    | 97.0 / 99.8 / 0.421 | 96.7 / 99.9 / 0.419  |
| 1                                    | 98.3 / 98.8 / 0.416 | 98.3 / 100.0 / 0.413 |
| 2                                    | 97.9 / 96.9 / 0.429 | 97.6 / 95.9 / 0.430  |
| 3                                    | 98.8 / 96.9 / 0.418 | 97.9 / 97.4 / 0.416  |
| 4                                    | 99.1 / 98.1 / 0.425 | 98.9 / 97.8 / 0.423  |
