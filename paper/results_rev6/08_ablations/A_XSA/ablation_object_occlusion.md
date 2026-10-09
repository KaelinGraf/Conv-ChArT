# A-XSA: standard MHSA (reference) vs exclusive self-attention — object_occlusion (real object occluders (n, SAM2 bank))

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| real object occluders (n, SAM2 bank) | reference           | XSA                 |
|--------------------------------------|---------------------|---------------------|
| 0                                    | 97.0 / 99.8 / 0.421 | 96.9 / 99.9 / 0.414 |
| 1                                    | 98.3 / 98.8 / 0.416 | 98.3 / 99.9 / 0.413 |
| 2                                    | 97.9 / 96.9 / 0.429 | 97.7 / 96.1 / 0.430 |
| 3                                    | 98.8 / 96.9 / 0.418 | 98.1 / 97.1 / 0.419 |
| 4                                    | 99.1 / 98.1 / 0.425 | 98.8 / 98.0 / 0.423 |
