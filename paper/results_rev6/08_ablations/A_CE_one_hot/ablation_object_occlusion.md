# A-CE: Gaussian target (reference) vs strict one-hot BCE — object_occlusion (real object occluders (n, SAM2 bank))

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| real object occluders (n, SAM2 bank) | reference           | A-CE                |
|--------------------------------------|---------------------|---------------------|
| 0                                    | 97.0 / 99.8 / 0.421 | 95.7 / 99.3 / 0.405 |
| 1                                    | 98.3 / 98.8 / 0.416 | 96.7 / 97.7 / 0.402 |
| 2                                    | 97.9 / 96.9 / 0.429 | 93.9 / 95.2 / 0.412 |
| 3                                    | 98.8 / 96.9 / 0.418 | 93.9 / 97.9 / 0.402 |
| 4                                    | 99.1 / 98.1 / 0.425 | 96.2 / 98.2 / 0.409 |
