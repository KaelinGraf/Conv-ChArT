# A-XSA: standard MHSA (reference) vs exclusive self-attention — occlusion (rectangular occluders (n))

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| rectangular occluders (n) | reference           | XSA                 |
|---------------------------|---------------------|---------------------|
| 0                         | 98.1 / 99.9 / 0.419 | 98.0 / 98.7 / 0.418 |
| 2                         | 97.2 / 96.6 / 0.417 | 96.7 / 97.0 / 0.421 |
| 4                         | 96.0 / 98.5 / 0.420 | 96.0 / 98.3 / 0.428 |
| 6                         | 97.2 / 97.3 / 0.416 | 96.9 / 96.6 / 0.414 |
| 8                         | 99.8 / 99.5 / 0.422 | 99.7 / 98.4 / 0.431 |
| 10                        | 95.7 / 99.3 / 0.421 | 95.5 / 98.7 / 0.424 |
