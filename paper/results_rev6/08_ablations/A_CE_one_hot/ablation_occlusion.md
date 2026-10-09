# A-CE: Gaussian target (reference) vs strict one-hot BCE — occlusion (rectangular occluders (n))

Cells are `recall% / ID% / median err px`. Identical frames across arms; coarse arm only (the refiner is shared, so a refined comparison would re-measure the refiner rather than the architecture).

| rectangular occluders (n) | reference           | A-CE                |
|---------------------------|---------------------|---------------------|
| 0                         | 98.1 / 99.9 / 0.419 | 96.1 / 98.7 / 0.403 |
| 2                         | 97.2 / 96.6 / 0.417 | 92.2 / 97.0 / 0.399 |
| 4                         | 96.0 / 98.5 / 0.420 | 93.0 / 98.6 / 0.410 |
| 6                         | 97.2 / 97.3 / 0.416 | 94.9 / 95.7 / 0.403 |
| 8                         | 99.8 / 99.5 / 0.422 | 97.3 / 97.3 / 0.410 |
| 10                        | 95.7 / 99.3 / 0.421 | 93.6 / 96.2 / 0.414 |
