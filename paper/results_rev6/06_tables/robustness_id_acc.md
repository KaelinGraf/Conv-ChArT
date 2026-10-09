# id_acc by factor and method (higher is better)

Cells are `mean (worst)` across the factor's swept range.

**Reading notes.** Classical OpenCV only reports corners it has already identified, so its ID accuracy is ~100% among matched corners by construction and is not comparable -- read its **recall** instead (its ID cells show `n/c`). Deep ChArUco is **unrefined** (their RefineNet is not run), so its localisation belongs against our **coarse** column, not our refined one. All arms are scored on identical frames.

| Factor           | Ours (coarse) | Ours (refined) | Deep ChArUco | Classical |
|------------------|---------------|----------------|--------------|-----------|
| brightness       | 98.0 (95.4)   | 98.0 (95.4)    | 85.0 (77.8)  | n/c       |
| contrast         | 97.7 (95.1)   | 97.7 (95.1)    | 86.0 (84.4)  | n/c       |
| darkness         | 94.3 (83.6)   | 94.3 (83.6)    | 56.6 (17.1)  | n/c       |
| defocus_blur     | 98.6 (97.1)   | 98.6 (97.1)    | 87.0 (85.0)  | n/c       |
| diff_ambient     | 99.2 (98.0)   | 99.2 (98.0)    | 81.8 (74.2)  | n/c       |
| diff_ghosting    | 99.3 (97.6)   | 99.3 (97.6)    | 83.2 (77.5)  | n/c       |
| diff_ratio       | 99.0 (97.9)   | 99.0 (97.9)    | 78.9 (66.5)  | n/c       |
| distance         | 98.8 (96.6)   | 98.8 (96.6)    | 79.0 (49.5)  | n/c       |
| distance_extrap  | 65.8 (0.0)    | 65.8 (0.0)     | 32.9 (0.0)   | n/c       |
| droplets         | 98.0 (97.0)   | 98.1 (97.1)    | 86.7 (85.6)  | n/c       |
| ink_contrast     | 95.9 (86.6)   | 95.9 (86.7)    | 80.0 (68.3)  | n/c       |
| motion_blur      | 98.3 (94.6)   | 98.3 (94.9)    | 85.1 (78.8)  | n/c       |
| object_occlusion | 98.6 (97.4)   | 98.6 (97.4)    | 87.3 (85.1)  | n/c       |
| occlusion        | 98.3 (96.5)   | 98.3 (96.5)    | 85.0 (82.5)  | n/c       |
| rotation         | 99.4 (98.7)   | 99.4 (98.7)    | 91.1 (89.9)  | n/c       |
| sensor_noise_K   | 96.2 (88.5)   | 96.2 (88.4)    | 81.7 (73.8)  | n/c       |
| specular         | 98.4 (96.8)   | 98.4 (96.8)    | 84.9 (82.1)  | n/c       |
| tilt             | 99.7 (98.9)   | 99.7 (98.9)    | 92.9 (91.0)  | n/c       |
| vignette         | 98.9 (97.6)   | 98.9 (97.6)    | 87.0 (85.9)  | n/c       |
