# err_median by factor and method (lower is better)

Cells are `mean (worst)` across the factor's swept range.

**Reading notes.** Classical OpenCV only reports corners it has already identified, so its ID accuracy is ~100% among matched corners by construction and is not comparable -- read its **recall** instead (its ID cells show `n/c`). Deep ChArUco is **unrefined** (their RefineNet is not run), so its localisation belongs against our **coarse** column, not our refined one. All arms are scored on identical frames.

| Factor           | Ours (coarse) | Ours (refined) | Deep ChArUco  | Classical     |
|------------------|---------------|----------------|---------------|---------------|
| brightness       | 0.424 (0.445) | 0.104 (0.124)  | 1.595 (1.628) | 0.718 (0.724) |
| contrast         | 0.422 (0.433) | 0.091 (0.110)  | 1.614 (1.650) | 0.716 (0.725) |
| darkness         | 0.449 (0.506) | 0.268 (0.506)  | 1.452 (1.616) | 0.420 (0.716) |
| defocus_blur     | 0.425 (0.446) | 0.087 (0.134)  | 1.637 (1.674) | 0.707 (0.716) |
| diff_ambient     | 0.518 (0.596) | 0.262 (0.360)  | 1.633 (1.687) | 0.751 (0.783) |
| diff_ghosting    | 0.495 (0.566) | 0.242 (0.293)  | 1.648 (1.697) | 0.720 (0.750) |
| diff_ratio       | 0.576 (0.717) | 0.346 (0.583)  | 1.648 (1.714) | 0.755 (0.781) |
| distance         | 0.424 (0.435) | 0.099 (0.140)  | 1.619 (1.841) | 0.680 (0.728) |
| distance_extrap  | 0.382 (0.455) | 0.093 (0.146)  | 1.723 (3.290) | 0.498 (0.769) |
| droplets         | 0.423 (0.434) | 0.092 (0.101)  | 1.602 (1.616) | 0.717 (0.726) |
| ink_contrast     | 0.444 (0.497) | 0.162 (0.293)  | 1.638 (1.682) | 0.714 (0.720) |
| motion_blur      | 0.448 (0.542) | 0.182 (0.468)  | 1.638 (1.664) | 0.709 (0.734) |
| object_occlusion | 0.421 (0.436) | 0.095 (0.103)  | 1.600 (1.611) | 0.722 (0.732) |
| occlusion        | 0.420 (0.443) | 0.091 (0.101)  | 1.617 (1.646) | 0.713 (0.719) |
| rotation         | 0.420 (0.431) | 0.090 (0.099)  | 1.587 (1.623) | 0.712 (0.722) |
| sensor_noise_K   | 0.462 (0.609) | 0.263 (0.614)  | 1.621 (1.668) | 0.798 (1.083) |
| specular         | 0.420 (0.424) | 0.097 (0.113)  | 1.609 (1.652) | 0.719 (0.734) |
| tilt             | 0.427 (0.433) | 0.088 (0.098)  | 1.579 (1.608) | 0.719 (0.730) |
| vignette         | 0.420 (0.441) | 0.090 (0.103)  | 1.623 (1.650) | 0.713 (0.721) |
