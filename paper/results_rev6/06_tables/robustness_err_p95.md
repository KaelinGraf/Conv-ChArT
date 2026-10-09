# err_p95 by factor and method (lower is better)

Cells are `mean (worst)` across the factor's swept range.

**Reading notes.** Classical OpenCV only reports corners it has already identified, so its ID accuracy is ~100% among matched corners by construction and is not comparable -- read its **recall** instead (its ID cells show `n/c`). Deep ChArUco is **unrefined** (their RefineNet is not run), so its localisation belongs against our **coarse** column, not our refined one. All arms are scored on identical frames.

| Factor           | Ours (coarse) | Ours (refined) | Deep ChArUco  | Classical     |
|------------------|---------------|----------------|---------------|---------------|
| brightness       | 0.843 (1.020) | 0.757 (1.051)  | 2.554 (2.616) | 1.044 (1.117) |
| contrast         | 0.853 (0.970) | 0.768 (0.969)  | 2.586 (2.634) | 1.022 (1.098) |
| darkness         | 1.169 (1.850) | 1.131 (1.850)  | 2.539 (2.663) | 0.554 (1.041) |
| defocus_blur     | 0.845 (0.944) | 0.735 (0.965)  | 2.596 (2.606) | 1.068 (1.253) |
| diff_ambient     | 1.063 (1.271) | 0.999 (1.318)  | 2.740 (2.893) | 1.240 (1.366) |
| diff_ghosting    | 1.029 (1.205) | 0.966 (1.199)  | 2.743 (2.852) | 1.186 (1.298) |
| diff_ratio       | 1.284 (1.742) | 1.251 (1.899)  | 2.811 (3.075) | 1.247 (1.410) |
| distance         | 0.800 (0.860) | 0.694 (0.798)  | 2.663 (3.541) | 1.001 (1.153) |
| distance_extrap  | 0.764 (1.101) | 0.663 (1.173)  | 2.642 (3.868) | 0.757 (1.465) |
| droplets         | 0.829 (0.866) | 0.744 (0.832)  | 2.563 (2.601) | 1.040 (1.169) |
| ink_contrast     | 1.023 (1.426) | 1.043 (1.578)  | 2.659 (2.841) | 1.167 (1.418) |
| motion_blur      | 0.999 (1.660) | 0.941 (1.865)  | 2.615 (2.692) | 1.042 (1.225) |
| object_occlusion | 0.822 (0.864) | 0.690 (0.738)  | 2.556 (2.607) | 1.065 (1.166) |
| occlusion        | 0.812 (0.946) | 0.709 (0.888)  | 2.589 (2.625) | 1.109 (1.286) |
| rotation         | 0.803 (0.895) | 0.674 (0.826)  | 2.467 (2.525) | 1.033 (1.106) |
| sensor_noise_K   | 1.070 (1.788) | 1.151 (2.229)  | 2.661 (2.843) | 1.372 (2.315) |
| specular         | 0.803 (0.879) | 0.715 (0.833)  | 2.575 (2.634) | 1.085 (1.129) |
| tilt             | 0.792 (0.839) | 0.633 (0.711)  | 2.453 (2.475) | 1.079 (1.113) |
| vignette         | 0.802 (0.942) | 0.697 (0.939)  | 2.575 (2.652) | 1.046 (1.100) |
