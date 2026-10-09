# Robustness: recall from each method's BEST step to its WORST, per factor

`best -> worst (drop, pp)`. This is the robustness number proper -- a method can lead on mean accuracy and still collapse at one end of a factor. Best-to-worst rather than first-to-last step because several axes (rotation, brightness) sweep outward in both directions from benign.

**Reading notes.** Classical OpenCV only reports corners it has already identified, so its ID accuracy is ~100% among matched corners by construction and is not comparable -- read its **recall** instead (its ID cells show `n/c`). Deep ChArUco is **unrefined** (their RefineNet is not run), so its localisation belongs against our **coarse** column, not our refined one. All arms are scored on identical frames.

| Factor           | Ours (coarse)         | Ours (refined)        | Deep ChArUco          | Classical             |
|------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| brightness       | 99.9 -> 91.0  (-9.0)  | 99.9 -> 90.9  (-9.0)  | 90.6 -> 74.1  (-16.5) | 38.9 -> 19.3  (-19.6) |
| contrast         | 99.4 -> 97.3  (-2.1)  | 99.3 -> 97.2  (-2.1)  | 88.1 -> 81.2  (-6.9)  | 42.9 -> 32.9  (-10.0) |
| darkness         | 99.3 -> 73.5  (-25.8) | 99.3 -> 73.5  (-25.8) | 87.2 -> 0.2  (-87.0)  | 37.9 -> 0.0  (-37.9)  |
| defocus_blur     | 99.4 -> 96.9  (-2.5)  | 99.4 -> 96.2  (-3.2)  | 88.8 -> 84.9  (-3.9)  | 48.9 -> 19.1  (-29.9) |
| diff_ambient     | 99.9 -> 98.0  (-1.9)  | 99.9 -> 98.0  (-1.8)  | 89.3 -> 83.1  (-6.3)  | 39.2 -> 21.2  (-18.0) |
| diff_ghosting    | 99.9 -> 98.2  (-1.6)  | 99.9 -> 98.2  (-1.6)  | 89.9 -> 86.3  (-3.6)  | 37.8 -> 27.0  (-10.8) |
| diff_ratio       | 99.8 -> 97.9  (-1.9)  | 99.7 -> 97.9  (-1.8)  | 89.2 -> 75.7  (-13.5) | 40.4 -> 9.8  (-30.6)  |
| distance         | 99.8 -> 93.5  (-6.3)  | 99.7 -> 93.5  (-6.2)  | 94.0 -> 42.7  (-51.3) | 56.7 -> 0.6  (-56.2)  |
| distance_extrap  | 99.0 -> 0.0  (-99.0)  | 98.9 -> 0.0  (-98.9)  | 89.9 -> 0.0  (-89.9)  | 47.4 -> 0.0  (-47.4)  |
| droplets         | 99.6 -> 97.1  (-2.6)  | 99.5 -> 97.0  (-2.5)  | 87.2 -> 83.0  (-4.1)  | 39.0 -> 31.9  (-7.1)  |
| ink_contrast     | 98.3 -> 94.7  (-3.6)  | 98.3 -> 94.3  (-3.9)  | 85.7 -> 58.5  (-27.2) | 39.2 -> 11.3  (-27.9) |
| motion_blur      | 99.4 -> 95.9  (-3.5)  | 99.4 -> 95.1  (-4.3)  | 88.6 -> 76.8  (-11.8) | 48.9 -> 12.5  (-36.4) |
| object_occlusion | 99.9 -> 96.7  (-3.1)  | 99.8 -> 96.6  (-3.1)  | 87.5 -> 85.7  (-1.8)  | 42.5 -> 31.6  (-11.0) |
| occlusion        | 99.8 -> 96.6  (-3.2)  | 99.8 -> 96.5  (-3.2)  | 89.0 -> 82.7  (-6.3)  | 41.1 -> 25.1  (-16.0) |
| rotation         | 98.9 -> 97.4  (-1.5)  | 98.9 -> 97.4  (-1.5)  | 94.9 -> 89.5  (-5.4)  | 55.2 -> 48.8  (-6.4)  |
| sensor_noise_K   | 99.3 -> 91.6  (-7.7)  | 99.2 -> 90.0  (-9.2)  | 88.2 -> 13.8  (-74.5) | 41.4 -> 2.6  (-38.9)  |
| specular         | 99.2 -> 96.0  (-3.2)  | 99.1 -> 95.9  (-3.1)  | 87.6 -> 84.0  (-3.7)  | 36.1 -> 30.3  (-5.8)  |
| tilt             | 99.0 -> 94.4  (-4.6)  | 99.0 -> 94.3  (-4.8)  | 95.6 -> 90.0  (-5.6)  | 55.6 -> 51.6  (-4.0)  |
| vignette         | 99.0 -> 97.2  (-1.8)  | 98.9 -> 97.1  (-1.8)  | 89.1 -> 80.4  (-8.7)  | 40.4 -> 34.2  (-6.2)  |
