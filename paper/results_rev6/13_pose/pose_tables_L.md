---
title: "Conv-ChArT --- pose accuracy, four systems"
geometry: margin=1.6cm
fontsize: 10pt
---

Rotation in **degrees** (deg) --- the geodesic angle between true and estimated board
rotation. Translation in **board squares** (sq): the evaluation set fixes
square\_length\_m = 1.0, so the unit is board squares, not metres. 1500 images.

**All four algorithms share the identical downstream pipeline** --- lattice gate, recovery
pass, then PnP (IPPE) --- so the only thing that differs between rows is the corners
each detector produced. Classical OpenCV and Deep ChArUco are therefore *not* using
their own pose solvers; these numbers will not match their native output, by design.
A **refusal** (a pose the pipeline declines to corroborate) is counted in the solve
rate and excluded from the error statistics --- it is a correct outcome, not an error.

| algorithm | solve % | rot median (deg) | rot mean (deg) | rot p95 (deg) | trans median (sq) | trans mean (sq) | trans p95 (sq) |
|:-------------------------|-------:|--------:|-------:|-------:|---------:|--------:|-------:|
| Conv-ChArT (coarse) | 95.5 | 0.362 | 0.902 | 2.311 | 0.0289 | 0.0746 | 0.2722 |
| **Conv-ChArT + refiner** | 95.5 | 0.133 | 0.380 | 1.035 | 0.0101 | 0.0347 | 0.1448 |
| Deep ChArUco (fine-tuned) | 94.5 | 0.873 | 10.771 | 90.468 | 0.0775 | 1.2615 | 7.3668 |
| classical OpenCV | 54.9 | 0.148 | 0.991 | 2.209 | 0.0191 | 0.0720 | 0.2230 |
| *lead: refined vs Deep ChArUco* | 1.0x | 6.5x | 28.4x | 87.4x | 7.7x | 36.3x | 50.9x |
| *lead: refined vs classical* | 1.7x | 1.1x | 2.6x | 2.1x | 1.9x | 2.1x | 1.5x |

# Reading the table

- **The two `lead` rows read as "how many times better our refined arm is"**, computed
  per column in that column's own direction: solve % is higher-is-better, every error
  column is lower-is-better. A value below 1.0 means we are behind on that column.
- **The refiner's contribution at pose level**: 2.7x on rotation median and
  2.9x on translation median. Pose is what a docking controller consumes.
- **Classical OpenCV is accurate when it succeeds** --- its median rotation is
  competitive with our coarse detector --- but it only solves 55% of frames. The
  claim against it is coverage, not precision.
- **Deep ChArUco has a heavy tail.** Its median is respectable, but p95 rotation is
  90 degrees: confidently wrong poses on several percent of frames. For a
  docking robot a wrong pose is more dangerous than no pose, which makes the p95
  column more operationally important than the median.
- **Solve rate and p95 must be read together.** Solve rate alone flatters Deep
  ChArUco; p95 alone hides that classical refuses nearly half the time.
