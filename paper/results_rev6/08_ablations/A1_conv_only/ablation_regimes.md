# A1: bottleneck attention (reference) vs conv-only (attn_blocks=0) — per-regime ID accuracy (%), deltas vs `reference`

**Judged per regime, not on aggregate.** The aggregate averages away exactly the structure the hypothesis names. Distance regimes are read at their named `s`; the occlusion regimes are read at each arm's WORST step, since that is where an occlusion claim lives.

Identical frames across arms; coarse arm only.

| Regime                  | reference | conv-only      |
|-------------------------|-----------|----------------|
| FAR  (s=12)             | 96.19     | 97.52  (+1.34) |
| FAR  (s=16)             | 93.98     | 94.42  (+0.44) |
| VERY CLOSE  (s=96)      | 99.73     | 98.91  (-0.82) |
| VERY CLOSE  (s=128)     | 99.20     | 91.82  (-7.39) |
| OCCLUDED  (max holes)   | 96.64     | 96.32  (-0.32) |
| OCCLUDED  (max objects) | 96.88     | 95.92  (-0.96) |
