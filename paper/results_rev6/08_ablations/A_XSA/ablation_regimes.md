# A-XSA: standard MHSA (reference) vs exclusive self-attention — per-regime ID accuracy (%), deltas vs `reference`

**Judged per regime, not on aggregate.** The aggregate averages away exactly the structure the hypothesis names. Distance regimes are read at their named `s`; the occlusion regimes are read at each arm's WORST step, since that is where an occlusion claim lives.

Identical frames across arms; coarse arm only.

| Regime                  | reference | XSA            |
|-------------------------|-----------|----------------|
| FAR  (s=12)             | 96.19     | 96.28  (+0.10) |
| FAR  (s=16)             | 93.98     | 94.28  (+0.31) |
| VERY CLOSE  (s=96)      | 99.73     | 99.91  (+0.18) |
| VERY CLOSE  (s=128)     | 99.20     | 98.75  (-0.45) |
| OCCLUDED  (max holes)   | 96.64     | 96.62  (-0.02) |
| OCCLUDED  (max objects) | 96.88     | 96.07  (-0.80) |
