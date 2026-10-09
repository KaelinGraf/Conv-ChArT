# A-CE: Gaussian target (reference) vs strict one-hot BCE — per-regime ID accuracy (%), deltas vs `reference`

**Judged per regime, not on aggregate.** The aggregate averages away exactly the structure the hypothesis names. Distance regimes are read at their named `s`; the occlusion regimes are read at each arm's WORST step, since that is where an occlusion claim lives.

Identical frames across arms; coarse arm only.

| Regime                  | reference | A-CE           |
|-------------------------|-----------|----------------|
| FAR  (s=12)             | 96.19     | 96.86  (+0.68) |
| FAR  (s=16)             | 93.98     | 98.72  (+4.74) |
| VERY CLOSE  (s=96)      | 99.73     | 99.17  (-0.55) |
| VERY CLOSE  (s=128)     | 99.20     | 95.14  (-4.06) |
| OCCLUDED  (max holes)   | 96.64     | 95.69  (-0.96) |
| OCCLUDED  (max objects) | 96.88     | 95.16  (-1.72) |
