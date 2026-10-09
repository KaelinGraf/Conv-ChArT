# Ablations — folder layout

One folder per ablation, self-contained. Comparison figures and tables live INSIDE
the ablation's folder, never loose at this level (Kaelin, 2026-07-29: they were
"just floating in the ablations folder", which does not scale to four arms).

```
08_ablations/
  00_README_LAYOUT.md          this file
  _reference_sweeps/           the SHARED reference arm's sweep JSON
  A1_conv_only/                one folder per ablation
    README.md                  what it tests, the one key, the verdict
    sweeps/                    THIS arm's sweep JSON
    ablation_<factor>.png      curves, one panel per metric, one curve per arm
    ablation_<factor>.md       per-step table for that factor
    ablation_regimes.md        THE table: the regimes the hypothesis names
    ablation_all_steps.csv     every arm at every step
  A_CE_one_hot/                (pending)
  A_XSA/                       (pending)
  A_SIGMA1/                    (pending)
```

**`_reference_sweeps/` is shared on purpose.** The ablation protocol pins ONE
reference run serving every arm (`abl_reference_50k_rev6`), so its sweeps are
computed once and reused. Re-running them per ablation would cost GPU and — worse —
risk the arms being compared against *different* reference measurements.

## Regenerating an ablation

```
python tools/plot_ablation.py \
    reference=paper/results_rev6/08_ablations/_reference_sweeps \
    <label>=paper/results_rev6/08_ablations/<ABLATION>/sweeps \
    --out-dir paper/results_rev6/08_ablations/<ABLATION> \
    --title "<ablation>: reference vs <arm>"
```

The FIRST arm listed is the reference that deltas are taken against. Reads JSON
only — no model, no GPU — so restyling never needs a re-run.

## Producing a new ablation's sweeps

Both arms must be swept on **identical seeds**, and each arm needs **its own
config** — `robustness_sweep.py` builds the model from the config, so an arm with a
different architecture key fails to load against the base config:

```
Missing key(s) in state_dict: "blocks.0.n1.weight", ...
```

That is exactly what happened to A1 conv-only on the first attempt (`attn_blocks: 0`
against `rev640.yaml`'s 2). The configs differ in one key and none of them is a data
key, so the generated frames stay bit-identical across arms.

```
python tools/robustness_sweep.py --config configs/<arm>.yaml \
    --ckpt runs/<arm_run>/ckpt_latest.pt --refiner-ckpt runs/refiner_rev6/ckpt_0013000.pt \
    --factors distance distance_extrap occlusion object_occlusion \
    --seed 20260729 --n 100 --out paper/results_rev6/08_ablations/<ABLATION>/sweeps
```

## Note on the arm read

Curves use the **coarse** arm. The refiner is shared and identical across ablation
arms, so a refined-arm comparison would re-measure the refiner rather than the
architecture under test.
