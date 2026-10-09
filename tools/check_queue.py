"""Validate runs/queue.txt BEFORE the supervisor launches anything.

WHY THIS EXISTS. On 2026-08-01 two sigma_cls arms were queued as a pair. sigma_cls is a
loader-determining key -- the worker renders the class TARGET, so two values cannot share a
stream -- and train_pair refused at launch, exactly as designed. The constraint had been checked
by hand an hour earlier and the check's own output said "sigma_cls arm pairs with them: False
(correct)". The knowledge was there; the queue edit still violated it.

train_pair's guard is the backstop and it worked. This is the cheaper gate: the supervisor's
DRY_RUN prints what it WOULD launch but validates nothing about whether those groups are legal,
so a bad line looks fine until a slot frees and a launch dies. Run this after every queue edit.

Checks, per line:
  1. every config path exists and parses
  2. every arm builds a DetectorNet and produces the right output shapes
  3. all arms in a '|' group agree on train_pair.shared_signature -- the actual launch condition
  4. steps parse as ints; resume checkpoints, if named, exist

Exits nonzero on any failure, listing the disagreeing keys, so it can gate a queue edit.
"""
import argparse
import importlib.util as iu
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _train_pair():
    """Import train_pair as a module for its shared_signature -- the SAME function the launcher
    uses. Re-implementing the key list here would let this check and the launcher drift apart,
    which is the one failure mode that would make this tool worse than useless."""
    spec = iu.spec_from_file_location("tp", Path(__file__).with_name("train_pair.py"))
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queue", default="runs/queue.txt")
    p.add_argument("--no-build", action="store_true",
                   help="skip the model-build check (fast path; pairing is still validated)")
    a = p.parse_args()

    import torch

    from dcc.dataset import load_config
    from dcc.model import DetectorNet, detector_kwargs
    tp = _train_pair()

    bad = 0
    for lineno, raw in enumerate(Path(a.queue).read_text().splitlines(), 1):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        groups = [g.strip() for g in line.split("|")]
        arms, sigs = [], []
        for g in groups:
            parts = g.split(":")
            if len(parts) < 3:
                print(f"  line {lineno}: BAD FORMAT {g!r}"); bad += 1; continue
            name, cfgp, steps = parts[0], parts[1], parts[2]
            resume = parts[3] if len(parts) > 3 and parts[3] else None
            if not Path(cfgp).is_file():
                print(f"  line {lineno}: {name}: config {cfgp} MISSING"); bad += 1; continue
            try:
                int(steps)
            except ValueError:
                print(f"  line {lineno}: {name}: steps {steps!r} not an int"); bad += 1; continue
            if resume and not Path(resume).is_file():
                print(f"  line {lineno}: {name}: resume {resume} MISSING"); bad += 1; continue
            cfg = load_config(cfgp)
            if not a.no_build:
                W, H = cfg["input_size"]
                net = DetectorNet(H, W, n_cls=16, **detector_kwargs(cfg))
                with torch.no_grad():
                    hm, cl = net(torch.randn(1, 1, H, W))
                if hm.shape[-2:] != (H, W) or cl.shape[-2:] != (H // 4, W // 4):
                    print(f"  line {lineno}: {name}: BAD SHAPES {tuple(hm.shape)} {tuple(cl.shape)}")
                    bad += 1
                    continue
            arms.append(name)
            sigs.append(tp.shared_signature(cfg))
        if len(sigs) > 1:
            ref = sigs[0]
            for name, s in zip(arms[1:], sigs[1:]):
                diff = sorted(k for k in set(ref) | set(s) if ref.get(k) != s.get(k))
                if diff:
                    print(f"  line {lineno}: ILLEGAL PAIR -- {arms[0]} vs {name} disagree on {diff}")
                    print("      a '|' group shares ONE DataLoader; these cannot. Split the line.")
                    bad += 1
        if arms:
            print(f"  line {lineno}: OK  {len(arms)} arm(s): {', '.join(arms)}")
    print("\nQUEUE OK" if not bad else f"\n{bad} PROBLEM(S) -- fix runs/queue.txt before launching")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
