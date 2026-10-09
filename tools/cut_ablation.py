"""Cut an ablation config from a base, changing EXACTLY the keys you name -- and prove it.

WHY THIS EXISTS. dcc/dataset.py:18 load_config is a bare yaml.safe_load: there is no
inheritance, so every arm is a full copy of its base. On 2026-07-29 a set of arms was
hand-cut from a stale base and silently differed in TEN keys, so the conv-only arm trained
on easier rev-5 data and "beat" the reference -- an hour of GPU and a false architectural
finding. CLAUDE.md's rule ("VERIFY the one-key invariant PROGRAMMATICALLY BEFORE LAUNCHING",
"regenerate ablation configs after ANY edit to the base config") is enforced here rather
than left to whoever is cutting the next arm at 2am.

The copy is TEXTUAL, not a yaml round-trip: the base's comments are load-bearing
documentation and a round-trip would drop every one of them. Overrides rewrite the key's
own line in place (or append it under a header if the key is absent from the base).

    python tools/cut_ablation.py --base configs/rev640.yaml --out configs/abl_foo.yaml \
        --set focal_beta=0 --note "A-BETA0: isolates the Gaussian penalty discount."

Exits nonzero if the resulting flat-key diff against the base is anything other than the
keys you asked for -- which is the whole point.
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcc.trainutil import flat_cfg as flat  # single canonical leaf-flattener; see its docstring


def parse_value(s):
    """YAML-parse the RHS so --set width_mult=0.5 gives a float, =false a bool, =ce a str."""
    return yaml.safe_load(s)


def apply_override(text, key, raw):
    """Rewrite `key:` in place, preserving any trailing comment on that line. Appends the key
    if absent.

    A DOTTED key (`synth.train_seed`, `train.lr`) rewrites a nested leaf, SCOPED TO ITS PARENT
    BLOCK: the parent line is located at column 0 and the block runs until the next unindented
    line, so `train.lr` cannot touch `refiner_train.lr` -- both exist in every config. Both the
    parent and the leaf-within-parent must be unique or the cut is refused rather than guessed
    at. Dotted keys are never appended: a leaf absent from the base means the wrong base was
    chosen. The gate at the bottom of main() re-parses the result either way, so a wrong-scope
    rewrite could not survive even if this matching were wrong."""
    if "." in key:
        # SECTION-SCOPED, not merely leaf-unique. `train.lr` must not match refiner_train's lr --
        # both exist in every config, so a bare leaf search finds 2 lines and (correctly) refused,
        # which blocked every legitimate hyperparameter cut. Slice the parent's block by
        # indentation and rewrite inside it, so the two sections stay independent.
        parent, leaf = key.rsplit(".", 1)
        lines = text.split("\n")
        starts = [i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(parent)}\s*:", ln)]
        if len(starts) != 1:
            raise SystemExit(f"--set {key}: parent {parent!r} matches {len(starts)} blocks; must be unique")
        i0 = starts[0]
        i1 = next((j for j in range(i0 + 1, len(lines))
                   if lines[j].strip() and not lines[j][:1].isspace()), len(lines))
        hits = [j for j in range(i0 + 1, i1) if re.match(rf"^\s+{re.escape(leaf)}\s*:", lines[j])]
        if len(hits) != 1:
            raise SystemExit(f"--set {key}: leaf {leaf!r} matches {len(hits)} lines inside {parent!r}")
        j = hits[0]
        m = re.match(rf"^(\s*{re.escape(leaf)}\s*:\s*)([^#\n]*)(.*)$", lines[j])
        lines[j] = f"{m.group(1)}{raw}{(' ' + m.group(3).strip()) if m.group(3).strip() else ''}"
        return "\n".join(lines)
    pat = re.compile(rf"^({re.escape(key)}\s*:\s*)([^#\n]*)(.*)$", re.M)
    if pat.search(text):
        return pat.sub(lambda m: f"{m.group(1)}{raw}{(' ' + m.group(3).strip()) if m.group(3).strip() else ''}", text, count=1)
    return text.rstrip("\n") + f"\n\n# ABLATION LEVER (added by tools/cut_ablation.py -- absent from the base).\n{key}: {raw}\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="top-level key to override; repeatable")
    p.add_argument("--note", default="", help="one-line rationale for the header")
    a = p.parse_args()

    base_text = Path(a.base).read_text()
    base_cfg = yaml.safe_load(base_text)

    overrides = {}
    for s in a.set:
        if "=" not in s:
            raise SystemExit(f"--set {s!r}: need KEY=VALUE")
        k, _, v = s.partition("=")
        overrides[k.strip()] = v.strip()

    text = base_text
    for k, raw in overrides.items():
        text = apply_override(text, k, raw)

    header = [f"# CUT FROM {a.base} by tools/cut_ablation.py -- do not hand-edit; regenerate.",
              f"# Changes EXACTLY {len(overrides)} key(s): " +
              ", ".join(f"{k}={v}" for k, v in overrides.items()) + ".",
              "# The one-key invariant below was verified programmatically at cut time; re-run this",
              f"# command after ANY edit to {a.base} (CLAUDE.md ablation protocol)."]
    if a.note:
        header += ["#"] + [f"# {ln}" for ln in a.note.split("\n")]
    text = "\n".join(header) + "\n" + text

    # THE GATE. Parse what we just built and diff its leaves against the base's.
    new_cfg = yaml.safe_load(text)
    fb, fn = flat(base_cfg), flat(new_cfg)
    diff = {k: (fb.get(k, "<absent>"), fn[k]) for k in fn if fb.get(k) != fn[k]}
    diff.update({k: (fb[k], "<absent>") for k in fb if k not in fn})

    want = {k: parse_value(v) for k, v in overrides.items()}
    unexpected = {k: v for k, v in diff.items() if k not in want}
    wrong = {k: (want[k], fn.get(k, "<absent>")) for k in want if fn.get(k) != want[k]}
    if unexpected or wrong:
        for k, (was, now) in sorted(unexpected.items()):
            print(f"  UNEXPECTED  {k}: {was!r} -> {now!r}")
        for k, (wanted, got) in sorted(wrong.items()):
            print(f"  NOT APPLIED {k}: wanted {wanted!r}, got {got!r}")
        raise SystemExit(f"ABORT: {a.out} not written -- diff is not exactly the requested keys")

    Path(a.out).write_text(text)
    print(f"-> {a.out}  ({len(diff)} key(s) vs {a.base}: " +
          ", ".join(f"{k} {v[0]!r}->{v[1]!r}" for k, v in sorted(diff.items())) + ")")


if __name__ == "__main__":
    main()
