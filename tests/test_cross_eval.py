import importlib.util
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def ce():
    """tools.cross_eval, loaded by absolute path -- an unrelated detectron2
    'tools' package shadows any `import tools.cross_eval` (same footgun as
    every other tools/*.py test fixture in this repo). Covers only the
    self-contained trainer-health duplicate (see cross_eval.py's module
    docstring for why it can't just import tools/factor_sweep.py -- that
    file doesn't exist in the rev-2 worktree this script must also run
    from); the GPU/dataset-touching parts of main() are exercised manually
    per the standing GO-gated concurrency rules, not by this suite."""
    spec = importlib.util.spec_from_file_location("_cross_eval_under_test", ROOT / "tools" / "cross_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_trainer_health_healthy(ce, tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(json.dumps({"step": 1, "wall": time.time(), "samples_per_s": 150.0}) + "\n")
    ok, age, sps = ce._check_trainer_health(str(p))
    assert ok and age < 5 and sps == 150.0


def test_check_trainer_health_stale_or_slow(ce, tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(json.dumps({"step": 1, "wall": time.time() - 999, "samples_per_s": 150.0}) + "\n")
    ok, age, _sps = ce._check_trainer_health(str(p))
    assert not ok and age > 900

    p.write_text(json.dumps({"step": 1, "wall": time.time(), "samples_per_s": 5.0}) + "\n")
    ok, _age, sps = ce._check_trainer_health(str(p))
    assert not ok and sps == 5.0


def test_check_trainer_health_torn_last_line(ce, tmp_path):
    """A live trainer can crash mid-write on its own last line (memory.md
    2026-07-28 10:45's actual postmortem) -- must fall back to the
    second-to-last complete line rather than raising on a truncated one."""
    p = tmp_path / "metrics.jsonl"
    good = json.dumps({"step": 1, "wall": time.time(), "samples_per_s": 120.0})
    torn = '{"step": 2, "wall": ' + str(time.time())  # deliberately truncated
    p.write_text(good + "\n" + torn)
    ok, _age, sps = ce._check_trainer_health(str(p))
    assert ok and sps == 120.0


def test_parser_requires_ckpt_and_tag(ce):
    """--ckpt/--tag must be required with no default (team-lead 2026-07-28:
    a convenient default is exactly how a stale-checkpoint result gets
    silently reused; --tag is required so a 2x2-table cell can never be
    written without an unambiguous label)."""
    parser = ce.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--ckpt", "/some/path.pt"])  # missing --tag
    args = parser.parse_args(["--ckpt", "/some/path.pt", "--tag", "modelA_on_Y"])
    assert args.ckpt == "/some/path.pt" and args.tag == "modelA_on_Y"
