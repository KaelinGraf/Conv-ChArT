import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def pf():
    """tools.plot_figures, loaded by absolute path -- an unrelated
    detectron2 'tools' package shadows any `import tools.plot_figures`
    (same footgun as every other tools/*.py test fixture in this repo)."""
    spec = importlib.util.spec_from_file_location("_plot_figures_under_test", ROOT / "tools" / "plot_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


# --------------------------------------------------------- load_metrics_series

def test_dedupes_replayed_segment_last_occurrence_wins(pf, tmp_path, capsys):
    """Synthetic reproduction of the real 2026-07-28 incident: a run writes
    steps 1..5, crashes, resumes from a checkpoint and re-writes 3..7 (its
    own steps 3/4/5 duplicated, then genuinely new steps 6/7). The dead
    run's values for 3/4/5 must be DISCARDED in favour of the replay's --
    last occurrence in file order wins -- and the series must come back
    step-sorted, not in write order (which would show a backwards jump)."""
    p = tmp_path / "metrics.jsonl"
    _write_jsonl(p, [
        {"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.9}, {"step": 3, "loss": 0.8},
        {"step": 4, "loss": 0.7}, {"step": 5, "loss": 0.6},
        # crash here; resume from a checkpoint before step 3 replays 3..7
        {"step": 3, "loss": 0.75}, {"step": 4, "loss": 0.65}, {"step": 5, "loss": 0.55},
        {"step": 6, "loss": 0.5}, {"step": 7, "loss": 0.45},
    ])
    series = pf.load_metrics_series(p, "loss")
    assert [s for s, _ in series] == [1, 2, 3, 4, 5, 6, 7]  # sorted, no duplicate steps
    by_step = dict(series)
    # the REPLAY's values (second pass) must win, not the dead run's
    assert by_step[3] == 0.75
    assert by_step[4] == 0.65
    assert by_step[5] == 0.55
    assert by_step[1] == 1.0 and by_step[6] == 0.5 and by_step[7] == 0.45  # untouched steps unaffected


def test_logs_dedupe_and_backtrack_never_silent(pf, tmp_path, capsys):
    p = tmp_path / "metrics.jsonl"
    _write_jsonl(p, [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.9},
                      {"step": 1, "loss": 0.95}])  # one duplicate, one backtrack
    pf.load_metrics_series(p, "loss")
    out = capsys.readouterr().out
    assert "deduped 1 record" in out
    assert "backtrack detected at step 1" in out


def test_no_backtrack_no_log(pf, tmp_path, capsys):
    """A clean, monotonic file (the common case) must not print anything --
    the dedupe/backtrack log exists to flag the unusual case, not to add
    noise to every run."""
    p = tmp_path / "metrics.jsonl"
    _write_jsonl(p, [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.9}, {"step": 3, "loss": 0.8}])
    series = pf.load_metrics_series(p, "loss")
    assert series == [(1, 1.0), (2, 0.9), (3, 0.8)]
    assert capsys.readouterr().out == ""


def test_dedupe_is_scoped_per_key_not_per_record(pf, tmp_path):
    """A single step can legitimately carry two DIFFERENT record kinds in
    one real pass (a per-step train_fields log and a periodic val log both
    at the same step) -- deduping the whole line by step would silently
    drop one kind. Scoping to one key path at a time must keep both."""
    p = tmp_path / "metrics.jsonl"
    _write_jsonl(p, [
        {"step": 100, "loss": 0.5},
        {"step": 100, "val": {"m04": {"accuracy": 0.97}}},
    ])
    loss_series = pf.load_metrics_series(p, "loss")
    val_series = pf.load_metrics_series(p, "val.m04.accuracy")
    assert loss_series == [(100, 0.5)]
    assert val_series == [(100, 0.97)]


def test_missing_key_and_missing_step_are_skipped(pf, tmp_path):
    p = tmp_path / "metrics.jsonl"
    _write_jsonl(p, [{"step": 1, "loss": 1.0}, {"step": 2, "lr": 3e-4}, {"loss": 0.5}])
    assert pf.load_metrics_series(p, "loss") == [(1, 1.0)]


def test_get_path_nested_and_missing(pf):
    assert pf._get_path({"a": {"b": 5}}, "a.b") == 5
    assert pf._get_path({"a": {"b": 5}}, "a.c") is None
    assert pf._get_path({"a": 5}, "a.b") is None  # a is not a dict, can't descend
    assert pf._get_path({}, "a") is None
