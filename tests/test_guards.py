import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import importlib.util
import json
import shutil

import pytest

from dcc.dataset import load_config
from dcc.trainutil import generator_fingerprint

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
FINGERPRINT_RELS = ("dcc/board.py", "dcc/synth.py", "dcc/targets.py", "dcc/dataset.py")


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def preflight():
    """tools.preflight, loaded by absolute path -- an unrelated detectron2
    'tools' package on sys.path shadows any `import tools.preflight`."""
    spec = importlib.util.spec_from_file_location("_preflight_under_test", ROOT / "tools" / "preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_report(path, fingerprint, all_gates_passed=True):
    path.write_text(json.dumps({"generator_fingerprint": fingerprint, "all_gates_passed": all_gates_passed}))


def test_fingerprint_stable_and_keys(cfg):
    fp1 = generator_fingerprint(cfg, ROOT)
    fp2 = generator_fingerprint(cfg, ROOT)
    assert fp1 == fp2
    assert set(fp1["files"]) == set(FINGERPRINT_RELS)
    assert isinstance(fp1["config_sha1"], str) and len(fp1["config_sha1"]) == 40
    for h in fp1["files"].values():
        assert isinstance(h, str) and len(h) == 40


def test_fingerprint_detects_single_file_change(cfg, tmp_path):
    (tmp_path / "dcc").mkdir()
    for rel in FINGERPRINT_RELS:
        shutil.copy(ROOT / rel, tmp_path / rel)

    fp_before = generator_fingerprint(cfg, tmp_path)
    doctored = tmp_path / "dcc" / "synth.py"
    doctored.write_text(doctored.read_text() + "\n# doctored for test_fingerprint_detects_single_file_change\n")
    fp_after = generator_fingerprint(cfg, tmp_path)

    assert fp_after["files"]["dcc/synth.py"] != fp_before["files"]["dcc/synth.py"]
    for rel in FINGERPRINT_RELS:
        if rel != "dcc/synth.py":
            assert fp_after["files"][rel] == fp_before["files"][rel]
    assert fp_after["config_sha1"] == fp_before["config_sha1"]
    assert fp_after != fp_before


def test_generator_lock_pass(cfg, preflight, tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path, generator_fingerprint(cfg, ROOT))

    status, numbers = preflight.check_generator_lock(cfg, ROOT, report_path)
    assert status == "PASS"


def test_generator_lock_fingerprint_mismatch(cfg, preflight, tmp_path):
    report_path = tmp_path / "report.json"
    fp = generator_fingerprint(cfg, ROOT)
    fp["files"]["dcc/synth.py"] = "0" * 40  # doesn't match the real repo's synth.py hash
    _write_report(report_path, fp)

    status, numbers = preflight.check_generator_lock(cfg, ROOT, report_path)
    assert status == "FAIL"
    assert numbers["reason"] == "fingerprint_mismatch"
    assert numbers["changed"] == ["dcc/synth.py"]


def test_generator_lock_no_audit_report(cfg, preflight, tmp_path):
    status, numbers = preflight.check_generator_lock(cfg, ROOT, tmp_path / "does_not_exist.json")
    assert status == "FAIL"
    assert numbers["reason"] == "no_audit_report"


def test_generator_lock_audit_gates_failed(cfg, preflight, tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path, generator_fingerprint(cfg, ROOT), all_gates_passed=False)

    status, numbers = preflight.check_generator_lock(cfg, ROOT, report_path)
    assert status == "FAIL"
    assert numbers["reason"] == "audit_gates_failed"


def test_generator_lock_no_fingerprint_in_report(cfg, preflight, tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"some_pre_existing_field": True}))

    status, numbers = preflight.check_generator_lock(cfg, ROOT, report_path)
    assert status == "FAIL"
    assert numbers["reason"] == "no_fingerprint_in_report"
