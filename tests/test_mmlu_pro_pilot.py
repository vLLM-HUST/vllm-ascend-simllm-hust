"""Validate the diagnostic MMLU-Pro answer parser."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "mmlu_pro_pilot.py"
SPEC = importlib.util.spec_from_file_location("mmlu_pro_pilot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_strict_letter_accepts_only_one_answer() -> None:
    assert MODULE.grade_prediction(" I ", "I") == ("I", True)
    assert MODULE.grade_prediction("I.", "I") == ("I", True)
    assert MODULE.grade_prediction("A", "I") == ("A", False)
    assert MODULE.grade_prediction("I or H", "I") == (None, False)
    assert MODULE.grade_prediction("The answer is I", "I") == (None, False)


def test_summary_requires_aligned_tasks_and_counts_reuse(tmp_path: Path) -> None:
    baseline = tmp_path / "b0.jsonl"
    plugin = tmp_path / "b1.jsonl"
    log = tmp_path / "plugin.log"
    output = tmp_path / "summary.json"
    baseline.write_text(json.dumps({"id": "q1", "answer": "A", "correct": True}) + "\n")
    plugin.write_text(
        json.dumps({"id": "q1", "answer": "A", "correct": False, "request_id": "r1"})
        + "\n"
    )
    log.write_text("SimLLM: reusing 128 semantic tokens for request r1\n")
    MODULE.summarize(
        SimpleNamespace(baseline=baseline, plugin=plugin, plugin_log=log, output=output)
    )
    result = json.loads(output.read_text())
    assert result["runs"]["baseline"]["accuracy"] == 1
    assert result["runs"]["plugin"]["accuracy"] == 0
    assert result["plugin_control_path"] == {
        "requests_with_reuse_event": 1,
        "reused_tokens": 128,
    }
