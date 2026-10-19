"""Lifecycle evidence must reject lost, reordered, or changed rollback outputs."""

import json
import os
import runpy
import zipfile
from pathlib import Path

import pytest

PROBE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "benchmarks/verify_inference_rollback.py")
)


def result(request_id="a", text="answer"):
    return {
        "id": request_id,
        "prompt_tokens": 42,
        "completion_tokens": 2,
        "output_text": text,
    }


def test_rollback_comparison_ignores_latency_noise():
    before = {**result(), "latency_s": 1.0, "request_id": "server1"}
    after = {**result(), "latency_s": 1.2, "request_id": "server2"}
    assert PROBE["compare_outputs"]([before], [after])["exactly_equal"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("id", "different"),
        ("prompt_tokens", 41),
        ("completion_tokens", 1),
        ("output_text", "different"),
    ],
)
def test_rollback_comparison_detects_changed_inference(key, value):
    comparison = PROBE["compare_outputs"]([result()], [{**result(), key: value}])
    assert not comparison["exactly_equal"]
    assert comparison["different_ids"] == ["a"]


def test_reordered_results_fail():
    before = [result("a"), result("b")]
    assert not PROBE["compare_outputs"](before, list(reversed(before)))["exactly_equal"]


@pytest.mark.parametrize("before,after", [([], []), ([result()], [])])
def test_empty_or_missing_results_fail(before, after):
    with pytest.raises(AssertionError, match="Missing or unequal"):
        PROBE["compare_outputs"](before, after)


def test_cann_path_survives_while_source_shadow_is_removed(tmp_path):
    cann = tmp_path / "cann"
    cann.mkdir()
    source = tmp_path / "source"
    (source / "vllm_ascend_simllm").mkdir(parents=True)
    original = {
        "PYTHONPATH": os.pathsep.join((str(cann), str(source), "")),
        "VLLM_PLUGINS": "foreign",
        "VLLM_ASCEND_SIMLLM_ENABLED": "1",
        "LD_LIBRARY_PATH": "qualified-runtime",
    }
    clean = PROBE["runtime_environment"](original)
    assert clean["PYTHONPATH"] == str(cann)
    assert clean["LD_LIBRARY_PATH"] == "qualified-runtime"
    assert "VLLM_PLUGINS" not in clean
    assert "VLLM_ASCEND_SIMLLM_ENABLED" not in clean


def stage_evidence(tmp_path, text, enabled=True):
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vllm_ascend_simllm/__init__.py", "# qualified wheel")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "installed.json").write_text(
        json.dumps(
            {
                "files": PROBE["wheel_files"](wheel),
            }
        )
    )
    (stage / "launch.json").write_text(
        json.dumps(
            {
                "settings": {"VLLM_ASCEND_SIMLLM_ENABLED": str(int(enabled))},
            }
        )
    )
    (stage / "outputs.jsonl").write_text(json.dumps(result()) + "\n")
    (stage / "server.log").write_text(text)
    return stage, wheel


def test_runtime_reuse_proves_activation_without_startup_banner(tmp_path):
    stage, wheel = stage_evidence(tmp_path, "SimLLM: reusing 128 semantic tokens\n")
    evidence = PROBE["validate_stage"](stage, [result()], wheel, True)
    assert evidence["patch_exercised"]
    assert not evidence["startup_banner_observed"]
    assert evidence["semantic_reused_tokens"] == 128


def test_startup_banner_without_reuse_is_insufficient(tmp_path):
    stage, wheel = stage_evidence(tmp_path, "Sim-LLM patch applied\n")
    with pytest.raises(AssertionError, match="did not exercise"):
        PROBE["validate_stage"](stage, [result()], wheel, True)


def test_disabled_stage_must_not_store_cache(tmp_path):
    stage, wheel = stage_evidence(tmp_path, "SimLLM extract_kv: stored 1 tasks", False)
    with pytest.raises(AssertionError, match="unexpectedly activated"):
        PROBE["validate_stage"](stage, [result()], wheel, False)


def test_revalidation_rejects_wrong_installed_wheel(tmp_path):
    stage, wheel = stage_evidence(tmp_path, "SimLLM: reusing 128 semantic tokens\n")
    (stage / "installed.json").write_text('{"files": {}}')
    with pytest.raises(AssertionError, match="do not match"):
        PROBE["validate_stage"](stage, [result()], wheel, True)
