import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fetch_replay", ROOT / "scripts/fetch_benchmark_replay.py"
)
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


def test_integrity_fail_closed():
    raw = b"synthetic integrity fixture"
    digest = hashlib.sha256(raw).hexdigest()
    assert FETCH.checked(raw, digest, len(raw)) == raw
    with pytest.raises(ValueError, match="checksum"):
        FETCH.checked(raw + b"changed", digest)
    with pytest.raises(ValueError, match="length"):
        FETCH.checked(raw, digest, len(raw) + 1)


def test_external_empty_destination(tmp_path):
    assert FETCH.destination(tmp_path) == tmp_path
    with pytest.raises(ValueError, match="outside"):
        FETCH.destination(ROOT / "benchmark-copy")
    (tmp_path / "existing").touch()
    with pytest.raises(ValueError, match="empty"):
        FETCH.destination(tmp_path)


@pytest.mark.parametrize(
    "path", ["/absolute", "benchmarks/results/../other", "unrelated"]
)
def test_unsafe_members(path):
    with pytest.raises(ValueError, match="Unsafe"):
        FETCH.member(path)


def test_migrated_directory_stays_empty():
    tracked = subprocess.check_output(
        ["git", "ls-files", "benchmarks/results/qwen35_latency_20261008"], cwd=ROOT
    )
    assert not tracked.strip(), (
        "Restore evidence externally from benchmark, not into Git"
    )
