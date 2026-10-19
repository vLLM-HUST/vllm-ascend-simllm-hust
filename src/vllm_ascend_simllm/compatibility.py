"""Conservative activation checks for the private Ascend runner API."""

from __future__ import annotations

import inspect
import sys
from importlib import metadata
from typing import Any

RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"
QUALIFIED_VERSIONS = {
    "vllm": frozenset(
        {"0.28.1.post1.dev143+gf18cf803c.empty", "0.28.1.post1.dev143+gf18cf803c"}
    ),
    "vllm-ascend": frozenset({"0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27"}),
    "torch": frozenset({"2.13.0+cpu"}),
    "torch-npu": frozenset({"2.13.0rc1"}),
}


def check_runtime() -> None:
    """Reject unqualified hosts without importing torch or initializing an NPU."""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "SimLLM activation requires the qualified Python 3.11 runtime"
        )
    for package, qualified in QUALIFIED_VERSIONS.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"SimLLM activation requires {package}") from exc
        if installed not in qualified:
            raise RuntimeError(
                f"SimLLM unqualified {package} version {installed!r}; "
                f"expected one of {sorted(qualified)}"
            )


def check_runner(runner: Any) -> None:
    """Detect missing private methods and foreign replacements before mutation."""
    for name, argument in (
        ("execute_model", "scheduler_output"),
        ("_model_forward", "num_tokens_padded"),
    ):
        method = getattr(runner, name, None)
        if not callable(method):
            raise RuntimeError(f"SimLLM incompatible runner: missing {name}")
        if getattr(inspect.unwrap(method), "__module__", None) != RUNNER_MODULE:
            raise RuntimeError(f"SimLLM runner conflict: {name} has a foreign owner")
        parameters = tuple(inspect.signature(method).parameters)
        if parameters[:2] != ("self", argument):
            raise RuntimeError(f"SimLLM incompatible runner signature: {name}")
