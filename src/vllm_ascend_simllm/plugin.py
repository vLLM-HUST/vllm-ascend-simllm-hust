"""vLLM general-plugin entry point for the Ascend model runner.

The worker module is patched after import completes. Importing it eagerly at
plugin discovery time can initialize NPU code in the API or scheduler process.
"""

from __future__ import annotations

import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from types import ModuleType
from typing import Any

from .config import SimLLMConfig

RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"


def _patch_runner(module: ModuleType) -> None:
    runner = getattr(module, "NPUModelRunner", None)
    if runner is None:
        raise RuntimeError("SimLLM requires vLLM Ascend's NPUModelRunner")
    from .patch.patch_model_runner import apply_simllm_patch

    apply_simllm_patch(runner)


class _RunnerLoader(Loader):
    def __init__(self, original: Loader) -> None:
        self.original = original

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        create = getattr(self.original, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self.original.exec_module(module)
        _patch_runner(module)


class _RunnerFinder(MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Any = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname != RUNNER_MODULE:
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _RunnerLoader(spec.loader)
        return spec


def register() -> None:
    """Register the worker patch when explicitly enabled by Extension Manager."""
    if not SimLLMConfig.from_env().enabled:
        return
    loaded = sys.modules.get(RUNNER_MODULE)
    if loaded is not None and hasattr(loaded, "NPUModelRunner"):
        _patch_runner(loaded)
        return
    if not any(isinstance(finder, _RunnerFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _RunnerFinder())
