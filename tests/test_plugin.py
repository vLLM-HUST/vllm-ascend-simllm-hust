"""The installed vLLM entry point patches the worker only when enabled."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

from vllm_ascend_simllm import plugin


def test_register_is_inert_without_enablement(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_ASCEND_SIMLLM_ENABLED", raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    plugin.register()
    assert not any(isinstance(item, plugin._RunnerFinder) for item in sys.meta_path)


def test_register_patches_loaded_runner(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_ENABLED", "1")
    module = ModuleType(plugin.RUNNER_MODULE)
    module.NPUModelRunner = type("Runner", (), {})
    monkeypatch.setitem(sys.modules, plugin.RUNNER_MODULE, module)
    calls = []
    monkeypatch.setattr(plugin, "_patch_runner", lambda loaded: calls.append(loaded))
    plugin.register()
    assert calls == [module]


def test_register_patches_runner_after_module_import(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_ENABLED", "1")
    monkeypatch.delitem(sys.modules, plugin.RUNNER_MODULE, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    calls = []
    monkeypatch.setattr(plugin, "_patch_runner", lambda loaded: calls.append(loaded))

    class FakeLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.NPUModelRunner = type("Runner", (), {})

    monkeypatch.setattr(
        plugin.PathFinder,
        "find_spec",
        lambda fullname, path=None, target=None: ModuleSpec(fullname, FakeLoader()),
    )
    plugin.register()
    plugin.register()
    finders = [item for item in sys.meta_path if isinstance(item, plugin._RunnerFinder)]
    assert len(finders) == 1
    spec = finders[0].find_spec(plugin.RUNNER_MODULE)
    assert spec is not None and spec.loader is not None
    module = ModuleType(plugin.RUNNER_MODULE)
    spec.loader.exec_module(module)
    assert calls == [module]
