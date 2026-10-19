"""Activation rejection must leave the host untouched."""

from types import SimpleNamespace

import pytest

from vllm_ascend_simllm import compatibility


@pytest.fixture
def qualified(monkeypatch):
    monkeypatch.setattr(compatibility, "sys", SimpleNamespace(version_info=(3, 11)))
    versions = {k: sorted(v)[0] for k, v in compatibility.QUALIFIED_VERSIONS.items()}
    monkeypatch.setattr(compatibility.metadata, "version", versions.__getitem__)
    return versions


def test_qualified_runtime(qualified):
    compatibility.check_runtime()


@pytest.mark.parametrize("package", compatibility.QUALIFIED_VERSIONS)
def test_reject_unqualified_revision(qualified, package):
    qualified[package] = "999.0"
    with pytest.raises(RuntimeError, match=package):
        compatibility.check_runtime()


def test_missing_dependency(qualified, monkeypatch):
    def missing(_):
        raise compatibility.metadata.PackageNotFoundError("vllm")

    monkeypatch.setattr(compatibility.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="requires vllm"):
        compatibility.check_runtime()


def test_python_rejection(qualified, monkeypatch):
    monkeypatch.setattr(compatibility, "sys", SimpleNamespace(version_info=(3, 13)))
    with pytest.raises(RuntimeError, match="Python 3.11"):
        compatibility.check_runtime()


def make_runner():
    class Runner:
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return scheduler_output

        def _model_forward(self, num_tokens_padded, **kwargs):
            return num_tokens_padded

    for name in ("execute_model", "_model_forward"):
        getattr(Runner, name).__module__ = compatibility.RUNNER_MODULE
    return Runner


def test_runner_contract():
    compatibility.check_runner(make_runner())


@pytest.mark.parametrize("name", ["execute_model", "_model_forward"])
def test_missing_runner_method(name):
    runner = make_runner()
    delattr(runner, name)
    with pytest.raises(RuntimeError, match="missing " + name):
        compatibility.check_runner(runner)


def test_foreign_patch_rejected():
    runner = make_runner()
    runner.execute_model = lambda *args: None
    with pytest.raises(RuntimeError, match="foreign owner"):
        compatibility.check_runner(runner)


def test_changed_signature_rejected():
    runner = make_runner()

    def incompatible(self, new_argument):
        pass

    incompatible.__module__ = compatibility.RUNNER_MODULE
    runner.execute_model = incompatible
    with pytest.raises(RuntimeError, match="signature"):
        compatibility.check_runner(runner)
