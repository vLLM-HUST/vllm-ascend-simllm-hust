"""Dependency failures and patch conflicts must not partially activate reuse."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend_simllm import compatibility
from vllm_ascend_simllm.patch import patch_model_runner as patch


@pytest.fixture
def runner_class(monkeypatch):
    class Runner:
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return scheduler_output

        def _model_forward(self, num_tokens_padded, **kwargs):
            return num_tokens_padded

    for name in ("execute_model", "_model_forward"):
        getattr(Runner, name).__module__ = compatibility.RUNNER_MODULE
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_ENABLED", "1")
    monkeypatch.setattr(patch, "check_runtime", lambda: None)
    for name in (
        "_simllm_config",
        "_kv_manager",
        "_simhash_hasher",
        "_similarity_identifier",
        "_sandwich_config",
        "_kv_reuse_engine",
        "_original_execute_model",
        "_original_model_forward",
        "_patched_runner_cls",
    ):
        monkeypatch.setattr(patch, name, None)
    return Runner


def test_constructor_failure_does_not_mutate_host(runner_class, monkeypatch):
    from vllm_ascend_simllm import lsh

    originals = (runner_class.execute_model, runner_class._model_forward)

    def unavailable(**kwargs):
        raise ImportError("projection dependency unavailable")

    monkeypatch.setattr(lsh, "SimHashHasher", unavailable)
    with pytest.raises(ImportError, match="dependency unavailable"):
        patch.apply_simllm_patch(runner_class)
    assert (runner_class.execute_model, runner_class._model_forward) == originals
    assert patch._simllm_config is None
    assert patch._kv_manager is None
    assert patch._patched_runner_cls is None


def test_activation_idempotence_and_later_conflict(runner_class):
    original = runner_class.execute_model
    patch.apply_simllm_patch(runner_class)
    patch.apply_simllm_patch(runner_class)
    assert patch._original_execute_model is original
    runner_class._model_forward = lambda *args: None
    with pytest.raises(RuntimeError, match="active patch was replaced"):
        patch.apply_simllm_patch(runner_class)
    assert patch._original_execute_model is original


def test_embedding_failure_clears_batch_state(runner_class, monkeypatch, caplog):
    from vllm_ascend_simllm.config import SimLLMConfig

    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(enabled=True))

    def fail(_):
        raise ImportError("embedding provider missing")

    monkeypatch.setattr(patch, "_reconcile_hasher_dim", fail)
    runner = SimpleNamespace(_simllm_batch_hashes=torch.tensor([42]))
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="new",
                prompt_token_ids=[1, 2],
                mm_features=None,
                lora_request=None,
                prompt_embeds=None,
            )
        ]
    )
    patch._simllm_preprocess_from_scheduler(runner, output)
    assert runner._simllm_batch_hashes is None
    assert runner._simllm_batch_embeddings is None
    assert "processing as unmatched" in caplog.text


def test_restore_failure_never_forwards_shortened_prompt(runner_class, monkeypatch):
    from vllm_ascend_simllm.config import SimLLMConfig

    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(enabled=True))
    calls = []
    monkeypatch.setattr(patch, "_original_model_forward", lambda *args: calls.append(1))

    def fail(_):
        raise RuntimeError("KV restore failed")

    monkeypatch.setattr(patch, "_simllm_prepopulate_prefix_kv", fail)
    with pytest.raises(RuntimeError, match="KV restore failed"):
        patch._simllm_model_forward(SimpleNamespace(), 1)
    assert calls == []
