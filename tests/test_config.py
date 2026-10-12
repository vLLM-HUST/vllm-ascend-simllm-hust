"""Plugin configuration no longer depends on host-owned env declarations."""

import pytest

from vllm_ascend_simllm.config import SimLLMConfig


def test_defaults_are_disabled(monkeypatch) -> None:
    for name in (
        "ENABLED",
        "COSINE_THRESHOLD",
        "LSH_NUM_BITS",
        "LSH_BATCH_THRESHOLD",
        "KV_CACHE_SIZE",
        "KV_CACHE_MAX_BYTES",
        "LSH_HAMMING_RADIUS",
        "HYBRID_REUSE_MODE",
        "HYBRID_RECOMPUTE_TOKENS",
        "SANDWICH_BOTTOM",
        "SANDWICH_TOP",
        "EMBEDDING_POOLING",
        "DEFERRAL_RATIO",
        "MAX_DEFERRALS",
        "PROFILE",
    ):
        monkeypatch.delenv(f"VLLM_ASCEND_SIMLLM_{name}", raising=False)
    assert SimLLMConfig.from_env() == SimLLMConfig()
    assert SimLLMConfig.from_env().hybrid_reuse_mode == "tail_anchor"


def test_worker_environment_is_read_by_plugin(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_ENABLED", "1")
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_KV_CACHE_SIZE", "17")
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_EMBEDDING_POOLING", "last")
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_PROFILE", "1")
    config = SimLLMConfig.from_env()
    assert config.enabled
    assert config.kv_cache_size == 17
    assert config.embedding_pooling == "last"
    assert config.profile


@pytest.mark.parametrize(
    "mode", ["aggressive", "exact", "aligned", "tail", "tail_anchor"]
)
def test_hybrid_policy_environment(monkeypatch, mode) -> None:
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_HYBRID_REUSE_MODE", mode)
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_HYBRID_RECOMPUTE_TOKENS", "1024")
    config = SimLLMConfig.from_env()
    assert config.hybrid_reuse_mode == mode
    assert config.hybrid_recompute_tokens == 1024


def test_unknown_hybrid_mode_fails_instead_of_using_aggressive(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_HYBRID_REUSE_MODE", "taill")
    with pytest.raises(ValueError, match="HYBRID_REUSE_MODE"):
        SimLLMConfig.from_env()


@pytest.mark.parametrize("tokens", ["0", "-1"])
def test_tail_requires_positive_recompute_window(monkeypatch, tokens) -> None:
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_HYBRID_RECOMPUTE_TOKENS", tokens)
    with pytest.raises(ValueError, match="HYBRID_RECOMPUTE_TOKENS"):
        SimLLMConfig.from_env()


def test_unknown_setting_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_SIMLLM_HYBRID_RECOMPUTE_TOKEN", "512")
    with pytest.raises(ValueError, match="HYBRID_RECOMPUTE_TOKEN"):
        SimLLMConfig.from_env()


@pytest.mark.parametrize("name", ["ENABLED", "PROFILE"])
@pytest.mark.parametrize("value", ["true", "2", "-1"])
def test_boolean_settings_require_zero_or_one(monkeypatch, name, value) -> None:
    monkeypatch.setenv(f"VLLM_ASCEND_SIMLLM_{name}", value)
    with pytest.raises(ValueError, match=name):
        SimLLMConfig.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("COSINE_THRESHOLD", "nan"),
        ("COSINE_THRESHOLD", "1.1"),
        ("DEFERRAL_RATIO", "inf"),
        ("LSH_NUM_BITS", "0"),
        ("LSH_HAMMING_RADIUS", "65"),
        ("KV_CACHE_SIZE", "0"),
        ("KV_CACHE_MAX_BYTES", "-1"),
        ("EMBEDDING_POOLING", "middle"),
    ],
)
def test_invalid_settings_fail_before_activation(monkeypatch, name, value) -> None:
    monkeypatch.setenv(f"VLLM_ASCEND_SIMLLM_{name}", value)
    with pytest.raises(ValueError, match=name):
        SimLLMConfig.from_env()
