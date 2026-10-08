"""Regressions for the worker-side prefix reuse failures found on Ascend."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend_simllm.config import SimLLMConfig
from vllm_ascend_simllm.kv_manager import CachedTask, KVManager
from vllm_ascend_simllm.kv_reuse import KVReuseEngine
from vllm_ascend_simllm.patch import patch_model_runner as patch
from vllm_ascend_simllm.similarity import MatchResult, SimilarityIdentifier


def _task(prompt: tuple[int, ...], layer_values: tuple[float, ...]) -> CachedTask:
    layers = tuple(
        (
            torch.full((1, 8, 1, 2), value),
            torch.full((1, 8, 1, 2), -value),
        )
        for value in layer_values
    )
    return CachedTask(
        task_id="source",
        embedding=torch.tensor([[1.0, 0.0]]),
        lsh_hash=7,
        top_k=layers[-1][0],
        top_v=layers[-1][1],
        last_access_time=time.monotonic(),
        seq_len=len(prompt),
        prompt_token_ids=prompt,
        layer_kv=layers,
    )


def test_prefix_selection_requires_identical_tokens(monkeypatch):
    manager = KVManager()
    manager.store(_task((1, 2, 3, 4, 5), (1.0, 2.0)))
    monkeypatch.setattr(patch, "_kv_manager", manager)
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(prompt_token_ids=[1, 2, 3, 9], num_computed_tokens=0),
            SimpleNamespace(prompt_token_ids=[1, 9, 3, 4], num_computed_tokens=0),
            SimpleNamespace(
                prompt_token_ids=[1, 2, 3, 4],
                num_computed_tokens=0,
                mm_features=[object()],
            ),
            SimpleNamespace(
                prompt_token_ids=[1, 2, 3, 4],
                num_computed_tokens=0,
                lora_request=object(),
            ),
        ]
    )
    results = patch._simllm_identify_prefixes(output)
    assert results[0].prefix_len == 3
    assert 1 not in results
    assert 2 not in results
    assert 3 not in results


@pytest.mark.parametrize("mode", ["exact", "aligned", "tail"])
def test_token_only_snapshot_preparation_avoids_model_and_device(monkeypatch, mode):
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(hybrid_reuse_mode=mode))
    # No model or device attribute: guarded matching must not perform an
    # embedding lookup merely to register a cold request for KV capture.
    runner = SimpleNamespace(_has_gdn=True)
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="cold", prompt_token_ids=[1, 2, 3]),
            SimpleNamespace(req_id="mm", prompt_token_ids=[1, 2], mm_features=[1]),
        ]
    )
    patch._simllm_preprocess_from_scheduler(runner, output)
    assert set(runner._simllm_pending_prompts) == {"cold"}
    embedding, _, prompt = runner._simllm_pending_prompts["cold"]
    assert embedding.device.type == "cpu"
    assert embedding.numel() == 0
    assert prompt == (1, 2, 3)
    assert runner._simllm_batch_hashes is None


def test_hybrid_exact_mode_requires_complete_shorter_source(monkeypatch):
    manager = KVManager()
    manager.store(_task((1, 2, 3, 4, 5), (1.0, 2.0)))
    monkeypatch.setattr(patch, "_kv_manager", manager)
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(prompt_token_ids=[1, 2, 3, 9], num_computed_tokens=0),
            SimpleNamespace(prompt_token_ids=[1, 2, 3, 4, 5, 6], num_computed_tokens=0),
        ]
    )
    matches = patch._simllm_identify_prefixes(output, complete_source_only=True)
    assert 0 not in matches
    assert matches[1].prefix_len == 5


def test_hybrid_exact_mode_skips_complete_source_span(monkeypatch):
    monkeypatch.setattr(
        patch, "_simllm_config", SimLLMConfig(hybrid_reuse_mode="exact")
    )
    source = _task((1, 2, 3, 4, 5), (1.0, 2.0))
    runner = SimpleNamespace(
        _has_gdn=True,
        _simllm_match_results={
            0: MatchResult(
                matched=True,
                layer_kv=source.layer_kv,
                prefix_len=128,
                match_type="exact",
            )
        },
    )
    req = SimpleNamespace(
        req_id="continuation", prompt_token_ids=[1] * 256, num_computed_tokens=0
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[req],
        num_scheduled_tokens={"continuation": 256},
        total_num_scheduled_tokens=256,
    )
    patch._simllm_rewrite_scheduler_output(runner, output)
    assert req.num_computed_tokens == 128
    assert output.num_scheduled_tokens == {"continuation": 128}


def test_hybrid_tail_mode_keeps_recompute_window(monkeypatch):
    monkeypatch.setattr(
        patch,
        "_simllm_config",
        SimLLMConfig(hybrid_reuse_mode="tail", hybrid_recompute_tokens=128),
    )
    source = _task((1, 2, 3, 4, 5), (1.0, 2.0))
    runner = SimpleNamespace(
        _has_gdn=True,
        _simllm_match_results={
            0: MatchResult(
                matched=True,
                layer_kv=source.layer_kv,
                prefix_len=400,
                match_type="semantic",
            )
        },
    )
    req = SimpleNamespace(
        req_id="target", prompt_token_ids=[1] * 500, num_computed_tokens=0
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[req],
        num_scheduled_tokens={"target": 500},
        total_num_scheduled_tokens=500,
    )
    patch._simllm_rewrite_scheduler_output(runner, output)
    assert req.num_computed_tokens == 372
    assert output.num_scheduled_tokens == {"target": 128}


@pytest.mark.parametrize("mode", ["exact", "aligned", "tail"])
def test_complete_hybrid_prefix_is_not_truncated_by_chunk(monkeypatch, mode):
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(hybrid_reuse_mode=mode))
    source = _task(tuple(range(256)), (1.0,))
    runner = SimpleNamespace(
        _has_gdn=True,
        _simllm_match_results={
            0: MatchResult(
                matched=True,
                layer_kv=source.layer_kv,
                prefix_len=256,
                match_type="exact",
            )
        },
    )
    req = SimpleNamespace(
        req_id="target", prompt_token_ids=list(range(512)), num_computed_tokens=0
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[req],
        num_scheduled_tokens={"target": 256},
        total_num_scheduled_tokens=256,
    )
    patch._simllm_rewrite_scheduler_output(runner, output)
    assert req.num_computed_tokens == 0
    assert output.num_scheduled_tokens["target"] == 256
    assert not runner._simllm_reuse_tokens_by_req_id


def test_tail_exact_prefix_keeps_complete_recurrent_position(monkeypatch):
    monkeypatch.setattr(
        patch,
        "_simllm_config",
        SimLLMConfig(hybrid_reuse_mode="tail", hybrid_recompute_tokens=512),
    )
    source = _task(tuple(range(256)), (1.0,))
    runner = SimpleNamespace(
        _has_gdn=True,
        _simllm_match_results={
            0: MatchResult(
                matched=True,
                layer_kv=source.layer_kv,
                prefix_len=256,
                match_type="exact",
            )
        },
    )
    req = SimpleNamespace(
        req_id="target", prompt_token_ids=list(range(400)), num_computed_tokens=0
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[req],
        num_scheduled_tokens={"target": 400},
        total_num_scheduled_tokens=400,
    )
    patch._simllm_rewrite_scheduler_output(runner, output)
    assert req.num_computed_tokens == 256
    assert output.num_scheduled_tokens["target"] == 144


def test_semantic_match_crosses_nearby_hash_bucket_and_skips_different_tokens(
    monkeypatch,
):
    manager = KVManager()
    manager.store(_task((1, 2, 3, 4, 5), (1.0, 2.0)))
    monkeypatch.setattr(patch, "_kv_manager", manager)
    identifier = SimilarityIdentifier(
        embedding_dim=2, cosine_threshold=0.9, lsh_num_bits=4
    )
    semantic = identifier.identify_reusable(
        torch.tensor([[0.995, 0.1]]), torch.tensor([6]), manager, hamming_radius=1
    )
    assert semantic[0].match_type == "semantic"
    assert semantic[0].source_task_id == "source"

    req = SimpleNamespace(
        req_id="different",
        prompt_token_ids=[1, 9, 3, 4, 5],
        num_computed_tokens=0,
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[req],
        num_scheduled_tokens={"different": 5},
        total_num_scheduled_tokens=5,
    )
    selected = patch._simllm_select_reuse(output, semantic)
    assert selected[0].match_type == "semantic"
    runner = SimpleNamespace(_simllm_match_results=selected)
    patch._simllm_rewrite_scheduler_output(runner, output)
    assert req.num_computed_tokens == 4
    assert output.num_scheduled_tokens == {"different": 1}


def test_semantic_reuse_rejects_distant_hash_or_low_cosine():
    manager = KVManager()
    manager.store(_task((1, 2, 3, 4, 5), (1.0, 2.0)))
    identifier = SimilarityIdentifier(
        embedding_dim=2, cosine_threshold=0.9, lsh_num_bits=4
    )
    assert not identifier.identify_reusable(
        torch.tensor([[1.0, 0.0]]), torch.tensor([0]), manager, hamming_radius=1
    )
    assert not identifier.identify_reusable(
        torch.tensor([[0.0, 1.0]]), torch.tensor([6]), manager, hamming_radius=1
    )


def test_qwen_token_overlap_finds_reordered_prompt_without_embedding(monkeypatch):
    manager = KVManager()
    manager.store(_task((1, 2, 3, 4, 5), (1.0, 2.0)))
    monkeypatch.setattr(patch, "_kv_manager", manager)
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(prompt_token_ids=[5, 4, 3, 2, 1], num_computed_tokens=0),
            SimpleNamespace(prompt_token_ids=[6, 7, 8, 9, 10], num_computed_tokens=0),
        ]
    )
    matches = patch._simllm_identify_token_overlap(output)
    assert matches[0].source_task_id == "source"
    assert matches[0].match_type == "semantic"
    assert 1 not in matches


def test_aligned_overlap_requires_ordered_prefix_and_shorter_source():
    manager = KVManager()
    source = tuple(range(400))
    manager.store(_task(source, (1.0,)))
    options = {
        "shorter_source_only": True,
        "min_common_prefix": 256,
        "min_ordered_overlap": 0.55,
    }
    assert manager.best_token_overlap(source + tuple(range(400, 480)), 0.9, **options)
    assert manager.best_token_overlap(source, 0.9, **options) is None
    assert (
        manager.best_token_overlap(
            tuple(range(1000, 1020)) + source[20:] + tuple(range(400, 480)),
            0.9,
            **options,
        )
        is None
    )


def test_topic_anchor_rejects_cross_topic_candidate(monkeypatch):
    manager = KVManager()
    task = _task(tuple(range(400)), (1.0,))
    task.topic_anchor = "分布式缓存系统"
    manager.store(task)
    target = tuple(range(400, 480)) + task.prompt_token_ids
    assert (
        manager.best_token_overlap(target, 0.5, required_anchor="数据库查询优化")
        is None
    )
    assert manager.best_token_overlap(target, 0.5, required_anchor="分布式缓存系统")

    class Tokenizer:
        def decode(self, _ids, **_kwargs):
            return (
                "对于“云原生微服务架构”先观察。"
                "在“分布式缓存系统”中，针对“分布式缓存系统”处理。"
            )

    runner = SimpleNamespace(_simllm_quality_tokenizer=Tokenizer())
    assert patch._simllm_topic_anchor(runner, "new", (1, 2)) == "分布式缓存系统"


@pytest.mark.parametrize("mode", ["tail", "tail_anchor", "aligned"])
def test_guarded_overlap_rejects_template_mismatch(monkeypatch, mode):
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(hybrid_reuse_mode=mode))
    manager = KVManager()
    task = _task(tuple(range(400)), (1.0,))
    task.topic_anchor = "分布式缓存系统"
    task.template_signature = ()  # Raw completion source.
    task.context_topics = ("分布式缓存系统",)
    manager.store(task)
    monkeypatch.setattr(patch, "_kv_manager", manager)

    class Tokenizer:
        def decode(self, _ids, *, skip_special_tokens):
            assert not skip_special_tokens
            return (
                "<|im_start|>user\n请分析“分布式缓存系统”。<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n</think>\n"
            )

    runner = SimpleNamespace(_simllm_quality_tokenizer=Tokenizer())
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="chat", prompt_token_ids=list(range(410)), num_computed_tokens=0
            )
        ]
    )
    assert not patch._simllm_identify_token_overlap(output, runner)
    task.template_signature = patch._simllm_prompt_features(
        runner, "chat", tuple(range(410))
    )[1]
    assert patch._simllm_identify_token_overlap(output, runner)


def test_tail_rejects_stale_secondary_context_with_same_main_topic(monkeypatch):
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(hybrid_reuse_mode="tail"))
    manager = KVManager()
    source = _task(tuple(range(400)), (1.0,))
    source.topic_anchor = "数据库查询优化"
    source.context_topics = ("分布式缓存系统", "数据库查询优化")
    source.template_signature = ("<|im_start|>user\n",)
    manager.store(source)
    monkeypatch.setattr(patch, "_kv_manager", manager)

    class Tokenizer:
        def decode(self, _ids, *, skip_special_tokens):
            assert not skip_special_tokens
            return (
                "<|im_start|>user\n"
                "【共享上下文摘要：主题】\n智能制造调度\n"
                "【背景材料：主题】\n数据库查询优化\n"
                "请比较“数据库查询优化”与“数据库查询优化”。"
            )

    runner = SimpleNamespace(_simllm_quality_tokenizer=Tokenizer())
    target = tuple(range(400)) + tuple(range(20))
    features = patch._simllm_prompt_features(runner, "target", target)
    assert features[0] == "数据库查询优化"
    assert features[2] == ("数据库查询优化", "智能制造调度")
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="target", prompt_token_ids=list(target), num_computed_tokens=0
            )
        ]
    )
    assert not patch._simllm_identify_token_overlap(output, runner)
    source.context_topics = features[2]
    assert patch._simllm_identify_token_overlap(output, runner)


def test_tail_anchor_allows_secondary_context_mismatch(monkeypatch):
    monkeypatch.setattr(
        patch, "_simllm_config", SimLLMConfig(hybrid_reuse_mode="tail_anchor")
    )
    manager = KVManager()
    source = _task(tuple(range(400)), (1.0,))
    source.topic_anchor = "数据库查询优化"
    source.context_topics = ("分布式缓存系统", "数据库查询优化")
    source.template_signature = ("<|im_start|>user\n",)
    manager.store(source)
    monkeypatch.setattr(patch, "_kv_manager", manager)

    class Tokenizer:
        def decode(self, _ids, *, skip_special_tokens):
            return (
                "<|im_start|>user\n"
                "【共享上下文摘要：主题】\n智能制造调度\n"
                "【背景材料：主题】\n数据库查询优化\n"
                "请比较“数据库查询优化”与“数据库查询优化”。"
            )

    runner = SimpleNamespace(_simllm_quality_tokenizer=Tokenizer())
    target = tuple(range(400)) + tuple(range(20))
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="target", prompt_token_ids=list(target), num_computed_tokens=0
            )
        ]
    )
    assert patch._simllm_identify_token_overlap(output, runner)


def test_prompt_features_preserve_roles_and_thinking_and_decode_once():
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = [
        "<|im_start|>user\n“主题”<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|im_start|>user\n“主题”<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n",
        "<|im_start|>system\n“主题”<|im_end|>\n<|im_start|>assistant\n<think>\n",
    ]
    runner = SimpleNamespace(_simllm_quality_tokenizer=tokenizer)
    first = patch._simllm_prompt_features(runner, "thinking", (1, 2))
    assert patch._simllm_prompt_features(runner, "thinking", (1, 2)) == first
    disabled = patch._simllm_prompt_features(runner, "nonthinking", (3, 4))
    role_changed = patch._simllm_prompt_features(runner, "system", (5, 6))
    assert first[0] == disabled[0] == role_changed[0] == "主题"
    assert first[1] != disabled[1]
    assert first[1] != role_changed[1]
    assert tokenizer.decode.call_count == 3


def test_qwen_skips_short_exact_prefix_copy():
    runner = SimpleNamespace(
        _has_gdn=True,
        _simllm_match_results={
            0: MatchResult(
                matched=True,
                layer_kv=_task((1,) * 40, (1.0,)).layer_kv,
                prefix_len=40,
                match_type="exact",
            )
        },
    )
    req = SimpleNamespace(
        req_id="short-prefix", prompt_token_ids=[1] * 300, num_computed_tokens=0
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[req],
        num_scheduled_tokens={"short-prefix": 300},
        total_num_scheduled_tokens=300,
    )
    patch._simllm_rewrite_scheduler_output(runner, output)
    assert runner._simllm_reuse_tokens_by_req_id == {}
    assert req.num_computed_tokens == 0


def test_hamming_distance_uses_signed_hash_bits():
    manager = KVManager()
    source = _task((1, 2, 3), (1.0,))
    source.lsh_hash = -(1 << 63)
    manager.store(source)
    assert manager.candidates_within_hamming(-(1 << 63) + 1, 1, 64) == [source]
    assert manager.candidates_within_hamming(0, 1, 64) == [source]


def test_rewrite_updates_both_worker_schedule_counts():
    runner = SimpleNamespace(
        _simllm_match_results={
            0: MatchResult(
                matched=True, layer_kv=_task((1,) * 6, (1.0,)).layer_kv, prefix_len=6
            ),
        }
    )
    new_req = SimpleNamespace(
        req_id="new", prompt_token_ids=[1] * 8, num_computed_tokens=0
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[new_req],
        num_scheduled_tokens={"new": 8},
        total_num_scheduled_tokens=8,
    )
    patch._simllm_rewrite_scheduler_output(runner, output)
    assert new_req.num_computed_tokens == 5
    assert output.num_scheduled_tokens == {"new": 3}
    assert output.total_num_scheduled_tokens == 3
    assert runner._simllm_reuse_tokens_by_req_id["new"][1] == 5


def test_prepopulate_restores_each_layer_instead_of_top_layer(monkeypatch):
    task = _task((1, 2, 3, 4, 5, 6), (1.0, 2.0))
    runner = MagicMock()
    runner.input_batch.req_ids = ["new"]
    runner.input_batch.num_reqs = 1
    runner.input_batch.block_table[0].get_device_tensor.return_value = torch.tensor(
        [[0]]
    )
    runner.kv_caches = [
        (torch.zeros(1, 8, 1, 2), torch.zeros(1, 8, 1, 2)) for _ in range(2)
    ]
    runner._simllm_reuse_tokens_by_req_id = {
        "new": (MatchResult(matched=True, layer_kv=task.layer_kv, prefix_len=6), 5)
    }
    monkeypatch.setattr(patch, "_kv_reuse_engine", KVReuseEngine(block_size=8))
    patch._simllm_prepopulate_prefix_kv(runner)
    for idx, (key, value) in enumerate(runner.kv_caches, 1):
        assert torch.all(key[0] == idx)
        assert torch.all(value[0] == -idx)


def test_physical_block_copy_keeps_vendor_layout():
    cache = torch.arange(4 * 8 * 2 * 4).reshape(4, 8, 2, 4)
    saved = KVReuseEngine.gather_blocks(cache, [1, 3])
    cache[1].zero_()
    cache[3].zero_()
    KVReuseEngine.write_blocks(cache, [0, 2], saved)
    assert torch.equal(cache[0], saved[0])
    assert torch.equal(cache[2], saved[1])
    assert torch.all(cache[1] == 0)


def test_vectorized_block_copy_updates_noncontiguous_cache_view():
    cache = torch.zeros(4, 2, 8, 1, 2)
    values = torch.arange(2 * 8 * 1 * 2).reshape(2, 8, 1, 2).float()
    KVReuseEngine.write_blocks(cache[:, 0], torch.tensor([1, 3]), values)
    assert torch.equal(cache[1, 0], values[0])
    assert torch.equal(cache[3, 0], values[1])
    assert torch.all(cache[:, 1] == 0)


def test_mismatched_scheduler_and_physical_block_sizes_disable_reuse():
    runner = SimpleNamespace(
        block_size=16,
        cache_config=SimpleNamespace(block_size=128),
        kv_caches=[(torch.zeros(2, 128, 1, 2), torch.zeros(2, 128, 1, 2))],
    )
    assert not patch._simllm_cache_layout_compatible(runner)
    runner.block_size = 128
    assert patch._simllm_cache_layout_compatible(runner)


def test_gdn_model_does_not_reuse_attention_kv_without_recurrent_state():
    runner = SimpleNamespace(
        _has_gdn=True,
        block_size=128,
        cache_config=SimpleNamespace(block_size=128),
        kv_caches=[(torch.zeros(2, 128, 1, 2), torch.zeros(2, 128, 1, 2))],
    )
    assert not patch._simllm_cache_layout_compatible(runner)


class MambaSpec:
    block_size = 8


class AttentionSpec:
    block_size = 8


def _hybrid_runner():
    runner = MagicMock()
    runner._has_gdn = True
    runner.kv_caches = [
        [torch.zeros(4, 2), torch.zeros(4, 3)],
        (torch.zeros(4, 8, 1, 2), torch.zeros(4, 8, 1, 2)),
    ]
    runner.kv_cache_config.kv_cache_groups = [
        SimpleNamespace(
            kv_cache_spec=MambaSpec(), layer_names=["model.layers.0.linear_attn"]
        ),
        SimpleNamespace(
            kv_cache_spec=AttentionSpec(), layer_names=["model.layers.1.self_attn"]
        ),
    ]
    runner.input_batch.num_reqs = 1
    runner.input_batch.req_ids = ["new"]
    runner.input_batch.block_table = [MagicMock(), MagicMock()]
    runner.input_batch.block_table[0].get_device_tensor.return_value = torch.tensor(
        [[2, 3]]
    )
    runner.input_batch.block_table[1].get_device_tensor.return_value = torch.tensor(
        [[1, 2]]
    )
    runner.seq_lens = torch.tensor([12])
    return runner


def test_hybrid_restores_recurrent_state_to_current_block_and_attention_prefix():
    runner = _hybrid_runner()
    saved = (
        (torch.full((1, 2), 3.0), torch.full((1, 3), 4.0)),
        (torch.full((2, 8, 1, 2), 5.0), torch.full((2, 8, 1, 2), 6.0)),
    )
    runner._simllm_reuse_tokens_by_req_id = {
        "new": (MatchResult(matched=True, layer_kv=saved, hybrid_states=True), 10)
    }
    assert patch._simllm_cache_layout_compatible(runner)
    patch._simllm_prepopulate_prefix_kv(runner)
    assert torch.all(runner.kv_caches[0][0][3] == 3)
    assert torch.all(runner.kv_caches[0][1][3] == 4)
    assert torch.all(runner.kv_caches[0][0][2] == 0)
    assert torch.all(runner.kv_caches[1][0][1:3] == 5)
    assert torch.all(runner.kv_caches[1][1][1:3] == 6)


@pytest.mark.parametrize("match_type", ["exact", "semantic"])
def test_tail_restores_exact_state_but_resets_semantic_state(monkeypatch, match_type):
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(hybrid_reuse_mode="tail"))
    runner = _hybrid_runner()
    for tensor in runner.kv_caches[0]:
        tensor.fill_(99)  # Simulate preprocess_mamba copying an old state.
    saved = (
        (torch.full((1, 2), 3.0), torch.full((1, 3), 4.0)),
        (torch.full((2, 8, 1, 2), 5.0), torch.full((2, 8, 1, 2), 6.0)),
    )
    runner._simllm_reuse_tokens_by_req_id = {
        "new": (
            MatchResult(
                matched=True,
                layer_kv=saved,
                hybrid_states=True,
                match_type=match_type,
            ),
            10,
        )
    }
    patch._simllm_prepopulate_prefix_kv(runner)
    assert torch.all(runner.kv_caches[0][0][3] == (3 if match_type == "exact" else 0))
    assert torch.all(runner.kv_caches[0][1][3] == (4 if match_type == "exact" else 0))
    assert torch.all(runner.kv_caches[1][0][1:3] == 5)


def test_hybrid_restore_reads_host_block_table_without_device_round_trip():
    runner = _hybrid_runner()
    runner.input_batch.block_table[0].get_cpu_tensor.return_value = torch.tensor(
        [[2, 3]]
    )
    runner.input_batch.block_table[1].get_cpu_tensor.return_value = torch.tensor(
        [[1, 2]]
    )
    for table in runner.input_batch.block_table:
        table.get_device_tensor.side_effect = AssertionError("NPU table read")
    saved = (
        (torch.full((1, 2), 3.0), torch.full((1, 3), 4.0)),
        (torch.full((2, 8, 1, 2), 5.0), torch.full((2, 8, 1, 2), 6.0)),
    )
    runner._simllm_reuse_tokens_by_req_id = {
        "new": (MatchResult(matched=True, layer_kv=saved, hybrid_states=True), 10)
    }
    patch._simllm_prepopulate_prefix_kv(runner)
    assert torch.all(runner.kv_caches[0][0][3] == 3)
    assert torch.all(runner.kv_caches[1][0][1:3] == 5)


def test_hybrid_restore_batches_three_disjoint_matches():
    runner = _hybrid_runner()
    runner.input_batch.num_reqs = 3
    runner.input_batch.req_ids = ["a", "b", "c"]
    runner.seq_lens = torch.tensor([12, 12, 12])
    runner.kv_caches = [
        [torch.zeros(8, 2), torch.zeros(8, 3)],
        (torch.zeros(8, 8, 1, 2), torch.zeros(8, 8, 1, 2)),
    ]
    runner.input_batch.block_table[0].get_cpu_tensor.return_value = torch.tensor(
        [[0, 1], [2, 3], [4, 5]]
    )
    runner.input_batch.block_table[1].get_cpu_tensor.return_value = torch.tensor(
        [[0, 1], [2, 3], [4, 5]]
    )
    runner._simllm_reuse_tokens_by_req_id = {
        req_id: (
            MatchResult(
                matched=True,
                hybrid_states=True,
                layer_kv=(
                    (torch.full((1, 2), value), torch.full((1, 3), value)),
                    (
                        torch.full((2, 8, 1, 2), value),
                        torch.full((2, 8, 1, 2), value),
                    ),
                ),
            ),
            10,
        )
        for req_id, value in zip(("a", "b", "c"), (1.0, 2.0, 3.0), strict=True)
    }
    patch._simllm_prepopulate_prefix_kv(runner)
    for row, value in enumerate((1.0, 2.0, 3.0)):
        assert torch.all(runner.kv_caches[0][0][2 * row + 1] == value)
        assert torch.all(runner.kv_caches[1][0][2 * row : 2 * row + 2] == value)

    # Shared physical blocks must keep the original request write order.
    runner.input_batch.block_table[0].get_cpu_tensor.return_value[2] = torch.tensor(
        [2, 3]
    )
    runner.input_batch.block_table[1].get_cpu_tensor.return_value[2] = torch.tensor(
        [2, 3]
    )
    patch._simllm_prepopulate_prefix_kv(runner)
    assert torch.all(runner.kv_caches[0][0][3] == 3)
    assert torch.all(runner.kv_caches[1][0][2:4] == 3)


def test_hybrid_attention_uses_expanded_kernel_block_width():
    runner = _hybrid_runner()
    runner.kv_cache_config.kv_cache_groups[1].kv_cache_spec.block_size = 16
    runner.input_batch.block_table[1].block_size = 8
    assert patch._simllm_hybrid_layout(runner)[1] == (1, 8, False)


def test_hybrid_snapshot_saves_only_latest_recurrent_block(monkeypatch):
    manager = KVManager()
    monkeypatch.setattr(patch, "_kv_manager", manager)
    monkeypatch.setattr(
        patch,
        "_simllm_config",
        SimLLMConfig(enabled=True, hybrid_reuse_mode="aggressive"),
    )
    runner = _hybrid_runner()
    runner.kv_caches[0][0][3] = 7
    runner.kv_caches[0][1][3] = 8
    runner.kv_caches[1][0][1:3] = 9
    runner.kv_caches[1][1][1:3] = 10
    runner._simllm_batch_hashes = torch.tensor([11])
    runner._simllm_batch_embeddings = torch.tensor([[1.0, 0.0]])
    runner._simllm_batch_req_ids = ["new"]
    runner._simllm_prompt_token_ids = {"new": tuple(range(12))}
    runner._simllm_match_results = {}
    runner._simllm_reuse_tokens_by_req_id = {}
    patch._simllm_extract_kv(runner, torch.zeros(1))
    stored = manager.lookup_by_hash(11)[0]
    assert stored.hybrid_states
    assert stored.layer_kv is not None
    assert stored.layer_kv[0][0].shape == (1, 2)
    assert torch.all(stored.layer_kv[0][0] == 7)
    assert torch.all(stored.layer_kv[1][0] == 9)


def test_tail_snapshot_omits_recurrent_copy_but_keeps_attention(monkeypatch):
    manager = KVManager()
    monkeypatch.setattr(patch, "_kv_manager", manager)
    monkeypatch.setattr(
        patch, "_simllm_config", SimLLMConfig(enabled=True, hybrid_reuse_mode="tail")
    )
    runner = _hybrid_runner()
    runner._simllm_quality_tokenizer = SimpleNamespace(
        decode=lambda *_args, **_kwargs: "请分析“数据库查询优化”。"
    )
    runner.kv_caches[1][0][1:3] = 9
    prompt = tuple(range(12))
    runner._simllm_pending_prompts = {"new": (torch.empty((1, 0)), 0, prompt)}
    runner._simllm_match_results = {}
    runner._simllm_reuse_tokens_by_req_id = {}
    patch._simllm_extract_kv(runner, torch.zeros(1))
    stored = manager.get_task("new")
    assert stored is not None and stored.layer_kv is not None
    assert not stored.recurrent_valid
    assert all(blocks.shape[0] == 0 for blocks in stored.layer_kv[0])
    assert torch.all(stored.layer_kv[1][0] == 9)
    assert (
        manager.longest_prefix_match(prompt + (12,), complete_source_only=True) is None
    )


def test_hybrid_snapshot_reads_host_block_table(monkeypatch):
    manager = KVManager()
    monkeypatch.setattr(patch, "_kv_manager", manager)
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(enabled=True))
    runner = _hybrid_runner()
    runner.input_batch.block_table[0].get_cpu_tensor.return_value = torch.tensor(
        [[2, 3]]
    )
    runner.input_batch.block_table[1].get_cpu_tensor.return_value = torch.tensor(
        [[1, 2]]
    )
    for table in runner.input_batch.block_table:
        table.get_device_tensor.side_effect = AssertionError("NPU table read")
    runner._simllm_pending_prompts = {
        "new": (torch.tensor([[1.0, 0.0]]), 11, tuple(range(12)))
    }
    runner._simllm_match_results = {}
    runner._simllm_reuse_tokens_by_req_id = {}
    patch._simllm_extract_kv(runner, torch.zeros(1))
    assert manager.size() == 1


def test_chunked_prefill_snapshots_only_after_complete_prompt(monkeypatch):
    manager = KVManager()
    monkeypatch.setattr(patch, "_kv_manager", manager)
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(enabled=True))
    runner = _hybrid_runner()
    runner._simllm_pending_prompts = {
        "new": (torch.tensor([[1.0, 0.0]]), 11, tuple(range(12)))
    }
    runner._simllm_batch_hashes = None
    runner._simllm_match_results = {}
    runner._simllm_reuse_tokens_by_req_id = {}
    runner.seq_lens = torch.tensor([4])
    patch._simllm_extract_kv(runner, torch.zeros(1))
    assert manager.size() == 0
    assert "new" in runner._simllm_pending_prompts
    runner.seq_lens = torch.tensor([12])
    patch._simllm_extract_kv(runner, torch.zeros(1))
    assert manager.size() == 1
    assert manager.lookup_by_hash(11)[0].seq_len == 12
    assert runner._simllm_pending_prompts == {}


def test_full_exact_hit_skips_embedding_preprocessing(monkeypatch):
    manager = KVManager()
    manager.store(_task((1, 2, 3, 4), (1.0,)))
    monkeypatch.setattr(patch, "_kv_manager", manager)
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(enabled=True))
    monkeypatch.setattr(patch, "_simllm_cache_layout_compatible", lambda _: True)
    monkeypatch.setattr(
        patch,
        "_simllm_preprocess_from_scheduler",
        lambda *_: (_ for _ in ()).throw(AssertionError("unexpected embeddings")),
    )
    monkeypatch.setattr(patch, "_original_execute_model", lambda *_a, **_k: 7)
    request = SimpleNamespace(
        req_id="repeat", prompt_token_ids=[1, 2, 3, 4], num_computed_tokens=0
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[request],
        num_scheduled_tokens={"repeat": 4},
        total_num_scheduled_tokens=4,
    )
    runner = SimpleNamespace()
    assert patch._simllm_execute_model(runner, output) == 7
    assert request.num_computed_tokens == 3
    assert runner._simllm_batch_embeddings is None


def test_qwen_high_overlap_hit_skips_embedding_preprocessing(monkeypatch):
    manager = KVManager()
    manager.store(_task(tuple(range(256)), (1.0,)))
    monkeypatch.setattr(patch, "_kv_manager", manager)
    monkeypatch.setattr(
        patch,
        "_simllm_config",
        SimLLMConfig(enabled=True, hybrid_reuse_mode="aggressive"),
    )
    monkeypatch.setattr(patch, "_simllm_cache_layout_compatible", lambda _: True)
    monkeypatch.setattr(
        patch,
        "_simllm_preprocess_from_scheduler",
        lambda *_: (_ for _ in ()).throw(AssertionError("unexpected embeddings")),
    )
    monkeypatch.setattr(patch, "_original_execute_model", lambda *_a, **_k: 7)
    request = SimpleNamespace(
        req_id="reworded",
        prompt_token_ids=list(reversed(range(256))),
        num_computed_tokens=0,
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[request],
        num_scheduled_tokens={"reworded": 256},
        total_num_scheduled_tokens=256,
    )
    runner = SimpleNamespace(_has_gdn=True)
    assert patch._simllm_execute_model(runner, output) == 7
    assert runner._simllm_match_results[0].match_type == "semantic"
    assert request.num_computed_tokens == 255
    assert runner._simllm_batch_embeddings is None


def test_decode_step_bypasses_plugin_but_chunked_prefill_does_not(monkeypatch):
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(enabled=True))
    monkeypatch.setattr(patch, "_original_execute_model", lambda *_a, **_k: 11)
    monkeypatch.setattr(patch, "_original_model_forward", lambda *_a, **_k: 13)
    output = SimpleNamespace(scheduled_new_reqs=[], num_scheduled_tokens={"running": 1})
    runner = SimpleNamespace(
        requests={
            "running": SimpleNamespace(num_computed_tokens=10, num_prompt_tokens=10)
        },
        _simllm_reuse_tokens_by_req_id={"stale": object()},
    )
    assert patch._simllm_execute_model(runner, output) == 11
    assert runner._simllm_skip_forward
    assert runner._simllm_reuse_tokens_by_req_id == {}
    assert patch._simllm_model_forward(runner, 1) == 13
    runner.requests["running"].num_computed_tokens = 9
    assert not patch._simllm_decode_only_step(runner, output)


def test_chunked_semantic_target_is_not_cached_as_new_source(monkeypatch):
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(enabled=True))
    monkeypatch.setattr(patch, "_simllm_cache_layout_compatible", lambda _: True)
    monkeypatch.setattr(patch, "_simllm_identify_prefixes", lambda _: {})
    monkeypatch.setattr(patch, "_original_execute_model", lambda *_a, **_k: 7)
    request = SimpleNamespace(
        req_id="target", prompt_token_ids=list(range(8)), num_computed_tokens=0
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[request],
        num_scheduled_tokens={"target": 4},
        total_num_scheduled_tokens=4,
    )
    runner = SimpleNamespace(_simllm_pending_prompts={})

    def preprocess(runner, _output):
        runner._simllm_pending_prompts["target"] = (
            torch.tensor([[1.0]]),
            1,
            tuple(range(8)),
        )

    monkeypatch.setattr(patch, "_simllm_preprocess_from_scheduler", preprocess)
    monkeypatch.setattr(
        patch,
        "_simllm_identify",
        lambda _: {
            0: MatchResult(
                matched=True,
                layer_kv=_task(tuple(range(8)), (1.0,)).layer_kv,
                prefix_len=8,
                match_type="semantic",
            )
        },
    )
    assert patch._simllm_execute_model(runner, output) == 7
    assert request.num_computed_tokens == 3
    assert runner._simllm_pending_prompts == {}


def test_store_uses_input_embedding_and_complete_layer_kv(monkeypatch):
    manager = KVManager()
    monkeypatch.setattr(patch, "_kv_manager", manager)
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(enabled=True))
    runner = MagicMock()
    runner.input_batch.num_reqs = 1
    runner.input_batch.req_ids = ["new"]
    runner.input_batch.block_table[0].get_device_tensor.return_value = torch.tensor(
        [[0]]
    )
    runner.seq_lens = torch.tensor([4])
    runner._simllm_batch_hashes = torch.tensor([7])
    runner._simllm_batch_embeddings = torch.tensor([[1.0, 0.0]])
    runner._simllm_batch_req_ids = ["new"]
    runner._simllm_prompt_token_ids = {"new": (1, 2, 3, 4)}
    runner._simllm_match_results = {}
    runner._simllm_reuse_tokens_by_req_id = {}
    runner.kv_caches = [
        (torch.full((1, 8, 1, 2), value), torch.full((1, 8, 1, 2), -value))
        for value in (1.0, 2.0)
    ]
    patch._simllm_extract_kv(runner, torch.full((4, 2), 99.0))
    stored = manager.lookup_by_hash(7)[0]
    assert torch.equal(stored.embedding, runner._simllm_batch_embeddings)
    assert stored.prompt_token_ids == (1, 2, 3, 4)
    assert stored.layer_kv is not None
    assert torch.all(stored.layer_kv[0][0] == 1.0)
    assert torch.all(stored.layer_kv[1][0] == 2.0)


def test_approximate_kv_is_not_stored_as_new_source(monkeypatch):
    manager = KVManager()
    monkeypatch.setattr(patch, "_kv_manager", manager)
    monkeypatch.setattr(patch, "_simllm_config", SimLLMConfig(enabled=True))
    runner = MagicMock()
    runner.input_batch.num_reqs = 1
    runner.input_batch.req_ids = ["new"]
    runner.input_batch.block_table[0].get_device_tensor.return_value = torch.tensor(
        [[0]]
    )
    runner.seq_lens = torch.tensor([4])
    runner._simllm_batch_hashes = torch.tensor([7])
    runner._simllm_batch_embeddings = torch.tensor([[1.0, 0.0]])
    runner._simllm_batch_req_ids = ["new"]
    runner._simllm_prompt_token_ids = {"new": (1, 9, 3, 4)}
    runner._simllm_reuse_tokens_by_req_id = {
        "new": (MatchResult(matched=True, match_type="semantic"), 3)
    }
    runner.kv_caches = [(torch.zeros(1, 8, 1, 2), torch.zeros(1, 8, 1, 2))]
    patch._simllm_extract_kv(runner, torch.zeros(4, 2))
    assert manager.size() == 0


def test_long_hybrid_semantic_target_promotes_after_full_prefill(monkeypatch):
    manager = KVManager()
    source = _task(tuple(range(8)), (1.0, 2.0))
    source.hybrid_states = True
    manager.store(source)
    monkeypatch.setattr(patch, "_kv_manager", manager)
    monkeypatch.setattr(
        patch,
        "_simllm_config",
        SimLLMConfig(enabled=True, hybrid_reuse_mode="aggressive"),
    )
    monkeypatch.setattr(patch, "_QWEN35_PROMOTION_MIN_GROWTH", 4)
    runner = _hybrid_runner()
    runner._simllm_pending_prompts = {}
    runner._simllm_promotions = set()
    runner._simllm_batch_hashes = None
    runner._simllm_batch_embeddings = None
    runner._simllm_batch_req_ids = None
    runner._simllm_match_results = {}
    runner._simllm_reuse_tokens_by_req_id = {
        "new": (
            MatchResult(
                matched=True,
                source_task_id="source",
                hybrid_states=True,
                match_type="semantic",
                prefix_len=8,
            ),
            7,
        )
    }
    short_request = SimpleNamespace(req_id="new", prompt_token_ids=list(range(10)))
    patch._simllm_prepare_promotions(
        runner, SimpleNamespace(scheduled_new_reqs=[short_request])
    )
    assert runner._simllm_promotions == set()
    assert runner._simllm_pending_prompts == {}

    request = SimpleNamespace(req_id="new", prompt_token_ids=list(range(12)))
    output = SimpleNamespace(scheduled_new_reqs=[request])
    patch._simllm_prepare_promotions(runner, output)
    assert runner._simllm_promotions == {"new"}
    assert runner._simllm_pending_prompts["new"][1] == source.lsh_hash

    runner.seq_lens = torch.tensor([4])
    patch._simllm_extract_kv(runner, torch.zeros(1))
    assert manager.size() == 1
    assert runner._simllm_promotions == {"new"}

    runner.seq_lens = torch.tensor([12])
    patch._simllm_extract_kv(runner, torch.zeros(1))
    assert manager.size() == 2
    promoted = manager.get_task("new")
    assert promoted is not None
    assert promoted.seq_len == 12
    assert promoted.hybrid_states
    assert runner._simllm_promotions == set()
    assert runner._simllm_pending_prompts == {}
