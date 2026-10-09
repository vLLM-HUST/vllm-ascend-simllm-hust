#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Worker-side exact and approximate task KV reuse for the v1 NPU runner.

The cache retains every layer's KV for a prompt. Exact token-prefix matches
are lossless; semantic matches reuse another task's KV as an approximation.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections import Counter
from typing import Any

import torch

from vllm_ascend_simllm.config import SimLLMConfig
from vllm_ascend_simllm.kv_reuse import KVReuseEngine
from vllm_ascend_simllm.utils import (
    cumsum_to_ranges,
    resolve_input_embedding_dim,
    tensor_to_int_list,
    tensor_to_int_matrix,
)

logger = logging.getLogger(__name__)


def get_forward_context() -> Any:
    """Resolve the host context only while executing in a vLLM worker."""
    from vllm.forward_context import get_forward_context as host_context

    return host_context()


# Regex to parse layer index from attention layer names like
# "model.layers.5.self_attn".
_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")
_QWEN35_TOKEN_OVERLAP_THRESHOLD = 0.95
_QWEN35_MIN_USEFUL_EXACT_PREFIX = 128
_QWEN35_BATCH_RESTORE_MIN_HITS = 3
_QWEN35_PROMOTION_MIN_GROWTH = 128
_QUOTED_ANCHOR_RE = re.compile(r"“([^”]{2,24})”")
_TOPIC_HEADING_RE = re.compile(r"【[^】\n]*：主题】\s*([^\n]+)")
_QWEN_TEMPLATE_RE = re.compile(r"<\|im_start\|>[^\n]*(?:\n|$)|<\|[^>]+\|>|</?think>")

# ---------------------------------------------------------------------------
# Module-level singletons — initialised once at patch-apply time, reused
# across every execute_model / _model_forward call within the worker process.
# ---------------------------------------------------------------------------
_simllm_config: SimLLMConfig | None = None
_kv_manager: Any = None  # KVManager
_simhash_hasher: Any = None  # SimHashHasher
_similarity_identifier: Any = None  # SimilarityIdentifier
_sandwich_config: Any = None  # SandwichConfig
_kv_reuse_engine: Any = None  # KVReuseEngine
_original_execute_model: Any = None
_original_model_forward: Any = None

# Per-forward injection map — built before _model_forward, consumed by the
# hijacked do_kv_cache_update inside every attention layer.
#   dict[batch_idx → (k_flat, v_flat, tok_start, covered)]
# where k_flat / v_flat have shape [L_kv, num_kv_heads, head_size].
_simllm_injection_map: dict[int, tuple] | None = None


def _patch_do_kv_cache_update() -> None:
    """Inactive legacy helper for attention-backend KV writes.

    Replaces matched-token slices of *key* / *value* with cached KV so that
    ``reshape_and_cache`` writes injected KV into the cache through the
    normal path.  All layers get the same top-layer cached KV.

    Patches both the Ascend NPU backend and the CUDA FlashAttention backend
    so Sim-LLM works on either hardware.
    """
    global _original_ascend_kv_update, _original_flash_kv_update

    # -- Ascend NPU backend ------------------------------------------------
    # The active path restores complete per-layer physical blocks directly.
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    _original_ascend_kv_update = AscendAttentionBackendImpl.do_kv_cache_update

    def _ascend_kv_update(
        self_impl: Any,
        layer: Any,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Any,
        slot_mapping: torch.Tensor,
    ) -> None:
        _inject_into_kv(key, value)
        _original_ascend_kv_update(
            self_impl,
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
        )

    AscendAttentionBackendImpl.do_kv_cache_update = _ascend_kv_update  # type: ignore[method-assign]
    logger.info("SimLLM: patched AscendAttentionBackendImpl.do_kv_cache_update.")

    # -- CUDA FlashAttention backend ---------------------------------------
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

        _original_flash_kv_update = FlashAttentionImpl.do_kv_cache_update

        def _flash_kv_update(
            self_impl: Any,
            layer: Any,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            _inject_into_kv(key, value)
            _original_flash_kv_update(
                self_impl,
                layer,
                key,
                value,
                kv_cache,
                slot_mapping,
            )

        FlashAttentionImpl.do_kv_cache_update = _flash_kv_update  # type: ignore[method-assign]
        logger.info("SimLLM: patched FlashAttentionImpl.do_kv_cache_update.")
    except Exception:
        logger.debug("SimLLM: FlashAttention backend not available, skipping.")


def _inject_into_kv(key: torch.Tensor, value: torch.Tensor) -> None:
    """Replace matched-token slices of *key* / *value* with cached KV.

    Called from hijacked ``do_kv_cache_update`` in every attention layer.
    *key* / *value* have shape ``[num_tokens, num_kv_heads, head_size]``.
    """
    global _simllm_injection_map
    inj_map = _simllm_injection_map
    if inj_map is None:
        return
    for _batch_idx, (k_flat, v_flat, tok_start, covered) in inj_map.items():
        # k_flat: [L_kv, H, D] — same shape as key[tok_start:tok_start+covered]
        if covered > 0:
            key[tok_start : tok_start + covered] = k_flat.to(
                device=key.device,
                dtype=key.dtype,
                non_blocking=True,
            )
            value[tok_start : tok_start + covered] = v_flat.to(
                device=value.device,
                dtype=value.dtype,
                non_blocking=True,
            )


# Stash originals so tests can restore them.
_original_ascend_kv_update: Any = None
_original_flash_kv_update: Any = None


def apply_simllm_patch(model_runner_cls: Any | None = None) -> None:
    """Apply the Sim-LLM patch to NPUModelRunner.

    Called once per worker after ``NPUModelRunner`` is defined.
    When ``VLLM_ASCEND_SIMLLM_ENABLED=0`` this is a silent no-op.

    Patches both ``execute_model`` (for proactive matching/rewrite) and
    ``_model_forward`` (for KV injection / extraction at the right point
    in the execution pipeline).
    """
    global _simllm_config, _kv_manager, _simhash_hasher
    global _similarity_identifier, _sandwich_config, _kv_reuse_engine
    global _original_execute_model, _original_model_forward

    config = SimLLMConfig.from_env()
    if not config.enabled:
        return

    if model_runner_cls is None:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        model_runner_cls = NPUModelRunner

    if getattr(model_runner_cls, "execute_model", None) is _simllm_execute_model:
        return

    logger.info("Applying Sim-LLM patch to NPUModelRunner …")
    logger.setLevel(logging.INFO)

    _simllm_config = config

    from vllm_ascend_simllm.kv_manager import KVManager
    from vllm_ascend_simllm.kv_reuse import KVReuseEngine
    from vllm_ascend_simllm.lsh import SimHashHasher
    from vllm_ascend_simllm.sandwich import SandwichConfig
    from vllm_ascend_simllm.similarity import SimilarityIdentifier

    _kv_manager = KVManager(
        max_cache_size=config.kv_cache_size,
        max_cache_bytes=config.kv_cache_max_bytes,
    )
    _simhash_hasher = SimHashHasher(
        dim=4096,  # default for Qwen2.5-7B; overridden after model load
        num_bits=config.lsh_num_bits,
    )
    _similarity_identifier = SimilarityIdentifier(
        cosine_threshold=config.cosine_threshold,
        lsh_batch_threshold=config.lsh_batch_threshold,
        lsh_num_bits=config.lsh_num_bits,
    )
    _sandwich_config = SandwichConfig(
        bottom_layers=config.sandwich_bottom,
        top_layers=config.sandwich_top,
    )
    _kv_reuse_engine = KVReuseEngine(
        block_size=128,  # Ascend 910B optimal; overridden from kv_cache_config later
        num_kv_heads=8,  # overridden after model load
        head_size=128,  # overridden after model load
    )

    # Patch execute_model (lightweight — triggers the full pipeline).
    _original_execute_model = model_runner_cls.execute_model
    model_runner_cls.execute_model = _simllm_execute_model  # type: ignore[method-assign]

    # Patch _model_forward to restore and snapshot per-layer KV.
    _original_model_forward = model_runner_cls._model_forward
    model_runner_cls._model_forward = _simllm_model_forward  # type: ignore[method-assign]

    logger.info(
        "Sim-LLM patch applied (cache_size=%d, threshold=%.2f, "
        "sandwich_bottom=%d, sandwich_top=%d, lsh_bits=%d).",
        config.kv_cache_size,
        config.cosine_threshold,
        config.sandwich_bottom,
        config.sandwich_top,
        config.lsh_num_bits,
    )


# ===========================================================================
# Proactive preprocessing — runs BEFORE the original execute_model so we
# can rewrite scheduler_output and avoid full prefill for matched requests.
# ===========================================================================


def _simllm_preprocess_from_scheduler(self: Any, scheduler_output: Any) -> None:
    """Extract embeddings and LSH hashes from *scheduler_output* directly.

    Runs before ``_original_execute_model`` so we can identify matches and
    rewrite ``num_computed_tokens`` before the batch is built.  Does NOT
    depend on ``input_batch`` (which is stale / not yet populated).
    """
    new_reqs = scheduler_output.scheduled_new_reqs
    self._simllm_prompt_token_ids = {
        req.req_id: tuple(req.prompt_token_ids)
        for req in new_reqs
        if _simllm_request_cacheable(req)
    }
    if not new_reqs:
        self._simllm_batch_embeddings = None
        self._simllm_batch_hashes = None
        self._simllm_batch_req_ids = None
        return

    if getattr(self, "_has_gdn", False) is True and getattr(
        _simllm_config, "hybrid_reuse_mode", "aggressive"
    ) in {"exact", "aligned", "tail", "tail_anchor"}:
        # These policies match CPU token features only; their embedding/LSH
        # fallback is disabled. Keep snapshot bookkeeping without embedding
        # lookup, NPU hashing, or the hash-to-host synchronization on misses.
        self._simllm_batch_embeddings = None
        self._simllm_batch_hashes = None
        self._simllm_batch_req_ids = None
        pending = getattr(self, "_simllm_pending_prompts", None)
        if not isinstance(pending, dict):
            pending = self._simllm_pending_prompts = {}
        token_only_embedding = torch.empty((1, 0), device="cpu")
        for req_id, prompt in self._simllm_prompt_token_ids.items():
            if prompt:
                pending[req_id] = (token_only_embedding, 0, prompt)
        return

    try:
        _reconcile_hasher_dim(self)

        # Build flat input_ids and query_start_loc from scheduler_output.
        all_ids: list[int] = []
        qsl = [0]
        req_ids: list[str] = []
        for req in new_reqs:
            req_ids.append(req.req_id)
            ids = req.prompt_token_ids or []
            all_ids.extend(ids)
            qsl.append(qsl[-1] + len(ids))

        if not all_ids:
            self._simllm_batch_embeddings = None
            self._simllm_batch_hashes = None
            self._simllm_batch_req_ids = None
            return

        input_ids = torch.tensor(all_ids, device=self.device)
        query_start_loc = torch.tensor(qsl, device=self.device)

        from vllm_ascend_simllm.hooks.preprocess import SimLLMPreprocessor

        preprocessor = SimLLMPreprocessor(
            pooling=_simllm_config.embedding_pooling,  # type: ignore[union-attr]
        )
        embeddings = preprocessor.extract_embeddings(
            self.model,
            input_ids,
            query_start_loc,
        )

        hashes = _simhash_hasher.hash(embeddings)  # type: ignore[misc]

        self._simllm_batch_embeddings = embeddings
        self._simllm_batch_hashes = hashes
        self._simllm_batch_req_ids = req_ids
        pending = getattr(self, "_simllm_pending_prompts", None)
        if not isinstance(pending, dict):
            pending = self._simllm_pending_prompts = {}
        hash_values = tensor_to_int_list(hashes)
        for idx, req_id in enumerate(req_ids):
            prompt = self._simllm_prompt_token_ids.get(req_id)
            if prompt:
                pending[req_id] = (embeddings[idx : idx + 1], hash_values[idx], prompt)

    except Exception:
        logger.exception(
            "SimLLM preprocess_from_scheduler failed — processing as unmatched."
        )
        self._simllm_batch_embeddings = None
        self._simllm_batch_hashes = None
        self._simllm_batch_req_ids = None


def _simllm_rewrite_scheduler_output(self: Any, scheduler_output: Any) -> None:
    """Keep the worker's token counts consistent when reusing a prefix."""
    self._simllm_reuse_tokens_by_req_id = {}
    match_results = getattr(self, "_simllm_match_results", None)
    if not match_results:
        return

    new_reqs = scheduler_output.scheduled_new_reqs
    for batch_idx, m in match_results.items():
        if not m.matched or m.layer_kv is None:
            continue
        if batch_idx >= len(new_reqs):
            continue

        req = new_reqs[batch_idx]
        if req.num_computed_tokens != 0:
            continue
        if (
            getattr(self, "_has_gdn", False) is True
            and m.match_type == "exact"
            and m.prefix_len < _QWEN35_MIN_USEFUL_EXACT_PREFIX
        ):
            continue
        scheduled = scheduler_output.num_scheduled_tokens.get(req.req_id, 0)
        hybrid_mode = getattr(_simllm_config, "hybrid_reuse_mode", "aggressive")
        complete_hybrid_prefix = getattr(self, "_has_gdn", False) is True and (
            hybrid_mode in {"exact", "aligned"}
            or (hybrid_mode in {"tail", "tail_anchor"} and m.match_type == "exact")
        )
        if complete_hybrid_prefix and scheduled <= m.prefix_len:
            # A recurrent snapshot describes the complete source. It cannot
            # seed a shorter chunk, even when its attention blocks can.
            continue
        if (
            getattr(self, "_has_gdn", False) is True
            and hybrid_mode in {"tail", "tail_anchor"}
            and not complete_hybrid_prefix
        ):
            skipped = min(
                m.prefix_len - 1,
                len(req.prompt_token_ids) - _simllm_config.hybrid_recompute_tokens,
                scheduled - 1,
            )
        else:
            skipped = min(
                m.prefix_len if complete_hybrid_prefix else m.prefix_len - 1,
                scheduled - 1,
            )
        if skipped <= 0:
            continue

        req.num_computed_tokens = skipped
        scheduler_output.num_scheduled_tokens[req.req_id] = scheduled - skipped
        scheduler_output.total_num_scheduled_tokens -= skipped
        self._simllm_reuse_tokens_by_req_id[req.req_id] = (m, skipped)
        logger.info(
            "SimLLM: reusing %d %s tokens for request %s (source=%s, cosine=%s)",
            skipped,
            m.match_type,
            req.req_id,
            m.source_task_id,
            m.similarity_score,
        )


def _simllm_build_injection_map_from_scheduler(
    self: Any, scheduler_output: Any
) -> None:
    """Inactive legacy helper for attention-backend injection."""
    global _simllm_injection_map
    _simllm_injection_map = None

    match_results = getattr(self, "_simllm_match_results", None)
    if not match_results:
        return

    _reconcile_kv_reuse_engine(self)

    new_reqs = scheduler_output.scheduled_new_reqs

    # Build query_start_loc from (possibly rewritten) scheduler_output.
    qsl = [0]
    for req in new_reqs:
        qsl.append(qsl[-1] + len(req.prompt_token_ids or []))

    inj_map: dict[int, tuple] = {}
    for batch_idx, m in match_results.items():
        if not m.matched or m.cached_k is None:
            continue
        if batch_idx >= len(new_reqs):
            continue

        req = new_reqs[batch_idx]
        prompt_len = len(req.prompt_token_ids or [])
        cached_len = m.cached_k.shape[2]
        covered = min(cached_len, prompt_len)
        if covered == 0:
            continue

        # Align + flatten cached KV.
        k_aligned, v_aligned = _kv_reuse_engine.prepare_injection(  # type: ignore[misc]
            m.cached_k,
            m.cached_v,
            covered,
        )
        k_flat = k_aligned.squeeze(0).permute(1, 0, 2).contiguous()
        v_flat = v_aligned.squeeze(0).permute(1, 0, 2).contiguous()
        tok_start = qsl[batch_idx]

        inj_map[batch_idx] = (k_flat, v_flat, tok_start, covered)

    if inj_map:
        _simllm_injection_map = inj_map


def _parse_layer_idx(layer_name: str) -> int | None:
    """Extract the index from a name like ``model.layers.5.self_attn``."""
    m = _LAYER_IDX_RE.search(layer_name)
    return int(m.group(1)) if m else None


def _simllm_apply_sandwich_slots(self: Any) -> None:
    """Set ``slot_mapping=-1`` on MIDDLE layers for UNMATCHED requests.

    Middle layers (not in ``keep_layers``) skip KV cache writes, saving
    ~81% of BlockTable memory per unique request.  Only ``keep_layers``
    (bottom-N + top-N) retain KV for future matching.
    """
    match_results = getattr(self, "_simllm_match_results", {})
    num_reqs = self.input_batch.num_reqs
    if num_reqs == 0:
        return

    # Build set of UNMATCHED batch indices.
    unmatched = {
        i
        for i in range(num_reqs)
        if i not in match_results or not match_results[i].matched
    }
    if not unmatched:
        return

    try:
        ctx = get_forward_context()
        slot_mapping_dict = ctx.slot_mapping
    except Exception:
        logger.debug("SimLLM sandwich: forward context not available, skipping.")
        return

    if not isinstance(slot_mapping_dict, dict):
        return  # spec-decode list path — skip for now

    keep_layers = _sandwich_config.keep_layers  # type: ignore[union-attr]

    qsl = self.query_start_loc
    if hasattr(qsl, "gpu"):
        query_start_loc = qsl.gpu[: num_reqs + 1]
    else:
        query_start_loc = qsl[: num_reqs + 1]
    seq_lens = self.seq_lens[:num_reqs]
    query_ranges = cumsum_to_ranges(query_start_loc)
    seq_len_values = tensor_to_int_list(seq_lens)

    disabled = 0
    for layer_name, sm_tensor in slot_mapping_dict.items():
        layer_idx = _parse_layer_idx(layer_name)
        if layer_idx is None or layer_idx in keep_layers:
            continue

        for batch_idx in unmatched:
            s_len = seq_len_values[batch_idx]
            if s_len == 0:
                continue
            tok_start, _ = query_ranges[batch_idx]
            tok_end = tok_start + s_len
            sm_tensor[tok_start:tok_end] = -1
        disabled += 1

    if disabled:
        logger.debug(
            "SimLLM sandwich: disabled KV cache for %d middle layers "
            "(%d unique requests, keep_layers=%s).",
            disabled,
            len(unmatched),
            sorted(keep_layers),
        )


# ===========================================================================
# execute_model wrapper — proactive preprocessing + scheduler rewrite
# ===========================================================================


def _simllm_execute_model(
    self: Any,
    scheduler_output: Any,
    intermediate_tensors: Any = None,
    **kwargs: Any,
) -> Any:
    """Wrapped ``NPUModelRunner.execute_model`` with proactive preprocessing.

    Identifies matched requests BEFORE the batch is built so we can rewrite
    ``num_computed_tokens`` and skip prefill for matched requests.
    """
    if not _simllm_config or not _simllm_config.enabled:
        return _original_execute_model(
            self, scheduler_output, intermediate_tensors, **kwargs
        )

    profile_start = (
        time.perf_counter() if getattr(_simllm_config, "profile", False) else 0.0
    )

    pending = getattr(self, "_simllm_pending_prompts", None)
    if isinstance(pending, dict) and pending:
        for req_id in getattr(scheduler_output, "finished_req_ids", ()):
            pending.pop(req_id, None)
    promotions = getattr(self, "_simllm_promotions", None)
    if isinstance(promotions, set) and promotions:
        promotions.difference_update(getattr(scheduler_output, "finished_req_ids", ()))
    features = getattr(self, "_simllm_prompt_features", None)
    if isinstance(features, dict) and features:
        for req_id in getattr(scheduler_output, "finished_req_ids", ()):
            features.pop(req_id, None)

    if not scheduler_output.scheduled_new_reqs and _simllm_decode_only_step(
        self, scheduler_output
    ):
        # Decode-only steps do not create or consume task snapshots. Avoid
        # rebuilding the hybrid layout on every generated token.
        self._simllm_skip_forward = True
        self._simllm_reuse_tokens_by_req_id = {}
        return _original_execute_model(
            self, scheduler_output, intermediate_tensors, **kwargs
        )
    self._simllm_skip_forward = False

    if not _simllm_cache_layout_compatible(self):
        self._simllm_reuse_tokens_by_req_id = {}
        self._simllm_batch_hashes = None
        return _original_execute_model(
            self, scheduler_output, intermediate_tensors, **kwargs
        )

    # Full identical prompts need no embedding lookup. This is especially
    # useful for Qwen3.5, where prefill is fast enough that matching overhead
    # can consume a meaningful share of TTFT.
    guarded_hybrid = getattr(
        self, "_has_gdn", False
    ) is True and _simllm_config.hybrid_reuse_mode in {
        "exact", "aligned", "tail", "tail_anchor"
    }
    exact = (
        _simllm_identify_prefixes(scheduler_output, complete_source_only=True)
        if guarded_hybrid
        else _simllm_identify_prefixes(scheduler_output)
    )
    new_reqs = scheduler_output.scheduled_new_reqs
    if guarded_hybrid and _simllm_config.hybrid_reuse_mode == "exact":
        _simllm_preprocess_from_scheduler(self, scheduler_output)
        self._simllm_semantic_matches = {}
        self._simllm_match_results = exact
    elif guarded_hybrid:
        token_matches = _simllm_identify_token_overlap(scheduler_output, self)
        if not new_reqs or len(token_matches) < len(new_reqs):
            _simllm_preprocess_from_scheduler(self, scheduler_output)
        else:
            self._simllm_batch_embeddings = None
            self._simllm_batch_hashes = None
            self._simllm_batch_req_ids = None
        self._simllm_semantic_matches = token_matches
        self._simllm_match_results = _simllm_select_reuse(
            scheduler_output, token_matches, exact_matches=exact
        )
    elif (
        new_reqs
        and len(exact) == len(new_reqs)
        and all(
            exact[idx].prefix_len >= len(req.prompt_token_ids)
            for idx, req in enumerate(new_reqs)
        )
    ):
        self._simllm_batch_embeddings = None
        self._simllm_batch_hashes = None
        self._simllm_batch_req_ids = None
        self._simllm_semantic_matches = {}
        self._simllm_match_results = exact
    else:
        token_matches = (
            _simllm_identify_token_overlap(scheduler_output)
            if getattr(self, "_has_gdn", False) is True
            else {}
        )
        if new_reqs and len(token_matches) == len(new_reqs):
            # The common Qwen3.5 workload has long near-duplicate prompts.
            # A CPU token sketch avoids embedding, SimHash, and NPU syncs.
            self._simllm_batch_embeddings = None
            self._simllm_batch_hashes = None
            self._simllm_batch_req_ids = None
            self._simllm_semantic_matches = token_matches
        else:
            _simllm_preprocess_from_scheduler(self, scheduler_output)
            semantic = _simllm_identify(self)
            for idx, fast in token_matches.items():
                if idx not in semantic or fast.prefix_len >= semantic[idx].prefix_len:
                    semantic[idx] = fast
            self._simllm_semantic_matches = semantic
        self._simllm_match_results = _simllm_select_reuse(
            scheduler_output, self._simllm_semantic_matches
        )

    # -- Phase 0b: Rewrite scheduler_output for matched requests ----------
    _simllm_rewrite_scheduler_output(self, scheduler_output)
    if profile_start:
        logger.info(
            "SimLLM profile match_ms=%.3f new_reqs=%d hits=%d",
            (time.perf_counter() - profile_start) * 1000,
            len(scheduler_output.scheduled_new_reqs),
            len(self._simllm_reuse_tokens_by_req_id),
        )
    _simllm_prepare_promotions(self, scheduler_output)

    self._simllm_scheduler_output = scheduler_output
    self._simllm_deferrals: set[int] = set()

    # -- Original execute_model (sees modified num_computed_tokens) -------
    outputs = _original_execute_model(
        self, scheduler_output, intermediate_tensors, **kwargs
    )

    _simllm_handle_deferrals(self)
    return outputs


def _simllm_decode_only_step(self: Any, scheduler_output: Any) -> bool:
    """Check cached requests are past prefill before skipping plugin work."""
    requests = self.requests
    for req_id in scheduler_output.num_scheduled_tokens:
        state = requests.get(req_id)
        if state is None or state.num_computed_tokens < state.num_prompt_tokens:
            return False
    return True


def _simllm_prepare_promotions(self: Any, scheduler_output: Any) -> None:
    """Retain a longer semantic target as an approximate future source.

    The target's un-reused tail is forwarded before snapshotting. Promoting
    only when it grows by at least one scheduler page limits snapshot work.
    """
    pending = getattr(self, "_simllm_pending_prompts", None)
    if not isinstance(pending, dict):
        pending = self._simllm_pending_prompts = {}
    promotions = getattr(self, "_simllm_promotions", None)
    if not isinstance(promotions, set):
        promotions = self._simllm_promotions = set()
    new_reqs = {req.req_id: req for req in scheduler_output.scheduled_new_reqs}
    for req_id, (match, skipped) in self._simllm_reuse_tokens_by_req_id.items():
        if match.match_type != "semantic":
            continue
        req = new_reqs.get(req_id)
        prompt = tuple(req.prompt_token_ids or ()) if req is not None else ()
        source = (
            _kv_manager.get_task(match.source_task_id)
            if match.source_task_id is not None
            else None
        )
        if (
            getattr(self, "_has_gdn", False) is True
            and _simllm_config.hybrid_reuse_mode == "aggressive"
            and match.hybrid_states
            and source is not None
            and len(prompt) - skipped >= _QWEN35_PROMOTION_MIN_GROWTH
        ):
            pending.setdefault(req_id, (source.embedding, source.lsh_hash, prompt))
            promotions.add(req_id)
        else:
            pending.pop(req_id, None)
            promotions.discard(req_id)


# ===========================================================================
# _model_forward wrapper — restore and snapshot physical KV blocks
# ===========================================================================


def _simllm_model_forward(
    self: Any,
    num_tokens_padded: int,
    input_ids: Any = None,
    positions: Any = None,
    intermediate_tensors: Any = None,
    inputs_embeds: Any = None,
    **model_kwargs: Any,
) -> Any:
    """Patched ``_model_forward`` for task KV reuse.

    Preprocessing and identification now happen in ``_simllm_execute_model``
    before this is called. The cached prefix is restored, the remaining
    tokens are forwarded normally, and complete KV blocks are snapshotted.
    """
    global _simllm_injection_map

    if (
        not _simllm_config
        or not _simllm_config.enabled
        or getattr(self, "_simllm_skip_forward", False)
    ):
        return _original_model_forward(
            self,
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )

    profile = getattr(_simllm_config, "profile", False)
    if profile:
        torch.npu.synchronize()
        restore_start = time.perf_counter()

    # The active path keeps all layers so a cached prefix is complete.
    _simllm_prepopulate_prefix_kv(self)
    if profile:
        torch.npu.synchronize()
        restore_end = time.perf_counter()

    # -- Original forward processes only the tokens left by the rewrite. --
    try:
        hidden_states = _original_model_forward(
            self,
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )
    finally:
        _simllm_injection_map = None

    if profile:
        torch.npu.synchronize()
        forward_end = time.perf_counter()

    # -- Extract KV + store in KVManager ----------------------------------
    _simllm_extract_kv(self, hidden_states)
    if profile:
        torch.npu.synchronize()
        logger.info(
            "SimLLM profile restore_ms=%.3f forward_ms=%.3f snapshot_ms=%.3f hits=%d",
            (restore_end - restore_start) * 1000,
            (forward_end - restore_end) * 1000,
            (time.perf_counter() - forward_end) * 1000,
            len(getattr(self, "_simllm_reuse_tokens_by_req_id", {})),
        )

    return hidden_states


# ===========================================================================
# Hook implementations
# ===========================================================================


def _simllm_preprocess(self: Any) -> None:
    """Inactive legacy helper for embedding extraction after batch setup."""
    num_reqs = self.input_batch.num_reqs
    if num_reqs == 0:
        self._simllm_batch_embeddings = None
        self._simllm_batch_hashes = None
        self._simllm_batch_req_ids = None
        return

    try:
        # Reconcile SimHashHasher dimension with actual model embedding dim.
        _reconcile_hasher_dim(self)

        # Access populated input data.
        num_tokens = self.input_batch.num_tokens[:num_reqs].sum()
        if num_tokens == 0:
            self._simllm_batch_embeddings = None
            self._simllm_batch_hashes = None
            self._simllm_batch_req_ids = None
            return

        input_ids = self.input_batch.input_ids[:num_tokens]

        qsl = self.query_start_loc
        if hasattr(qsl, "gpu"):
            query_start_loc = qsl.gpu[: num_reqs + 1]
        else:
            query_start_loc = qsl[: num_reqs + 1]

        from vllm_ascend_simllm.hooks.preprocess import SimLLMPreprocessor

        preprocessor = SimLLMPreprocessor(
            pooling=_simllm_config.embedding_pooling,  # type: ignore[union-attr]
        )
        embeddings = preprocessor.extract_embeddings(
            self.model, input_ids, query_start_loc
        )  # [num_reqs, D]

        hashes = _simhash_hasher.hash(embeddings)  # type: ignore[misc]

        self._simllm_batch_embeddings = embeddings
        self._simllm_batch_hashes = hashes
        self._simllm_batch_req_ids = list(self.input_batch.req_ids[:num_reqs])

    except Exception:
        logger.exception("SimLLM preprocess failed — falling back to normal forward.")
        self._simllm_batch_embeddings = None
        self._simllm_batch_hashes = None
        self._simllm_batch_req_ids = None


def _simllm_identify(self: Any) -> dict[int, Any]:
    """Find approximate candidates with complete KV in nearby LSH buckets."""
    embeddings = getattr(self, "_simllm_batch_embeddings", None)
    hashes = getattr(self, "_simllm_batch_hashes", None)

    if embeddings is None or hashes is None or embeddings.shape[0] == 0:
        return {}

    try:
        return _similarity_identifier.identify_reusable(
            embeddings,
            hashes,
            _kv_manager,  # type: ignore[misc]
            hamming_radius=_simllm_config.lsh_hamming_radius,
        )
    except Exception:
        logger.exception("SimLLM identify failed — processing all as unmatched.")
        return {}


def _simllm_identify_prefixes(
    scheduler_output: Any, *, complete_source_only: bool = False
) -> dict[int, Any]:
    """Select only token-identical prefixes with a complete layer snapshot."""
    from vllm_ascend_simllm.similarity import MatchResult

    matches: dict[int, MatchResult] = {}
    for batch_idx, req in enumerate(scheduler_output.scheduled_new_reqs):
        if not _simllm_request_cacheable(req):
            continue
        token_ids = tuple(req.prompt_token_ids or ())
        if req.num_computed_tokens or len(token_ids) < 2:
            continue
        result = _kv_manager.longest_prefix_match(
            token_ids, complete_source_only=complete_source_only
        )
        if result is None:
            continue
        task, prefix_len = result
        layer_kv = task.layer_kv
        if layer_kv is None:
            continue
        matches[batch_idx] = MatchResult(
            matched=True,
            source_task_id=task.task_id,
            cached_k=task.top_k,
            cached_v=task.top_v,
            layer_kv=layer_kv,
            hybrid_states=task.hybrid_states,
            prefix_len=prefix_len,
            match_type="exact",
        )
    return matches


def _simllm_prompt_features(
    self: Any, req_id: str, token_ids: tuple[int, ...]
) -> tuple[str | None, tuple[str, ...] | None, tuple[str, ...] | None]:
    """Decode once to identify the main topic, context topics and template."""
    features = getattr(self, "_simllm_prompt_features", None)
    if not isinstance(features, dict):
        features = self._simllm_prompt_features = {}
    if req_id in features:
        return features[req_id]
    try:
        tokenizer = getattr(self, "_simllm_quality_tokenizer", None)
        if tokenizer is None:
            from transformers import AutoTokenizer

            model_config = self.model_config
            tokenizer_name = model_config.tokenizer or model_config.model
            tokenizer = self._simllm_quality_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
        # Keep role delimiters and thinking markers: token-count cosine alone
        # accepts raw prompts and chat templates with incompatible boundaries.
        prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
        template = tuple(_QWEN_TEMPLATE_RE.findall(prompt))
        counts = Counter(_QUOTED_ANCHOR_RE.findall(prompt))
        most_common = counts.most_common(2)
        anchor = (
            most_common[0][0]
            if most_common
            and (len(most_common) == 1 or most_common[0][1] > most_common[1][1])
            else None
        )
        headings = tuple(sorted(set(_TOPIC_HEADING_RE.findall(prompt))))
        context_topics = headings if headings else ((anchor,) if anchor else None)
    except Exception:
        logger.exception("SimLLM: prompt feature extraction failed")
        anchor = None
        template = None
        context_topics = None
    features[req_id] = (anchor, template, context_topics)
    return anchor, template, context_topics


def _simllm_topic_anchor(
    self: Any, req_id: str, token_ids: tuple[int, ...]
) -> str | None:
    """Find a dominant quoted subject in a prompt for quality-aware reuse."""
    return _simllm_prompt_features(self, req_id, token_ids)[0]


def _simllm_identify_token_overlap(
    scheduler_output: Any, self: Any = None
) -> dict[int, Any]:
    """Select Qwen3.5 near duplicates without an NPU embedding round trip."""
    from vllm_ascend_simllm.similarity import MatchResult

    matches: dict[int, MatchResult] = {}
    aligned = getattr(_simllm_config, "hybrid_reuse_mode", "aggressive") == "aligned"
    anchored = getattr(_simllm_config, "hybrid_reuse_mode", "aggressive") in {
        "tail", "tail_anchor"
    }
    full_context_guard = (
        getattr(_simllm_config, "hybrid_reuse_mode", "aggressive") == "tail"
    )
    for batch_idx, req in enumerate(scheduler_output.scheduled_new_reqs):
        if not _simllm_request_cacheable(req) or req.num_computed_tokens:
            continue
        token_ids = tuple(req.prompt_token_ids)
        anchor, template, context_topics = (None, None, None)
        if anchored or aligned:
            if self is None:
                continue
            anchor, template, context_topics = _simllm_prompt_features(
                self, req.req_id, token_ids
            )
            if template is None:
                continue
        if anchored and (anchor is None or context_topics is None):
            continue
        candidate = _kv_manager.best_token_overlap(
            token_ids,
            _QWEN35_TOKEN_OVERLAP_THRESHOLD,
            shorter_source_only=aligned,
            min_common_prefix=256 if aligned else 0,
            min_ordered_overlap=0.55 if aligned else 0.0,
            required_anchor=anchor if anchored else None,
            required_template=template,
            required_context_topics=context_topics if full_context_guard else None,
        )
        if candidate is None:
            continue
        task, score = candidate
        matches[batch_idx] = MatchResult(
            matched=True,
            source_task_id=task.task_id,
            cached_k=task.top_k,
            cached_v=task.top_v,
            similarity_score=score,
            layer_kv=task.layer_kv,
            hybrid_states=task.hybrid_states,
            prefix_len=task.seq_len,
            match_type="semantic",
        )
    return matches


def _simllm_select_reuse(
    scheduler_output: Any,
    semantic_matches: dict[int, Any],
    *,
    exact_matches: dict[int, Any] | None = None,
) -> dict[int, Any]:
    """Use the longest reusable span; prefer exact KV when spans tie."""
    selected = (
        dict(exact_matches)
        if exact_matches is not None
        else _simllm_identify_prefixes(scheduler_output)
    )
    for batch_idx, semantic in semantic_matches.items():
        if batch_idx >= len(scheduler_output.scheduled_new_reqs):
            continue
        req = scheduler_output.scheduled_new_reqs[batch_idx]
        if not _simllm_request_cacheable(req) or req.num_computed_tokens:
            continue
        available = min(semantic.prefix_len, len(req.prompt_token_ids))
        if (
            not semantic.matched
            or semantic.layer_kv is None
            or available < 2
            or (batch_idx in selected and selected[batch_idx].prefix_len >= available)
        ):
            continue
        semantic.prefix_len = available
        selected[batch_idx] = semantic
    return selected


def _simllm_request_cacheable(req: Any) -> bool:
    """Only plain token prompts have KV determined by token IDs alone."""
    token_ids = getattr(req, "prompt_token_ids", None)
    token_mask = getattr(req, "prompt_is_token_ids", None)
    return bool(
        token_ids
        and not getattr(req, "mm_features", None)
        and getattr(req, "prompt_embeds", None) is None
        and getattr(req, "lora_request", None) is None
        and getattr(req, "pooling_params", None) is None
        and (token_mask is None or all(token_mask))
    )


def _split_kv_cache(layer_cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Get physical key and value block views for both host cache formats."""
    if isinstance(layer_cache, tuple | list):
        return layer_cache[0], layer_cache[1]
    if layer_cache.ndim == 5 and layer_cache.shape[0] == 2:
        return layer_cache[0], layer_cache[1]
    if layer_cache.ndim == 4 and layer_cache.shape[1] == 2:
        return layer_cache[:, 0], layer_cache[:, 1]
    raise ValueError(f"Unsupported KV cache layout: {tuple(layer_cache.shape)}")


def _simllm_block_ids(table: Any, row: int, start: int, count: int) -> list[int]:
    """Read the host's authoritative CPU block table without an NPU sync."""
    get_cpu_tensor = getattr(table, "get_cpu_tensor", None)
    cpu_table = get_cpu_tensor() if callable(get_cpu_tensor) else None
    if isinstance(cpu_table, torch.Tensor) and cpu_table.device.type == "cpu":
        return tensor_to_int_list(cpu_table[row, start : start + count])
    return tensor_to_int_list(table.get_device_tensor()[row, start : start + count])


def _simllm_block_table_rows(table: Any, num_reqs: int) -> list[list[int]]:
    """Materialize snapshot block IDs from the host table when available."""
    get_cpu_tensor = getattr(table, "get_cpu_tensor", None)
    cpu_table = get_cpu_tensor() if callable(get_cpu_tensor) else None
    if isinstance(cpu_table, torch.Tensor) and cpu_table.device.type == "cpu":
        return tensor_to_int_matrix(cpu_table[:num_reqs])
    return tensor_to_int_matrix(table.get_device_tensor()[:num_reqs])


def _simllm_cache_layout_compatible(self: Any) -> bool:
    """Reject unsupported recurrent caches and mismatched physical blocks."""
    if getattr(self, "_has_gdn", False) is True:
        try:
            _simllm_hybrid_layout(self)
            return True
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            if not getattr(self, "_simllm_layout_warning_emitted", False):
                logger.warning("SimLLM: unsupported hybrid cache layout: %s", exc)
                self._simllm_layout_warning_emitted = True
            return False
    try:
        configured = self.cache_config.block_size
        runner_block_size = getattr(self, "block_size", configured)
        physical = _split_kv_cache(self.kv_caches[0])[0].shape[1]
        if configured == physical == runner_block_size:
            return True
    except (AttributeError, IndexError, TypeError, ValueError):
        configured = None
        runner_block_size = None
        physical = None
    if not getattr(self, "_simllm_layout_warning_emitted", False):
        logger.warning(
            "SimLLM: disabled KV reuse because cache, runner, and physical "
            "block sizes differ (%s, %s, %s)",
            configured,
            runner_block_size,
            physical,
        )
        self._simllm_layout_warning_emitted = True
    return False


def _simllm_hybrid_layout(self: Any) -> tuple[tuple[int, int, bool], ...]:
    """Map runner layer order to cache group, block width and recurrent state."""
    groups = self.kv_cache_config.kv_cache_groups
    by_layer: dict[int, tuple[int, int, bool]] = {}
    for group_id, group in enumerate(groups):
        spec = group.kv_cache_spec
        block_size = spec.block_size
        recurrent = type(spec).__name__.endswith("MambaSpec")
        for name in group.layer_names:
            layer_idx = _parse_layer_idx(name)
            if layer_idx is None or layer_idx in by_layer:
                raise ValueError(f"ambiguous hybrid layer {name}")
            by_layer[layer_idx] = (group_id, block_size, recurrent)
    if len(by_layer) != len(self.kv_caches):
        raise ValueError("cache group and runner layer counts differ")
    layout = []
    for layer, (group_id, block_size, recurrent) in zip(
        self.kv_caches,
        (by_layer[idx] for idx in sorted(by_layer)),
        strict=True,
    ):
        if not isinstance(layer, tuple | list) or not layer:
            raise ValueError("hybrid cache layer is not a tensor tuple")
        if not all(isinstance(t, torch.Tensor) for t in layer):
            raise ValueError("hybrid cache contains a non-tensor state")
        if any(t.shape[0] != layer[0].shape[0] for t in layer):
            raise ValueError("hybrid state block counts differ")
        if not recurrent:
            if len(layer) != 2 or layer[0].ndim < 2:
                raise ValueError("unsupported attention cache tensor shape")
            physical_width = layer[0].shape[1]
            # Ascend expands one scheduler page into smaller kernel blocks.
            # The group's block table already contains the expanded IDs.
            if block_size % physical_width:
                raise ValueError("attention kernel block does not divide page")
            table_width = getattr(
                self.input_batch.block_table[group_id], "block_size", None
            )
            if isinstance(table_width, int) and table_width != physical_width:
                raise ValueError("attention block table and tensor widths differ")
            block_size = physical_width
        layout.append((group_id, block_size, recurrent))
    return tuple(layout)


def _simllm_prepopulate_prefix_kv(self: Any) -> None:
    """Seed every attention layer before the shortened forward executes."""
    reuse = getattr(self, "_simllm_reuse_tokens_by_req_id", None)
    if not reuse:
        return
    if getattr(self, "_has_gdn", False) is True:
        _simllm_prepopulate_hybrid_states(self, reuse)
        return
    kv_caches = self.kv_caches
    block_table = self.input_batch.block_table[0]
    req_id_to_row = {
        req_id: row
        for row, req_id in enumerate(
            self.input_batch.req_ids[: self.input_batch.num_reqs]
        )
    }
    for req_id, (match, skipped) in reuse.items():
        row = req_id_to_row.get(req_id)
        if row is None or match.layer_kv is None:
            raise RuntimeError("SimLLM: matched request missing from worker batch")
        if len(match.layer_kv) != len(kv_caches):
            raise RuntimeError("SimLLM: cached layer count differs from worker")
        block_size = _split_kv_cache(kv_caches[0])[0].shape[1]
        num_blocks = KVReuseEngine.num_blocks_needed(skipped, block_size)
        block_ids = _simllm_block_ids(block_table, row, 0, num_blocks)
        if len(block_ids) != num_blocks or any(block_id < 0 for block_id in block_ids):
            raise RuntimeError("SimLLM: missing physical blocks for reused prefix")
        block_indices = torch.tensor(
            block_ids, dtype=torch.long, device=_split_kv_cache(kv_caches[0])[0].device
        )
        for layer_cache, (cached_k, cached_v) in zip(
            kv_caches, match.layer_kv, strict=True
        ):
            k_cache, v_cache = _split_kv_cache(layer_cache)
            if cached_k.shape[0] < num_blocks or cached_v.shape[0] < num_blocks:
                raise RuntimeError("SimLLM: cached KV has too few physical blocks")
            KVReuseEngine.write_blocks(k_cache, block_indices, cached_k[:num_blocks])
            KVReuseEngine.write_blocks(v_cache, block_indices, cached_v[:num_blocks])


def _simllm_prepopulate_hybrid_states(self: Any, reuse: dict[str, Any]) -> None:
    """Restore attention blocks and the current DeltaNet running state."""
    layout = _simllm_hybrid_layout(self)
    tail_mode = getattr(_simllm_config, "hybrid_reuse_mode", None) in {
        "tail", "tail_anchor"
    }
    zero_recurrent_states: dict[int, tuple[torch.Tensor, ...]] = {}
    req_ids = self.input_batch.req_ids[: self.input_batch.num_reqs]
    req_rows = {req_id: row for row, req_id in enumerate(req_ids)}
    seq_lens = tensor_to_int_list(self.seq_lens[: self.input_batch.num_reqs])
    block_tables = self.input_batch.block_table
    batch_writes = len(reuse) >= _QWEN35_BATCH_RESTORE_MIN_HITS
    updates: list[list[tuple[torch.Tensor, tuple[torch.Tensor, ...]]]] = [
        [] for _ in layout
    ]
    seen_ids: dict[tuple[int, bool], set[int]] = {}
    disjoint_destinations = True
    for req_id, (match, skipped) in reuse.items():
        row = req_rows.get(req_id)
        if row is None or not match.hybrid_states or match.layer_kv is None:
            raise RuntimeError("SimLLM: hybrid match has no complete state")
        if len(match.layer_kv) != len(layout):
            raise RuntimeError("SimLLM: hybrid state layer count differs")
        indices_by_group: dict[tuple[int, bool], torch.Tensor] = {}
        for layer_idx, (layer, saved, (group_id, block_size, recurrent)) in enumerate(
            zip(self.kv_caches, match.layer_kv, layout, strict=True)
        ):
            if len(saved) != len(layer):
                raise RuntimeError("SimLLM: hybrid state tensor count differs")
            if recurrent and tail_mode and match.match_type != "exact":
                if layer_idx not in zero_recurrent_states:
                    zero_recurrent_states[layer_idx] = tuple(
                        torch.zeros_like(blocks[:1]) for blocks in layer
                    )
                saved = zero_recurrent_states[layer_idx]
            key = (group_id, recurrent)
            block_indices = indices_by_group.get(key)
            if block_indices is None:
                if recurrent:
                    # preprocess_mamba already ran before _model_forward. Its
                    # destination is the final block for this scheduled step.
                    state_block = (
                        KVReuseEngine.num_blocks_needed(seq_lens[row], block_size) - 1
                    )
                    block_ids = _simllm_block_ids(
                        block_tables[group_id], row, state_block, 1
                    )
                    expected = 1
                else:
                    expected = KVReuseEngine.num_blocks_needed(skipped, block_size)
                    block_ids = _simllm_block_ids(
                        block_tables[group_id], row, 0, expected
                    )
                if len(block_ids) != expected or any(
                    block_id < 0 for block_id in block_ids
                ):
                    raise RuntimeError("SimLLM: missing hybrid cache blocks")
                if batch_writes:
                    prior_ids = seen_ids.setdefault(key, set())
                    if prior_ids.intersection(block_ids):
                        disjoint_destinations = False
                    prior_ids.update(block_ids)
                block_indices = torch.tensor(
                    block_ids, dtype=torch.long, device=layer[0].device
                )
                indices_by_group[key] = block_indices
            for blocks in saved:
                if blocks.shape[0] < len(block_indices):
                    raise RuntimeError("SimLLM: hybrid snapshot is too short")
            if batch_writes:
                updates[layer_idx].append((block_indices, saved))
            else:
                for destination, blocks in zip(layer, saved, strict=True):
                    KVReuseEngine.write_blocks(
                        destination, block_indices, blocks[: len(block_indices)]
                    )

    if batch_writes:
        for layer, layer_updates in zip(self.kv_caches, updates, strict=True):
            if disjoint_destinations:
                indices = torch.cat([entry[0] for entry in layer_updates])
                for tensor_idx, destination in enumerate(layer):
                    blocks = torch.cat(
                        [
                            saved[tensor_idx][: len(index)]
                            for index, saved in layer_updates
                        ]
                    )
                    KVReuseEngine.write_blocks(destination, indices, blocks)
            else:
                # index_copy_ with duplicate indices has undefined write order.
                for index, saved in layer_updates:
                    for destination, blocks in zip(layer, saved, strict=True):
                        KVReuseEngine.write_blocks(
                            destination, index, blocks[: len(index)]
                        )


def _simllm_inject_kv(self: Any) -> None:
    """Inactive legacy helper for top-layer KV injection.

    The active path restores complete per-layer physical blocks before forward.
    """
    match_results = getattr(self, "_simllm_match_results", None)
    if not match_results:
        return

    if not hasattr(self, "kv_caches") or not self.kv_caches:
        logger.debug("SimLLM inject_kv: kv_caches not available yet, skipping.")
        return

    try:
        num_reqs = self.input_batch.num_reqs
        blk_table = self.input_batch.block_table[0]
        blk_table_tensor = blk_table.get_device_tensor()
        block_size = _kv_reuse_engine._block_size

        _reconcile_kv_reuse_engine(self)

        # Determine target layers: top-N only.
        num_layers = len(self.kv_caches)
        top_n = _sandwich_config.top_layers  # type: ignore[union-attr]
        if top_n <= 0 or top_n >= num_layers:
            target_layers = self.kv_caches  # fallback: all layers
        else:
            target_layers = self.kv_caches[num_layers - top_n :]

        matched_count = 0
        for batch_idx, m in match_results.items():
            if not m.matched or batch_idx >= num_reqs:
                continue
            if m.cached_k is None or m.cached_v is None:
                continue

            seq_len = int(self.seq_lens[batch_idx].item())
            k_aligned, v_aligned = _kv_reuse_engine.prepare_injection(
                m.cached_k, m.cached_v, seq_len
            )

            num_blocks = KVReuseEngine.num_blocks_needed(seq_len, block_size)
            block_ids = blk_table_tensor[batch_idx, :num_blocks].tolist()
            if not block_ids:
                continue

            # Write cached KV into top-N layers' kv_cache at those blocks.
            for layer_kv in target_layers:
                if isinstance(layer_kv, tuple):
                    k_cache, v_cache = layer_kv
                else:
                    k_cache, v_cache = layer_kv[0], layer_kv[1]
                _kv_reuse_engine.write_to_cache(
                    k_cache,
                    v_cache,
                    block_ids,
                    k_aligned,
                    v_aligned,
                )

            matched_count += 1

        if matched_count:
            logger.debug(
                "SimLLM inject_kv: injected cached KV for %d matched requests "
                "(top-%d of %d layers).",
                matched_count,
                top_n,
                num_layers,
            )

    except Exception:
        logger.exception("SimLLM inject_kv failed — continuing with normal forward.")


def _simllm_extract_kv(self: Any, hidden_states: Any) -> None:
    """Store complete per-layer prompt KV and its input embedding."""
    if hidden_states is None:
        return

    num_reqs = self.input_batch.num_reqs
    if num_reqs == 0:
        return

    hashes = getattr(self, "_simllm_batch_hashes", None)
    embeddings = getattr(self, "_simllm_batch_embeddings", None)
    batch_req_ids = getattr(self, "_simllm_batch_req_ids", None)
    pending = getattr(self, "_simllm_pending_prompts", None)
    if not isinstance(pending, dict):
        pending = {}
    if not pending and (
        hashes is None
        or embeddings is None
        or getattr(hashes, "shape", (0,))[0] == 0
        or not batch_req_ids
    ):
        logger.debug(
            "SimLLM extract_kv: no prefill hashes/req_ids for this step, skipping."
        )
        return

    try:
        kv_caches = getattr(self, "kv_caches", None)
        if not kv_caches:
            logger.debug("SimLLM extract_kv: kv_caches not available, skipping.")
            return

        # -- Build CachedTask per request -------------------------------
        req_ids = list(self.input_batch.req_ids[:num_reqs])
        match_results = getattr(self, "_simllm_match_results", {})
        if pending:
            store_plan = [
                (row_idx, -1)
                for row_idx, req_id in enumerate(req_ids)
                if req_id in pending
            ]
        else:
            hash_values = tensor_to_int_list(hashes)
            prompt_token_ids = getattr(self, "_simllm_prompt_token_ids", {})
            store_plan = _simllm_build_store_plan(
                req_ids, batch_req_ids, len(hash_values)
            )

        if not store_plan:
            logger.debug(
                "SimLLM extract_kv: no current input_batch rows matched "
                "prefill req_ids, skipping."
            )
            return

        seq_len_values = tensor_to_int_list(self.seq_lens[:num_reqs])
        hybrid = getattr(self, "_has_gdn", False) is True
        if hybrid:
            layout = _simllm_hybrid_layout(self)
            block_table_rows_by_group = [
                _simllm_block_table_rows(
                    self.input_batch.block_table[group_id], num_reqs
                )
                for group_id in range(len(self.kv_cache_config.kv_cache_groups))
            ]
        else:
            all_kv = [_split_kv_cache(layer) for layer in kv_caches]
            block_size = all_kv[0][0].shape[1]
            block_table_rows = _simllm_block_table_rows(
                self.input_batch.block_table[0], num_reqs
            )

        now = time.monotonic()
        stored = 0

        for row_idx, hash_idx in store_plan:
            req_id = req_ids[row_idx]
            if req_id in pending:
                emb, hsh, prompt = pending[req_id]
            else:
                prompt = prompt_token_ids.get(req_id, ())
                emb = embeddings[hash_idx : hash_idx + 1]
                hsh = hash_values[hash_idx]
            s_len = min(seq_len_values[row_idx], len(prompt))
            if s_len < len(prompt):
                continue
            reuse = getattr(self, "_simllm_reuse_tokens_by_req_id", {}).get(
                req_ids[row_idx]
            )
            if reuse is not None:
                match = reuse[0]
                # A longer semantic target can upgrade the reusable span
                # after its remaining tokens have been forwarded. Other
                # approximate hits do not need another snapshot.
                promotions = getattr(self, "_simllm_promotions", set())
                if (match.match_type == "semantic" and req_id not in promotions) or (
                    match.match_type == "exact" and match.prefix_len >= len(prompt)
                ):
                    pending.pop(req_id, None)
                    continue

            if hybrid:
                layer_states = []
                for layer, (group_id, group_block_size, recurrent) in zip(
                    kv_caches, layout, strict=True
                ):
                    if recurrent and _simllm_config.hybrid_reuse_mode in {
                        "tail", "tail_anchor"
                    }:
                        # Approximate tail hits always reset this state. Avoid
                        # gathering every GDN layer for cold source snapshots.
                        layer_states.append(tuple(tensor[:0] for tensor in layer))
                        continue
                    count = KVReuseEngine.num_blocks_needed(s_len, group_block_size)
                    ids = block_table_rows_by_group[group_id][row_idx]
                    block_ids = ids[count - 1 : count] if recurrent else ids[:count]
                    if len(block_ids) != (1 if recurrent else count) or any(
                        block_id < 0 for block_id in block_ids
                    ):
                        raise RuntimeError("SimLLM: missing hybrid source blocks")
                    layer_states.append(
                        tuple(
                            KVReuseEngine.gather_blocks(tensor, block_ids)
                            for tensor in layer
                        )
                    )
                layer_kv = tuple(layer_states)
                last_attention = max(
                    idx for idx, (_, _, recurrent) in enumerate(layout) if not recurrent
                )
                k_per_req, v_per_req = layer_kv[last_attention]
            else:
                num_blk = KVReuseEngine.num_blocks_needed(s_len, block_size)
                block_ids = block_table_rows[row_idx][:num_blk]
                if not block_ids:
                    continue
                layer_kv = tuple(
                    (
                        KVReuseEngine.gather_blocks(k_cache, block_ids),
                        KVReuseEngine.gather_blocks(v_cache, block_ids),
                    )
                    for k_cache, v_cache in all_kv
                )
                k_per_req, v_per_req = layer_kv[-1]
            from vllm_ascend_simllm.kv_manager import CachedTask

            task = CachedTask(
                task_id=req_ids[row_idx],
                embedding=emb,
                lsh_hash=hsh,
                top_k=k_per_req,
                top_v=v_per_req,
                last_access_time=now,
                seq_len=s_len,
                prompt_token_ids=prompt[:s_len],
                layer_kv=layer_kv,
                hybrid_states=hybrid,
                recurrent_valid=not (
                    hybrid
                    and _simllm_config.hybrid_reuse_mode in {"tail", "tail_anchor"}
                ),
                topic_anchor=(
                    _simllm_topic_anchor(self, req_id, prompt)
                    if hybrid
                    and _simllm_config.hybrid_reuse_mode in {"tail", "tail_anchor"}
                    else None
                ),
                template_signature=(
                    _simllm_prompt_features(self, req_id, prompt)[1]
                    if hybrid
                    and _simllm_config.hybrid_reuse_mode
                    in {"tail", "tail_anchor", "aligned"}
                    else None
                ),
                context_topics=(
                    _simllm_prompt_features(self, req_id, prompt)[2]
                    if hybrid
                    and _simllm_config.hybrid_reuse_mode in {"tail", "tail_anchor"}
                    else None
                ),
            )
            _kv_manager.store(task)  # type: ignore[misc]
            pending.pop(req_id, None)
            promotions = getattr(self, "_simllm_promotions", None)
            if isinstance(promotions, set):
                promotions.discard(req_id)
            if _kv_manager.get_kv(task.task_id) is not None:
                stored += 1
            else:
                logger.info(
                    "SimLLM: task %s snapshot (%d bytes) exceeds cache budget",
                    task.task_id,
                    _kv_manager._task_bytes(task),
                )

        # -- Compute diagnostic-only deferral decisions -------------------
        from vllm_ascend_simllm.hooks.postprocess import SimLLMPostprocessor

        postprocessor = SimLLMPostprocessor(
            kv_manager=_kv_manager,  # type: ignore[misc]
            pooling=_simllm_config.embedding_pooling,  # type: ignore[union-attr]
            deferral_ratio=_simllm_config.deferral_ratio,  # type: ignore[union-attr]
            max_deferrals=_simllm_config.max_deferrals,  # type: ignore[union-attr]
        )
        self._simllm_deferrals = postprocessor.compute_deferrals(
            match_results,
            num_reqs,
        )

        if stored:
            logger.info(
                "SimLLM extract_kv: stored %d tasks (cache size=%d, layers=%d).",
                stored,
                _kv_manager.size(),
                len(kv_caches),
            )

    except Exception:
        logger.exception("SimLLM extract_kv failed — KV not stored for this batch.")


def _simllm_protect_kv_slots(self: Any) -> None:
    """Set slot_mapping to -1 for tokens already covered by cached KV injection.

    Legacy/test-support helper for the earlier pre-population path.  The
    current primary path no longer calls this helper.

    Prevents ``unified_kv_cache_update`` from overwriting pre-populated
    cached KV positions inside ``self.kv_caches``.  ``flash_attn_varlen_func``
    reads from *block_table* (which is untouched), so it still finds the
    injected KV at those blocks.

    Must run inside ``_model_forward`` where ``set_forward_context`` is active
    and ``slot_mapping`` is accessible via ``get_forward_context()``.
    """
    match_results = getattr(self, "_simllm_match_results", None)
    if not match_results:
        return

    try:
        ctx = get_forward_context()
        slot_mapping_raw = ctx.slot_mapping
    except Exception:
        logger.debug(
            "SimLLM protect_kv_slots: forward context not available, skipping."
        )
        return

    if slot_mapping_raw is None:
        return

    # Normalise to list-of-dicts (spec-decode path uses a list).
    if isinstance(slot_mapping_raw, list):
        mappings_list: list[dict] = slot_mapping_raw
    else:
        mappings_list = [slot_mapping_raw]

    num_reqs = self.input_batch.num_reqs

    qsl = self.query_start_loc
    if hasattr(qsl, "gpu"):
        query_start_loc = qsl.gpu[: num_reqs + 1]
    else:
        query_start_loc = qsl[: num_reqs + 1]

    seq_lens = self.seq_lens[:num_reqs]

    protected_total = 0
    matched_count = 0

    for batch_idx, m in match_results.items():
        if not m.matched or m.cached_k is None:
            continue
        if batch_idx >= num_reqs:
            continue

        cached_len = m.cached_k.shape[2]  # L_kv in [1, H, L, D]
        req_seq_len = int(seq_lens[batch_idx].item())

        # Only protect tokens that have REAL cached KV (not zero-padding).
        covered = min(cached_len, req_seq_len)
        if covered == 0:
            continue

        tok_start = int(query_start_loc[batch_idx].item())
        tok_end = tok_start + covered

        # Write -1 across every layer's slot_mapping tensor so
        # reshape_and_cache_flash skips those positions.
        for sm_dict in mappings_list:
            for sm in sm_dict.values():
                sm[tok_start:tok_end] = -1

        # Tell vLLM internals that these tokens are already cached.
        with contextlib.suppress(AttributeError, IndexError):
            self.input_batch.num_computed_tokens_cpu[batch_idx] = covered

        protected_total += covered
        matched_count += 1

    if protected_total:
        logger.debug(
            "SimLLM protect_kv_slots: protected %d token slots across "
            "%d matched requests.",
            protected_total,
            matched_count,
        )


def _simllm_handle_deferrals(self: Any) -> None:
    """Log diagnostic deferral decisions from the just-completed forward.

    Phase 3 keeps deferral as future/backlog input only.  This helper must not
    re-queue, delay, drop, or reorder requests.
    """
    deferrals: set[int] = getattr(self, "_simllm_deferrals", set())
    if deferrals:
        logger.debug(
            "SimLLM: %d tasks flagged for future deferral diagnostics; "
            "processing continues in the current batch.",
            len(deferrals),
        )


# ===========================================================================
# Internal helpers
# ===========================================================================


def _reconcile_hasher_dim(self: Any) -> None:
    """Re-create SimHashHasher if the model embedding dim differs from default."""
    global _simhash_hasher
    try:
        embed_dim = resolve_input_embedding_dim(self.model)
    except Exception:
        return
    if _simhash_hasher.dim != embed_dim:
        from vllm_ascend_simllm.lsh import SimHashHasher

        _simhash_hasher = SimHashHasher(
            dim=embed_dim,
            num_bits=_simllm_config.lsh_num_bits,  # type: ignore[union-attr]
        )
        logger.info("SimLLM: re-created SimHashHasher with dim=%d.", embed_dim)


def _simllm_build_store_plan(
    input_batch_req_ids: list[str],
    prefill_req_ids: list[str],
    num_hashes: int,
) -> list[tuple[int, int]]:
    """Map prefill hash rows to current input_batch row indices."""
    req_id_to_row = {req_id: idx for idx, req_id in enumerate(input_batch_req_ids)}
    store_plan: list[tuple[int, int]] = []
    for hash_idx, req_id in enumerate(prefill_req_ids[:num_hashes]):
        row_idx = req_id_to_row.get(req_id)
        if row_idx is not None:
            store_plan.append((row_idx, hash_idx))
    return store_plan


def _reconcile_kv_reuse_engine(self: Any) -> None:
    """Update KVReuseEngine block_size / num_kv_heads / head_size from actual caches."""
    kv_caches = getattr(self, "kv_caches", None)
    if not kv_caches:
        return
    sample = kv_caches[-1][0]
    # sample: [num_blocks, block_size, num_kv_heads, head_size]
    bs = sample.shape[1]
    nh = sample.shape[2]
    hs = sample.shape[3]
    if (
        _kv_reuse_engine._block_size != bs
        or _kv_reuse_engine._num_kv_heads != nh
        or _kv_reuse_engine._head_size != hs
    ):
        _kv_reuse_engine._block_size = bs
        _kv_reuse_engine._num_kv_heads = nh
        _kv_reuse_engine._head_size = hs
        logger.debug(
            "SimLLM: KVReuseEngine reconciled: block=%d, heads=%d, dim=%d.",
            bs,
            nh,
            hs,
        )


def _per_request_embeddings(
    hidden_states: Any,
    query_start_loc: Any,
    pooling: str = "mean",
) -> Any | None:
    """Compute per-request L2-normalized embeddings from flat hidden states."""
    ranges = cumsum_to_ranges(query_start_loc)
    num_reqs = len(ranges)
    if num_reqs == 0:
        return None
    max_len = 0
    slices: list[Any] = []
    for start, end in ranges:
        if end > start:
            sl = hidden_states[start:end]
            slices.append(sl)
            max_len = max(max_len, sl.shape[0])
    if not slices:
        return None
    D = slices[0].shape[-1]
    padded = hidden_states.new_zeros(len(slices), max_len, D)
    for i, s in enumerate(slices):
        padded[i, : s.shape[0], :] = s
    from vllm_ascend_simllm.embedding import extract_embedding

    return extract_embedding(padded, pooling=pooling)
