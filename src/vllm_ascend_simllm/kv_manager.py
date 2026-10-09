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

"""KVManager — LRU-evicted store for task embeddings, LSH hashes, and top-layer KV.

Operates at the *task/semantic* level, distinct from vLLM's token/block-level
BlockManager.  Uses ``collections.OrderedDict`` for O(1) LRU eviction and a
separate ``dict[int, list[str]]`` for LSH bucket indexing.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import TypeAlias

import torch

LayerKV: TypeAlias = tuple[tuple[torch.Tensor, ...], ...]
DEFAULT_MAX_KV_CACHE_BYTES = 512 * 1024 * 1024


@dataclass
class CachedTask:
    """A single task snapshot stored in KVManager.

    Attributes
    ----------
    task_id:
        Unique request/task identifier.
    embedding:
        Pooled task embedding ``[1, D]`` (L2-normalized).
        Token-only hybrid policies use an empty CPU ``[1, 0]`` tensor and
        never pass these entries to the embedding/LSH matcher.
    lsh_hash:
        Packed int64 SimHash of the embedding.
    top_k:
        Top-layer Key snapshot. The active worker path stores physical blocks.
    top_v:
        Top-layer Value snapshot, with the same layout as ``top_k``.
    last_access_time:
        Monotonic timestamp for LRU (seconds, from ``time.monotonic()``).
    seq_len:
        Original sequence length — used for shape compatibility checks.
    prompt_token_ids:
        Token IDs represented by the cached prompt KV.
    layer_kv:
        Key and value physical blocks for every attention layer.
    """

    task_id: str
    embedding: torch.Tensor
    lsh_hash: int
    top_k: torch.Tensor
    top_v: torch.Tensor
    last_access_time: float
    seq_len: int
    prompt_token_ids: tuple[int, ...] = ()
    layer_kv: LayerKV | None = None
    hybrid_states: bool = False
    recurrent_valid: bool = True
    token_counts: Counter[int] | None = None
    token_norm: float = 0.0
    topic_anchor: str | None = None
    template_signature: tuple[str, ...] | None = None
    context_topics: tuple[str, ...] | None = None


class KVManager:
    """LRU-evicted store with LSH bucket indexing.

    Parameters
    ----------
    max_cache_size:
        Maximum number of CachedTask entries.  When exceeded the
        least-recently-accessed entry is evicted automatically.
    """

    def __init__(
        self,
        max_cache_size: int = 1024,
        max_cache_bytes: int = DEFAULT_MAX_KV_CACHE_BYTES,
    ) -> None:
        self._max_cache_size = max_cache_size
        self._max_cache_bytes = max_cache_bytes
        self._cache_bytes = 0
        # OrderedDict for O(1) LRU: most-recently-used at the right end.
        self._cache: OrderedDict[str, CachedTask] = OrderedDict()
        # LSH bucket index: hash → [task_id, ...]
        self._buckets: dict[int, list[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, task: CachedTask) -> None:
        """Insert (or overwrite) a task and maintain LRU + bucket invariants.

        If the task_id already exists the old entry is replaced (including
        bucket membership).  If the cache is over capacity after insertion,
        ``evict_lru()`` is called.
        """
        # Remove old bucket entry if overwriting.
        if task.task_id in self._cache:
            old = self._cache[task.task_id]
            self._remove_from_bucket(old.lsh_hash, task.task_id)
            self._cache_bytes -= self._task_bytes(old)

        task_bytes = self._task_bytes(task)
        if task_bytes > self._max_cache_bytes:
            self._cache.pop(task.task_id, None)
            return

        if task.prompt_token_ids:
            task.token_counts = Counter(task.prompt_token_ids)
            task.token_norm = math.sqrt(
                sum(count * count for count in task.token_counts.values())
            )

        self._cache[task.task_id] = task
        self._cache_bytes += task_bytes
        self._cache.move_to_end(task.task_id)

        self._buckets.setdefault(task.lsh_hash, []).append(task.task_id)

        while (
            len(self._cache) > self._max_cache_size
            or self._cache_bytes > self._max_cache_bytes
        ):
            self.evict_lru()

    def longest_prefix_match(
        self,
        token_ids: tuple[int, ...],
        min_tokens: int = 2,
        complete_source_only: bool = False,
    ) -> tuple[CachedTask, int] | None:
        """Find the longest identical token prefix with complete per-layer KV."""
        best: CachedTask | None = None
        best_len = min_tokens - 1
        for task in self._cache.values():
            if task.layer_kv is None or not task.prompt_token_ids:
                continue
            if complete_source_only and not task.recurrent_valid:
                continue
            limit = min(len(token_ids), len(task.prompt_token_ids), task.seq_len)
            if limit <= best_len or token_ids[0] != task.prompt_token_ids[0]:
                continue
            matched = 0
            for left, right in zip(
                token_ids[:limit], task.prompt_token_ids[:limit], strict=True
            ):
                if left != right:
                    break
                matched += 1
            if complete_source_only and (
                matched != task.seq_len or matched >= len(token_ids)
            ):
                continue
            if matched > best_len:
                best = task
                best_len = matched
        if best is None:
            return None
        best.last_access_time = time.monotonic()
        self._cache.move_to_end(best.task_id)
        return best, best_len

    def best_token_overlap(
        self,
        token_ids: tuple[int, ...],
        threshold: float,
        *,
        shorter_source_only: bool = False,
        min_common_prefix: int = 0,
        min_ordered_overlap: float = 0.0,
        required_anchor: str | None = None,
        required_template: tuple[str, ...] | None = None,
        required_context_topics: tuple[str, ...] | None = None,
    ) -> tuple[CachedTask, float] | None:
        """Find a cached prompt with high token-count cosine on the CPU.

        This cheap path avoids an NPU embedding lookup for nearly identical
        long prompts. Among accepted candidates, prefer the longest reusable
        span, then the highest overlap score.
        """
        query = Counter(token_ids)
        query_norm = math.sqrt(sum(count * count for count in query.values()))
        if query_norm == 0:
            return None
        best: CachedTask | None = None
        best_score = threshold
        best_span = 0
        query_ngrams = (
            set(
                zip(
                    token_ids, token_ids[1:], token_ids[2:], token_ids[3:], strict=False
                )
            )
            if min_ordered_overlap
            else None
        )
        for task in self._cache.values():
            counts = task.token_counts
            if task.layer_kv is None or not counts or task.token_norm == 0:
                continue
            if required_anchor is not None and task.topic_anchor != required_anchor:
                continue
            if (
                required_context_topics is not None
                and task.context_topics != required_context_topics
            ):
                continue
            if (
                required_template is not None
                and task.template_signature != required_template
            ):
                continue
            if shorter_source_only and task.seq_len >= len(token_ids):
                continue
            if min_common_prefix:
                common = 0
                for left, right in zip(token_ids, task.prompt_token_ids, strict=False):
                    if left != right:
                        break
                    common += 1
                if common < min_common_prefix:
                    continue
            if query_ngrams is not None:
                source = task.prompt_token_ids
                source_ngrams = set(
                    zip(source, source[1:], source[2:], source[3:], strict=False)
                )
                union = len(query_ngrams | source_ngrams)
                if (
                    not union
                    or len(query_ngrams & source_ngrams) / union < min_ordered_overlap
                ):
                    continue
            dot = sum(count * counts.get(token, 0) for token, count in query.items())
            score = dot / (query_norm * task.token_norm)
            span = min(task.seq_len, len(token_ids))
            if score >= threshold and (
                span > best_span or (span == best_span and score > best_score)
            ):
                best = task
                best_span = span
                best_score = score
        if best is None:
            return None
        self.get_kv(best.task_id)
        return best, best_score

    def lookup_by_hash(self, lsh_hash: int) -> list[CachedTask]:
        """Return all cached tasks whose LSH hash matches *exactly*.

        Updates ``last_access_time`` on every returned task (LRU refresh).
        """
        task_ids = self._buckets.get(lsh_hash, [])
        results: list[CachedTask] = []
        now = time.monotonic()
        for tid in task_ids:
            if tid in self._cache:
                task = self._cache[tid]
                task.last_access_time = now
                # Move to right end (most-recently-used).
                self._cache.move_to_end(tid)
                results.append(task)
        return results

    def candidates_within_hamming(
        self, lsh_hash: int, radius: int, num_bits: int
    ) -> list[CachedTask]:
        """Find cached tasks in nearby SimHash buckets without changing LRU order."""
        if radius < 0 or not 1 <= num_bits <= 64:
            raise ValueError("Invalid SimHash radius or width")
        mask = (1 << num_bits) - 1
        return [
            task
            for task in self._cache.values()
            if (((task.lsh_hash ^ lsh_hash) & mask).bit_count() <= radius)
        ]

    def get_kv(self, task_id: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return ``(top_k, top_v)`` for *task_id*, or None if not found.

        Access refreshes the LRU position.
        """
        task = self.get_task(task_id)
        if task is None:
            return None
        return task.top_k, task.top_v

    def get_task(self, task_id: str) -> CachedTask | None:
        """Return a cached task and refresh its LRU position."""
        task = self._cache.get(task_id)
        if task is not None:
            task.last_access_time = time.monotonic()
            self._cache.move_to_end(task_id)
        return task

    def evict_lru(self) -> None:
        """Evict the least-recently-accessed entry (O(1))."""
        if not self._cache:
            return
        task_id, task = self._cache.popitem(last=False)
        self._cache_bytes -= self._task_bytes(task)
        self._remove_from_bucket(task.lsh_hash, task_id)

    def size(self) -> int:
        """Return the current number of cached tasks."""
        return len(self._cache)

    def clear(self) -> None:
        """Reset all state (useful for model reload / testing)."""
        self._cache.clear()
        self._buckets.clear()
        self._cache_bytes = 0

    @staticmethod
    def _task_bytes(task: CachedTask) -> int:
        tensors = (
            (tensor for pair in task.layer_kv for tensor in pair)
            if task.layer_kv is not None
            else (task.top_k, task.top_v)
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_from_bucket(self, lsh_hash: int, task_id: str) -> None:
        """Remove *task_id* from the bucket for *lsh_hash*.

        Cleans up the bucket list and the bucket key if it becomes empty.
        """
        bucket = self._buckets.get(lsh_hash)
        if bucket is None:
            return
        with contextlib.suppress(ValueError):
            bucket.remove(task_id)
        if not bucket:
            del self._buckets[lsh_hash]
