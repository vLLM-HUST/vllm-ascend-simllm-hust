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

"""SimilarityIdentifier — inter-task similarity matching.

The active worker path probes nearby SimHash buckets and ranks candidates by
cosine score. The legacy ``identify`` helper retains two batch strategies:

* **Small batch** (``batch_size < lsh_batch_threshold``): exhaustive cosine
  similarity against all candidates in the LSH bucket.
* **Large batch** (``batch_size >= lsh_batch_threshold``): LSH bucket membership
  with KV merging (average K and V across all tasks in the same bucket).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_ascend_simllm.kv_manager import CachedTask, KVManager, LayerKV
from vllm_ascend_simllm.lsh import SimHashHasher, cosine_similarity
from vllm_ascend_simllm.utils import tensor_to_float_list, tensor_to_int_list


@dataclass
class MatchResult:
    """Result of similarity matching for one batch element.

    Attributes
    ----------
    matched:
        True if a similar cached task was found.
    source_task_id:
        Task id of the matched cached task, or None.
    cached_k:
        Top-layer K from the matched task. Layout depends on cache producer.
    cached_v:
        Top-layer V from the matched task.
    similarity_score:
        Cosine similarity score, or None if LSH-merge was used.
    match_type:
        ``"exact"`` for a token prefix or ``"semantic"`` for approximate
        task matching in the active worker path.
    """

    matched: bool
    source_task_id: str | None = None
    cached_k: torch.Tensor | None = None
    cached_v: torch.Tensor | None = None
    similarity_score: float | None = None
    layer_kv: LayerKV | None = None
    hybrid_states: bool = False
    prefix_len: int = 0
    match_type: str | None = None


class SimilarityIdentifier:
    """Adaptive similarity matching engine.

    Parameters
    ----------
    cosine_threshold:
        Cosine-similarity threshold θ.  Embeddings with cosine >= θ are
        considered a match.  Paper default 0.8.
    lsh_batch_threshold:
        Batch size at which the strategy switches from exhaustive cosine to
        LSH bucket + KV merge.  Default 32.
    lsh_num_bits:
        Number of SimHash bits.  Default 64.
    embedding_dim:
        Dimensionality of the task embeddings.  Default 4096 (Qwen2.5-7B).
    """

    def __init__(
        self,
        cosine_threshold: float = 0.8,
        lsh_batch_threshold: int = 32,
        lsh_num_bits: int = 64,
        embedding_dim: int = 4096,
    ) -> None:
        self.threshold = cosine_threshold
        self.lsh_batch_threshold = lsh_batch_threshold
        self.lsh_hasher = SimHashHasher(dim=embedding_dim, num_bits=lsh_num_bits)

    @torch.no_grad()
    def identify(
        self,
        batch_embeddings: torch.Tensor,
        batch_hashes: torch.Tensor,
        kv_manager: KVManager,
    ) -> dict[int, MatchResult]:
        """Identify similarity matches for a batch.

        Parameters
        ----------
        batch_embeddings:
            ``[B, D]`` L2-normalized task embeddings.
        batch_hashes:
            ``[B]`` int64 SimHash values.
        kv_manager:
            The KVManager instance to query for cached tasks.

        Returns
        -------
        ``dict[int, MatchResult]`` mapping batch index → MatchResult.
        """
        batch_size = batch_embeddings.shape[0]
        results: dict[int, MatchResult] = {}
        hash_values = tensor_to_int_list(batch_hashes)

        if batch_size < self.lsh_batch_threshold:
            # -- Small batch: exhaustive cosine per candidate --------------
            for i in range(batch_size):
                emb = batch_embeddings[i : i + 1]  # [1, D]
                hsh = hash_values[i]
                candidates = kv_manager.lookup_by_hash(hsh)

                if not candidates:
                    results[i] = MatchResult(matched=False)
                    continue

                candidate_embs = torch.cat(
                    [c.embedding for c in candidates], dim=0
                )  # [N, D]
                scores = cosine_similarity(emb, candidate_embs)  # [N]

                score_values = tensor_to_float_list(scores)
                best_idx = max(range(len(score_values)), key=score_values.__getitem__)
                best_score = score_values[best_idx]

                if best_score >= self.threshold:
                    best_candidate = candidates[best_idx]
                    results[i] = MatchResult(
                        matched=True,
                        source_task_id=best_candidate.task_id,
                        cached_k=best_candidate.top_k,
                        cached_v=best_candidate.top_v,
                        similarity_score=best_score,
                    )
                else:
                    results[i] = MatchResult(matched=False)
        else:
            # -- Large batch: LSH bucket + KV merge ------------------------
            # Group batch indices by hash.
            hash_to_indices: dict[int, list[int]] = {}
            for i in range(batch_size):
                hsh = hash_values[i]
                hash_to_indices.setdefault(hsh, []).append(i)

            for hsh, indices in hash_to_indices.items():
                candidates = kv_manager.lookup_by_hash(hsh)
                if not candidates:
                    for idx in indices:
                        results[idx] = MatchResult(matched=False)
                    continue

                merged_k, merged_v = self._merge_kv(candidates)
                for idx in indices:
                    results[idx] = MatchResult(
                        matched=True,
                        source_task_id=None,  # merged — no single source
                        cached_k=merged_k,
                        cached_v=merged_v,
                        similarity_score=None,
                    )

        return results

    @torch.no_grad()
    def identify_reusable(
        self,
        batch_embeddings: torch.Tensor,
        batch_hashes: torch.Tensor,
        kv_manager: KVManager,
        hamming_radius: int = 2,
    ) -> dict[int, MatchResult]:
        """Find approximate tasks with complete per-layer KV for active reuse.

        A strict 64-bit hash equality loses near neighbours. Probe nearby
        buckets, then require the configured cosine similarity threshold.
        """
        results: dict[int, MatchResult] = {}
        hashes = tensor_to_int_list(batch_hashes)
        for idx, lsh_hash in enumerate(hashes):
            candidates = [
                task
                for task in kv_manager.candidates_within_hamming(
                    lsh_hash, hamming_radius, self.lsh_hasher.num_bits
                )
                if task.layer_kv is not None and task.seq_len >= 2
            ]
            if not candidates:
                continue
            candidate_embeddings = torch.cat(
                [task.embedding for task in candidates], dim=0
            )
            scores = cosine_similarity(
                batch_embeddings[idx : idx + 1], candidate_embeddings
            )
            score_values = tensor_to_float_list(scores)
            best_idx = max(range(len(score_values)), key=score_values.__getitem__)
            best_score = score_values[best_idx]
            if best_score < self.threshold:
                continue
            best = candidates[best_idx]
            kv_manager.get_kv(best.task_id)
            results[idx] = MatchResult(
                matched=True,
                source_task_id=best.task_id,
                cached_k=best.top_k,
                cached_v=best.top_v,
                similarity_score=best_score,
                layer_kv=best.layer_kv,
                hybrid_states=best.hybrid_states,
                prefix_len=best.seq_len,
                match_type="semantic",
            )
        return results

    @staticmethod
    def _merge_kv(tasks: list[CachedTask]) -> tuple[torch.Tensor, torch.Tensor]:
        """Average K and V across all tasks in the same LSH bucket.

        Parameters
        ----------
        tasks:
            Non-empty list of CachedTask in the same bucket.

        Returns
        -------
        ``(merged_k, merged_v)`` — element-wise mean of top-layer K and V.
        """
        if not tasks:
            raise ValueError("_merge_kv requires at least one task")

        ks = torch.stack([t.top_k for t in tasks])  # [N, ...]
        vs = torch.stack([t.top_v for t in tasks])
        return ks.mean(dim=0), vs.mean(dim=0)
