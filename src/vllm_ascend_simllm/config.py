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

"""SimLLM configuration dataclass — populated from VLLM_ASCEND_SIMLLM_* env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _environment(name: str, default: str) -> str:
    return os.environ.get(f"VLLM_ASCEND_SIMLLM_{name}", default)


@dataclass
class SimLLMConfig:
    """Centralized Sim-LLM configuration.

    All fields are populated from VLLM_ASCEND_SIMLLM_* environment variables
    directly from the worker process environment.

    Typical usage::

        config = SimLLMConfig.from_env()
        if config.enabled:
            kv_mgr = KVManager(max_cache_size=config.kv_cache_size)
    """

    # -- Feature gate -------------------------------------------------------
    enabled: bool = False

    # -- Similarity ---------------------------------------------------------
    cosine_threshold: float = 0.8
    lsh_num_bits: int = 64
    lsh_batch_threshold: int = 32
    lsh_hamming_radius: int = 2

    # -- KV cache -----------------------------------------------------------
    kv_cache_size: int = 1024
    kv_cache_max_bytes: int = 512 * 1024 * 1024
    # Hybrid policies: aggressive, exact, aligned, tail, or tail_anchor.
    hybrid_reuse_mode: str = "tail_anchor"
    # Positive number of target suffix tokens to recompute in tail mode.
    hybrid_recompute_tokens: int = 512

    # -- Sandwich config ----------------------------------------------------
    sandwich_bottom: int = 3
    sandwich_top: int = 3

    # -- Embedding ----------------------------------------------------------
    embedding_pooling: str = "mean"

    # -- Deferral -----------------------------------------------------------
    deferral_ratio: float = 0.5
    max_deferrals: int = 3
    profile: bool = False

    def __post_init__(self) -> None:
        if self.hybrid_reuse_mode not in {
            "aggressive", "exact", "aligned", "tail", "tail_anchor"
        }:
            raise ValueError(
                "HYBRID_REUSE_MODE must be aggressive, exact, aligned, tail, "
                "or tail_anchor"
            )
        if self.hybrid_recompute_tokens < 1:
            raise ValueError("HYBRID_RECOMPUTE_TOKENS must be positive")

    @classmethod
    def from_env(cls) -> SimLLMConfig:
        """Populate a SimLLMConfig from VLLM_ASCEND_SIMLLM_* env vars.

        Returns a configuration snapshot — repeated calls may reflect env-var changes
        (useful for testing), but production code should call once at init.
        """
        return cls(
            enabled=bool(int(_environment("ENABLED", "0"))),
            cosine_threshold=float(_environment("COSINE_THRESHOLD", "0.8")),
            lsh_num_bits=int(_environment("LSH_NUM_BITS", "64")),
            lsh_batch_threshold=int(_environment("LSH_BATCH_THRESHOLD", "32")),
            lsh_hamming_radius=int(_environment("LSH_HAMMING_RADIUS", "2")),
            kv_cache_size=int(_environment("KV_CACHE_SIZE", "1024")),
            kv_cache_max_bytes=int(
                _environment("KV_CACHE_MAX_BYTES", str(512 * 1024 * 1024))
            ),
            hybrid_reuse_mode=_environment("HYBRID_REUSE_MODE", "tail_anchor"),
            hybrid_recompute_tokens=int(_environment("HYBRID_RECOMPUTE_TOKENS", "512")),
            sandwich_bottom=int(_environment("SANDWICH_BOTTOM", "3")),
            sandwich_top=int(_environment("SANDWICH_TOP", "3")),
            embedding_pooling=_environment("EMBEDDING_POOLING", "mean"),
            deferral_ratio=float(_environment("DEFERRAL_RATIO", "0.5")),
            max_deferrals=int(_environment("MAX_DEFERRALS", "3")),
            profile=bool(int(_environment("PROFILE", "0"))),
        )
