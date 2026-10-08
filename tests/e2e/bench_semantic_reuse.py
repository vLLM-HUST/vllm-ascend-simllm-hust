"""Measure repeated similar requests with and without SimLLM on one NPU.

Run this script twice in separate processes, once with
VLLM_ASCEND_SIMLLM_ENABLED=0 and once with it set to 1. The input JSONL must
contain at least two rows with a ``prompt`` string. The second row should be
semantically related to, but token-different from, the first row.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.prompts_jsonl.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) < 2 or args.repeats < 1:
        parser.error("provide two prompt rows and at least one repeat")
    source, target = rows[:2]
    if source["prompt"] == target["prompt"]:
        parser.error("source and target prompts must differ")

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=2,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    enabled = os.environ.get("VLLM_ASCEND_SIMLLM_ENABLED") == "1"
    for run, row in enumerate([source] + [target] * args.repeats):
        start = time.monotonic()
        result = llm.generate([row["prompt"]], params, use_tqdm=False)[0]
        print(
            json.dumps(
                {
                    "enabled": enabled,
                    "run": run,
                    "id": row.get("id", run),
                    "latency_s": round(time.monotonic() - start, 4),
                    "output_token_ids": list(result.outputs[0].token_ids),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
