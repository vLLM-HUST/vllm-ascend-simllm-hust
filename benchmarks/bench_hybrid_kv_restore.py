"""Compare per-request and batched physical-cache writes on one NPU.

The shapes approximate Qwen3.5-35B-A3B at the selected tensor-parallel size:
30 recurrent layers, ten full-attention layers, two cache tensors per layer,
and 24 attention blocks per prompt. This benchmark includes Python and NPU
dispatch time.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("npu:0")
    # Qwen3.5-35B-A3B: 16 key heads, 32 value heads, 128-dimensional
    # recurrent state, 4-wide convolution, and two KV heads of width 256.
    recurrent_shapes = (
        (32 // args.tp_size, 128, 128),
        (8192 // args.tp_size, 3),
    )
    attention_shape = (128, 2 // args.tp_size, 256)
    block_counts = [1 if layer % 4 != 3 else 24 for layer in range(40)]
    shapes = [
        recurrent_shapes if count == 1 else (attention_shape, attention_shape)
        for count in block_counts
    ]
    caches = [
        tuple(
            torch.zeros(
                (max(8, count * (args.requests + 1)), *shape),
                dtype=torch.bfloat16,
                device=device,
            )
            for shape in pair
        )
        for count, pair in zip(block_counts, shapes, strict=True)
    ]
    indices = [
        [
            torch.arange(req * count, (req + 1) * count, device=device)
            for count in block_counts
        ]
        for req in range(args.requests)
    ]
    saved = [
        [
            tuple(
                torch.full(
                    (count, *shape), req + 1, dtype=torch.bfloat16, device=device
                )
                for shape in pair
            )
            for count, pair in zip(block_counts, shapes, strict=True)
        ]
        for req in range(args.requests)
    ]

    def per_request() -> None:
        for req in range(args.requests):
            for layer, destinations in enumerate(caches):
                for destination, blocks in zip(
                    destinations, saved[req][layer], strict=True
                ):
                    destination.index_copy_(0, indices[req][layer], blocks)

    def batched() -> None:
        for layer, destinations in enumerate(caches):
            index = torch.cat([indices[req][layer] for req in range(args.requests)])
            for tensor_idx, destination in enumerate(destinations):
                blocks = torch.cat(
                    [saved[req][layer][tensor_idx] for req in range(args.requests)]
                )
                destination.index_copy_(0, index, blocks)

    for name, operation in (("per_request", per_request), ("batched", batched)):
        for _ in range(args.warmups):
            operation()
        torch.npu.synchronize()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            operation()
            torch.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        print(
            f"{name} tp_size={args.tp_size} requests={args.requests} "
            f"median_ms={statistics.median(times):.3f} "
            f"min_ms={min(times):.3f} max_ms={max(times):.3f}"
        )


if __name__ == "__main__":
    main()
