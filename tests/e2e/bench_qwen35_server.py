"""Measure sequential TTFT and concurrent throughput on an OpenAI server.

The first request fills SimLLM's task cache. Subsequent requests repeat the
same long prompt. Run separately against plugin-enabled and baseline servers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from urllib.request import Request, urlopen


def completion(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = Request(
        f"{url.rstrip('/')}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first_token = None
    token_count = 0
    with urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            chunk = line[6:].strip()
            if chunk == b"[DONE]":
                break
            event = json.loads(chunk)
            if event.get("usage"):
                token_count = event["usage"]["completion_tokens"]
            if any(choice.get("text") for choice in event.get("choices", [])):
                first_token = first_token or time.monotonic()
    end = time.monotonic()
    return {
        "ttft_s": round((first_token or end) - start, 4),
        "latency_s": round(end - start, 4),
        "completion_tokens": token_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8102")
    parser.add_argument("--model", default="qwen35-a3b-w8a8")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--paragraph-repeats", type=int, default=64)
    parser.add_argument("--vary-paragraph", action="store_true")
    args = parser.parse_args()

    paragraph = (
        "A service ingests event records, deduplicates identifiers, checks "
        "causal ordering, and publishes a compact report. Explain how to "
        "diagnose delayed events, retry storms, missing keys, and duplicate "
        "writes while preserving observability and bounded memory. "
    )
    prompt = (
        "You are reviewing a distributed event pipeline.\n"
        + paragraph * args.paragraph_repeats
        + "\nGive a concise operational plan."
    )
    repeat_prompt = (
        prompt.replace("event records", "event messages")
        if args.vary_paragraph
        else prompt
    )
    for run in range(args.repeats):
        result = completion(
            args.url, args.model, prompt if run == 0 else repeat_prompt, args.max_tokens
        )
        print(json.dumps({"phase": "sequential", "run": run, **result}), flush=True)
    batch_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as executor:
        results = list(
            executor.map(
                lambda _: completion(
                    args.url, args.model, repeat_prompt, args.max_tokens
                ),
                range(args.concurrency),
            )
        )
    batch_seconds = time.monotonic() - batch_start
    print(
        json.dumps(
            {
                "phase": "concurrent",
                "requests": args.concurrency,
                "wall_s": round(batch_seconds, 4),
                "output_tokens_per_s": round(
                    sum(result["completion_tokens"] for result in results)
                    / batch_seconds,
                    2,
                ),
                "results": results,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
