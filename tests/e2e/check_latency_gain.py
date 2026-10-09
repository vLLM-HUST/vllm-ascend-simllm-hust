"""Validate paired one-token benchmark files and the required speedup."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> tuple[dict[str, dict], dict]:
    records: dict[str, dict] = {}
    summary = None
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("phase") == "summary":
            if summary is not None:
                raise ValueError(f"multiple summaries in {path}")
            summary = row
            continue
        request_id = row.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"missing request ID in {path}")
        if request_id in records:
            raise ValueError(f"duplicate request ID {request_id} in {path}")
        if row.get("completion_tokens") != 1:
            raise ValueError(f"expected one completion token for {request_id}")
        if not isinstance(row.get("ttft_s"), (int, float)):
            raise ValueError(f"missing TTFT for {request_id}")
        records[request_id] = row
    if summary is None or summary.get("requests") != len(records):
        raise ValueError(f"invalid request count or missing summary in {path}")
    return records, summary


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--minimum-gain-pct", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline, base_summary = load(args.baseline)
    plugin, plugin_summary = load(args.plugin)
    if len(baseline) != args.expected_requests or baseline.keys() != plugin.keys():
        raise ValueError("benchmark inputs are incomplete or request IDs differ")
    if any(
        baseline[request_id].get("prompt_tokens")
        != plugin[request_id].get("prompt_tokens")
        for request_id in baseline
    ):
        raise ValueError("paired prompt token counts differ")

    base_ttft = [row["ttft_s"] for row in baseline.values()]
    plugin_ttft = [row["ttft_s"] for row in plugin.values()]
    metrics = {
        "requests": len(baseline),
        "baseline": {
            "requests_per_s": base_summary["requests_per_s"],
            "ttft_p50_s": percentile(base_ttft, 0.5),
            "ttft_p95_s": percentile(base_ttft, 0.95),
        },
        "plugin": {
            "requests_per_s": plugin_summary["requests_per_s"],
            "ttft_p50_s": percentile(plugin_ttft, 0.5),
            "ttft_p95_s": percentile(plugin_ttft, 0.95),
        },
    }
    metrics["gain_pct"] = {
        "throughput": 100
        * (
            metrics["plugin"]["requests_per_s"]
            / metrics["baseline"]["requests_per_s"]
            - 1
        ),
        "ttft_p50": 100
        * (
            1
            - metrics["plugin"]["ttft_p50_s"]
            / metrics["baseline"]["ttft_p50_s"]
        ),
        "ttft_p95": 100
        * (
            1
            - metrics["plugin"]["ttft_p95_s"]
            / metrics["baseline"]["ttft_p95_s"]
        ),
    }
    metrics["minimum_gain_pct"] = args.minimum_gain_pct
    metrics["passed"] = all(
        gain >= args.minimum_gain_pct for gain in metrics["gain_pct"].values()
    )
    result = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")
    if not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
