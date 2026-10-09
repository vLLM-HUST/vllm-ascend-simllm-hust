"""Pinned zero-shot MMLU-Pro pilot for checking SimLLM control-path coverage.

This is a local diagnostic adapter, not the canonical dataset-program grader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path


def grade_prediction(text: str, answer: str) -> tuple[str | None, bool]:
    found = re.fullmatch(r"\s*([A-J])\s*[.。]?\s*", text)
    prediction = found.group(1) if found else None
    return prediction, prediction == answer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> None:
    import pyarrow.parquet as parquet

    table = parquet.read_table(args.parquet)
    rows = table.to_pylist()
    indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.limit))
    with args.output.open("w", encoding="utf-8") as target:
        for index in indices:
            row = rows[index]
            options = "\n".join(
                f"{chr(65 + i)}. {value}" for i, value in enumerate(row["options"])
            )
            prompt = (
                "Answer the following multiple-choice question. Reply with only the "
                "single letter of the best option.\n\n"
                f"Question: {row['question']}\n{options}\nAnswer:"
            )
            record = {
                "id": f"mmlu-pro-test-{row['question_id']}",
                "source_row": index,
                "category": row["category"],
                "answer": row["answer"],
                "prompt": prompt,
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "dataset_revision": args.revision,
                "source_parquet_sha256": sha256(args.parquet),
                "sample_seed": args.seed,
                "sample_size": args.limit,
                "input_sha256": sha256(args.output),
                "grader": "strict-single-letter-v1",
                "mode": "zero-shot-diagnostic",
            },
            indent=2,
        )
    )


def run(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    with args.output.open("w", encoding="utf-8") as target:
        for row in rows:
            request = urllib.request.Request(
                f"{args.url.rstrip('/')}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": args.model,
                        "messages": [{"role": "user", "content": row["prompt"]}],
                        "temperature": 0,
                        "max_tokens": 16,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    output = json.load(response)
                text = output["choices"][0]["message"]["content"] or ""
                prediction, correct = grade_prediction(text, row["answer"])
                result = {
                    "id": row["id"],
                    "request_id": output["id"],
                    "answer": row["answer"],
                    "prediction": prediction,
                    "correct": correct,
                    "output_text": text,
                    "usage": output.get("usage"),
                }
            except Exception as exc:
                result = {"id": row["id"], "answer": row["answer"], "error": str(exc)}
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            print(result["id"], "correct", result.get("correct"), flush=True)


def summarize(args: argparse.Namespace) -> None:
    baseline = [json.loads(line) for line in args.baseline.read_text().splitlines()]
    plugin = [json.loads(line) for line in args.plugin.read_text().splitlines()]
    assert [item["id"] for item in baseline] == [item["id"] for item in plugin]
    reuse_by_request = {}
    if args.plugin_log:
        pattern = re.compile(r"SimLLM: reusing (\d+) semantic tokens for request (\S+)")
        for match in pattern.finditer(args.plugin_log.read_text(errors="replace")):
            reuse_by_request[match.group(2)] = int(match.group(1))
    metrics = {"input_ids_aligned": True, "runs": {}}
    for name, rows in (("baseline", baseline), ("plugin", plugin)):
        errors = [item for item in rows if "error" in item]
        metrics["runs"][name] = {
            "output_sha256": sha256(
                args.baseline if name == "baseline" else args.plugin
            ),
            "tasks": len(rows),
            "correct": sum(bool(item.get("correct")) for item in rows),
            "accuracy": sum(bool(item.get("correct")) for item in rows) / len(rows),
            "errors": len(errors),
            "invalid_output": sum(
                item.get("prediction") is None and "error" not in item for item in rows
            ),
        }
    metrics["plugin_control_path"] = {
        "requests_with_reuse_event": sum(
            item.get("request_id") in reuse_by_request for item in plugin
        ),
        "reused_tokens": sum(
            reuse_by_request.get(item.get("request_id"), 0) for item in plugin
        ),
    }
    result = json.dumps(metrics, indent=2) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("--parquet", type=Path, required=True)
    prep.add_argument("--revision", required=True)
    prep.add_argument("--limit", type=int, default=20)
    prep.add_argument("--seed", type=int, default=20261009)
    prep.add_argument("--output", type=Path, required=True)
    execute = modes.add_parser("run")
    execute.add_argument("--input", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--url", required=True)
    execute.add_argument("--model", default="qwen35-a3b-w8a8")
    summary = modes.add_parser("summarize")
    summary.add_argument("--baseline", type=Path, required=True)
    summary.add_argument("--plugin", type=Path, required=True)
    summary.add_argument("--plugin-log", type=Path)
    summary.add_argument("--output", type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.mode](args)


if __name__ == "__main__":
    main()
