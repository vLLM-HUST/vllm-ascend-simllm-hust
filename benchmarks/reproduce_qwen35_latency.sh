#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
baseline_url="${BASELINE_URL:-http://127.0.0.1:8107}"
plugin_url="${PLUGIN_URL:-http://127.0.0.1:8106}"
output_dir="${1:?usage: $0 OUTPUT_DIR}"
workload="${repo_root}/benchmarks/fixtures/semantic_similar.jsonl.gz"
warmup30="${repo_root}/benchmarks/fixtures/qwen35_warmup30.jsonl"
latency300="${repo_root}/benchmarks/fixtures/qwen35_latency300.jsonl"
client="${repo_root}/tests/e2e/bench_similarity_workload.py"
check="${repo_root}/tests/e2e/check_latency_gain.py"

if [[ -e "$output_dir" && -n "$(ls -A "$output_dir")" ]]; then
    echo "Output directory must be empty: $output_dir" >&2
    exit 1
fi
mkdir -p "$output_dir"

WORKLOAD="$workload" "$python_bin" - <<'PY'
import gzip
import hashlib
import os

expected = "eecb069a51af828f3388525fcaf84f82cb15a56116fdffc293a69dd77bc9a746"
digest = hashlib.sha256()
with gzip.open(os.environ["WORKLOAD"], "rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected:
    raise SystemExit("Similarity workload SHA-256 mismatch")
print(f"Verified workload SHA-256: {expected}")
PY

curl -fsS "${baseline_url}/health" >/dev/null
curl -fsS "${plugin_url}/health" >/dev/null

# This order matches the 2026-10-08 run. The plugin's 30- and 300-request
# passes warm its task cache before the complete 5,000-request pass.
"$python_bin" "$client" --input "$latency300" --url "$baseline_url" \
    --max-tokens 1 --chat --concurrency 16 --global-concurrency \
    > "${output_dir}/baseline300.jsonl"
"$python_bin" "$client" --input "$warmup30" --url "$plugin_url" \
    --max-tokens 128 --chat --concurrency 4 \
    > "${output_dir}/plugin-warmup30.jsonl"
"$python_bin" "$client" --input "$latency300" --url "$plugin_url" \
    --max-tokens 1 --chat --concurrency 16 --global-concurrency \
    > "${output_dir}/plugin300.jsonl"
"$python_bin" "$client" --input "$workload" --url "$baseline_url" \
    --max-tokens 1 --chat --concurrency 16 --global-concurrency \
    > "${output_dir}/baseline5000.jsonl"
"$python_bin" "$client" --input "$workload" --url "$plugin_url" \
    --max-tokens 1 --chat --concurrency 16 --global-concurrency \
    > "${output_dir}/plugin5000.jsonl"

"$python_bin" "$check" --baseline "${output_dir}/baseline300.jsonl" \
    --plugin "${output_dir}/plugin300.jsonl" --expected-requests 300 \
    --minimum-gain-pct 10 --output "${output_dir}/latency300.json"
"$python_bin" "$check" --baseline "${output_dir}/baseline5000.jsonl" \
    --plugin "${output_dir}/plugin5000.jsonl" --expected-requests 5000 \
    --minimum-gain-pct 10 --output "${output_dir}/latency5000.json"
