# Qwen3.5-35B-A3B similarity workload, 2026-09-29

## Input and setup

Source: `resources/workload/final_workloaddata_v2/semantic_similar.jsonl`
(5000 synthetic requests, 1000 groups of five). The tuning sample was the
first 12 groups. Before the CPU token-overlap fast path, that tuning sample
showed a regression: 3.0312 requests/s with SimLLM versus 3.2493 requests/s
without it, despite 53 semantic hits among 60 requests. This exposed the
embedding/matching overhead at approximately 3000 prompt tokens. The held-out
performance sample contains 20 groups from each
`reuse_tier` (`low`, `medium`, `high`), chosen with Python
`random.Random(20260929).sample` after excluding those first 12 groups. The
selected group indices were replayed in source order. The 300-request input
SHA-256 was `d7fa728dae0052a44529086a9fb46dd1b2aa4c8e2c7f99e8d346d8ec6c99676d`.

Model: `/data/jxd/Qwen3.5-35B-A3B-W8A8` on Ascend 910B3, TP2, CANN 9.1,
8192 context, block size 128, eager execution, `max_num_seqs=4`, host prefix
caching off, chunked prefill off. In each group, the first request ran to
completion before the other four were submitted concurrently. The baseline
disabled SimLLM; every other serving flag was identical. Each card assignment
was then reversed and the sample replayed again with fresh servers.

## 300-request prefill comparison

Each request used `max_tokens=1` to isolate first-token latency and prefill
throughput. Both runs produced 300 output tokens. `reuse_300.jsonl` and
`reuse_crossover_300.jsonl` map every request to its worker-side reused token
count; both assignments logged 296 semantic hits.

| Plugin cards / baseline cards | Baseline requests/s | SimLLM requests/s | Change | Warm TTFT median, baseline → SimLLM |
| --- | ---: | ---: | ---: | ---: |
| 2,3 / 4,5 | 4.0935 | 4.2118 | +2.9% | 0.7043 → 0.6599 s |
| 4,5 / 2,3 | 4.1998 | 4.3773 | +4.2% | 0.6946 → 0.6356 s |

The per-request JSONL files are included here. The result is a held-out,
stratified sample rather than a full 5000-request replay. Its shared prompt template can produce high token overlap across groups.
Worker logs confirm cache reuse events.

## Original output-budget comparison

Fifteen more requests from three held-out groups (one per reuse tier) were
replayed after the short-output sample. Each request used its original
`metadata.output_tokens_target` (96 or 192 here). Both the baseline and
SimLLM generated 2208 tokens in each card assignment.

| Plugin cards / baseline cards | Baseline output tokens/s | SimLLM output tokens/s | Change | Warm TTFT median, baseline → SimLLM |
| --- | ---: | ---: | ---: | ---: |
| 2,3 / 4,5 | 6.9397 | 9.2287 | +33.0% | 1.1219 → 0.9043 s |
| 4,5 / 2,3 | 6.8550 | 9.4665 | +38.1% | 1.1385 → 0.8652 s |

These are warm-cache measurements: the preceding 300-request run populated
SimLLM's task cache, while native vLLM prefix caching was disabled on both
servers. The same ordered requests and token budgets were used in both runs.
The selected subset is small, so its output-throughput result should not be
extrapolated to every output bucket of the 5000-request source.

## Full 5000-request prefill replay

The entire `semantic_similar.jsonl` file was then replayed in original order
with 16 client requests in flight. Each request used `max_tokens=1`; both
servers generated 5000 output tokens, and the prompt lengths reported by the
Qwen tokenizer ranged from 1585 to 4366 tokens. The two servers were exercised
sequentially. This measured the full source file's prefill throughput, while
the original output budgets were represented by the 15-request experiment
above.

| Metric | Baseline | SimLLM | Change |
| --- | ---: | ---: | ---: |
| Requests/s | 5.1985 | 5.3646 | +3.2% |
| TTFT median | 3.0673 s | 2.9891 s | -2.5% |
| TTFT mean | 3.0726 s | 2.9774 s | -3.1% |
| TTFT p95 | 3.1729 s | 3.3002 s | +4.0% |

The worker log matched 4997 of the 5000 requests to a cached source: 4990
approximate token-overlap hits and seven exact-prefix hits. The median skipped
span was 3010 tokens. The p95 regression is a remaining limitation at 16
client requests in flight, despite improved median and throughput. At the
four-request concurrency used in the held-out sample, p95 improved in both
card assignments. The full-run raw records are `baseline_full5000.jsonl`,
`simllm_full5000.jsonl`, and `reuse_full5000.jsonl`.

## Client

`tests/e2e/bench_similarity_workload.py` records each request's streaming
TTFT, total latency, actual prompt and completion token counts, and the server
request ID. Use the same ordered input and `--concurrency 4 --max-tokens 1`
against both ports for this comparison. Use `--target-output` to honor each
request's `metadata.output_tokens_target` for an end-to-end comparison. The
full-file replay uses `--concurrency 16 --global-concurrency --max-tokens 1`.
