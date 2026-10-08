# Qwen3.5-35B-A3B SimLLM latency investigation, 2026-09-30

## Scope

The input is `resources/workload/final_workloaddata_v2/semantic_similar.jsonl`.
Only the first 12 groups (60 requests) and the previously defined held-out,
stratified 60-group sample (300 requests) were replayed. No 5000-request
replay was performed. The model was
`/data/jxd/Qwen3.5-35B-A3B-W8A8`, TP2 on Ascend 910B3 with CANN 9.1,
8192 context, 128-token scheduler pages, eager execution, four maximum model
sequences, native prefix caching disabled, and chunked prefill disabled.
All requests generated one token.

## Where time went

`VLLM_ASCEND_SIMLLM_PROFILE=1` synchronizes the NPU around KV restore,
model forward, and snapshot solely for diagnosis. The same 60 ordered
requests were replayed on cards 2,3 before and after the block-table change.
The figures below are worker-side medians per model batch, not additive
client latency components.

| Stage | Before | After |
| --- | ---: | ---: |
| CPU matching | 1.50 ms | 1.46 ms |
| KV restore, one hit | 21.87 ms | 8.18 ms |
| KV restore, two hits | 42.84 ms | 15.74 ms |
| Model forward, one-hit batch | 264.92 ms | 267.65 ms |
| Snapshot, cold batch | 19.79 ms | 19.74 ms |

The old hybrid restore read the NPU block table once per layer and created a
fresh NPU index tensor for every K/V write. The host block table already
maintains an authoritative CPU buffer. Restore now reads that buffer and
creates one index tensor per cache group and request. Snapshot reads the CPU
buffer too and avoids table materialization when there is no task to store.
The snapshot's remaining cost is mainly gathering and cloning per-layer state;
this change did not materially reduce it.

In diagnostic mode, 60-request throughput changed from 4.0424 to 4.1703
requests/s and warm TTFT median from 0.6520 to 0.5958 s. Because diagnostic
synchronization changes scheduling, the uninstrumented comparisons below
are the serving results.

## Uninstrumented serving comparisons

The plugin ran on cards 2,3 and baseline on cards 4,5. Both used identical
serving flags. The 60-request comparison started with fresh servers. The
300-request comparison followed on the same servers; its plugin cache was
therefore warm from the preceding 60 requests. The baseline had native prefix
caching disabled throughout.

| Input and concurrency | Metric | Baseline | SimLLM | Change |
| --- | --- | ---: | ---: | ---: |
| 60, group concurrency 4 | Requests/s | 3.9838 | 4.2193 | +5.9% |
| 60, group concurrency 4 | Warm TTFT median | 0.6837 s | 0.6205 s | -9.2% |
| 300, global concurrency 16 | Requests/s | 5.8395 | 6.5328 | +11.9% |
| 300, global concurrency 16 | TTFT p50 | 2.7170 s | 2.4170 s | -11.0% |
| 300, global concurrency 16 | TTFT p95 | 2.9404 s | 2.5221 s | -14.2% |

The worker logged 57 reuse hits in the first 60 requests and 299 in the
following 300. These are local samples and one card assignment; they do not
replace a full-workload or cross-card replay.

## Validation and records

The two changed block-table paths have regression tests in
`tests/test_prefix_reuse.py`. The remote CANN environment passed 161 unit
tests (`pytest -q -m 'not model' tests --ignore=tests/e2e`; four model tests
deselected), and local Ruff checks passed. The eight neighboring JSONL/log
files contain raw client records and worker diagnostic lines for these
comparisons.
