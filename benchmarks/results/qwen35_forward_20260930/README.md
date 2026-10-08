# Qwen3.5-35B-A3B forward and serving follow-up, 2026-09-30

## Scope and setup

Input: `resources/workload/final_workloaddata_v2/semantic_similar.jsonl`.
The ordered first 12 groups (60 requests) warmed each fresh server. A
previously selected, stratified 60-group sample (300 requests) then ran at
global concurrency 16. That sample contains 20 groups from each `reuse_tier`,
selected with `random.Random(20260929).sample` after excluding the first 12
groups and replayed in source order. Each request generated one token, so the
300-request figures primarily measure prefill. Five original-length requests
(960 output tokens total) provided a small decode check. After source
promotion, the full 5000-request replay used the same ordered source file,
60-request warmup, and global concurrency 16 on each server.

The model was `/data/jxd/Qwen3.5-35B-A3B-W8A8` on Ascend 910B3 with CANN
9.1, `--max-model-len 8192 --block-size 128 --max-num-seqs 4
--no-enable-prefix-caching --no-enable-chunked-prefill`. Plugin and baseline
used the same model and serving flags within each pair. TP2 plugin used cards
2,3 and TP2 baseline cards 4,5. TP1 plugin used card 6 and TP1 baseline card
7, with `--gpu-memory-utilization 0.85`. Eager means `--enforce-eager`;
PIECEWISE means `--compilation-config
'{"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":4}'`.
The baseline had `VLLM_ASCEND_SIMLLM_ENABLED=0`; the plugin had it set to 1.

## Forward profile and restore change

The host PyTorch/NPU profiler captured a five-request TP2 eager plugin replay
with `--profiler-config
'{"profiler":"torch","torch_profiler_dir":"/tmp/simllm-forward-profile","torch_profiler_with_stack":false}'`.
The captured device window contained 324 HCCL all-reduces, totaling 766.1 ms
of communication; median call was 2.308 ms. Leading compute operators by
cumulative time were `GroupedMatmulSwigluQuant` (32.9 ms),
`ChunkGatedDeltaRule` (29.1 ms), `MatMulV2` (24.3 ms), `GroupedMatmul`
(16.5 ms), and `ScatterUpdate` (14.7 ms). The profiler-distorted stage
summary lists 209.8 ms compute, 767.7 ms communication, and 541.6 ms free
time across its captured window. These totals are **not** a decomposition of
client wall latency. Raw aggregates are in the adjacent CSV and JSON files.

The server also warned that `causal_conv1d_update_npu` was unavailable and a
fallback synchronizes each request. Full decode graph capture was therefore
not available; PIECEWISE capture succeeded.

`benchmarks/bench_hybrid_kv_restore.py` models the 30 recurrent plus ten
full-attention layers on one NPU and compares per-request `index_copy_` with
one concatenated write per layer. The first proxy run motivated the
three-hit threshold. A follow-up corrected the recurrent-state dimensions
to match the Qwen3.5 config and tested both TP1 and TP2 on one 910B3 card
(three warmups, ten timed repeats). Median milliseconds from that follow-up:

| TP size | Simultaneous cache hits | Per-request writes | Batched writes |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 6.384 | 9.658 |
| 1 | 3 | 11.495 | 9.888 |
| 1 | 4 | 12.944 | 11.295 |
| 2 | 2 | 6.824 | 9.359 |
| 2 | 3 | 9.554 | 9.414 |
| 2 | 4 | 12.962 | 9.654 |

The worker now batches restore writes at three or more hits when destination
block IDs are disjoint. It retains sequential writes for one or two hits or
overlapping destinations, where `index_copy_` write order would otherwise be
undefined. This reduces dispatch overhead without changing cache selection.

## Paired serving measurements

Each 300-request comparison used fresh plugin and baseline servers, each
first warmed with the same 60 requests. The client sent the same 300 prompts
to both members of each pair. TTFT includes client-side queueing at
concurrency 16. RPS is completed requests divided by replay wall time.

| Configuration | Baseline RPS | Plugin RPS | RPS change | Baseline TTFT p50 / p95 | Plugin TTFT p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP2 eager | 5.8534 | 6.4421 | +10.1% | 2.7089 / 2.9409 s | 2.4569 / 2.5704 s |
| TP2 PIECEWISE | 5.5627 | 7.0659 | +27.0% | 2.8535 / 3.0346 s | 2.2869 / 2.7814 s |
| TP1 PIECEWISE | 5.7016 | 7.3168 | +28.3% | 2.7778 / 3.1172 s | 2.1717 / 2.6350 s |

TP1 PIECEWISE gave the highest measured plugin throughput and p50 while using
one NPU. TP2 eager gave the lowest plugin p95 by 65 ms. The separate TP2
graph run does not isolate the batching change from graph mode and server-run
variation. The one-NPU restore microbenchmark supports batching at three or
more hits; a same-card, controlled unbatched-versus-batched serving comparison
has not been performed. Cross-configuration results also used different NPU
cards and are directional.

## TP1 batch-capacity follow-up

Two fresh TP1 plugin servers replayed the same 60-request warmup and
300-request sample. Both had PIECEWISE capture and separate compile-cache
directories. The only serving differences were `--max-num-seqs` and
`max_cudagraph_capture_size`, both set to 4 on card 7 or both set to 8 on
card 6. A PyTorch AOT Autograd cache incompatibility required
`TORCHINDUCTOR_AUTOGRAD_CACHE=0` for both servers. The initial 60-request
warmups included first-use compilation spikes (maximum TTFT over 24 s), so
the table uses the following warmed 300 requests.

| Maximum sequences / graph capture | RPS | TTFT p50 | TTFT p95 | Confirmed reuse hits across 60 + 300 |
| ---: | ---: | ---: | ---: | ---: |
| 4 / 4 | 7.7107 | 2.0649 s | 2.4828 s | 355 |
| 8 / 8 | 7.7240 | 2.0806 s | 2.6703 s | 355 |

Capacity 8 produced no meaningful throughput gain and had worse p95 on this
fixed card assignment, so the tested recommendation remains 4. The two
configurations were not swapped between cards. Raw files are
`simllm-seq{4,8}-{60,300}.jsonl`.

A further TP1 eager run on card 7 used the same cache setting, warmup, and
300-request input as the card-7 PIECEWISE run. Eager reached 6.7617 RPS,
TTFT p50 2.3525 s, and p95 2.4505 s, with 355 logged reuse hits and no
restore error. PIECEWISE at capacity 4 delivered 14.0% more requests per
second and 0.288 s lower p50; eager had 0.032 s lower p95 in this single
comparison. The throughput recommendation is TP1 PIECEWISE at capacity 4.
Raw eager records are `simllm-tp1-eager{60,300}.jsonl`.

## Independent 1000-request sample and source promotion

The earlier full replay showed that a 300-request sample could overstate a
speedup. We therefore selected another 200 five-request groups from the same
source file: 67 low, 67 medium, and 66 high `reuse_tier` groups, excluding
the first 12 warmup groups and all groups in the 300-request sample. Selection
used `random.Random(20260930).sample` within each tier; selected groups were
replayed in original source order. The 1000-request input SHA-256 is
`54c56590251381235362002cbdefe61b0a21119e6f311399906f86d9652d5d57`.
Each fresh server first ran the same 60-request warmup. The 1000-request
replays used global concurrency 16 and `max_tokens=1`.

With the original 512 MiB task cache, three consecutive plugin replays on
card 6 showed a loss of effective coverage even though nearly every request
still logged a hit. The un-reused prompt-token total grew from 112,555 to
168,616 to 313,782; RPS fell from 12.0117 to 10.1344 to 5.8795. Restarting
the service restored the first-round result (12.0518 RPS and exactly 112,555
un-reused tokens). NPU memory stayed near 55,440 MiB in the slow third round.
Increasing `KV_CACHE_MAX_BYTES` to 2 GiB kept more tasks but selected sources
that left about 273,000 un-reused tokens per round; throughput was 7.0474,
7.1134, then 6.5048 RPS. Cache capacity alone did not fix coverage.

The new Qwen3.5 hybrid path promotes a semantic target into a new cached
source only when its prompt exceeds the source's reused span by at least 128
tokens. Its remaining tokens must finish model forward before the worker
snapshots all 40 layers. Shorter semantic hits do not create a new snapshot.
This lets later similar prompts reuse a longer span. The source prefix in a
promoted snapshot is approximate.

| Configuration and 1000-request round | RPS | TTFT p50 / p95 | Confirmed hits | Un-reused prompt tokens |
| --- | ---: | ---: | ---: | ---: |
| Baseline, first | 5.2777 | 3.0812 / 3.2445 s | — | 2,951,635 |
| Baseline, repeated | 5.0324 | 3.1752 / 3.2754 s | — | 2,951,635 |
| Original plugin, first | 12.0117 | 1.1787 / 2.4234 s | 994 | 112,555 |
| Original plugin, second | 10.1344 | 1.2090 / 3.2412 s | 987 | 168,616 |
| Original plugin, third | 5.8795 | 2.7397 / 3.2423 s | 993 | 313,782 |
| Promoting plugin, first | 17.7534 | 0.7103 / 1.4777 s | 993 | 30,555 |
| Promoting plugin, second | 17.9901 | 0.7001 / 1.4700 s | 994 | 22,813 |
| Promoting plugin, third | 20.4451 | 0.6971 / 1.2063 s | 993 | 21,805 |

All rows generated exactly 1000 output tokens. The baseline ran on card 7,
the plugin on card 6; this comparison has not been card-swapped. The plugin
comparison used the same card and input across fresh configurations. The
promoting worker logged 58 snapshot-store events across the 60-request
warmup and three 1000-request replays and no restore errors. After those
replays, 15 requests with their original output targets completed all 2976
tokens at 32.235 output tokens/s, with no restore error. These 15 requests
are a functional decode check, not a paired decode-throughput comparison.
The raw client records are `simllm-heldout1000-*.jsonl`,
`simllm-promotion1000-r{1,2,3}.jsonl`, and
`simllm-promotion15-output.jsonl`.

## Full 5000-request validation after promotion

The complete `semantic_similar.jsonl` source file (SHA-256
`eecb069a51af828f3388525fcaf84f82cb15a56116fdffc293a69dd77bc9a746`)
was replayed once to each fresh TP1 PIECEWISE server. The baseline ran on
card 7 and the plugin on card 6, with identical model and serving flags except
for the plugin. Each server first processed the same ordered 60-request
warmup. The measured 5000 requests used `max_tokens=1`, global concurrency
16, `--max-num-seqs 4`, and graph capture size 4. Host prefix caching and
chunked prefill were disabled on both sides.

| Metric | Baseline | Promoting plugin | Change |
| --- | ---: | ---: | ---: |
| Completed requests | 5000 / 5000 | 5000 / 5000 | — |
| Wall time | 980.4366 s | 241.4197 s | -75.4% |
| Throughput | 5.0998 requests/s | 20.7108 requests/s | 4.061x (+306.1%) |
| TTFT p50 | 3.1543 s | 0.6903 s | -78.1% |
| TTFT p95 | 3.2770 s | 0.9873 s | -69.9% |
| TTFT p99 | 3.3815 s | 1.2213 s | -63.9% |

Each side generated exactly 5000 output tokens; neither client record set
contains an error. The plugin worker logged reuse for 4972 of 5000 measured
requests, totaling 14,691,148 reused tokens out of 14,789,413 prompt tokens
(99.34%). The residual prompt-token total was 98,265. By workload tier,
confirmed hits were 917/920 low, 2318/2330 medium, and 1737/1750 high.
The worker logged 75 snapshot-store events across warmup and full replay and
no injection or extraction failures. These hit counts come from worker logs
matched to client request IDs, rather than workload `matched_flag` metadata.

This fixed-card, single-pass comparison shows a large prefill improvement on
the complete workload. It does not isolate card effects or establish output
long-generation throughput. Raw client records are
`simllm-full-promotion-{baseline,plugin}{60,5000}.jsonl`; the worker trace is
`simllm-full-promotion-plugin-server.log`.

The five-request original-length decode check generated 960 tokens per
member. TP2 PIECEWISE baseline and plugin delivered 26.7124 and 26.3743
output tokens/s, respectively. TP1 PIECEWISE baseline and plugin delivered
28.6000 and 30.8463 output tokens/s. This tiny sample does not establish a
decode throughput improvement; the optimization targets prefill.

The `matched_flag` field in client JSONL is workload metadata, not a confirmed
worker cache hit. The worker reuse log is the source for confirmed hits. Raw
per-request records for every run are in
this directory.

## Validation

After source promotion was added, the remote CANN environment passed 163
non-model unit tests with four model tests deselected
(`pytest -q -m 'not model' tests --ignore=tests/e2e`). Local targeted tests
passed 61 cases and Ruff passed. After the full replay, the focused local
prefix-reuse and cache-manager suite passed 40 cases, Ruff check and format
passed, and `git diff --check` was clean. Both TP1 and TP2 serving
configurations completed their end-to-end replays without a restore error.
