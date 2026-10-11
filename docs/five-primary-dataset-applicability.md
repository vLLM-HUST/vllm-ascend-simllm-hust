# Five primary datasets: SimLLM applicability

This is a planning and blocker record for
[simllm-migration issue #7](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/issues/7)
under the [canonical dataset program](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254).
It contains no dataset score or optimization claim. The existing
`semantic_similar` measurements are supplementary serving evidence only.

The SimLLM control path is exercised only when requests reach a vLLM Ascend
worker with `VLLM_ASCEND_SIMLLM_ENABLED=1`, the compatible model/host runtime,
and a logged KV-reuse event for the evaluated request. A shared endpoint or
similar looking prompts alone does not establish a hit.

| Dataset | Status | Concrete reason | Owner and next step |
| --- | --- | --- | --- |
| MMLU-Pro | not-exercised | The source snapshot is frozen. A pinned 200-item diagnostic B0/B1 run stored tasks but logged zero request-level KV-reuse events. Independent questions did not exercise the optimization path. Its task manifest and canonical scorer remain pending. | @GuMorming: review the zero-hit conclusion and integrate the frozen five-shot task contract when approved; benchmark maintainers: freeze the formal task manifest/scorer. Run matched formal B0/B1 only after the execution gate is met. |
| HLE-Verified | blocked | The 2,500-row source snapshot is frozen. Formal rights review, full-versus-Gold choice, judge/scorer contract, image handling, and execution image remain pending. Gold includes image tasks and cannot silently become a text-only score. | Benchmark maintainers: approve rights and judge/evaluation contract; @GuMorming: add request-to-task reuse telemetry and B0/B1 adapter once the contract is executable. |
| SWE-bench-Pro | blocked | The source and task/image mapping are frozen, but rights review, an AMD64 fresh-sandbox backend, agent/tool/budget contract, and isolated regrader remain pending. | Benchmark maintainers and execution-environment provider: freeze rights, sandbox, agent and regrader; @GuMorming: integrate the SimLLM service and request-to-task trace once available. |
| FrontierScience | blocked | Olympiad and Research source inputs are frozen and preliminary track-specific protocol code exists, but the actual judge and semantic calibration are not complete. No SimLLM task run exists. | Benchmark maintainers: approve separate track judge/scorer contracts; @GuMorming: implement the paired SimLLM runner and separate track outcomes once approved. |
| Terminal-Bench 2.1 | blocked | The 89-task source and image digest audit and a candidate Harbor plan exist; actual AMD64 Docker/Harbor execution, model connectivity from that backend, verifier validation, rights review, and final agent/scorer contract are pending. | Execution-environment provider and benchmark maintainers: complete the Harbor and contract gates; @GuMorming: connect SimLLM and collect per-task resolution and reuse traces. |

For every future comparison, B0 must be the same Qwen3.5-35B-A3B weights and
Ascend serving stack with SimLLM disabled; B1 changes only SimLLM activation
and its documented policy. Record immutable model/runtime/dataset/evaluator
revisions, hardware and topology, inference settings, agent budget where
applicable, commands, per-task outputs, failures, hashes, and reuse events.
The dataset task metric is primary; TTFT and throughput are supporting data.
Publish canonical artifacts in `vllm-hust-benchmark` before any website sync.

The [initial asset audit](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/236cf9f4a87348e56fbc08316892f8e0f7598633/reports/pujiang-specified-datasets-20261009)
recorded zero contract-ready datasets. [Benchmark PR #257](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/257)
subsequently froze and checked the five source snapshots, so all five are now
`asset-frozen`, but none is thereby `executable`. The benchmark program JSON
is the current machine-readable authority. No near-name substitute was used.
These remaining contract and environment blockers prevent a canonical
five-dataset score today.

The active [V5.5 acceptance boundary](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/main/docs/ACCEPTANCE_V5_5.md)
requires Qwen3.5-35B-A3B BF16, TP=2 on two Ascend 910B2 devices. B0 and B1
also require signed release and OCI digests plus actual startup/effective
configuration receipts from release/evaluation engineering. The 20- and
200-item diagnostics below used W8A8, TP=1 on 910B3 and cannot be promoted to
formal V5.5 evidence. The currently accessible `train8-docker` host has 910B3
devices and does not contain the separately verified five-dataset asset bundle.

## MMLU-Pro control-path diagnostic, 2026-10-09

This is a **20-item diagnostic**, not a program score or speedup result. Source:
`TIGER-Lab/MMLU-Pro` cached revision
`b189ec765aa7ed75c8acfea42df31fdae71f97be`, `data/test-00000-of-00001.parquet`
(SHA-256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`).
The script `benchmarks/mmlu_pro_pilot.py` selected 20 of 12,032 rows with seed
`20261009`, sorted by source row, and used zero-shot, temperature 0,
`max_tokens=16`, thinking disabled, and a strict single-letter grader.
Generated input SHA-256:
`da7b8960073ef1122ff7bf85e27d5b5eb088042cd25cb46f7d2d0ec10effab15`.

Both endpoints used `/data/jxd/Qwen3.5-35B-A3B-W8A8` and the same vLLM
Ascend environment (`vLLM 0.28.1.post1.dev143+gf18cf803c`, TP=1,
max model length 8192, block size 128, prefix caching disabled, chunked
prefill disabled, max sequences 4). B0 served on NPU 7, port 8107;
B1 served on NPU 6, port 8106 with `VLLM_ASCEND_SIMLLM_ENABLED=1`.
The source branch base was `b64586cd145551e614372d51423c3150d1bbce92`.
The diagnostic runner completed with 0 request errors and 0 invalid outputs
in each arm. Both arms returned 14/20 correct under this diagnostic grader.
Raw output SHA-256: B0
`d2a99597529f8f07635710d46ed0f1f8bf2b5d740c5b4dc771f330d593b3ab78`,
B1 `f55d16314f12c771e5d1d90ca53e9ddc68c79c5ac1357554150eab746d55b52d`.
The B1 server log shows 20 `SimLLM extract_kv` task stores, but zero
`SimLLM: reusing ... semantic tokens` events for these requests. Thus the
control path did not reuse any KV tokens. The 20-item score is too small and
uses a noncanonical prompt/grader; it supports no task-quality or latency
claim. Raw questions and answers are not committed. The local diagnostic
summary and output files are retained outside the repository under `/tmp`;
canonical artifacts await the benchmark program's frozen contract.

## MMLU-Pro 200-item diagnostic, 2026-10-09

To expand the control-path check without running an unfrozen full benchmark,
the same source Parquet and runner selected 200 rows with seed `20261009`.
Input JSONL SHA-256:
`193096be2779523b20d1abd8c124f1a302829d562de8089fdacece3f3aff100e`.
The committed [`sample200-manifest.jsonl`](../benchmarks/results/mmlu_pro_pilot_20261009/sample200-manifest.jsonl)
records task IDs and source rows without republishing question text.
Both arms used the same model, runtime, launch flags, request order, and
strict-letter scorer as the 20-item run. The per-task B0 and B1 outputs and
machine-readable [`summary200.json`](../benchmarks/results/mmlu_pro_pilot_20261009/summary200.json)
are committed alongside the manifest.

| Diagnostic measure | B0 | B1 |
| --- | ---: | ---: |
| Tasks | 200 | 200 |
| Correct under local strict-letter grader | 118 (59%) | 118 (59%) |
| Request errors / invalid outputs | 0 / 0 | 0 / 0 |
| Sequential request time (s) | 84.09 | 80.56 |

The B1 server log SHA-256 is
`bf6705c40d238ef80f9ec36651ac2299a80a570404cbaf69966786341ccc6b46`.
It records zero request-level `SimLLM: reusing ... semantic tokens` events.
The output SHA-256 values are
`2e140d47f2982153bb671a3f02217ffe0e49a23dc21a462026c4bb5c990b835b`
for B0 and
`907d62b31eea3af8a99254c8b809eae6b49ab1e9b711b25390d22318586334dc`
for B1. These were independent questions, so the plugin did not exercise its
reuse path. The timing difference is **not** attributed to SimLLM: there was
no KV reuse, and the endpoints ran on separate NPUs while the other service
could be starting. This local scorer and 200-item sample are not the program's
frozen MMLU-Pro contract and must not be entered as canonical task accuracy.
