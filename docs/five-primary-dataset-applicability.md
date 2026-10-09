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
| MMLU-Pro | not-exercised | A pinned 20-item diagnostic B0/B1 run reached the plugin and stored 20 tasks, but logged zero request-level KV-reuse events. Independent questions did not exercise the optimization path. The canonical item/evaluator contract remains unfrozen. | @GuMorming with benchmark maintainers: freeze the official split, prompt/scorer and task manifest; define a repeated or genuinely similar MMLU-Pro workload if this dataset is expected to exercise reuse, then repeat matched B0/B1 with request-level evidence. |
| HLE-Verified | blocked | The full verified set versus verified-gold subset and grader are not frozen; no adapter or task-level run exists. | @GuMorming: agree on subset and grader revision, implement per-item scoring and reuse trace, then run matched B0/B1. |
| SWE-bench-Pro | blocked | A validated release, fresh-sandbox regrading path, agent scaffold/tool policy, and request-to-task trace are absent. | @GuMorming with benchmark maintainers: pin release and isolated grader, integrate the serving endpoint into the agent runner, then run matched B0/B1 with resolved-task rate. |
| FrontierScience | blocked | Olympiad and Research require separate pinned answer/scoring contracts; neither runner is wired to this plugin. | @GuMorming with benchmark maintainers: pin both tracks and graders, implement separate per-track outcomes and reuse trace, then run matched B0/B1. |
| Terminal-Bench 2.1 | blocked | The Harbor 2.1 task release, environment, agent scaffold, tool policy, budget, and request-to-task trace are not pinned in this repository. | @GuMorming with benchmark maintainers: freeze the Harbor contract, integrate the serving endpoint, then run matched B0/B1 with task resolution as primary. |

For every future comparison, B0 must be the same Qwen3.5-35B-A3B weights and
Ascend serving stack with SimLLM disabled; B1 changes only SimLLM activation
and its documented policy. Record immutable model/runtime/dataset/evaluator
revisions, hardware and topology, inference settings, agent budget where
applicable, commands, per-task outputs, failures, hashes, and reuse events.
The dataset task metric is primary; TTFT and throughput are supporting data.
Publish canonical artifacts in `vllm-hust-benchmark` before any website sync.

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
