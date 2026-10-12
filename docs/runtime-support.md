# SimLLM runtime support record

The plugin patches private `NPUModelRunner` methods. A successful wheel import
does not establish serving compatibility. The following combinations have
actual evidence; other combinations require a fresh activation and inference
test before they are described as supported.

| Combination | Evidence | Limit |
| --- | --- | --- |
| Qwen3.5-35B-A3B W8A8, Ascend 910B3 TP1, vLLM `0.28.1.post1.dev143+gf18cf803c.empty`, vLLM Ascend `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`, torch-npu `2.13.0rc1`, CANN 9.1 | [5,000-request paired similarity replay](../benchmarks/results/qwen35_latency_20261008/README.md) and non-model tests | Synthetic repeated-task workload; separate cards; task quality not established. |
| Qwen3.5-35B-A3B BF16, Ascend 910B3 TP2, vLLM `0.28.1.post1.dev143+gf18cf803c`, CANN 9.1 | [20-item MMLU-Pro diagnostic](../benchmarks/results/mmlu_pro_bf16_910b3_20261011/README.md) | Plugin stored KV but had zero reuse; neither performance benefit nor formal V5.5 task score established. |
| Clean Python 3.13 wheel without vLLM/torch/Ascend | `benchmarks/verify_wheel_lifecycle.py` | Entry-point discovery, opt-in activation, fresh-process disable, uninstall only; no model execution. |

The W8A8 replay provides measured latency and throughput evidence for that
exact setup. The BF16 run used the same cards for B0 and B1, but its local
zero-shot prompt and grader are not the dataset program's canonical contract.
Formal V5.5 acceptance specifies 910B2 BF16 TP2 and additional signed runtime
receipts. Neither 910B3 row is a substitute for that evaluation.

## Preflight and failure behavior

The extension is inert after installation. With `ENABLED=1`, the general
plugin entry point installs an import hook for the Ascend model runner.
Unknown configuration names, invalid numeric ranges, and malformed booleans
raise `ValueError` before that hook is installed. The wheel lifecycle probe
checks this opt-in boundary and verifies that a newly started disabled process
has no SimLLM hook. To disable or roll back a live service, stop and restart
it; the plugin does not offer live unpatching of a running model runner.

The current hook does not pin exact host package revisions at runtime and has
not passed the full compatibility-rejection, fault-degradation, or formal
910B2 hardware matrix required by [issue #2](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/issues/2).
Keep that issue open until those checks and end-to-end validation are recorded.
