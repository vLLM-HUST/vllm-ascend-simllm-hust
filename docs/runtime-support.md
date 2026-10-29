# SimLLM runtime support record

The plugin patches private `NPUModelRunner` methods. A successful wheel import
does not establish serving compatibility. The following combinations have
actual evidence; other combinations require a fresh activation and inference
test before they are described as supported.

| Combination | Evidence | Limit |
| --- | --- | --- |
| Qwen3.5-35B-A3B W8A8, Ascend 910B3 TP1, vLLM `0.28.1.post1.dev143+gf18cf803c.empty`, vLLM Ascend `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`, torch-npu `2.13.0rc1`, CANN 9.1 | [paired similarity replay](benchmark-custody.md) and non-model tests | Synthetic repeated-task workload; separate cards; task quality not established. |
| Qwen3.5-35B-A3B BF16, Ascend 910B3 TP2, vLLM `0.28.1.post1.dev143+gf18cf803c`, CANN 9.1 | [20-item MMLU-Pro diagnostic](../benchmarks/results/mmlu_pro_bf16_910b3_20261011/README.md) | Plugin stored KV but had zero reuse; neither performance benefit nor formal V5.5 task score established. |
| Clean Python 3.10/3.13 wheel without vLLM/torch/Ascend | `benchmarks/verify_wheel_lifecycle.py` | Entry-point discovery, inert installation, rejection of unqualified activation, fresh-process disable, uninstall; no model execution. |
| Final `0c004bd` wheel, Python 3.11.16, Qwen3.5-35B-A3B W8A8, 910B3 TP1 eager, same recorded host packages | [Installed-wheel inference and rollback](../benchmarks/results/issue2_inference_rollback_20261019/README.md) | 30 requests per arm; 24 enabled semantic hits; restoring the old wheel disabled exactly reproduces 30 baseline outputs. Functional lifecycle check. |

The W8A8 replay provides measured latency and throughput evidence for that
exact setup. The [existing-output audit](../benchmarks/results/issue3_quality_audit_20261029/README.md)
links 24 approximate hits to cached source IDs. Twenty of their 16-token
completion fragments differ from the disabled baseline; six misses match.
This is an observed text difference, not a graded quality score. The 5,000
request replay retained no decoded client text and used one output token, so
its quality effect cannot be inferred. Native APC was disabled and the two
fixed NPU cards were not swapped. The BF16 run used the same cards for B0 and B1,
but its local
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

Activation now requires Python 3.11 and the exact package revisions recorded in
`compatibility.QUALIFIED_VERSIONS`. The two recorded vLLM build strings are
accepted, alongside the recorded Ascend build, torch `2.13.0+cpu` and torch-npu
`2.13.0rc1`. The torch wheel's `+cpu` metadata is the actual installed value on
this Ascend host; torch-npu supplies its NPU backend. Missing packages and other
versions fail before hook registration. This allowlist is conservative and must
be updated with evidence when qualifying another environment. It checks package
metadata, not editable source integrity, CANN shared libraries, or hardware.

Once the runner loads, both private methods must exist, have the expected
leading parameters, and belong to the host module. Foreign replacements and a
changed active SimLLM patch are rejected. Dependencies and cache/index objects
are initialized before either method is replaced, so initialization failure
does not leave a partially patched runner. Decorators that deliberately conceal
their ownership cannot be identified by these checks.

Embedding/approximate-matcher failures use the existing unmatched-request path;
unsupported cache layouts bypass reuse. A KV restoration failure after the
schedule has been shortened propagates rather than forwarding with missing KV.
The latter is an intentional request/worker failure, not transparent recovery.

Run `python benchmarks/verify_host_lifecycle.py` from the qualified serving
environment after sourcing CANN and ATB. It installs a newly built wheel in a
disposable environment inheriting the host dependencies, checks real-runner
activation and fresh-process disable, and uninstalls only that overlay wheel.
For actual model inference and wheel replacement, run
`python benchmarks/verify_inference_rollback.py --help` and follow the
[recorded 30-request recipe](../benchmarks/results/issue2_inference_rollback_20261019/README.md).
That probe verifies exact installed files, actual semantic reuse, reinstalling
the old wheel, and output equality after restart. It keeps CANN/TBE import paths
while removing SimLLM source overrides. See the
[issue #2 audit](issue2-acceptance.md) for remaining closure gates. The 910B2
requirement belongs to the separate dataset program; issue #2 can record the
user-authorized 910B3 environment explicitly.
