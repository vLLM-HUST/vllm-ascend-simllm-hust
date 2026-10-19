# Issue #2 acceptance audit — 2026-10-19

Issue #2 is **not ready to close**. PR #10 is merged at `8101b03`; the
follow-up activation guards and tests now include a successful installed-wheel
inference and rollback diagnostic on the owner-approved 910B3 environment.
The source-snapshot and original provenance criteria remain unresolved.

| Requirement | Evidence and remaining work |
| --- | --- |
| Provenance / authorization | The owner instructed this migration to proceed without this gate. That instruction is not a published copyright-holder authorization or an independent implementation record; the original issue criterion remains formally unresolved. |
| Real adapter / entry point | The wheel exposes `vllm.general_plugins:simllm`; it wraps the Ascend runner after import. Existing device replays exercised KV storage/reuse. Manager lifecycle boundaries remain unchanged. |
| Compatibility matrix and rejection | Runtime support is narrowed to recorded Python 3.11 and package revisions. New tests reject unqualified/missing packages, missing/private-interface changes, and visible foreign runner replacements. Editable source integrity and hardware/CANN qualification remain separate requirements. |
| Configuration, conflicts and dependency failures | PR #10 added strict configuration parsing. This branch initializes dependencies before replacing either host method and tests constructor failure, repeated activation, later conflict, embedding failure and fatal KV-restore failure. |
| Clean wheel, disable, rollback and uninstall | Clean-wheel and real-runner probes pass. The final installed wheel also completed 30 model requests with 24 semantic reuse events; reinstalling the prior wheel and restarting disabled completed 30 requests exactly matching baseline. The original serving installation was unchanged. |
| Real hardware correctness / performance | The new 910B3 TP1 W8A8 diagnostic passes inference and package rollback for 30 requests per arm. It is a lifecycle check, not a new performance or task-quality benchmark. The committed full similarity replay and BF16 TP2 diagnostic retain their original scope. The separate dataset program's 910B2 gate is not imposed on this owner-approved 910B3 engineering check. |
| Executable lifecycle documentation | README, HOST_CONTRACT and runtime-support document installation, explicit enablement, stop, restart disable, rollback and uninstall, with concrete probe commands. |
| CI enabled | PR #10 enabled Python 3.10/3.13 package CI. This branch adds compatibility tests to it. These Python versions cover the inert/rejection path; activation is qualified only on Python 3.11. |

## Verification in this follow-up

- Local configuration/plugin/compatibility tests: **43 passed**.
- Inference-evidence, environment-filter and stage-validation tests: **13 passed**.
- Clean-wheel lifecycle: **PASS** on Python 3.13; no host packages installed.
- Remote non-model suite on `train8-docker`, Python 3.11.16 with CANN 9.1
  and ATB sourced: **227 passed, 2 skipped, 4 deselected**. The manifest test
  was excluded because `vllm_hust_ext` is absent on that environment. The four
  deselected model tests were not run.
- Ruff and formatting checks pass for source, tests and the lifecycle probes.
- Installed-wheel inference/rollback: **30 / 30 / 30 successful requests**,
  **24 semantic reuse events**, **53,704 skipped tokens** in the enabled arm;
  **30/30 rollback outputs exactly equal baseline**. Request IDs and original
  prompt token counts align across arms. The original serving installation
  is unchanged and all owned model services were stopped.

The initial direct-runner import and the initial host wheel probe encountered
an upstream `DeviceOperator` circular import before model loading. The probe
now initializes Ascend operators before loading the worker. Keep the failed
attempt as diagnostic evidence; it is not an inference failure or score.

On this remote environment, the Ascend source tree's `.git` points to an
unavailable worktree parent, so `git rev-parse HEAD` fails. Its reported package
metadata alone cannot establish the actual source revision. The observed
`model_runner_v1.py` SHA-256 is
`407da57242c5700e76d2b5980979c32f8d27af1ccf9a68f6ed6ef39e0baaa7ee`.

The real-host wheel probe passed all three fresh-process runner states
(disabled, enabled, disabled) and overlay uninstall. Its tested wheel SHA-256
is `12dc24c9992b12e19afd4a4bd1265d86c7990e262f641599489992d6053ab608`.
That wheel predates the final additional missing-runner rejection; this last
error-path change is covered by the final 43/227-test runs. Raw test logs,
the failed first probe and the corrected probe are preserved in
[`benchmarks/results/issue2_runtime_20261019`](../benchmarks/results/issue2_runtime_20261019/).

The subsequent real inference run used the exact final runtime from `0c004bd`,
wheel SHA-256
`7ae77f00e47a4de1ad34acbbd02db57c4d0d565af2184ceee62490f621566666`.
It includes the final missing-runner rejection. Its paired outputs, wheel
artifacts, model checksums, failure records, cleanup and reproduction commands
are retained in the
[installed-wheel inference report](../benchmarks/results/issue2_inference_rollback_20261019/README.md).

## Work before closure

1. Restore a reliable source revision record or archive/checksum the actual
   runtime source tree. The actual tested plugin wheels are now retained.
2. Resolve the original issue's provenance criterion explicitly in the public
   issue, or amend that criterion through its maintainer; do not silently mark
   it as satisfied by repository ownership.
3. Review the new PR and its CI, merge the accepted engineering changes, then
   update the issue checklist against the final evidence.
