# Installed-wheel inference and rollback, 2026-10-19

**PASS** on Qwen3.5-35B-A3B W8A8, Ascend 910B3 physical card 7, TP1.
This is a 30-request lifecycle diagnostic. The separate 5,000-request
performance measurements retain their original scope.

| Stage | Installed code | Enabled | Successful requests | Semantic reuse events | Skipped tokens |
| --- | --- | --- | ---: | ---: | ---: |
| Baseline | PR #10 main (`8101b03`) | 0 | 30 | 0 | 0 |
| Final | PR #12 runtime (`0c004bd`) | 1 | 30 | 24 | 53,704 |
| Rollback | Reinstalled PR #10 wheel | 0 | 30 | 0 | 0 |

All **30 rollback outputs exactly matched baseline** in request ID, prompt
and completion token counts, and decoded text. All three arms used the same
ordered input and model/service settings. The original serving installation's
Python source files and package versions were unchanged. Each overlay install
was checked against the Python-file hashes inside its wheel; replacing an
identical `0.1.0.dev0` version label alone was not considered sufficient.
`run5/summary.json` is the machine-readable result. `request-alignment.json`
links all 24 reuse events to client request IDs and checks prompt token counts.
The enabled arm's approximate output is not used as an equality target.

## Frozen inputs and runtime

- Exact final wheel SHA-256: `7ae77f00e47a4de1ad34acbbd02db57c4d0d565af2184ceee62490f621566666`.
- Rollback wheel SHA-256: `10c0db015afa70b34277c8272814e5ffca0715350344041a7c6b384c4498dfea`.
- Both wheels are retained in `wheels/`; built from `git archive` of the
  source commits recorded in `experiment.json`. Runtime Python files were
  unchanged by the subsequent probe/documentation changes.
- Input: committed `benchmarks/fixtures/qwen35_warmup30.jsonl`, six intact
  five-request similarity groups; SHA-256
  `90f3c3229d96237b389e2a3f45fa8f8618d270921091211166661fec4a4e2c41`.
- Python 3.11.16; vLLM `0.28.1.post1.dev143+gf18cf803c.empty`; Ascend
  `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`; torch `2.13.0+cpu`;
  torch-npu `2.13.0rc1`; CANN 9.1 and ATB. Per-stage versions and code hashes
  are in `run5/*/installed.json`.
- Same 910B3 card, eager execution, TP1, context 8192, block size 128,
  max sequences 4, memory utilization 0.85, seed 0, prefix caching off,
  chunked prefill off. Chat requests disable thinking, temperature 0,
  maximum 16 generated tokens, concurrency 1.
- A temporary local copy of the 37 GiB DPC checkpoint was used for all arms.
  All 22 files were SHA-256 verified while staging; the full model file
  manifest is `model-staging.json`. Both checkpoint reading and execution
  used eager mode. No model weights are published here.

## Reproduction on the measured host

Use a free device and port. Source the qualified runtime, create a local
checkpoint copy with adequate space, then run from a checkout of this PR:

```bash
source /data/jxd/Ascend-9.1.0/cann-9.1.0/set_env.sh
source /data/jxd/Ascend-9.1.0/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/data/jxd/cache/torch_ext
export TORCHINDUCTOR_CACHE_DIR=/data/jxd/cache/torchinductor
export TRITON_CACHE_DIR=/data/jxd/cache/triton
export VLLM_CACHE_ROOT=/tmp/simllm-inference-rollback-oct19/cache
mkdir -p /tmp/simllm-inference-rollback-oct19
/data/jxd/envs/vllm-hust-v1-cann91/bin/python \
  benchmarks/results/issue2_inference_rollback_20261019/stage-model.py
/data/jxd/envs/vllm-hust-v1-cann91/bin/python \
  benchmarks/verify_inference_rollback.py \
  --final-wheel benchmarks/results/issue2_inference_rollback_20261019/wheels/final/vllm_ascend_simllm-0.1.0.dev0-py3-none-any.whl \
  --rollback-wheel benchmarks/results/issue2_inference_rollback_20261019/wheels/rollback/vllm_ascend_simllm-0.1.0.dev0-py3-none-any.whl \
  --model /tmp/simllm-inference-rollback-oct19/model-local \
  --device 7 --port 8128 --output /tmp/simllm-rollback-replay-new
```

The staging helper refuses to overwrite an existing destination and reserves
20 GiB of free disk space beyond the copy. The probe uses a disposable
`--system-site-packages` virtual environment, uninstalls only its local
package before switching wheels, and stops its own service group between arms.
`run5/*/launch.json` and `client-command.json` retain actual commands/settings.

## Failure records and checker correction

`failed-attempt/` retains the first startup failure: the harness initially
removed CANN's PYTHONPATH along with source overrides, making `acl` unavailable.
The environment filter now preserves CANN/TBE paths and rejects SimLLM source
shadowing; a preflight checks both imports before model loading.

`aborted-startups.tgz` retains the DPC mmap, prefetch and eager-loading attempts,
which were interrupted during checkpoint loading before any client requests.
The observed wait was `odcs_dispatch_wait_complete`. Local staging resolved
that storage bottleneck. This is not an inference speedup measurement.

The initial checker also required a startup INFO banner. That banner was
absent even though 24 real reuse events and 30 successful responses were
recorded. The checker now requires runtime semantic-reuse events; a banner
alone is insufficient. Its tests cover both cases. `run5/pre-resume-summary.json`
and `run5.log` preserve the original false rejection. `--resume-completed`
revalidated the completed baseline/final records against the exact input,
wheels, recorded activation flags and unchanged base installation, then
actually installed and ran the rollback arm. The continuation is retained in
`run5-resume.log`. No model outputs were replaced during revalidation.

All services and the temporary model copy were cleaned up after success;
`cleanup.json` and `device7-after.txt` record the final state. `SHA256SUMS`
checksums the retained artifacts. The host's broken Git worktree reference
remains the separate source-snapshot gap recorded in the issue #2 audit.
