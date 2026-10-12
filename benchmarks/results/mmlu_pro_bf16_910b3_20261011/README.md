# MMLU-Pro BF16 TP2 on Ascend 910B3, 2026-10-11

This is a **compatibility and control-path diagnostic**, not a formal V5.5
acceptance result, a canonical MMLU-Pro score, or a speedup claim. V5.5
specifies two 910B2 devices. This accessible host had 910B3 devices; the user
authorized a clearly labeled diagnostic on them.

## Inputs and runtime

- Model: `/data/jxd/Qwen3.5-35B-A3B` BF16, `config.json` SHA-256
  `5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`.
- Host: `train8-docker`, Ascend 910B3 cards 0,1 for **both** arms, TP=2.
- vLLM: `0.28.1.post1.dev143+gf18cf803c`; CANN
  `/data/jxd/Ascend-9.1.0/cann-9.1.0/set_env.sh`; ATB
  `/data/jxd/Ascend-9.1.0/nnal/atb/set_env.sh`.
- Serve: model length 8192, block size 128, prefix cache off, chunked prefill
  off, max sequences 4, GPU memory utilization 0.85, PIECEWISE graph capture
  up to 4. This vLLM reported that disabling chunked prefill is not officially
  supported for this model, although both services completed this diagnostic.
- B0: `VLLM_ASCEND_SIMLLM_ENABLED=0`; B1: `=1`, `HYBRID_REUSE_MODE=tail_anchor`,
  `HYBRID_RECOMPUTE_TOKENS=512`. Start B0, run it, stop it, then start B1 on
  the same cards. See [`../../start_qwen35_bf16_diagnostic.sh`](../../start_qwen35_bf16_diagnostic.sh).
- MMLU-Pro source: cached `TIGER-Lab/MMLU-Pro` revision
  `b189ec765aa7ed75c8acfea42df31fdae71f97be`, test Parquet SHA-256
  `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`.
  Fixed 20-row sample seed `20261009`, input JSONL SHA-256
  `da7b8960073ef1122ff7bf85e27d5b5eb088042cd25cb46f7d2d0ec10effab15`.
  Zero-shot, temperature 0, thinking off, max output 16, strict single-letter
  local grader. The source prompt text is not republished here.

## Result

| Diagnostic measure | B0 | B1 |
| --- | ---: | ---: |
| Tasks | 20 | 20 |
| Correct under local strict-letter grader | 14 | 14 |
| Request errors / invalid outputs | 0 / 0 | 0 / 0 |
| Sum of sequential client request durations | 9.48 s | 10.40 s |
| Request-level KV reuse events | N/A | 0 |
| Reused semantic tokens | N/A | 0 |

All 20 predictions matched between arms. The B1 worker logged 20 stored tasks
per TP worker, so the plugin storage path ran. It logged no `SimLLM: reusing`
event. A separate control sent the same 518-token MMLU-Pro item five times;
it also recorded five correct responses and zero reuse events. Its input SHA-256
was `0e13d4430e36647ec915b1b097b0b1e2e9e1d7d5601b3d18257fe908d518af08`,
and the output is [`repeat5-plugin.jsonl`](repeat5-plugin.jsonl). The repeat
is only a control-path probe, not five independent benchmark tasks. Current
`tail_anchor` policy recomputes up to 512 tail tokens and has hybrid matching
guards, so an exact repeat of a 518-token prompt is not evidence that the
general semantic reuse path works or fails. It confirms that **this** probe
did not reuse KV.

The 20-item times are short sequential client durations, with no TTFT or
throughput measurement and no reuse. They provide **no latency benefit claim**.
The diagnostic prompt and grader differ from the benchmark program's frozen
five-shot MMLU-Pro candidate; no result here is a canonical dataset score.

Artifacts: [`summary20.json`](summary20.json),
[`baseline20.jsonl`](baseline20.jsonl), [`plugin20.jsonl`](plugin20.jsonl),
and the existing 20-item sample manifest in
[`../mmlu_pro_pilot_20261009`](../mmlu_pro_pilot_20261009).
SHA-256 of completed B0 and B1 server logs:
`a74bf380c3f169a00ed3df17e9dea07a399988511b6a232f2f5b8d1f976839a6`
and `8c905e40c4177c4418ab38fd0ceddd5a4764dce8bfb786cf5851866e5f6bb02a`.
The logs remain on the remote host under `/tmp/simllm-issue7-bf16/`.

The first B0 start used the host's default CANN path and failed in prewarm
with missing `aclnnAddRmsNormBias` from `libopapi.so`. Restarting both arms
with the pinned CANN 9.1 and ATB paths above resolved it. The failed run did
not process requests; its log SHA-256 is
`3d2b8e98ac04fd571f05cb2cc1a44d32d737b76583603124c0b2e9890f1dca64`.

## Reproduction

On this host, with the model/runtime/CANN paths above available:

```bash
bash benchmarks/start_qwen35_bf16_diagnostic.sh baseline /tmp/simllm-issue7-bf16
python3 benchmarks/mmlu_pro_pilot.py run \
  --input /tmp/simllm-issue7/simllm-mmlu-pro-pilot20.jsonl \
  --output /tmp/simllm-issue7-bf16/baseline20.jsonl \
  --url http://127.0.0.1:8117 --model qwen35-a3b-bf16
# Read the PID from the arm-specific file; verify its command, then stop its process group.
kill -TERM -- -"$(cat /tmp/simllm-issue7-bf16/baseline/swe_server_bf16.pid)"
bash benchmarks/start_qwen35_bf16_diagnostic.sh plugin /tmp/simllm-issue7-bf16
python3 benchmarks/mmlu_pro_pilot.py run \
  --input /tmp/simllm-issue7/simllm-mmlu-pro-pilot20.jsonl \
  --output /tmp/simllm-issue7-bf16/plugin20.jsonl \
  --url http://127.0.0.1:8117 --model qwen35-a3b-bf16
python3 benchmarks/mmlu_pro_pilot.py summarize \
  --baseline /tmp/simllm-issue7-bf16/baseline20.jsonl \
  --plugin /tmp/simllm-issue7-bf16/plugin20.jsonl \
  --plugin-log /tmp/simllm-issue7-bf16/plugin/swe_server_bf16_*.log
```

Regenerate the input using `mmlu_pro_pilot.py prepare --limit 20 --seed
20261009` and the pinned test Parquet, then check the input SHA above before
running. The remote `/tmp` paths are ephemeral, so preserve full logs in a
durable artifact store before any formal evaluation.
