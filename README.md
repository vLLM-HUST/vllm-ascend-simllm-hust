# SimLLM for vLLM Ascend

This package migrates the SimLLM implementation from the legacy
`vllm_ascend/simllm` tree into an installable plugin. It provides task embedding
extraction, SimHash and cosine matching, a bounded task KV cache, and
worker-side KV reuse. The active path restores complete per-layer cache state.
Standard attention can reuse identical token prefixes. Qwen3.5 recurrent
snapshots additionally require the complete source prefix and a longer target
for exact reuse; a partial prefix does not have a matching recurrent state.
Nearby SimHash buckets and cosine
similarity also allow different prompts to reuse a prior task's KV and skip
prefill approximately. The implementation targets the v1 `NPUModelRunner` in
`vllm-ascend-hust`.

The worker enables reuse only when the scheduler block size matches the
physical NPU KV block size. On the tested Ascend 910B host, specify
`--block-size 128`; a size mismatch makes the plugin fall back to normal
execution. For Qwen3.5-35B-A3B latency and throughput measurements, use
`--no-enable-chunked-prefill --no-enable-prefix-caching`. With chunked prefill
enabled, the worker can reuse only tokens within the current scheduler chunk;
the tested 2048-token chunk setting did not improve latency or throughput.
For Qwen3.5 hybrid Gated DeltaNet models, the worker snapshots the running
convolution/recurrent state from each linear-attention layer and the physical
KV blocks from each full-attention layer. It restores them through their
respective cache groups before the shortened model forward. This path
prioritizes prefill latency and throughput. Hybrid cache layouts that do not
expose these state tensors and group block tables fall back to normal execution.
For long Qwen3.5 near-duplicate prompts, a CPU token-count cosine score of at
least 0.95 selects a cached source before the NPU embedding/SimHash path. This
avoids device synchronization during hot-cache matching. A token-count match
is approximate even when the prompts have different token order. Exact
prefixes shorter than one 128-token cache block are not restored on this model.
When the host splits a prompt across prefill steps, the worker waits until the
full prompt has run before storing its reusable state.
If a Qwen3.5 semantic hit is at least 128 tokens longer than its cached
source, the worker saves the completed target as a new approximate source.
This keeps reusable spans from shrinking as the task cache turns over; these
promoted states remain approximate.

## Install and run

Install vLLM and vLLM Ascend for your NPU environment, then install this
package:

```bash
python -m pip install .
VLLM_ASCEND_SIMLLM_ENABLED=1 vllm serve /path/to/model
```

The package registers `simllm` in `vllm.general_plugins`. It is inert until
`VLLM_ASCEND_SIMLLM_ENABLED=1` is present when the worker loads. The hook
applies the SimLLM patch after the host's model runner module is imported.
No changes to the host source tree are needed. See [HOST_CONTRACT.md](HOST_CONTRACT.md)
for the integration details.

With the vLLM-HUST Extension Manager, the included manifest provides the
activation environment:

```bash
python -m pip install "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@main"
vllm-hust-ext extension inspect org.vllm-hust.simllm
vllm-hust-ext extension check org.vllm-hust.simllm
vllm-hust-ext extension enable org.vllm-hust.simllm
vllm-hust-ext run -- vllm serve /path/to/model
```

The Extension Manager records activation; the vLLM plugin entry point applies
the patch in the serving process. If `VLLM_PLUGINS` restricts plugin loading,
include both `ascend` (the platform plugin) and `simllm`, for example
`VLLM_PLUGINS=ascend,simllm`. Leaving it unset loads both automatically.
Loading only `simllm` prevents Ascend's Triton compatibility setup and can
stop the server during import.

## Configuration

All settings are read by the plugin from the worker process environment:

| Variable suffix after `VLLM_ASCEND_SIMLLM_` | Default | Purpose |
| --- | ---: | --- |
| `ENABLED` | `0` | Enable the worker patch |
| `COSINE_THRESHOLD` | `0.8` | Minimum cosine score for approximate task reuse |
| `LSH_NUM_BITS` | `64` | SimHash projection width |
| `LSH_BATCH_THRESHOLD` | `32` | Reserved for the legacy batch matcher |
| `LSH_HAMMING_RADIUS` | `2` | Maximum bit distance for approximate candidates |
| `KV_CACHE_SIZE` | `1024` | Maximum cached tasks |
| `KV_CACHE_MAX_BYTES` | `536870912` | Maximum bytes retained for prompt KV |
| `HYBRID_REUSE_MODE` | `tail_anchor` | `aggressive`, `exact`, `aligned`, `tail`, or `tail_anchor`; experimental hybrid policies described below |
| `HYBRID_RECOMPUTE_TOKENS` | `512` | Positive target suffix length to recompute for approximate `tail` and `tail_anchor` hits |
| `SANDWICH_BOTTOM` | `3` | Reserved for the inactive legacy sandwich path |
| `SANDWICH_TOP` | `3` | Reserved for the inactive legacy sandwich path |
| `EMBEDDING_POOLING` | `mean` | `mean`, `last`, or `cls` |
| `DEFERRAL_RATIO` | `0.5` | Diagnostic deferral threshold |
| `MAX_DEFERRALS` | `3` | Diagnostic deferral count |
| `PROFILE` | `0` | Synchronize and log worker stage timings for diagnosis; changes serving timing |

### Qwen3.5 hybrid reuse policies

`tail_anchor` is the default for Qwen3.5 hybrid models. It matches the quoted
main topic and chat template, restores cached full-attention KV, resets
recurrent state, and recomputes the final 512 target tokens. The
[5,000-request latency replay](benchmarks/results/qwen35_latency_20261008/README.md)
measured 20.39% higher throughput, 16.74% lower median TTFT, and 11.32% lower
p95 TTFT in the first run. The independent repeat measured 18.78%, 15.63%,
and 11.27% respectively.

Other policies are `aggressive`, `exact`, `aligned`, and `tail`. `tail` also
requires matching secondary context topics; `exact` requires a complete source
prefix and captured recurrent state. The quoted-topic heuristic is specific to
the synthetic similarity workload.

## Test

```bash
python -m pip install -e '.[test]'
pytest -q -m 'not model'
```

The `model` tests download TinyLlama and can be run separately. The migrated
NPU end-to-end scenarios require an Ascend device and a local model; CPU unit
tests do not prove device accuracy or performance. The worker patch uses host
internals, so check it on the exact vLLM Ascend version before production use.

For a latency comparison on one NPU, prepare a JSONL file with two different
but related `prompt` values and run
`tests/e2e/bench_semantic_reuse.py --model /path/to/model --prompts-jsonl prompts.jsonl`
in separate processes with `VLLM_ASCEND_SIMLLM_ENABLED=0` and `=1`. The script
prints each request's latency and output token IDs. A semantic hit should be
confirmed in the worker log; the cosine and hash gates alone do not guarantee
a speedup on small models.

For Qwen3.5-35B-A3B, compare the same W8A8 model and serving flags in two
processes, one with `VLLM_ASCEND_SIMLLM_ENABLED=0` and one with `=1`. The
following client measures streaming first-token latency and concurrent output
throughput against each server:

```bash
python tests/e2e/bench_qwen35_server.py --url http://127.0.0.1:8102 \
  --model qwen35-a3b-w8a8 --paragraph-repeats 128 --vary-paragraph \
  --max-tokens 1
```

The tested Ascend 910B3 setup used TP2, W8A8, an 8192-token context, eager
execution, disabled host prefix caching, and disabled chunked prefill. Inspect
the worker log for
`reusing ... semantic tokens`; an HTTP latency reduction without a logged
reuse event is not evidence of a SimLLM hit. The cache primarily removes
prefill work. Long generations can spend enough time decoding that total
latency or output-token throughput does not improve.

On the tested CANN environment, a fresh compiled-cache directory triggered
PyTorch's `expected OutputCode` error during AOT Autograd cache creation.
Setting `TORCHINDUCTOR_AUTOGRAD_CACHE=0` allowed PIECEWISE compilation to
complete. This setting was used for the later TP1 batch-capacity comparison.

The Qwen3.5-35B-A3B `semantic_similar.jsonl` measurements, including the full
5000-request prefill replay, card-swapped controls, original-output subset,
and per-request records, are in
[benchmarks/results/qwen35_similarity_20260929/README.md](benchmarks/results/qwen35_similarity_20260929/README.md).

The later [forward and serving follow-up](benchmarks/results/qwen35_forward_20260930/README.md)
profiles the hybrid cache path and measures a batched KV restore for three or
more simultaneous hits. On the sampled 300 requests, TP1 with PIECEWISE graph
capture delivered 7.3168 requests/s versus 5.7016 for the matched baseline;
TP2 eager delivered the lowest plugin TTFT p95 at 2.5704 s. These are small
samples on fixed card assignments. A follow-up TP1 comparison found no useful gain from increasing
`--max-num-seqs` and graph capture size from 4 to 8 on this 300-request sample.
On the same card, TP1 PIECEWISE at capacity 4 also delivered 14.0% more
requests/s than TP1 eager; eager had a 32 ms lower TTFT p95 in that one run.
On a separate held-out 1000-request sample, promoting longer semantic targets
kept throughput above 17.7 requests/s for three consecutive passes, while the
earlier cache policy fell from 12.0 to 5.9 requests/s. The promoting path
reduced TTFT p95 to 1.21–1.48 s across those passes. On the complete
5000-request workload, the promoting plugin completed 20.7108 requests/s
versus 5.0998 for the baseline (4.061x), and
reduced TTFT p95 from 3.2770 to 0.9873 s. The worker confirmed reuse on
4972 requests; the total un-reused prompt span was 98,265 of 14,789,413
tokens. This single-pass comparison used fixed NPU card assignments and
one-token outputs; it does not establish long-generation throughput. See the
linked report for serving flags and per-request records.

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
