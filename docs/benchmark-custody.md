# Archived replay inputs

The paired replay collection is owned by the benchmark repository, pinned at
`71c2d4ad5f34536d73f66c577e1017da59dba085`. Its
[manifest](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/71c2d4ad5f34536d73f66c577e1017da59dba085/archive/mod-git-20261022/website-source-vllm-ascend-simllm-hust/manifest.json)
and [original files](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/71c2d4ad5f34536d73f66c577e1017da59dba085/archive/mod-git-20261022/website-source-vllm-ascend-simllm-hust/snapshot)
preserve the original report, requests, diagnostics, repeat and failures unchanged.
This migration does not qualify any new hardware or change any historical result.

Download into an empty directory outside this checkout:

```bash
python scripts/fetch_benchmark_replay.py /absolute/external/simllm-replay
```

The downloader verifies the pinned manifest, every original byte length and SHA256,
rejects unsafe paths, and does not unpack or execute archived material. CI repeats
the remote checks. Existing historical directories outside this collection remain
a separate migration queue; they are not declared migrated by this change.
