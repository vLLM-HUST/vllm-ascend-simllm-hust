# SimLLM host contract

This document defines the seams that must exist before SimLLM can become an
activatable vLLM Ascend extension. It is an interface specification, not a copy
of the historical or public SimLLM implementation.

## Ownership boundary

- Extension Manager installs, discovers, checks compatibility, and renders a
  plan. It must not rewrite a running scheduler or load a model by itself.
- vLLM owns request lifecycle, block tables, scheduler mutation, worker RPC,
  model execution, and rollback.
- The extension may propose similarity reuse and slot mappings only through the
  versioned host protocols below.
- Embedding/index resources are extension-owned unless a future manifest names
  an external service and assigns its lifecycle to an external operator.

## Required protocols

### `vllm.scheduler.rewrite-policy.v1`

The host supplies immutable request snapshots and accepts a deterministic
proposal containing admission order, reused-token count, and an explanation.
The proposal must be validated before it changes scheduler state. Cancellation,
preemption, and finished requests invalidate pending proposals.

### `vllm.kv-reuse.index.v1`

The host exposes opaque request/block identities, cache generation, model ID,
adapter ID, KV dtype/layout, and device identity. The extension may return a
candidate match; the host remains responsible for bounds checks, block ownership,
reference counts, and rejecting stale generations.

### `vllm.embedding.provider.v1`

The extension receives token IDs and explicit model/revision metadata. The
provider must declare embedding dimension, normalization, batching limits,
device placement, and deterministic-version identity. No hidden model download
or network access is permitted without manifest permissions and configuration.

### `vllm.model-runner.slot-mapping.v1`

The host validates a proposed sandwich/injection mapping before execution. A
mapping includes source and destination ranges, cache generation, attention
layer range, and expected KV layout. Partial application must fail closed.

## Compatibility and lifecycle

Activation requires all four protocols, a pinned vLLM/vLLM Ascend range, a
supported model family, identical tokenizer/model revisions, compatible KV
layout, and device tests. Prefix caching, speculative decoding, LoRA, pipeline
parallelism, and disaggregated KV transfer are incompatible until individually
tested and declared. Disablement must stop new proposals, drain or invalidate
pending proposals, release extension-owned index state, and leave host-owned KV
blocks untouched.

## Source and license gate

Historical implementation code may enter this repository only through one of
two routes checked by `validate_source_evidence`:

1. a licensed migration with identified copyright holders and a reviewable
   license/permission record; or
2. a clean-room implementation from this behavior contract by independently
   recorded authors.

Until then, the package remains `import_only` and Extension Manager must refuse
enablement.

