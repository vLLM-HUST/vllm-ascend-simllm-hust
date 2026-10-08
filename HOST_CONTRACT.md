# SimLLM host integration

The plugin uses vLLM's `vllm.general_plugins` entry point and the existing
`NPUModelRunner` and Ascend attention backend in `vllm-ascend-hust`. It does not
require files to be copied into the host repository.

When `VLLM_ASCEND_SIMLLM_ENABLED=1`, `vllm_ascend_simllm.plugin:register`
installs a Python import hook. Once
`vllm_ascend.worker.model_runner_v1` finishes loading, the hook applies the
migrated SimLLM worker patch. The active patch wraps `execute_model` and
`_model_forward`; it seeds per-layer KV for an identical token prefix or a
similar cached task before the shortened forward. Similar-task reuse is an
approximation and may change output. The original methods are retained by the patch module
for tests and diagnostics.

The plugin owns its configuration, similarity index, task embeddings, and
cached KV tensors. The host still owns request scheduling, block allocation,
model execution, and attention cache storage. The current integration depends
on internal host APIs and must be checked against each host revision. Worker
schedule counts are adjusted together for reused prefix tokens; the engine
scheduler continues to own the original prompt length and block allocation.
The plugin checks that scheduler and physical KV block sizes agree before
reusing data. Each cached snapshot stores whole physical blocks per layer so
Ascend cache layouts are preserved without interpreting their token axis.

The Extension Manager bundle manifest enables the environment flag. Direct
`vllm serve` launches can enable the same plugin with that flag after the wheel
is installed. Installation alone leaves the patch inactive.
