# SimLLM for vLLM Ascend

Owner-led migration carrier for task-similarity KV reuse. Release is blocked until the owner confirms whether the legacy code is independent or derives from the public SimLLM repository, which has no declared license.

**Status: source-preservation and contract-design scaffold. There is no installable runtime implementation or support claim yet.**

Technical ownership belongs to @GuMorming. Source extraction must preserve exact authorship, license, tests, constraints, and evidence before activation is considered.

See [MAINTAINERS.md](MAINTAINERS.md) and [PROVENANCE.md](PROVENANCE.md).

## Extension framework

Extension ID: `org.vllm-hust.simllm`

This repository follows the vLLM-HUST Extension Template. The current package
is deliberately `import_only`: it can be built, installed, discovered, and
inspected, but Extension Manager must refuse enablement until the maintainers
land a real host contract, implementation, compatibility evidence, and tests.

```bash
python -m pip install "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@main"
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.simllm
vllm-hust-ext extension check org.vllm-hust.simllm
pytest -q
```

The static Manifest 0.2 descriptor lives inside the Python distribution under
`src/`. Installation alone changes no vLLM behavior.
