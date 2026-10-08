from pathlib import Path

from vllm_hust_ext.manifest import activation_blocker, load_manifest

import vllm_ascend_simllm


def test_runtime_bundle_is_activatable() -> None:
    manifest = load_manifest(
        Path(vllm_ascend_simllm.__file__).with_name("vllm-hust-extension-v0.2.json")
    )
    assert manifest.bundle_id == "org.vllm-hust.simllm"
    assert activation_blocker(manifest) is None
    assert manifest.activation.environment == (("VLLM_ASCEND_SIMLLM_ENABLED", "1"),)
