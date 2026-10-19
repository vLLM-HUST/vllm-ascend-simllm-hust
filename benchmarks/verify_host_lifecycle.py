"""Install a wheel in a disposable overlay of a qualified Ascend environment.

Run with the serving environment's Python after sourcing CANN/ATB. The probe
loads the real runner but does not load model weights or run inference.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import sys
import vllm.platforms
import vllm_ascend.ops
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend_simllm import plugin
plugin.register()
enabled = sys.argv[1] == '1'
assert (NPUModelRunner.execute_model.__module__ ==
        'vllm_ascend_simllm.patch.patch_model_runner') == enabled
assert (NPUModelRunner._model_forward.__module__ ==
        'vllm_ascend_simllm.patch.patch_model_runner') == enabled
if enabled:
    plugin.register()
print('host runner enabled=' + str(enabled) + ': PASS')
"""


def main() -> None:
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("VLLM_ASCEND_SIMLLM_") and key != "PYTHONPATH"
    }
    clean_env["VLLM_PLUGINS"] = "ascend,simllm"

    def run(*args: str, enabled: bool = False) -> None:
        env = {**clean_env, "VLLM_ASCEND_SIMLLM_ENABLED": str(int(enabled))}
        subprocess.run(args, check=True, cwd=ROOT, env=env)

    with tempfile.TemporaryDirectory(prefix="simllm-host-lifecycle-") as temp:
        directory = Path(temp)
        run(sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", temp, ".")
        wheel = next(directory.glob("vllm_ascend_simllm-*.whl"))
        overlay = directory / "venv"
        run(sys.executable, "-m", "venv", "--system-site-packages", str(overlay))
        python = str(overlay / "bin" / "python")
        run(
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--ignore-installed",
            str(wheel),
        )
        run(
            python,
            "-I",
            "-c",
            "import sys,vllm_ascend_simllm; "
            "assert vllm_ascend_simllm.__file__.startswith(sys.prefix)",
        )
        run(python, "-I", "-c", PROBE, "0")
        run(python, "-I", "-c", PROBE, "1", enabled=True)
        run(python, "-I", "-c", PROBE, "0")
        run(python, "-m", "pip", "uninstall", "-y", "vllm-ascend-simllm")
        # A pre-existing base installation may remain. Never uninstall or
        # modify the serving environment's package to perform this probe.
        run(
            python,
            "-I",
            "-c",
            "import importlib.util,sys; "
            "s=importlib.util.find_spec('vllm_ascend_simllm'); "
            "assert s is None or not s.origin.startswith(sys.prefix)",
        )
        print(
            "qualified host wheel activation, restart disable, overlay uninstall: PASS"
        )


if __name__ == "__main__":
    main()
