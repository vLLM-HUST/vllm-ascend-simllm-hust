"""Exercise clean-wheel discovery, opt-in activation, restart rollback and uninstall."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import importlib.metadata as metadata
import importlib.util
import sys
from vllm_ascend_simllm import plugin

eps = metadata.entry_points(group='vllm.general_plugins')
assert sum(ep.name == 'simllm' for ep in eps) == 1
plugin.register()
finders = [finder for finder in sys.meta_path
           if isinstance(finder, plugin._RunnerFinder)]
assert len(finders) == int(sys.argv[1]), len(finders)
assert importlib.util.find_spec('vllm_ascend') is None
"""


def run(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, check=True, cwd=ROOT, env=env)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="simllm-wheel-lifecycle-") as temp:
        temp_dir = Path(temp)
        wheel_dir = temp_dir / "wheel"
        wheel_dir.mkdir()
        run(
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            ".",
        )
        wheel = next(wheel_dir.glob("vllm_ascend_simllm-*.whl"))
        venv = temp_dir / "venv"
        run(sys.executable, "-m", "venv", str(venv))
        python = venv / "bin" / "python"
        if sys.platform == "win32":
            python = venv / "Scripts" / "python.exe"
        run(str(python), "-m", "pip", "install", "--no-deps", str(wheel))
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("VLLM_ASCEND_SIMLLM_") and key != "PYTHONPATH"
        }
        # Installation is inert; activation is explicit; rollback takes effect
        # in a fresh worker process after the flag is removed.
        run(str(python), "-c", PROBE, "0", env=clean_env)
        enabled_env = {**clean_env, "VLLM_ASCEND_SIMLLM_ENABLED": "1"}
        run(str(python), "-c", PROBE, "1", env=enabled_env)
        run(str(python), "-c", PROBE, "0", env=clean_env)
        run(str(python), "-m", "pip", "uninstall", "-y", "vllm-ascend-simllm")
        run(
            str(python),
            "-c",
            "import importlib.util; "
            "assert importlib.util.find_spec('vllm_ascend_simllm') is None",
            env=clean_env,
        )
        print(
            "wheel lifecycle: install, inert, enable, restart rollback, uninstall: PASS"
        )


if __name__ == "__main__":
    main()
