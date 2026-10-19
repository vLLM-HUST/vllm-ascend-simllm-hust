"""Freeze the Python source actually loaded from editable vLLM installations.

Run with the serving environment's Python. The output contains source files,
packaging metadata, and a hash manifest; it intentionally omits model weights,
compiled kernels, and Python bytecode.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import tarfile
from datetime import datetime, timezone
from importlib import metadata

PACKAGES = {"vllm": "vllm", "vllm_ascend": "vllm-ascend"}
METADATA_FILES = ("pyproject.toml", "setup.py", "setup.cfg", "LICENSE", "VERSION")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    files: list[tuple[str, bytes]] = []
    origins: dict[str, object] = {}
    for package, distribution in PACKAGES.items():
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(f"Cannot locate loaded package {package}")
        package_path = pathlib.Path(
            next(iter(spec.submodule_search_locations))
        ).resolve()
        root = package_path.parent
        origins[package] = {
            "distribution": distribution,
            "version": metadata.version(distribution),
            "package_path": str(package_path),
            "git_pointer": (root / ".git").read_text().strip()
            if (root / ".git").is_file()
            else None,
        }
        candidates = [p for p in package_path.rglob("*.py") if p.is_file()]
        candidates += [p for p in package_path.rglob("*.pyi") if p.is_file()]
        candidates += [
            root / name for name in METADATA_FILES if (root / name).is_file()
        ]
        for path in sorted(set(candidates)):
            if path.is_symlink() or "__pycache__" in path.parts:
                continue
            name = f"{package}/{path.relative_to(root)}"
            data = path.read_bytes()
            stat_after = path.stat()
            if stat_after.st_size != len(data):
                raise RuntimeError(f"File changed while reading: {path}")
            files.append((name, data))
            records.append({"path": name, "bytes": len(data), "sha256": digest(data)})

    expected_runner = "vllm_ascend/vllm_ascend/worker/model_runner_v1.py"
    runner = next((r for r in records if r["path"] == expected_runner), None)
    if runner is None:
        raise RuntimeError("The Ascend model runner was not included")
    archive = args.output / "host-python-source.tar.gz"
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as tar,
    ):
        for name, data in sorted(files):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))

    report = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "scope": (
            "loaded vllm and vllm_ascend Python sources plus root packaging metadata"
        ),
        "excludes": (
            "compiled binaries, kernels, model weights, bytecode, non-Python assets"
        ),
        "origins": origins,
        "runner_sha256": runner["sha256"],
        "file_count": len(records),
        "archive_sha256": digest(archive.read_bytes()),
        "files": sorted(records, key=lambda item: str(item["path"])),
    }
    (args.output / "host-source-manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
