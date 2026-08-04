"""One-time local build of the TFLM benchmark binary for exact mode."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .measured import TFLM_CACHE, find_benchmark_binary

TFLM_REPO = "https://github.com/tensorflow/tflite-micro.git"


class SetupError(Exception):
    pass


def _module_missing(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is None


def _make_command() -> str:
    # TFLM's Makefile needs GNU make >= 3.82; macOS ships 3.81 from 2006.
    for cmd in ("gmake", "make"):
        path = shutil.which(cmd)
        if not path:
            continue
        version = subprocess.run([cmd, "--version"], capture_output=True, text=True).stdout
        if "3.81" not in version.split("\n", 1)[0]:
            return cmd
    raise SetupError(
        "GNU make >= 3.82 not found. On macOS: `brew install make` (installs `gmake`)."
    )


def build_benchmark(cache: Path = TFLM_CACHE, log: Path | None = None, jobs: int = 8) -> Path:
    """Clone tflite-micro (if needed) and build the benchmark binary.

    Returns the binary path. Raises SetupError with a actionable message on
    any missing prerequisite.
    """
    existing = find_benchmark_binary(cache)
    if existing:
        return existing

    if not shutil.which("git"):
        raise SetupError("git is required to fetch tflite-micro.")
    make = _make_command()
    missing = [m for m in ("numpy", "PIL") if _module_missing(m)]
    if missing:
        packages = " ".join("pillow" if m == "PIL" else m for m in missing)
        raise SetupError(
            f"TFLM's build scripts need {packages}: run `pip install {packages}` and retry."
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        clone = subprocess.run(
            ["git", "clone", "--depth", "1", TFLM_REPO, str(cache)],
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            raise SetupError(f"git clone failed:\n{clone.stderr[-500:]}")

    # TFLM's build scripts call `python3` and need numpy; the interpreter
    # running mcufit always has it, so put it first on PATH.
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")

    log_file = open(log, "w") if log else subprocess.DEVNULL
    try:
        build = subprocess.run(
            [
                make, "-f", "tensorflow/lite/micro/tools/make/Makefile",
                "tflm_benchmark", f"-j{jobs}", "BUILD_TYPE=default",
            ],
            cwd=cache,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=45 * 60,
        )
    finally:
        if log:
            log_file.close()
    if build.returncode != 0:
        hint = f" - see log: {log}" if log else ""
        raise SetupError(f"TFLM build failed (a C++ toolchain is required){hint}")

    binary = find_benchmark_binary(cache)
    if not binary:
        raise SetupError("build finished but the benchmark binary was not found")
    return binary
