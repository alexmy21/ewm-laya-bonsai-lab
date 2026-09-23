# ewm.py — the ewm-scene subprocess client (JSON in, JSON out).
#
# This is the boundary between the controller world (Python) and the
# HLLSet lattice apparatus (Rust). It is a thin wrapper around the
# `ewm-scene` CLI: every call is one subprocess invocation that reads a
# JSONL file and prints one JSON document on stdout.
#
# Protocol (see docs/INTERACTION_MAP.md):
#   in:  one JSON object per line  {"id": 1, "tokens": ["tid12", ...]}
#   out: a single JSON value, pretty-printed
#
# Binary resolution order: EWM_SCENE_BIN env, `ewm-scene` on PATH,
# then the sibling ewm-state-machine repo's release/debug builds.

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIBLING_RELEASE = os.path.join(
    _REPO_ROOT, "ewm-state-machine", "target", "release", "ewm-scene"
)
SIBLING_DEBUG = os.path.join(
    _REPO_ROOT, "ewm-state-machine", "target", "debug", "ewm-scene"
)


def resolve_bin() -> str:
    env = os.environ.get("EWM_SCENE_BIN")
    if env:
        return env
    on_path = shutil.which("ewm-scene")
    if on_path:
        return on_path
    for candidate in (SIBLING_RELEASE, SIBLING_DEBUG):
        if os.path.exists(candidate):
            return candidate
    return "ewm-scene"  # let subprocess raise a clear error


class EwmScene:
    """Thin subprocess client for the ewm-scene CLI."""

    def __init__(self, bin_path: Optional[str] = None, timeout: int = 600):
        self.bin = bin_path or resolve_bin()
        self.timeout = timeout

    def __call__(self, *args, verbose: bool = False) -> dict:
        cmd = [self.bin, *args]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if verbose:
            print(f"$ {' '.join(cmd)}")
            print(r.stdout.strip()[:4000])
        if r.returncode != 0:
            raise RuntimeError(f"ewm-scene failed: {r.stderr[-800:]}")
        return json.loads(r.stdout.strip())

    # ── the flat-frame commands ──────────────────────────────────────────
    def ingest(self, path: str, **kw) -> dict:
        return self("ingest", path, **kw)

    def bss(self, path: str, **kw) -> dict:
        return self("bss", path, **kw)

    def ma(self, path: str, short: int = 1, long: int = 5, **kw) -> dict:
        return self("ma", path, "--short", str(short), "--long", str(long), **kw)

    def noether(self, path: str, **kw) -> dict:
        return self("noether", path, **kw)

    def materialize(self, path: str, beam: int = 2, **kw) -> dict:
        return self("materialize", path, "--beam", str(beam), **kw)

    def sidecar(self, path: str, cap: Optional[int] = None, freeze: Optional[int] = None, **kw) -> dict:
        args = ["sidecar", path]
        if cap is not None:
            args += ["--cap", str(cap)]
        if freeze is not None:
            args += ["--freeze", str(freeze)]
        return self(*args, **kw)

    # ── the structured-frame commands ────────────────────────────────────
    def pyramid(self, path: str, cap: Optional[int] = None, freeze: Optional[int] = None, **kw) -> dict:
        args = ["pyramid", path]
        if cap is not None:
            args += ["--cap", str(cap)]
        if freeze is not None:
            args += ["--freeze", str(freeze)]
        return self(*args, **kw)

    def project(self, path: str, frame_path: str, **kw) -> dict:
        return self("project", path, "--frame", frame_path, **kw)


def write_frames(path: str, token_lists: list[list[str]], start_id: int = 1) -> str:
    """Write flat frames `{"id": i, "tokens": [...]}`."""
    with open(path, "w") as fh:
        for i, tokens in enumerate(token_lists, start_id):
            fh.write(json.dumps({"id": i, "tokens": tokens}) + "\n")
    return path
