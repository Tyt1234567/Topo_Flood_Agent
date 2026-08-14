"""
Morse topology analysis backed by TTK (the Topology ToolKit,
https://topology-tool-kit.github.io/), a peer-reviewed, VTK-based
implementation of discrete Morse-Smale complexes -- used in place of the
from-scratch engine in morse_topology.py.

TTK's Python bindings (topologytoolkit, cp310, linked against VTK 9.4.1)
only import inside the "tda_env" conda environment on this machine, not
the main interpreter running the rest of geoai_agent (different Python
ABI: 3.10 vs 3.14). analyze_scalar_field_ttk therefore shells out to that
environment's python running _ttk_worker.py as a subprocess, handing the
field over as a temp .npz and reading the result back as JSON -- the two
environments never share a process, so this has no effect on the rest of
geoai_agent's dependencies.

The returned dict has the same shape as morse_topology.analyze_scalar_field
(critical_points/basins/peak_basins/euler_characteristic/etc, so existing
callers -- visualize.critical_points_to_geojson, web_app's rendering, the
prompts) work unchanged, plus one addition: "separatrices", the 1-cells of
the Morse-Smale complex (saddle<->extremum integral lines) that bound each
basin, i.e. the terrain skeleton -- used by flood_tools for the "this
depression is bounded by two separatrices" style of reasoning.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from geoai_agent.morse_topology import ScalarField

TDA_ENV_PYTHON = os.environ.get("TTK_PYTHON", r"D:\anaconda3\envs\tda_env\python.exe")
TTK_BIN_DIR = os.environ.get("TTK_BIN_DIR", r"D:\TTK\bin")
_WORKER_SCRIPT = str(Path(__file__).resolve().parent / "_ttk_worker.py")
_SUBPROCESS_TIMEOUT_SECONDS = 300


class TTKUnavailableError(RuntimeError):
    """Raised when the tda_env/TTK installation this machine relies on for
    Morse-Smale analysis can't be found. Set TTK_PYTHON / TTK_BIN_DIR env
    vars to point at a different install if the paths differ."""


def _check_available() -> None:
    if not Path(TDA_ENV_PYTHON).is_file():
        raise TTKUnavailableError(
            f"TTK python interpreter not found at '{TDA_ENV_PYTHON}' (set the TTK_PYTHON "
            "env var if it's installed somewhere else -- this needs the conda env with "
            "topologytoolkit + a matching vtk build installed)."
        )
    if not Path(TTK_BIN_DIR).is_dir():
        raise TTKUnavailableError(
            f"TTK native library directory not found at '{TTK_BIN_DIR}' (set the TTK_BIN_DIR "
            "env var if it's installed somewhere else -- this needs to be on PATH for the "
            "ttk*.dll dependencies of the compiled topologytoolkit .pyd modules to load)."
        )


def analyze_scalar_field_ttk(field: ScalarField, persistence_threshold: float | None = None) -> dict:
    """TTK-backed drop-in replacement for morse_topology.analyze_scalar_field."""
    _check_available()

    with tempfile.TemporaryDirectory(prefix="ttk_morse_") as tmp_dir:
        in_path = Path(tmp_dir) / "field.npz"
        out_path = Path(tmp_dir) / "result.json"
        np.savez(in_path, xs=field.xs, ys=field.ys, values=field.values, name=np.array(field.name))

        env = dict(os.environ)
        env["PATH"] = TTK_BIN_DIR + os.pathsep + env.get("PATH", "")
        threshold_arg = "" if persistence_threshold is None else str(persistence_threshold)

        proc = subprocess.run(
            [TDA_ENV_PYTHON, _WORKER_SCRIPT, str(in_path), str(out_path), threshold_arg],
            env=env, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"TTK worker process failed (exit {proc.returncode}) analyzing field "
                f"'{field.name}':\n{proc.stderr or proc.stdout}"
            )
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)
