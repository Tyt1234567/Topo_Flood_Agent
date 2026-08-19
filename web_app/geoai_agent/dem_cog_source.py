"""
Real DEM elevation as an actual raster, not point-sampled.

Streams the same USGS 3DEP 1/3 arc-second (~10m) GeoTIFF COG used in the
reference PyQGIS pipeline at D:\\PythonProject\\new (qgis_slope_college_park.py
/ qgis_agent_slope_chat.py) directly via GDAL's /vsicurl/ virtual
filesystem -- no full download, no per-point HTTP request -- clips it to
the exact requested bounding box, and resamples straight onto the analysis
grid, all in one GDAL call. Same data source, same clip operation
(gdal:cliprasterbyextent) as that reference pipeline; see
_gdal_dem_worker.py for exactly what runs.

Bridged the same way ttk_engine.py bridges to TTK: GDAL's Python bindings
aren't installed in this project's main environment, but QGIS ships a
complete GDAL install with its own bundled Python (python-qgis.bat), so
the actual raster work happens in a subprocess there and comes back over a
temp .npz file. Unlike TTK this needs no persistence/critical-point
computation -- it is a thin, fast streaming-clip-resample operation, and
notably has no per-request point-count cap the way USGS's Elevation Point
Query Service does (fetch_usgs_elevation_field in real_data.py): GDAL
reads only the byte ranges it needs from the COG regardless of how finely
the output is resampled.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from geoai_agent.morse_topology import ScalarField

QGIS_PYTHON = os.environ.get("QGIS_PYTHON", r"D:\QGIS\bin\python-qgis.bat")
_WORKER_SCRIPT = str(Path(__file__).resolve().parent / "_gdal_dem_worker.py")
_SUBPROCESS_TIMEOUT_SECONDS = 120


class DEMSourceUnavailableError(RuntimeError):
    pass


def _check_available() -> None:
    if not Path(QGIS_PYTHON).exists():
        raise DEMSourceUnavailableError(
            f"QGIS_PYTHON ('{QGIS_PYTHON}') does not exist -- set the QGIS_PYTHON "
            "environment variable to your QGIS install's python-qgis.bat, or install "
            "QGIS, to use the streamed-DEM-COG elevation source."
        )


def _fill_nan_nearest(values: np.ndarray) -> np.ndarray:
    """Fills any NaN cells (a clip window edge landing just past the source
    tile's real coverage) from their nearest valid neighbor -- same
    fallback approach as fetch_usgs_elevation_field in real_data.py, kept
    local here to avoid a circular import."""
    missing = np.isnan(values)
    if not missing.any():
        return values
    if missing.all():
        raise RuntimeError("USGS DEM COG clip returned no valid elevation data for this area.")
    ny, nx = values.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    valid_y, valid_x = yy[~missing], xx[~missing]
    valid_v = values[~missing]
    out = values.copy()
    for my, mx in zip(*np.where(missing)):
        dist2 = (valid_y - my) ** 2 + (valid_x - mx) ** 2
        out[my, mx] = valid_v[np.argmin(dist2)]
    return out


def fetch_usgs_dem_cog_field(
    bounds: tuple[float, float, float, float], resolution: int | tuple[int, int],
) -> ScalarField:
    """Live-streams real USGS 3DEP elevation for `bounds` as an actual
    clipped-and-resampled raster (see module docstring), returned as a
    ScalarField on the same (lat_min, lat_max, lon_min, lon_max) grid
    convention as every other real_data.py fetcher."""
    _check_available()
    from geoai_agent.real_data import _as_axis_counts, _grid  # local import: avoid a circular import at module load

    lat_min, lat_max, lon_min, lon_max = bounds
    nx, ny = _as_axis_counts(resolution)

    with tempfile.TemporaryDirectory(prefix="usgs_dem_cog_") as tmp_dir:
        out_path = Path(tmp_dir) / "dem.npz"
        proc = subprocess.run(
            [
                QGIS_PYTHON, _WORKER_SCRIPT, str(out_path),
                str(lat_min), str(lat_max), str(lon_min), str(lon_max), str(nx), str(ny),
            ],
            capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"USGS DEM COG worker process failed (exit {proc.returncode}): "
                f"{proc.stderr or proc.stdout}"
            )
        with np.load(out_path) as data:
            values = data["values"]

    # GDAL reads top row = north; the rest of this codebase's grids
    # (ys = linspace(lat_min, lat_max, ny)) run south-to-north.
    values = np.flipud(values)
    values = _fill_nan_nearest(values)

    xs, ys = _grid(bounds, resolution)
    return ScalarField(name="elevation", xs=xs, ys=ys, values=values)
