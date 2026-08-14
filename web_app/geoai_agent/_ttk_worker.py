"""
TTK (Topology ToolKit) worker process.

This file is NEVER imported by the main geoai_agent process -- it only runs
as a subprocess under a separate Python interpreter that has TTK's VTK
python bindings installed (see ttk_engine.py, which shells out to it). Do
not add imports here that assume the main environment's packages (this
process gets numpy + vtk + topologytoolkit only).

Usage: <tda_env python> _ttk_worker.py <input.npz> <output.json> [threshold]

input.npz holds a ScalarField: xs (nx,), ys (ny,), values (ny,nx), name
(0-d string array). threshold is a float persistence threshold, or the
literal string "auto" (0.05 * value range, matching the pure-python
engine's default) if omitted/empty.

Pipeline (the standard TTK persistence-simplification workflow):
1. ttkPersistenceDiagram on the RAW field -- true persistence value for
   every critical point, and the raw (pre-simplification) critical point
   counts, so noise-filtering is visible rather than silently hidden.
2. ttkTopologicalSimplificationByPersistence(threshold) -- flattens every
   critical point pair below the threshold out of the field entirely.
3. ttkMorseSmaleComplex on the SIMPLIFIED field -- its port-0 output gives
   the surviving (significant) critical points directly paired with their
   descending/ascending manifold size (ManifoldSize), so basin area comes
   for free without a separate full-grid segmentation pass. Port 1 gives
   the 1-separatrices (the saddle-to-extremum curves that bound each
   basin -- the "terrain skeleton").

Unlike the pure-python engine (morse_topology.py), insignificant critical
points are not merged into a neighboring basin and still reported with
significant=False -- TTK's simplification removes them from the topology
outright, so every basin/peak_basin this reports IS significant by
construction. num_raw_minima/num_raw_maxima (from step 1) still shows how
much was filtered.
"""

import json
import sys
import types

import numpy as np

# ttkDimensionReduction.pyd fails to load in this TTK build (missing native
# dependency unrelated to terrain analysis -- it wraps scikit-learn/t-SNE
# for high-dimensional point clouds). Stubbing it out before importing the
# package lets every other module (everything used below) import normally,
# since Python's import system checks sys.modules before the package's
# __init__.py tries to load the real .pyd.
sys.modules["topologytoolkit.ttkDimensionReduction"] = types.ModuleType(
    "topologytoolkit.ttkDimensionReduction"
)

import vtk  # noqa: E402
from vtkmodules.util import numpy_support  # noqa: E402
import topologytoolkit as ttk  # noqa: E402

# TTK's CriticalType (standalone ttkScalarFieldCriticalPoints / ttkPersistenceDiagram)
_CRITICAL_TYPE = {0: "minimum", 1: "saddle", 3: "maximum"}
# ttkMorseSmaleComplex's own port-0 CellDimension uses a different, 2D-domain
# enumeration: 0=minimum, 1=saddle, 2=maximum (there is no 2-saddle in 2D).
_MSC_DIM = {0: "minimum", 1: "saddle", 2: "maximum"}


def _build_image(xs: np.ndarray, ys: np.ndarray, values: np.ndarray, field_name: str):
    ny, nx = values.shape
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, 1)
    image.SetSpacing(float(xs[1] - xs[0]), float(ys[1] - ys[0]), 1.0)
    image.SetOrigin(float(xs[0]), float(ys[0]), 0.0)
    arr = numpy_support.numpy_to_vtk(values.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
    arr.SetName(field_name)
    image.GetPointData().SetScalars(arr)
    return image


def _vertex_to_grid(vid: int, nx: int) -> tuple[int, int]:
    return vid // nx, vid % nx


def _raw_persistence(image, field_name: str) -> tuple[dict[int, float], set[int], dict[str, int]]:
    """Runs ttkPersistenceDiagram on the raw (unsimplified) field.

    Returns (persistence_by_vertex, infinite_vertex_ids, raw_type_counts).
    infinite_vertex_ids holds the global min/max vertex ids (the
    non-finite birth-death pair) -- these get persistence=None, matching
    the "infinite persistence" convention morse_topology.py uses for the
    field's global extremum.
    """
    pd = ttk.ttkPersistenceDiagram()
    pd.SetInputData(image)
    pd.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field_name)
    pd.Update()
    out = pd.GetOutput()

    vsf = numpy_support.vtk_to_numpy(out.GetPointData().GetArray("ttkVertexScalarField"))
    ctype = numpy_support.vtk_to_numpy(out.GetPointData().GetArray("CriticalType"))
    persistence = numpy_support.vtk_to_numpy(out.GetCellData().GetArray("Persistence"))
    isfinite = numpy_support.vtk_to_numpy(out.GetCellData().GetArray("IsFinite"))

    persistence_by_vertex: dict[int, float] = {}
    infinite_ids: set[int] = set()
    raw_type_counts = {"minimum": 0, "saddle": 0, "maximum": 0}
    seen_vertices: set[int] = set()

    for i in range(out.GetNumberOfCells()):
        cell = out.GetCell(i)
        pids = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
        vids = [int(vsf[p]) for p in pids]
        p_val = float(persistence[i])
        finite = bool(isfinite[i])
        for p, vid in zip(pids, vids):
            if vid not in seen_vertices:
                seen_vertices.add(vid)
                kind = _CRITICAL_TYPE.get(int(ctype[p]))
                if kind:
                    raw_type_counts[kind] += 1
        if not finite:
            infinite_ids.update(vids)
            continue
        for vid in vids:
            if vid not in persistence_by_vertex or p_val > persistence_by_vertex[vid]:
                persistence_by_vertex[vid] = p_val

    return persistence_by_vertex, infinite_ids, raw_type_counts


def _simplify(image, field_name: str, threshold: float):
    simp = ttk.ttkTopologicalSimplificationByPersistence()
    simp.SetInputData(image)
    simp.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field_name)
    simp.SetPersistenceThreshold(threshold)
    simp.SetThresholdIsAbsolute(True)
    simp.Update()
    return simp.GetOutput()


def _morse_smale_complex(simplified_image, field_name: str):
    msc = ttk.ttkMorseSmaleComplex()
    msc.SetInputData(simplified_image)
    msc.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field_name)
    msc.SetComputeCriticalPoints(True)
    msc.SetComputeAscendingSeparatrices1(True)
    msc.SetComputeDescendingSeparatrices1(True)
    # ManifoldSize (port-0 basin/peak-catchment area) is only populated when
    # segmentation is actually computed internally -- even though the full
    # per-vertex segmentation arrays (port 3) are never read here.
    msc.SetComputeAscendingSegmentation(True)
    msc.SetComputeDescendingSegmentation(True)
    msc.SetComputeFinalSegmentation(False)
    msc.Update()
    return msc


def _separatrix_polylines(msc) -> list[dict]:
    """Groups the 1-separatrices (saddle<->extremum integral lines -- the
    boundaries between basins/peak-catchments, i.e. the terrain skeleton)
    by SeparatrixId into ordered polylines."""
    sep = msc.GetOutputDataObject(1)
    n_cells = sep.GetNumberOfCells()
    if n_cells == 0:
        return []
    pts = numpy_support.vtk_to_numpy(sep.GetPoints().GetData())
    sep_id = numpy_support.vtk_to_numpy(sep.GetCellData().GetArray("SeparatrixId"))
    src = numpy_support.vtk_to_numpy(sep.GetCellData().GetArray("SourceId"))
    dst = numpy_support.vtk_to_numpy(sep.GetCellData().GetArray("DestinationId"))

    by_id: dict[int, list] = {}
    endpoints: dict[int, tuple[int, int]] = {}
    for i in range(n_cells):
        cell = sep.GetCell(i)
        pids = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
        sid = int(sep_id[i])
        entry = by_id.setdefault(sid, [])
        for pid in pids:
            xy = (float(pts[pid][0]), float(pts[pid][1]))
            if not entry or entry[-1] != xy:
                entry.append(xy)
        endpoints.setdefault(sid, (int(src[i]), int(dst[i])))

    return [
        {"separatrix_id": sid, "source_vertex": endpoints[sid][0], "destination_vertex": endpoints[sid][1], "points": pts_}
        for sid, pts_ in by_id.items()
    ]


def analyze(xs: np.ndarray, ys: np.ndarray, values: np.ndarray, field_name: str, persistence_threshold: float | None) -> dict:
    ny, nx = values.shape
    value_min, value_max = float(values.min()), float(values.max())
    value_range = value_max - value_min
    threshold = persistence_threshold if persistence_threshold is not None else 0.05 * value_range

    image = _build_image(xs, ys, values, field_name)
    persistence_by_vertex, infinite_ids, raw_type_counts = _raw_persistence(image, field_name)

    simplified = _simplify(image, field_name, threshold)
    msc = _morse_smale_complex(simplified, field_name)
    cps = msc.GetOutputDataObject(0)

    def describe(vid: int, value: float) -> dict:
        i, j = _vertex_to_grid(vid, nx)
        persistence = None if vid in infinite_ids else persistence_by_vertex.get(vid)
        return {
            "grid_index": [i, j],
            "coords": [float(xs[j]), float(ys[i])],
            "value": float(value),
            "persistence": persistence,
            "significant": persistence is None or persistence >= threshold,
            "multiplicity": 0,
        }

    critical_summary = {"minimum": [], "maximum": [], "saddle": []}
    basins, peak_basins = [], []
    if cps.GetNumberOfPoints() > 0:
        pdata = cps.GetPointData()
        vsf = numpy_support.vtk_to_numpy(pdata.GetArray("ttkVertexScalarField"))
        celldim = numpy_support.vtk_to_numpy(pdata.GetArray("CellDimension"))
        elev = numpy_support.vtk_to_numpy(pdata.GetArray(field_name))
        manifold_size = numpy_support.vtk_to_numpy(pdata.GetArray("ManifoldSize"))

        for k in range(cps.GetNumberOfPoints()):
            vid = int(vsf[k])
            kind = _MSC_DIM.get(int(celldim[k]))
            if kind is None:
                continue
            entry = describe(vid, float(elev[k]))
            critical_summary[kind].append(entry)
            if kind == "minimum":
                basins.append({"minimum": entry, "size_cells": int(manifold_size[k]), "significant": True})
            elif kind == "maximum":
                peak_basins.append({"maximum": entry, "size_cells": int(manifold_size[k]), "significant": True})

    for k in critical_summary:
        critical_summary[k].sort(key=lambda e: e.get("persistence") or e["value"], reverse=True)
    basins.sort(key=lambda b: b["size_cells"], reverse=True)
    peak_basins.sort(key=lambda b: b["size_cells"], reverse=True)

    # domain Euler characteristic: mesh-only, identical formula to
    # morse_topology.euler_characteristic_summary (independent of which
    # engine classified the critical points).
    v_count = ny * nx
    f_count = 2 * (ny - 1) * (nx - 1)
    e_count = (ny * (nx - 1)) + (nx * (ny - 1)) + ((ny - 1) * (nx - 1))
    domain_euler_characteristic = v_count - e_count + f_count

    return {
        "field": field_name,
        "grid_shape": [ny, nx],
        "value_range": [value_min, value_max],
        "persistence_threshold": threshold,
        "engine": "ttk",
        "critical_points": critical_summary,
        "euler_characteristic": {
            "domain_euler_characteristic": domain_euler_characteristic,
            "domain_expected": 1,
            "domain_valid": domain_euler_characteristic == 1,
        },
        "num_significant_basins": len(basins),
        "num_raw_minima": raw_type_counts["minimum"],
        "basins": basins,
        "num_significant_peak_basins": len(peak_basins),
        "num_raw_maxima": raw_type_counts["maximum"],
        "peak_basins": peak_basins,
        "num_raw_saddles": raw_type_counts["saddle"],
        "separatrices": _separatrix_polylines(msc),
    }


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    threshold_arg = sys.argv[3] if len(sys.argv) > 3 else ""
    threshold = float(threshold_arg) if threshold_arg not in ("", "auto") else None

    with np.load(in_path, allow_pickle=False) as data:
        xs, ys, values = data["xs"], data["ys"], data["values"]
        field_name = str(data["name"])

    result = analyze(xs, ys, values, field_name, threshold)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surfaced to the caller's stderr, not swallowed
        import traceback

        traceback.print_exc()
        sys.exit(1)
