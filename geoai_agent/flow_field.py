"""
D8 flow direction and accumulation over a terrain ScalarField -- standard
raster hydrology (O'Callaghan & Mark, 1984, "The extraction of drainage
networks from digital elevation data," Computer Vision, Graphics, and
Image Processing 28(3)): every cell drains to whichever ONE of its 8
neighbors has the steepest real-world downhill slope, and each cell's
flow accumulation is the total contributing area (its own cell plus every
cell that -- directly, or through a chain of other cells -- drains into
it).

This is a different kind of "which way does water flow" than TTK's own
`separatrices` (geoai_agent/ttk_engine.py): those are the topological
SKELETON of the persistence-simplified Morse-Smale complex -- the
ridge/valley curves that bound each basin, used elsewhere in this project
for basin-boundary reasoning, and only defined where TTK's simplification
leaves a 1-separatrix at all. D8 flow accumulation is plain per-cell
raster hydrology, computed directly from the elevation grid, independent
of TTK/persistence entirely -- and it's the right tool for "how much of
the terrain's runoff funnels through this specific spot": a steep single
cell can have almost no upstream area, while a gentle valley-floor cell
can carry the drainage of a huge catchment, so slope alone can't tell
those two cases apart the way accumulated area does.

flow_field_geojson below turns this into a map layer of arrows, length
scaled by each sampled point's own accumulated drainage area -- short
where little water passes through, long where a lot does.
"""

from __future__ import annotations

import math

import numpy as np

from geoai_agent.morse_topology import ScalarField

_METERS_PER_DEGREE_LAT = 111_320

# (di, dj) grid-step offsets for the 8 D8 neighbors -- di is a step along
# `ys` (latitude: real_data.py builds ys ascending south-to-north, so +1
# is north), dj a step along `xs` (longitude, +1 is east).
_D8_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _cell_spacing_meters(field: ScalarField) -> tuple[float, float]:
    """(dx_m, dy_m): real ground spacing of one grid cell along
    longitude/latitude respectively. Same formula flood_tools._cell_
    area_km2 uses, duplicated here (a few lines) rather than imported,
    since that helper is private to flood_tools and this module otherwise
    has no dependency on it."""
    lon_spacing_deg = abs(float(field.xs[1] - field.xs[0]))
    lat_spacing_deg = abs(float(field.ys[1] - field.ys[0]))
    meters_per_deg_lon = _METERS_PER_DEGREE_LAT * math.cos(math.radians(float(field.ys.mean())))
    return lon_spacing_deg * meters_per_deg_lon, lat_spacing_deg * _METERS_PER_DEGREE_LAT


def compute_d8_flow(field: ScalarField) -> dict:
    """Returns {"downhill_di", "downhill_dj": (ny,nx) int8, "has_downhill":
    (ny,nx) bool, "accumulation_km2": (ny,nx) float64}.

    downhill_di/dj: the grid-step to this cell's single steepest real-
    world-distance-weighted-descent neighbor (one of the 8 D8 offsets --
    diagonal neighbors are genuinely farther than orthogonal ones, and
    more so along one axis than the other over anisotropic lon/lat cells,
    so comparing raw elevation drops without dividing by real distance
    would systematically favor whichever axis has coarser degree spacing).
    Meaningless where has_downhill is False -- every neighbor is higher or
    equal, i.e. this cell IS a local pit (one of this project's own TTK
    basin floors); water stops here rather than continuing to flow.

    accumulation_km2: real ground area (this field's own per-cell area
    times accumulated upstream cell count, INCLUDING the cell itself)
    that drains through this cell -- computed in one pass over cells
    sorted by elevation descending, since every cell that could possibly
    drain INTO a given cell is, by construction, higher elevation, and is
    therefore already fully accumulated by the time that given cell is
    reached in the sweep."""
    elev = field.values
    ny, nx = elev.shape
    dx_m, dy_m = _cell_spacing_meters(field)
    diag_m = math.hypot(dx_m, dy_m)
    dist_by_offset = {
        (-1, -1): diag_m, (-1, 1): diag_m, (1, -1): diag_m, (1, 1): diag_m,
        (-1, 0): dy_m, (1, 0): dy_m, (0, -1): dx_m, (0, 1): dx_m,
    }

    best_slope = np.zeros((ny, nx), dtype=np.float64)
    downhill_di = np.zeros((ny, nx), dtype=np.int8)
    downhill_dj = np.zeros((ny, nx), dtype=np.int8)
    has_downhill = np.zeros((ny, nx), dtype=bool)

    for di, dj in _D8_OFFSETS:
        dist = dist_by_offset[(di, dj)]
        neighbor = np.full((ny, nx), np.nan)
        # neighbor[i, j] = elev[i+di, j+dj] wherever that's in bounds, NaN
        # elsewhere: the DESTINATION slice (where we're writing) needs the
        # bound that keeps i+di in range, i.e. i in [max(0,-di), ny-max(0,di)).
        dst_i0, dst_i1 = max(0, -di), ny - max(0, di)
        dst_j0, dst_j1 = max(0, -dj), nx - max(0, dj)
        src_i0, src_i1 = max(0, di), ny - max(0, -di)
        src_j0, src_j1 = max(0, dj), nx - max(0, -dj)
        neighbor[dst_i0:dst_i1, dst_j0:dst_j1] = elev[src_i0:src_i1, src_j0:src_j1]
        slope = (elev - neighbor) / dist
        better = np.where(np.isnan(slope), False, slope > best_slope)
        best_slope = np.where(better, slope, best_slope)
        downhill_di = np.where(better, di, downhill_di).astype(np.int8)
        downhill_dj = np.where(better, dj, downhill_dj).astype(np.int8)
        has_downhill = has_downhill | better

    cell_area_km2 = (dx_m * dy_m) / 1_000_000.0
    accumulation_cells = np.ones((ny, nx), dtype=np.float64)
    order = np.argsort(elev, axis=None)[::-1]  # flat vertex indices, elevation descending
    di_flat, dj_flat, has_flat = downhill_di.ravel(), downhill_dj.ravel(), has_downhill.ravel()
    accum_flat = accumulation_cells.ravel()
    for idx in order:
        idx = int(idx)
        if not has_flat[idx]:
            continue
        i, j = divmod(idx, nx)
        target = (i + int(di_flat[idx])) * nx + (j + int(dj_flat[idx]))
        accum_flat[target] += accum_flat[idx]

    return {
        "downhill_di": downhill_di, "downhill_dj": downhill_dj, "has_downhill": has_downhill,
        "accumulation_km2": accumulation_cells.reshape(ny, nx) * cell_area_km2,
    }


def flow_field_geojson(
    field: ScalarField, boundary_mask=None, target_arrows_per_axis: int = 300, min_t: float = 0.1,
) -> dict:
    """A FeatureCollection of arrow-shaped MultiLineString features (a
    shaft plus two short head barbs, all one polyline geometry so a
    single Leaflet style paints the whole arrow) showing D8 downhill flow
    direction.

    Candidate arrow positions are sampled at up to target_arrows_per_axis
    per axis -- 300 by default, comfortably denser than every grid this
    project typically fetches (so in practice this samples EVERY cell,
    i.e. no decimation at all before filtering). Density is deliberately
    NOT the mechanism that keeps this readable -- `min_t` is (see below):
    sampling as densely as the underlying grid allows means the min_t
    filter has real per-cell resolution to work with, so what survives
    traces the actual shape of the drainage network (a proper branching
    channel), rather than being limited to whichever handful of widely-
    spaced lattice points happened to land near it. Lower
    target_arrows_per_axis only matters as a performance cap on an
    unusually large custom grid.

    Arrow LENGTH encodes each sampled cell's own accumulation_km2
    (log1p-compressed, since accumulated drainage area is heavily
    right-skewed -- a few basin-outlet cells can carry orders of
    magnitude more area than typical hillslope cells, and a raw linear
    scale would shrink nearly every arrow to invisible next to those few
    -- then min-max normalized across just the SAMPLED points, not the
    whole grid, so the arrows actually drawn always span the full visible
    length range): short where little area drains through that spot,
    long where a lot does -- literally "flow_accumulation_km2" in each
    feature's own properties, for a popup/legend to show the real number
    behind the length.

    A sampled cell that IS a local pit (has_downhill False -- a TTK basin
    floor, water stops there) is skipped: there's no downhill direction
    to draw, and that spot is already marked by the "All local minima"
    layer.

    `boundary_mask`, if given, skips a sample point outside the real
    place boundary -- the same convention every other boundary-aware
    layer in this project uses (see boundary.grid_inside_polygon_mask).

    `min_t` drops any sampled point whose normalized accumulation falls
    below it (0.1 by default) -- most of a dense sampling lattice sits on
    ordinary hillslope with barely any upstream area, and drawing an
    arrow at literally every sampled cell regardless produces a dense,
    mechanical-looking grid that doesn't read as "water flowing." Only
    keeping the higher-accumulation points thins that out into an actual
    (if still schematic) drainage network -- empty on ridges/flat ground,
    present and well-traced along where flow is actually concentrating,
    since (combined with the now near-native sampling density above)
    there are enough candidate points near any real channel for the
    surviving ones to trace its shape rather than just mark a few dots
    along it. Set to 0 to keep every sampled point."""
    ny, nx = field.shape
    flow = compute_d8_flow(field)
    downhill_di, downhill_dj = flow["downhill_di"], flow["downhill_dj"]
    has_downhill, accumulation_km2 = flow["has_downhill"], flow["accumulation_km2"]

    stride_i = max(1, round(ny / target_arrows_per_axis))
    stride_j = max(1, round(nx / target_arrows_per_axis))
    lon_spacing = abs(float(field.xs[1] - field.xs[0]))
    lat_spacing = abs(float(field.ys[1] - field.ys[0]))

    samples = []
    for i in range(0, ny, stride_i):
        for j in range(0, nx, stride_j):
            if boundary_mask is not None and not boundary_mask[i][j]:
                continue
            if not has_downhill[i, j]:
                continue
            samples.append((i, j))
    if not samples:
        return {"type": "FeatureCollection", "features": []}

    log_mags = [math.log1p(float(accumulation_km2[i, j])) for i, j in samples]
    lo, hi = min(log_mags), max(log_mags)
    span = (hi - lo) or 1.0

    # First pass: which sampled points actually clear the min_t cutoff (most
    # of a regular lattice sits on ordinary hillslope with barely any
    # upstream area -- this is what thins that out into something closer to
    # an actual drainage network instead of a dense, mechanical-looking grid
    # of arrows everywhere). Second pass re-spreads just the survivors
    # across the FULL [0, 1] display range using their own min/max, not the
    # full unfiltered range -- otherwise, whenever there's a real gap
    # between "ordinary ground" and "an actual channel" (the common case
    # this filter is built for), every surviving arrow would end up
    # clustered together near one end instead of spanning the full visual
    # range the length/opacity styling is designed for.
    survivors = [
        (i, j, log_mag) for (i, j), log_mag in zip(samples, log_mags)
        if (log_mag - lo) / span >= min_t
    ]
    if not survivors:
        return {"type": "FeatureCollection", "features": []}
    survivor_log_mags = [s[2] for s in survivors]
    d_lo, d_hi = min(survivor_log_mags), max(survivor_log_mags)
    d_span = (d_hi - d_lo) or 1.0

    step_len_deg = min(stride_i * lat_spacing, stride_j * lon_spacing) or max(lat_spacing, lon_spacing)
    min_len, max_len = 0.32 * step_len_deg, 0.85 * step_len_deg
    barb_angle = math.radians(25)
    barb_fraction = 0.35
    cos_a, sin_a = math.cos(barb_angle), math.sin(barb_angle)

    features = []
    for i, j, log_mag in survivors:
        t_display = (log_mag - d_lo) / d_span
        length = min_len + t_display * (max_len - min_len)
        di, dj = int(downhill_di[i, j]), int(downhill_dj[i, j])
        vx, vy = dj * lon_spacing, di * lat_spacing
        norm = math.hypot(vx, vy) or 1.0
        ux, uy = vx / norm, vy / norm

        x0, y0 = float(field.xs[j]), float(field.ys[i])
        x1, y1 = x0 + ux * length, y0 + uy * length

        back_x, back_y = -ux, -uy
        barb_len = length * barb_fraction
        barb1_x = x1 + barb_len * (back_x * cos_a - back_y * sin_a)
        barb1_y = y1 + barb_len * (back_x * sin_a + back_y * cos_a)
        barb2_x = x1 + barb_len * (back_x * cos_a + back_y * sin_a)
        barb2_y = y1 + barb_len * (-back_x * sin_a + back_y * cos_a)

        features.append({
            "type": "Feature",
            "properties": {"flow_accumulation_km2": round(float(accumulation_km2[i, j]), 4), "t": round(t_display, 4)},
            "geometry": {
                "type": "MultiLineString",
                "coordinates": [
                    [[x0, y0], [x1, y1]],
                    [[x1, y1], [barb1_x, barb1_y]],
                    [[x1, y1], [barb2_x, barb2_y]],
                ],
            },
        })
    return {"type": "FeatureCollection", "features": features}
