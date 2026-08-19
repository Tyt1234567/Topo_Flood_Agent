"""
Real administrative/place boundary lookup and polygon-based clipping.

geocode_place (real_data.py) only ever returns a point (lat/lon), so every
analysis so far has used a fixed-size square around that point -- fine for
"somewhere near this town" but not the same as the place's actual shape: a
long, narrow city gets the same square as a compact one, and the box
always includes area that isn't really "College Park" at all.

geocode_place_boundary looks up the real boundary polygon via Nominatim
(OpenStreetMap's free, keyless geocoder -- the only common free geocoder
that returns polygon geometry, not just a point). This is a courtesy
public instance with a strict usage policy (nominatim.org/release-docs/
latest/api/Search, and https://operations.osmfoundation.org/policies/nominatim/):
a descriptive User-Agent is required, and it may reject requests from
some networks/IPs outright (observed: consistently 403 from this
project's own sandboxed dev/test environment) regardless of headers --
which is exactly why every caller of this function must treat a None
return as normal and fall back to the point+fixed-box approach
(bounding_box_around_point in real_data.py), not as an error.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from shapely.geometry import Point, shape as shapely_shape
from shapely.geometry.base import BaseGeometry

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Required by Nominatim's usage policy -- identifies the application so a
# misbehaving client can be contacted/blocked individually rather than the
# policy just banning anonymous traffic outright.
_USER_AGENT = "GeoAI-Agent-FloodRiskTool/1.0 (educational/research project)"
# Small padding around the polygon's own bounding box so terrain right at
# the boundary edge isn't cut off mid-slope -- the polygon mask (not this
# box) is what actually determines what's "inside" the place for ranking.
_BOUNDARY_PADDING_FRACTION = 0.08


def geocode_place_boundary(place_name: str, timeout: float = 15.0) -> dict | None:
    """Live lookup of `place_name`'s real boundary polygon via Nominatim.
    Returns {"name", "display_name", "bounds", "polygon", "geojson"} where
    "polygon" is a shapely geometry (Polygon or MultiPolygon) and "geojson"
    is the raw GeoJSON geometry dict (for drawing on a Leaflet map without
    a shapely->GeoJSON round trip) -- or None if the lookup failed for any
    reason (no match, no polygon in the response, network/policy block).
    """
    query = urllib.parse.urlencode({
        "q": place_name, "format": "json", "polygon_geojson": 1, "limit": 1,
    })
    request = urllib.request.Request(
        f"{_NOMINATIM_URL}?{query}", headers={"User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            results = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    if not results:
        return None

    result = results[0]
    geojson_geometry = result.get("geojson")
    if not geojson_geometry or geojson_geometry.get("type") not in ("Polygon", "MultiPolygon"):
        return None

    try:
        polygon = shapely_shape(geojson_geometry)
    except Exception:
        return None
    if not polygon.is_valid or polygon.is_empty:
        return None

    south, north, west, east = (float(v) for v in result["boundingbox"])
    lat_span, lon_span = north - south, east - west
    bounds = (
        south - lat_span * _BOUNDARY_PADDING_FRACTION,
        north + lat_span * _BOUNDARY_PADDING_FRACTION,
        west - lon_span * _BOUNDARY_PADDING_FRACTION,
        east + lon_span * _BOUNDARY_PADDING_FRACTION,
    )
    return {
        "name": result.get("name") or place_name,
        "display_name": result.get("display_name", place_name),
        "bounds": bounds,
        "polygon": polygon,
        "geojson": geojson_geometry,
    }


def grid_inside_polygon_mask(xs: np.ndarray, ys: np.ndarray, polygon: BaseGeometry) -> np.ndarray:
    """Boolean (ny, nx) mask of which grid nodes fall inside `polygon`.
    shapely has no bulk vectorized contains check without an optional
    dependency (shapely.vectorized needs a separate compiled extra), so
    this uses a prepared geometry (caches polygon internals across repeated
    contains() calls) -- fast enough for the grid sizes this project uses
    (up to 80x80 = 6400 points)."""
    from shapely.prepared import prep

    prepared = prep(polygon)
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mask[j, i] = prepared.contains(Point(x, y))
    return mask


def tag_hotspots_within_boundary(hotspots: list[dict], polygon: BaseGeometry) -> None:
    """Mutates each hotspot dict in place, adding "within_boundary": bool --
    whether that basin's coordinate falls inside the real place polygon
    rather than just the padded rectangular analysis box around it."""
    for h in hotspots:
        x, y = h["coords"]
        h["within_boundary"] = bool(polygon.contains(Point(x, y)))


def clip_field_to_polygon(field, polygon: BaseGeometry, wall_margin_fraction: float = 0.5):
    """Returns a COPY of `field` (same class, e.g. ScalarField) with every
    grid cell outside `polygon` raised to "wall" ground well above the real
    data's max -- so TTK's Morse-Smale complex, which finds LOCAL MINIMA,
    can never place one outside the polygon: an outside cell can never be
    lower than its real-terrain neighbors once it's the highest thing on
    the grid. This is the standard watershed-delineation trick for
    constraining a topology analysis to an arbitrary region without needing
    an irregular (non-rectangular) grid, which TTK's VTK image-data
    pipeline doesn't support -- the analysis grid stays a complete
    rectangle (still required), only what counts as "low ground" changes.

    The wall isn't flat: outside cells climb with distance from the
    polygon (a shallow, smooth "hill" rather than a single flat plateau at
    a sharp cliff), so the wall region itself can never register as a
    degenerate minimum/plateau feature and the field stays a proper
    Morse function (no exactly-equal neighbor values) right up to the
    boundary.

    The base wall height is kept modest (real range's own span times
    `wall_margin_fraction` above the real max, not an arbitrary huge
    sentinel like 1e6) specifically so morse_result["value_range"] --
    computed FROM this clipped field, and used to normalize topological
    persistence scores in flood_tools.rank_flood_basins -- doesn't get
    wildly inflated by an unrealistic wall height, which would crush every
    real basin's persistence score toward zero. Callers that need the true
    (unclipped) value_range for reporting should keep the original field's
    min/max around separately -- see agent_analyze_stream's use of this
    function.

    A basin whose true lowest point is very close to the boundary edge can
    still get pulled toward the wall (the same edge-effect real watershed
    tools accept when clipping to a boundary that isn't itself a ridge) --
    worth disclosing, not a bug to "fix" away."""
    mask_inside = grid_inside_polygon_mask(field.xs, field.ys, polygon)
    real_min, real_max = float(field.values.min()), float(field.values.max())
    span = (real_max - real_min) or 1.0
    wall_base = real_max + span * wall_margin_fraction
    # per-grid-cell degree spacing, so the climb-with-distance term stays
    # proportional to grid resolution regardless of the box's real size.
    cell_span = max(float(field.xs[1] - field.xs[0]), float(field.ys[1] - field.ys[0])) if len(field.xs) > 1 and len(field.ys) > 1 else 1.0

    clipped_values = field.values.copy()
    for j, y in enumerate(field.ys):
        for i, x in enumerate(field.xs):
            if not mask_inside[j, i]:
                distance = polygon.distance(Point(x, y))
                clipped_values[j, i] = wall_base + span * 0.01 * (distance / cell_span)
    return field.__class__(name=field.name, xs=field.xs, ys=field.ys, values=clipped_values)
