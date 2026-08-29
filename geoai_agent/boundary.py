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
from shapely.geometry import Point, Polygon, mapping, shape as shapely_shape
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


def boundary_from_drawn_polygon(points: list, name: str = "Custom drawn area") -> dict:
    """Builds the same {"name", "display_name", "bounds", "polygon", "geojson"}
    shape geocode_place_boundary returns, but from a polygon the user drew by
    hand on the map instead of looking one up -- lets agent_analyze_stream
    treat "manually drawn boundary" and "real place boundary" identically for
    everything downstream (DEM clipping, hotspot within/outside tagging, the
    boundary outline drawn on the map).

    `points` is [[lat, lon], ...], the order Leaflet click events give them
    in (not GeoJSON's [lon, lat]) -- needs at least 3 vertices. Raises
    ValueError on anything that can't become a valid polygon (too few
    points, non-numeric coordinates, a ring so self-intersecting that even
    a zero-width buffer can't repair it), so the caller can turn that into a
    plain-language error for the user rather than a raw shapely exception.
    """
    if not points or len(points) < 3:
        raise ValueError("a boundary needs at least 3 points")
    try:
        coords_lonlat = [(float(lon), float(lat)) for lat, lon in points]
    except (TypeError, ValueError):
        raise ValueError("each point must be a [lat, lon] pair of numbers")
    if coords_lonlat[0] != coords_lonlat[-1]:
        coords_lonlat.append(coords_lonlat[0])  # shapely/GeoJSON polygon rings must close

    polygon = Polygon(coords_lonlat)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # repairs the self-intersecting rings a hand-drawn shape can produce
    if not polygon.is_valid or polygon.is_empty:
        raise ValueError("drawn boundary is not a valid polygon (likely self-intersecting)")

    lons = [c[0] for c in coords_lonlat]
    lats = [c[1] for c in coords_lonlat]
    south, north, west, east = min(lats), max(lats), min(lons), max(lons)
    lat_span, lon_span = (north - south) or 1e-4, (east - west) or 1e-4
    bounds = (
        south - lat_span * _BOUNDARY_PADDING_FRACTION,
        north + lat_span * _BOUNDARY_PADDING_FRACTION,
        west - lon_span * _BOUNDARY_PADDING_FRACTION,
        east + lon_span * _BOUNDARY_PADDING_FRACTION,
    )
    return {
        "name": name,
        "display_name": name,
        "bounds": bounds,
        "polygon": polygon,
        "geojson": mapping(polygon),
    }


def boundary_from_shapefile(path: str, name: str = "Uploaded boundary") -> dict:
    """Builds the same {"name", "display_name", "bounds", "polygon", "geojson"}
    shape geocode_place_boundary returns, but from a user-uploaded shapefile
    of POLYGON geometry (a .zip bundling .shp/.shx/.dbf, or a bare .shp
    alongside its sidecar files) instead of looking one up or drawing it by
    hand -- for a user who already has an exact boundary on hand (a parcel,
    a watershed delineation, an administrative unit not in OpenStreetMap...).
    Reads coordinates as-is (no reprojection from a .prj CRS), the same
    simplifying assumption real_data.py's load_shapefile_points already
    makes for point-sample shapefiles: the file's coordinates are expected
    to already be WGS84 lon/lat.

    Every polygon feature in the file is unioned into one boundary, so a
    shapefile of several adjoining parcels/tiles making up one site still
    works. Raises ValueError on anything that can't become a boundary (wrong
    shape type, no features, a ring even a zero-width buffer can't repair),
    so the caller can turn that into a plain-language error for the user.
    """
    import shapefile
    from shapely.ops import unary_union

    reader = shapefile.Reader(path)
    if reader.shapeType not in (shapefile.POLYGON, shapefile.POLYGONZ, shapefile.POLYGONM):
        raise ValueError(
            f"shapefile '{path}' has shape type {reader.shapeType!r}, expected POLYGON "
            "(the analysis boundary's outline)"
        )
    shapes = reader.shapes()
    if not shapes:
        raise ValueError(f"shapefile '{path}' has no features")

    polygon = unary_union([shapely_shape(s.__geo_interface__) for s in shapes])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # repairs the occasional self-intersecting ring
    if not polygon.is_valid or polygon.is_empty:
        raise ValueError("shapefile boundary is not a valid polygon (likely self-intersecting)")

    west, south, east, north = polygon.bounds
    lat_span, lon_span = (north - south) or 1e-4, (east - west) or 1e-4
    bounds = (
        south - lat_span * _BOUNDARY_PADDING_FRACTION,
        north + lat_span * _BOUNDARY_PADDING_FRACTION,
        west - lon_span * _BOUNDARY_PADDING_FRACTION,
        east + lon_span * _BOUNDARY_PADDING_FRACTION,
    )
    return {
        "name": name,
        "display_name": name,
        "bounds": bounds,
        "polygon": polygon,
        "geojson": mapping(polygon),
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


def clip_geojson_to_polygon(geojson: dict | None, polygon: BaseGeometry) -> dict | None:
    """Clips a GeoJSON FeatureCollection (points, lines, or polygons -- any
    mix) down to what actually falls inside `polygon`, the same real place
    boundary tag_hotspots_within_boundary/clip_field_to_polygon already use
    elsewhere. Added so every map layer this project draws respects the
    SAME boundary once one is resolved -- before this, elevation, FEMA
    flood zones, stormwater pipes, and river gauges were all fetched over
    the padded rectangular bounding box and shown in full, while hotspots/
    buildings/the elevation topology were already clipped to the real
    place shape -- an inconsistent, "why does this bit stick out past the
    boundary line but not that bit" look.

    - Point features: kept as-is if inside the polygon, dropped entirely
      otherwise (a gauge or marker either is or isn't in the place; there's
      no partial version of a point to keep).
    - Line/polygon features: replaced with their actual geometric
      intersection with `polygon` (not just an in/out test) -- a FEMA zone
      or pipe that only partly overlaps the boundary keeps the overlapping
      part, same as how a real GIS "clip to boundary" operation works,
      rather than being dropped whole or left un-clipped. A feature with NO
      overlap at all is dropped. `shapely.intersection` can occasionally
      return a GeometryCollection (e.g. a line that grazes the boundary at
      one point and crosses it properly elsewhere) -- only the components
      matching the original geometry's own type family are kept from that,
      so a stray zero-area Point component doesn't turn a polygon feature
      into a degenerate geometry.

    `geojson` may be None or empty (no data fetched this run) -- passed
    through unchanged, matching every other "nothing to do" case in this
    project's data pipeline."""
    if not geojson or not geojson.get("features"):
        return geojson
    from shapely.ops import unary_union
    from shapely.prepared import prep

    prepared = prep(polygon)
    _POLYGONAL = {"Polygon", "MultiPolygon"}
    _LINEAR = {"LineString", "MultiLineString"}

    def _matching_parts(geom, want_types: set) -> list:
        if geom.geom_type in want_types:
            return [] if geom.is_empty else [geom]
        if geom.geom_type == "GeometryCollection":
            return [g for g in geom.geoms if g.geom_type in want_types and not g.is_empty]
        return []

    clipped_features = []
    for feature in geojson["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        geom = shapely_shape(geometry)
        if geom.geom_type == "Point":
            if prepared.contains(geom):
                clipped_features.append(feature)
            continue

        want_types = _POLYGONAL if geom.geom_type in _POLYGONAL else _LINEAR
        try:
            intersection = polygon.intersection(geom)
        except Exception:
            continue
        parts = _matching_parts(intersection, want_types)
        if not parts:
            continue
        # unary_union of one part is just that part unchanged; of several
        # (a GeometryCollection's matching pieces) it combines them back
        # into a single Multi* geometry, same effect as manually building
        # a MultiPolygon/MultiLineString but without needing to pick which
        # constructor matches `want_types`.
        clipped_geom = parts[0] if len(parts) == 1 else unary_union(parts)
        new_feature = dict(feature)
        new_feature["geometry"] = mapping(clipped_geom)
        clipped_features.append(new_feature)

    return {"type": "FeatureCollection", "features": clipped_features}


def clip_field_to_polygon(field, polygon: BaseGeometry, wall_margin_fraction: float = 0.5, mask: np.ndarray | None = None):
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
    worth disclosing, not a bug to "fix" away.

    `mask`, if given, is a precomputed grid_inside_polygon_mask(field.xs,
    field.ys, polygon) -- lets a caller that already needs the same mask
    for something else (e.g. flood_tools.rank_flood_basins's
    boundary_mask, to keep contributing-area totals from counting this
    function's own synthetic wall cells) reuse it instead of paying the
    O(nx*ny) shapely contains-check cost twice."""
    mask_inside = mask if mask is not None else grid_inside_polygon_mask(field.xs, field.ys, polygon)
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
