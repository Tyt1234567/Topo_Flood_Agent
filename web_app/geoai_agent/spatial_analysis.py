"""
Spatial-analysis add-ons layered on top of the core flood-risk pipeline:

- Proximity to real infrastructure (OpenStreetMap, live via the Overpass
  API -- no key, same live-fetch philosophy as every other data source in
  this project, just a different OSM API than boundary.py's Nominatim
  lookup: raw tagged features by bounding box, not administrative shapes).
- Terrain metrics (aspect, local relief, flow length) computed directly
  from the elevation field already fetched for the main analysis, at no
  extra data-fetch cost -- aspect uses the same np.gradient approach
  real_data.py's derive_slope_field already does for slope magnitude, just
  keeping the direction instead of discarding it.

Every function here only ADDS fields to hotspot dicts already produced by
flood_tools.rank_flood_basins -- nothing here replaces or recomputes the
core risk ranking.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

import numpy as np

from geoai_agent.morse_topology import ScalarField

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Required -- a POST with no Content-Type/User-Agent gets a flat 406 Not
# Acceptable from this service (observed directly), the same kind of
# requirement Nominatim enforces for boundary.py's geocode_place_boundary.
_USER_AGENT = "GeoAI-Agent-FloodRiskTool/1.0 (educational/research project)"
_METERS_PER_DEGREE_LAT = 111_320.0


def _meters_per_degree_lon(lat_deg: float) -> float:
    return _METERS_PER_DEGREE_LAT * math.cos(math.radians(lat_deg))


# -- OSM infrastructure proximity ------------------------------------------

def fetch_osm_infrastructure(bounds: tuple[float, float, float, float], timeout: float = 25.0) -> dict | None:
    """Live Overpass API query for roads, buildings, and water features
    intersecting `bounds` (lat_min, lat_max, lon_min, lon_max). Returns
    {"roads": [...], "buildings": [...], "water": [...]} where each entry
    is a list of [(lat, lon), ...] way geometries, or None on any failure
    (network error, malformed response, empty area) -- Overpass's public
    instance is rate-limited/best-effort with no SLA, the same caveat as
    FEMA's NFHL (fema_flood.py) and Nominatim (boundary.py), so every
    caller must treat None as "not available", not a fatal error."""
    lat_min, lat_max, lon_min, lon_max = bounds
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    query = f"""
    [out:json][timeout:{max(int(timeout) - 5, 5)}];
    (
      way["highway"]({bbox});
      way["building"]({bbox});
      way["natural"="water"]({bbox});
      way["waterway"]({bbox});
    );
    out geom;
    """
    request = urllib.request.Request(
        _OVERPASS_URL,
        data=f"data={query}".encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
    elements = data.get("elements")
    if not elements:
        return None

    roads, buildings, water = [], [], []
    for el in elements:
        geometry = el.get("geometry")
        if not geometry or len(geometry) < 2:
            continue
        coords = [(pt["lat"], pt["lon"]) for pt in geometry]
        tags = el.get("tags", {})
        if "highway" in tags:
            roads.append(coords)
        elif "building" in tags:
            buildings.append(coords)
        elif tags.get("natural") == "water" or "waterway" in tags:
            water.append(coords)
    if not (roads or buildings or water):
        return None
    return {"roads": roads, "buildings": buildings, "water": water}


def tag_hotspots_with_nearest_infrastructure(hotspots: list[dict], infra: dict) -> None:
    """Mutates each hotspot dict in place, adding "nearest_road_m",
    "nearest_building_m", "nearest_water_m" -- straight-line distance in
    meters from the hotspot to the closest OSM-tagged feature in each
    category, or None if that category had no features in the queried
    area. Uses a shapely STRtree per category so this stays fast even with
    OSM's often-thousands of ways in a single bounding box (O(log n) per
    hotspot instead of checking every feature against every hotspot).
    Distances use a local equirectangular approximation (degrees -> meters
    via cos(latitude)) -- accurate enough at single-analysis-run scale, the
    same simplification report.py's _hillshade_spacing_meters already
    makes elsewhere in this project, not a true geodesic distance."""
    from shapely.geometry import LineString, Polygon
    from shapely.geometry import Point as ShapelyPoint
    from shapely.strtree import STRtree

    def _build_geoms(key: str, close_rings: bool) -> list:
        geoms = []
        for coords in infra.get(key, []):
            if len(coords) < 2:
                continue
            lonlat = [(lon, lat) for lat, lon in coords]
            try:
                if close_rings and len(lonlat) > 2 and lonlat[0] == lonlat[-1]:
                    geoms.append(Polygon(lonlat))
                else:
                    geoms.append(LineString(lonlat))
            except Exception:
                continue
        return geoms

    categories = {
        "nearest_road_m": (_build_geoms("roads", close_rings=False), None),
        "nearest_building_m": (_build_geoms("buildings", close_rings=True), None),
        "nearest_water_m": (_build_geoms("water", close_rings=True), None),
    }
    trees = {label: (STRtree(geoms) if geoms else None, geoms) for label, (geoms, _) in categories.items()}

    for h in hotspots:
        lon, lat = h["coords"]
        m_per_deg_lon = _meters_per_degree_lon(lat)
        point = ShapelyPoint(lon, lat)
        for label, (tree, geoms) in trees.items():
            if tree is None:
                h[label] = None
                continue
            nearest_idx = tree.nearest(point)
            nearest_geom = geoms[int(nearest_idx)]
            # shapely's own distance() is in degree-space, not meters --
            # find the actual nearest point on the geometry and convert
            # THAT pair of coordinates to meters, rather than scaling the
            # degree-distance directly (wrong whenever the nearest point
            # isn't due east/west of the hotspot).
            if nearest_geom.geom_type == "Polygon":
                # 0 if the hotspot falls inside one -- itself a real, useful
                # signal (e.g. "this basin is already inside a mapped water body").
                if nearest_geom.contains(point):
                    h[label] = 0.0
                    continue
                nearest_point = nearest_geom.exterior.interpolate(nearest_geom.exterior.project(point))
            else:
                nearest_point = nearest_geom.interpolate(nearest_geom.project(point))
            dist_m = math.hypot(
                (point.x - nearest_point.x) * m_per_deg_lon, (point.y - nearest_point.y) * _METERS_PER_DEGREE_LAT,
            )
            h[label] = round(dist_m, 1)


# -- terrain metrics: aspect, local relief, flow length ---------------------

_COMPASS_LABELS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _compass_label(bearing_deg: float) -> str:
    return _COMPASS_LABELS[round(bearing_deg / 45) % 8]


def compute_terrain_metrics(hotspots: list[dict], elevation: ScalarField, catchment_grid: list[list[int]] | None) -> None:
    """Mutates each hotspot dict in place, adding:
    - "aspect_deg" / "aspect_compass": compass bearing of steepest DESCENT
      at the hotspot's own grid cell (which way water actually flows away
      from/into this point), or None on a perfectly flat cell. Uses the
      same np.gradient approach real_data.py's derive_slope_field already
      uses for slope magnitude, converted to meters via cos(latitude) so
      the bearing isn't skewed by longitude/latitude degree-size mismatch.
    - "local_relief_m": max-min elevation across the hotspot's own
      catchment (every cell whose descending manifold drains into it, see
      ttk_engine's descending_manifold) -- the FULL elevation range of the
      whole contributing area, a different signal than "persistence_m"
      (which is about the specific saddle/path to the next basin, not the
      catchment's overall relief).
    - "flow_length_m": straight-line distance from the hotspot to the
      farthest cell in its own catchment -- an approximation of the
      longest flow path (a true flow-routed length would need actual flow
      accumulation, not just catchment membership; this is the same kind
      of honest simplification contributing_area_km2 already makes by
      counting catchment cells rather than modeling weighted flow
      accumulation).
    No-ops (sets fields to None) for a hotspot with no catchment_id or an
    empty catchment when `catchment_grid` isn't available."""
    ys, xs, values = elevation.ys, elevation.xs, elevation.values
    lat0 = float(np.mean(ys))
    m_per_deg_lon = _meters_per_degree_lon(lat0)
    m_per_deg_lat = _METERS_PER_DEGREE_LAT

    dz_dy, dz_dx = np.gradient(values, ys, xs)
    dz_dy_m = dz_dy / m_per_deg_lat
    dz_dx_m = dz_dx / m_per_deg_lon

    catchment_arr = np.array(catchment_grid) if catchment_grid is not None else None

    for h in hotspots:
        row, col = h["grid_index"]
        gx, gy = float(dz_dx_m[row, col]), float(dz_dy_m[row, col])
        if gx == 0.0 and gy == 0.0:
            h["aspect_deg"] = None
            h["aspect_compass"] = "flat"
        else:
            bearing = math.degrees(math.atan2(-gx, -gy)) % 360  # downhill direction, clockwise from north
            h["aspect_deg"] = round(bearing, 1)
            h["aspect_compass"] = _compass_label(bearing)

        h["local_relief_m"] = None
        h["flow_length_m"] = None
        if catchment_arr is not None and h.get("catchment_id") is not None:
            mask = catchment_arr == h["catchment_id"]
            if mask.any():
                cell_elevs = values[mask]
                h["local_relief_m"] = round(float(cell_elevs.max() - cell_elevs.min()), 2)
                rows, cols = np.where(mask)
                dy_m = (ys[rows] - ys[row]) * m_per_deg_lat
                dx_m = (xs[cols] - xs[col]) * m_per_deg_lon
                dists_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
                h["flow_length_m"] = round(float(dists_m.max()), 1)


# -- building-footprint exposure aggregation ---------------------------------

def compute_building_exposure(
    hotspots: list[dict], buildings: list, catchment_grid: list[list[int]] | None, elevation_field: ScalarField,
    boundary_mask: np.ndarray | None = None,
) -> dict[int, list]:
    """Mutates each hotspot dict in place, adding "buildings_in_catchment"
    (int count) and "building_density_per_km2" (count / contributing_
    area_km2), aggregating the OSM building footprints fetch_osm_
    infrastructure already pulls for this run per basin -- previously those
    buildings were only ever used for "nearest_building_m" (a single
    closest-distance metric), never counted or aggregated. This is a
    USAGE fix, not a new data source: `buildings` is the fetch already
    made this run for that distance computation, reused here at zero extra
    HTTP cost, same "corroborating signal from data already in memory" role
    river-gauge tagging and FEMA-zone tagging already play elsewhere.

    boundary_mask: a boolean (ny, nx) grid, True where a cell is actually
    inside the resolved place boundary (see boundary.grid_inside_polygon_
    mask and flood_tools.rank_flood_basins's own boundary_mask parameter).
    A basin's raw catchment_grid membership can legitimately extend past
    the real boundary into the artificially-raised "wall" cells outside it
    (see flood_tools.py's boundary_mask docstring for why) -- WITHOUT this
    mask, a building whose centroid falls in that wall-adjacent overflow
    region still gets counted against the basin, while `contributing_
    area_km2` (this function's density denominator, computed by rank_flood_
    basins) is ALREADY clipped to the real boundary, so the two would use
    inconsistent catchment definitions and can produce a wildly inflated
    density (observed directly: a 0.05 km2 catchment reporting 799
    buildings, 16,500/km2 -- physically impossible at real building sizes).
    Passing the SAME boundary_mask used for the ranking call keeps the
    numerator and denominator on the same footing. None (no boundary was
    resolved this run) skips this filtering entirely, same as before this
    parameter existed.

    Approach: each building's CENTROID (mean of its way's vertex lat/lons --
    good enough for a footprint this small, no need for a true polygon
    centroid) is snapped to its nearest grid cell and looked up in
    `catchment_grid` (same descending-manifold catchment-membership array
    compute_terrain_metrics above already uses) to find which basin's
    catchment, if any, it falls in. A CENTROID-in-cell test rather than a
    full polygon-intersects one: `catchment_grid` is only known as a RASTER
    cell-membership mask, not a vector polygon, so a true polygon overlap
    test would first need to vectorize the catchment boundary (extra work,
    its own extra approximation) for no real accuracy gain here -- this
    project's default analysis grid is already ~30m per cell (see web_app/
    app.py's _AGENT_TARGET_RESOLUTION_METERS), finer than most individual
    building footprints' own extent, so a centroid falling in the "wrong"
    neighboring cell at a catchment boundary is a rare edge case, not a
    systematic bias.

    Sets both fields to None on every hotspot when `buildings` is empty/
    falsy (OSM fetch unavailable this run) or `catchment_grid` is None --
    matches this file's existing "nearest_building_m is None when
    unavailable" convention rather than reporting a misleading zero.

    Returns a dict mapping catchment_id -> the list of building ways (same
    [(lat, lon), ...] shape `buildings` uses) actually counted toward that
    basin -- i.e. the exact footprints "buildings_in_catchment"/"building_
    density_per_km2" summarize. A count alone is hard to verify by eye; a
    caller (web_app/app.py builds this into a "buildings_geojson" map
    layer) can use this to draw the REAL building outlines a basin's
    exposure numbers came from -- e.g. on click, alongside the existing
    catchment-cell highlight -- rather than asking a reader to trust an
    aggregate number with nothing to check it against. Empty dict when
    `buildings`/`catchment_grid` are unavailable, same as the None fields
    above."""
    if not buildings or catchment_grid is None:
        for h in hotspots:
            h["buildings_in_catchment"] = None
            h["building_density_per_km2"] = None
        return {}

    xs, ys = elevation_field.xs, elevation_field.ys
    catchment_arr = np.array(catchment_grid)
    ny, nx = catchment_arr.shape

    counts_by_catchment: dict = {}
    ways_by_catchment: dict[int, list] = {}
    for way in buildings:
        if not way:
            continue
        lats = [pt[0] for pt in way]
        lons = [pt[1] for pt in way]
        centroid_lat = sum(lats) / len(lats)
        centroid_lon = sum(lons) / len(lons)
        col = int(np.argmin(np.abs(xs - centroid_lon)))
        row = int(np.argmin(np.abs(ys - centroid_lat)))
        if not (0 <= row < ny and 0 <= col < nx):
            continue
        if boundary_mask is not None and not boundary_mask[row, col]:
            continue  # outside the real boundary -- e.g. wall-region catchment overflow, see docstring
        catchment_id = int(catchment_arr[row, col])
        counts_by_catchment[catchment_id] = counts_by_catchment.get(catchment_id, 0) + 1
        ways_by_catchment.setdefault(catchment_id, []).append(way)

    for h in hotspots:
        catchment_id = h.get("catchment_id")
        count = counts_by_catchment.get(catchment_id, 0) if catchment_id is not None else 0
        h["buildings_in_catchment"] = count
        area_km2 = h.get("contributing_area_km2")
        h["building_density_per_km2"] = round(count / area_km2, 2) if area_km2 else None

    return ways_by_catchment


# Web-map-only geometry simplification: raw Overpass coordinates carry ~7
# decimal places (sub-centimeter precision -- pointless for a Leaflet
# overlay rendered at, at most, building-scale zoom) and roads/water ways
# especially can carry many near-collinear intermediate points a road's
# real curvature doesn't need at web-map scale. Checked directly on a real
# College Park, MD run: the raw embedded roads+buildings+water layers made
# the final downloadable report/live-map payload ~46MB; this cuts vertex
# count and coordinate string length with no visible difference at any
# zoom level a user would actually view the map at (~3m tolerance, well
# under a single web-map pixel except at extreme close-in zoom). This
# mirrors fema_flood.py's own geometryPrecision/maxAllowableOffset
# simplification of ArcGIS-served flood-zone polygons for the identical
# reason -- OSM's Overpass API has no server-side equivalent, so it's done
# here instead, once, at conversion time rather than repeatedly by every
# caller.
_WEBMAP_SIMPLIFY_TOLERANCE_DEG = 0.00003  # ~3m at this latitude range
_WEBMAP_COORD_NDIGITS = 5  # ~1.1m precision


def _simplify_line_coords(coords: list) -> list:
    """[[lon, lat], ...] -> simplified, rounded [[lon, lat], ...]. Falls
    back to the rounded-but-unsimplified input if shapely can't build a
    valid LineString from it (e.g. fewer than 2 distinct points)."""
    from shapely.geometry import LineString

    try:
        simplified = LineString(coords).simplify(_WEBMAP_SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
        pts = list(simplified.coords)
    except Exception:
        pts = coords
    return [[round(x, _WEBMAP_COORD_NDIGITS), round(y, _WEBMAP_COORD_NDIGITS)] for x, y in pts]


def _simplify_ring_coords(ring: list) -> list:
    """Same as _simplify_line_coords but for a closed polygon ring --
    simplifies via shapely's Polygon (not LineString) so the result stays a
    valid, closed ring, and re-closes it if simplification's own output
    isn't (shapely doesn't guarantee first==last on a simplified exterior
    unless the input already was, which every caller here already ensures)."""
    from shapely.geometry import Polygon

    try:
        simplified = Polygon(ring).simplify(_WEBMAP_SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
        pts = list(simplified.exterior.coords)
    except Exception:
        pts = ring
    rounded = [[round(x, _WEBMAP_COORD_NDIGITS), round(y, _WEBMAP_COORD_NDIGITS)] for x, y in pts]
    if rounded[0] != rounded[-1]:
        rounded.append(rounded[0])
    return rounded


def buildings_geojson(ways_by_catchment: dict[int, list]) -> dict:
    """`compute_building_exposure`'s returned catchment_id -> building-ways
    mapping, as a GeoJSON FeatureCollection of Polygons (one per building,
    tagged with its "catchment_id" property) -- for drawing the actual
    footprints on a map, same "each data source owns its own _to_geojson
    conversion" convention as drainage_db.pipes_to_geojson and streamflow.
    gauges_geojson. Meant to be filtered client-side by catchment_id (the
    same way the map's existing catchment-highlight interaction already
    filters field_geojson's cells -- see web_app/static/main.js's
    highlightCatchment), so clicking a basin shows the REAL buildings
    counted toward its "buildings_in_catchment"/"building_density_per_km2"
    figures, not just the aggregate number on its own."""
    # Rounded, not simplified: buildings are already small, few-vertex
    # rectangles (unlike roads/water below), so there's little redundant
    # point density to remove -- simplifying at the same tolerance risks
    # visibly distorting a small footprint's actual shape for little size
    # benefit. Coordinate rounding alone (still ~1.1m precision) is the
    # safe, still-worthwhile win here.
    features = []
    for catchment_id, ways in ways_by_catchment.items():
        for way in ways:
            if len(way) < 3:
                continue  # not enough vertices for a real polygon
            ring = [[round(lon, _WEBMAP_COORD_NDIGITS), round(lat, _WEBMAP_COORD_NDIGITS)] for lat, lon in way]
            if ring[0] != ring[-1]:
                ring.append(ring[0])  # GeoJSON polygons must be closed rings
            features.append({
                "type": "Feature",
                "properties": {"catchment_id": catchment_id},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })
    return {"type": "FeatureCollection", "features": features}


def roads_geojson(ways: list) -> dict:
    """fetch_osm_infrastructure's "roads" list ([(lat, lon), ...] ways) as a
    GeoJSON FeatureCollection of LineStrings, for drawing every road in the
    fetched area on the map -- same "own its own _to_geojson conversion"
    convention as buildings_geojson above, just unconditional (a run's
    roads are far fewer than its buildings, and unlike buildings there's no
    existing per-catchment aggregate to key this by). Geometry is
    simplified (see _simplify_line_coords) -- roads are this project's
    single largest map layer by feature count and point density, and the
    dominant contributor to a real run's response payload size."""
    features = []
    for way in ways:
        if len(way) < 2:
            continue
        coords = _simplify_line_coords([[lon, lat] for lat, lon in way])
        if len(coords) < 2:
            continue  # simplification collapsed this way to a single point
        features.append({
            "type": "Feature", "properties": {},
            "geometry": {"type": "LineString", "coordinates": coords},
        })
    return {"type": "FeatureCollection", "features": features}


def water_geojson(ways: list) -> dict:
    """fetch_osm_infrastructure's "water" list ([(lat, lon), ...] ways) as a
    GeoJSON FeatureCollection -- a Polygon for closed ways (lakes, ponds,
    reservoirs) and a LineString for open ones (rivers/streams tagged
    waterway=* often aren't closed rings), same closure test buildings_
    geojson doesn't need (OSM building ways are always closed) but a
    natural=water/waterway=* mix genuinely isn't. Geometry is simplified
    (see _simplify_line_coords/_simplify_ring_coords), same reasoning as
    roads_geojson above -- a river/shoreline way can carry as many
    intermediate points as a road."""
    features = []
    for way in ways:
        if len(way) < 2:
            continue
        raw_coords = [[lon, lat] for lat, lon in way]
        closed = len(raw_coords) >= 3 and raw_coords[0] == raw_coords[-1]
        if closed:
            coords = _simplify_ring_coords(raw_coords)
            if len(coords) < 4:
                continue  # simplification collapsed this ring below a valid polygon
            geometry = {"type": "Polygon", "coordinates": [coords]}
        else:
            coords = _simplify_line_coords(raw_coords)
            if len(coords) < 2:
                continue
            geometry = {"type": "LineString", "coordinates": coords}
        features.append({"type": "Feature", "properties": {}, "geometry": geometry})
    return {"type": "FeatureCollection", "features": features}


# -- topographic wetness index (TWI) -----------------------------------------

def compute_topographic_wetness_index(
    hotspots: list[dict], elevation: ScalarField, slope_field: ScalarField | None,
) -> None:
    """Mutates each hotspot dict in place, adding "twi": the Topographic
    Wetness Index, TWI = ln(specific catchment area / tan(slope))
    (Kirkby 1975 -- a standard terrain-analysis measure of a location's
    tendency to accumulate/retain water from its contributing area versus
    how readily it drains away given local slope). Repeatedly identified in
    the urban flood susceptibility literature as one of the single most
    influential terrain factors -- in at least one sensitivity study, more
    influential than rainfall or land use.

    Computed here at essentially zero extra fetch cost: specific catchment
    area is approximated from `contributing_area_km2` (already computed by
    flood_tools.rank_flood_basins from the basin's own descending-manifold
    cell count) divided by this field's own approximate cell width in
    meters. Slope is NOT read from `slope_field` directly -- that field
    (real_data.derive_slope_field) is a gradient magnitude in the
    elevation field's own coordinate units (meters of elevation change per
    DEGREE of lat/lon), fine for the purely relative min-max normalization
    flood_tools.rank_flood_basins already does with it, but not a true
    dimensionless slope (rise/run) that tan() can be meaningfully applied
    to. So the meters-per-meter gradient is recomputed here directly from
    `elevation`, the same way compute_terrain_metrics (above) already does
    for aspect -- duplicating that one small np.gradient call rather than
    changing derive_slope_field's contract for every other caller.

    DELIBERATELY NOT folded into flood_tools.rank_flood_basins' weighted
    risk_score: adding a sixth scored dimension would ripple into
    WEIGHT_PRESETS, the frontend weight-preset controls, and report.py's
    score-breakdown table, none of which this pass touches. TWI is
    surfaced here as additional, literature-grounded context alongside
    aspect/relief/flow-length (compute_terrain_metrics above), the same
    "corroborating signal, not a new score input" role FEMA NFHL zones and
    river-gauge readings play elsewhere in this pipeline -- not an
    oversight, a scoping choice for a possible future pass.

    Sets "twi" to None when `slope_field` wasn't supplied for this run
    (used only as the existing "was a slope layer part of this run at all"
    signal, matching flood_tools' own slope_field-presence check), or the
    hotspot has no positive contributing area to compute a specific
    catchment area from (e.g. a degenerate single-cell basin)."""
    if slope_field is None:
        for h in hotspots:
            h["twi"] = None
        return

    ys, xs, values = elevation.ys, elevation.xs, elevation.values
    lat0 = float(np.mean(ys))
    m_per_deg_lon = _meters_per_degree_lon(lat0)
    m_per_deg_lat = _METERS_PER_DEGREE_LAT

    dz_dy, dz_dx = np.gradient(values, ys, xs)
    dz_dy_m = dz_dy / m_per_deg_lat
    dz_dx_m = dz_dx / m_per_deg_lon
    # sqrt((dz/dx)^2 + (dz/dy)^2) is rise/run along the steepest-descent
    # direction, which is EXACTLY tan(slope_angle) by definition -- not the
    # angle itself, so this is already the "tan(slope)" term the TWI
    # formula below needs; applying tan() to it again would be wrong.
    tan_slope = np.sqrt(dz_dx_m ** 2 + dz_dy_m ** 2)

    lon_spacing_deg = abs(float(xs[1] - xs[0])) if len(xs) > 1 else 0.0
    lat_spacing_deg = abs(float(ys[1] - ys[0])) if len(ys) > 1 else 0.0
    dx_m = lon_spacing_deg * m_per_deg_lon
    dy_m = lat_spacing_deg * m_per_deg_lat
    # A single "effective" cell width regardless of grid aspect ratio --
    # same square-pixel-area assumption flood_tools._cell_area_km2 already
    # makes for contributing_area_km2 itself, so this stays consistent with
    # the area figure it's dividing.
    cell_width_m = math.sqrt(dx_m * dy_m) if dx_m > 0 and dy_m > 0 else 0.0

    for h in hotspots:
        area_km2 = h.get("contributing_area_km2")
        if not area_km2 or area_km2 <= 0 or cell_width_m <= 0:
            h["twi"] = None
            continue
        row, col = h["grid_index"]
        # Flat ground (tan_slope ~ 0) would send TWI -> +inf -- clamped to a
        # small epsilon, the same "flat is a real, handleable case, not a
        # crash" treatment compute_terrain_metrics already gives a
        # zero-gradient cell (its "flat" aspect_compass branch above).
        tan_slope_here = max(float(tan_slope[row, col]), 1e-4)
        specific_catchment_area_m = (area_km2 * 1_000_000.0) / cell_width_m
        h["twi"] = round(float(np.log(specific_catchment_area_m / tan_slope_here)), 3)
