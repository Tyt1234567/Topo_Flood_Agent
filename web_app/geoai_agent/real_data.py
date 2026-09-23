"""
Real-world scalar field data: multiple user-supplied or online-fetched
layers over an actual geographic bounding box, as an alternative to the
synthetic town in data.py. Feeds the same generic Morse engine
(geoai_agent/morse_topology.py) the synthetic fields do.

Two ways to get a layer's data onto the analysis grid:
- User-supplied scattered (x, y, value) samples -- a CSV with an x,y,value
  header, or an ESRI shapefile (POINT geometry + a "value" attribute
  field, as a .zip bundling .shp/.shx/.dbf) -- interpolated onto the grid
  with inverse-distance weighting (pure numpy, no scipy; shapefile
  parsing uses pyshp, a small pure-Python library with no further
  dependencies). A layer can also carry a free-text description of what
  it measures, passed through to the LLM recommendation step.
- Online fetch, live at request time (nothing is ever pre-downloaded or
  cached to disk): any of several free, keyless Open-Meteo endpoints --
  terrain elevation, or a live current-conditions weather variable
  (temperature, precipitation, wind speed, humidity, surface pressure).
  This is deliberately not elevation-only: OPEN_METEO_VARIABLES is a small
  registry so the same fetch/analysis path works for whichever topic a
  query is actually about (see prompts.REALWORLD_QUERY_EXTRACTION_SYSTEM
  for how a query is mapped onto one of these).

Elevation specifically has two further live sources, both USGS 3DEP-derived
and both US-only:
- The actual DEM raster, streamed and clipped directly from its source
  GeoTIFF COG via GDAL (see dem_cog_source.fetch_usgs_dem_cog_field) --
  genuinely ~10m resolution, no point-count cap, no per-point rate limit,
  since GDAL only reads the byte ranges a given clip+resample actually
  needs. This is tried FIRST by default within the continental US (see
  build_real_world_dataset's elevation_source="auto").
- USGS's Elevation Point Query Service, queried one point at a time rather
  than batched (often truly ~1m resolution where LiDAR-covered, but far
  slower and capped at a much smaller grid -- see fetch_usgs_elevation_field).
  Used as a fallback if the streamed-COG source is unavailable (see
  dem_cog_source.DEMSourceUnavailableError -- it needs a QGIS install for
  its bundled GDAL bindings) or fails outright.

Open-Meteo elevation is the final fallback if both USGS sources fail or
the location is outside the continental US.

A "slope" layer (gradient magnitude of elevation) is derived for free from
whatever elevation data is loaded, giving a second genuinely real layer
even when the user supplies nothing else -- a concrete demonstration of
the multi-layer path.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import time
import urllib.error
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from geoai_agent.morse_topology import ScalarField

GRID_RESOLUTION = 20
MIN_GRID_RESOLUTION = 6
# The Morse engine itself handles a 100x100 grid in well under a second;
# the real cost is live HTTP fetch batches (100 coordinates/request), which
# scale with the total point count. 80 keeps a worst-case single-axis
# request in the tens-of-seconds range rather than minutes, and well clear
# of Open-Meteo's per-minute rate limit.
MAX_GRID_RESOLUTION = 80
_FETCH_BATCH = 100  # conservative shared batch size, verified against the elevation endpoint's limit
# Open-Meteo's free tier enforces a per-minute request cap that a large
# grid (many batches issued back-to-back) can exhaust well before the
# fetch is done -- observed live, failing as early as batch 7 of 64 for an
# 80x80 grid. That 429 specifically says "try again in one minute", so it
# is worth waiting out and retrying rather than immediately giving up and
# falling back to a much coarser source (USGS's point-count cap is far
# tighter) -- a fine-resolution request should actually be delivered at
# the requested resolution, just slower, not silently degraded.
_OPEN_METEO_MINUTELY_RETRY_WAIT_SECONDS = 65
_OPEN_METEO_MINUTELY_MAX_RETRIES = 2
# Proactive spacing between batches, so a large grid never bursts past the
# per-minute cap in the first place -- the reactive retry above is only a
# safety net for whatever residual burst still slips through. Raised from
# 1.0 after a real 51-batch fetch at the old pacing still triggered several
# 65s retry-waits back to back (520s total for what should have been ~51s
# of pacing alone) -- the actual anonymous-tier per-minute ceiling is
# evidently tighter than 1 request/second sustained, so pacing more slowly
# up front is cheaper in expectation than eating repeated 65s penalties.
_OPEN_METEO_BATCH_PACING_SECONDS = 1.5
_METERS_PER_DEGREE_LAT = 111_320

# USGS EPQS is queried one point at a time (no batch endpoint). Measured
# live at ~0.8s/point even with 10-way concurrency (52s for 64 points),
# with a real error rate under load (~11% in testing) -- so it is only
# used for small grids. 10x10=100 points is roughly 80s in the worst case,
# the practical ceiling for a single synchronous web request without a
# progress indicator.
MAX_USGS_GRID_RESOLUTION = 10
_USGS_CONCURRENCY = 10
_USGS_MAX_RETRIES = 2
# Rough continental US bounding box (lat_min, lat_max, lon_min, lon_max) --
# does not cover Alaska, Hawaii, or the territories, all of which also have
# 3DEP coverage; deliberately conservative rather than wrong.
_CONTINENTAL_US_BOUNDS = (24.5, 49.5, -125.0, -66.9)

# Every entry is a live, keyless Open-Meteo endpoint. "elevation" hits the
# dedicated elevation API; everything else hits the forecast API's
# `current=<variable>` field, which returns live current-conditions data
# (not historical/cached) for each requested coordinate. Which one a given
# free-text query is about, and what place/coordinates it names, is
# figured out by an LLM extraction call (see
# prompts.REALWORLD_QUERY_EXTRACTION_SYSTEM) rather than regex/keyword
# matching here -- natural-language phrasing varies too much (prepositions,
# capitalization, trailing qualifier clauses) for a hand-written pattern to
# keep up with reliably.
OPEN_METEO_VARIABLES = [
    "elevation", "temperature_2m", "precipitation",
    "wind_speed_10m", "relative_humidity_2m", "surface_pressure",
]


# -- geocoding: turn a place name into a bounding box ---------------------
#
# So a query naming a real place ("College Park, MD") works without the
# user having to look up and type coordinates themselves. Live, keyless,
# same Open-Meteo family as the scalar-field fetches above -- nothing is
# pre-downloaded or cached.

_US_STATE_ABBREVIATIONS = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
_TRAILING_STATE_RE = re.compile(r",\s*([A-Za-z]{2})\s*$")


def _expand_trailing_state_abbreviation(place_name: str) -> str:
    """Open-Meteo's geocoding search matches full state names, not
    2-letter abbreviations -- "College Park, MD" returns nothing, but
    "College Park, Maryland" does. Expands a trailing ", XX" if XX is a
    known US state/DC abbreviation; leaves anything else untouched."""
    m = _TRAILING_STATE_RE.search(place_name)
    if not m:
        return place_name
    full = _US_STATE_ABBREVIATIONS.get(m.group(1).upper())
    if not full:
        return place_name
    return place_name[: m.start()] + ", " + full


def geocode_place(place_name: str) -> dict | None:
    """Live lookup of a place name via Open-Meteo's free geocoding API.
    Returns the best-matching {"name", "admin1", "country", "latitude",
    "longitude"} or None if nothing matched."""
    for candidate in (place_name, _expand_trailing_state_abbreviation(place_name)):
        query = urllib.parse.urlencode({"name": candidate, "count": 1, "format": "json"})
        with urllib.request.urlopen(
            f"https://geocoding-api.open-meteo.com/v1/search?{query}", timeout=15
        ) as resp:
            payload = json.loads(resp.read())
        results = payload.get("results") or []
        if results:
            r = results[0]
            return {
                "name": r["name"], "admin1": r.get("admin1"), "country": r.get("country"),
                "latitude": r["latitude"], "longitude": r["longitude"],
            }
    return None


def bounding_box_around_point(
    latitude: float, longitude: float, half_width_deg: float = 0.06
) -> tuple[float, float, float, float]:
    """A small lat/lon bounding box centered on a geocoded point (roughly
    town/neighborhood scale at typical mid-latitudes -- default
    half_width_deg=0.06 is about 13km tall)."""
    return (latitude - half_width_deg, latitude + half_width_deg,
            longitude - half_width_deg, longitude + half_width_deg)


def _as_axis_counts(resolution: int | tuple[int, int]) -> tuple[int, int]:
    """Normalizes a resolution to (nx, ny) -- points along longitude and
    latitude respectively. A plain int (from older/simpler callers, e.g.
    real_world_example.py) means a square nx == ny grid, same as before;
    a (nx, ny) tuple lets each axis have its own point count, so a
    non-square bounding box can get pixels that are genuinely square on
    the ground instead of both axes being forced to share one count."""
    if isinstance(resolution, tuple):
        return resolution
    return resolution, resolution


def ground_spacing_meters(
    bounds: tuple[float, float, float, float], resolution: int | tuple[int, int],
) -> tuple[float, float]:
    """The actual (lon_spacing, lat_spacing) meters-between-sample-points a
    grid resolution achieves over a bounding box -- one value per axis,
    since nx and ny are no longer forced to match each other, so a single
    shared number would hide a mismatch between the two."""
    nx, ny = _as_axis_counts(resolution)
    lat_min, lat_max, lon_min, lon_max = bounds
    mid_lat_rad = math.radians((lat_min + lat_max) / 2)
    meters_per_deg_lon = _METERS_PER_DEGREE_LAT * math.cos(mid_lat_rad)
    lon_spacing = abs(lon_max - lon_min) / (nx - 1) * meters_per_deg_lon
    lat_spacing = abs(lat_max - lat_min) / (ny - 1) * _METERS_PER_DEGREE_LAT
    return lon_spacing, lat_spacing


def resolution_for_target_spacing(
    bounds: tuple[float, float, float, float], target_meters: float,
    min_resolution: int = MIN_GRID_RESOLUTION, max_resolution: int = MAX_GRID_RESOLUTION,
) -> tuple[int, int]:
    """Grid resolution (nx, ny) -- points along longitude and latitude
    respectively -- needed so ground_spacing_meters is approximately
    target_meters on EACH axis independently: pixel count follows the
    box's own shape (nx ~= lon_span_m / target_meters, ny ~= lat_span_m /
    target_meters), so a pixel is genuinely ~target_meters x
    target_meters on the ground instead of a single shared point count
    distorting non-square boxes into non-square pixels.

    Each axis is floored at min_resolution and ceiled at max_resolution
    independently -- a large box + a fine target (e.g. "30 meters" over a
    13km box) can need more points than the cap allows on one or both
    axes; the caller should compare the requested target against
    ground_spacing_meters(bounds, result) to see if it was actually
    achievable, and tell the user if not, rather than silently returning a
    coarser grid than they asked for. Since both axes share the same
    per-axis cap, the total point count (nx * ny) is always bounded by
    max_resolution ** 2, the same worst-case budget the old square-only
    grid had.

    min_resolution/max_resolution default to the module-level MIN_GRID_
    RESOLUTION/MAX_GRID_RESOLUTION (tuned for the slow per-point sources --
    Open-Meteo's per-minute rate limit, USGS EPQS's one-point-per-request
    cost, see their definitions below). A caller whose actual fetch path
    doesn't have that cost (e.g. web_app's one-box agent flow, which always
    uses the streamed/clipped GDAL COG source -- constant ~3.5s regardless
    of resolution) can pass a higher max_resolution explicitly; this
    function has no way to know which source a caller will end up using,
    so it never assumes a higher cap is safe on its own."""
    if target_meters <= 0:
        return GRID_RESOLUTION, GRID_RESOLUTION
    lat_min, lat_max, lon_min, lon_max = bounds
    mid_lat_rad = math.radians((lat_min + lat_max) / 2)
    meters_per_deg_lon = _METERS_PER_DEGREE_LAT * math.cos(mid_lat_rad)
    lat_span_m = abs(lat_max - lat_min) * _METERS_PER_DEGREE_LAT
    lon_span_m = abs(lon_max - lon_min) * meters_per_deg_lon

    def axis_points(span_m: float) -> int:
        return max(min_resolution, min(math.ceil(span_m / target_meters) + 1, max_resolution))

    return axis_points(lon_span_m), axis_points(lat_span_m)


def resolution_for_target_spacing_scaled(
    bounds: tuple[float, float, float, float], target_meters: float,
    max_resolution: int = MAX_GRID_RESOLUTION, min_resolution: int = MIN_GRID_RESOLUTION,
) -> tuple[int, int]:
    """Like resolution_for_target_spacing, but instead of clamping each
    axis to max_resolution independently -- which, for a non-square box,
    can hit the cap on only the longer axis and leave the other axis at
    its true (smaller) point count, silently turning what should be a
    square ground pixel into a rectangular one -- this scales the
    EFFECTIVE target spacing UP uniformly on both axes just enough that
    the longer axis lands exactly at max_resolution. Ground pixels stay
    square (same real-world meters on both axes) even when the requested
    target_meters isn't achievable for a box this large; the caller gets a
    uniformly coarser grid instead of a distorted one.

    Use this (rather than resolution_for_target_spacing) whenever the
    caller wants "as fine as the point-count budget allows, but never
    lopsided" -- e.g. web_app's one-box agent flow, which lets a user ask
    for an arbitrary target resolution via chat and needs the result to
    stay sane for boxes much larger than the target was tuned for."""
    if target_meters <= 0:
        return GRID_RESOLUTION, GRID_RESOLUTION
    lat_min, lat_max, lon_min, lon_max = bounds
    mid_lat_rad = math.radians((lat_min + lat_max) / 2)
    meters_per_deg_lon = _METERS_PER_DEGREE_LAT * math.cos(mid_lat_rad)
    lat_span_m = abs(lat_max - lat_min) * _METERS_PER_DEGREE_LAT
    lon_span_m = abs(lon_max - lon_min) * meters_per_deg_lon

    ideal_nx = lon_span_m / target_meters + 1
    ideal_ny = lat_span_m / target_meters + 1
    longest_axis_points = max(ideal_nx, ideal_ny)
    effective_target_meters = (
        target_meters * (longest_axis_points / max_resolution) if longest_axis_points > max_resolution else target_meters
    )

    def axis_points(span_m: float) -> int:
        return max(min_resolution, min(math.ceil(span_m / effective_target_meters) + 1, max_resolution))

    return axis_points(lon_span_m), axis_points(lat_span_m)


class RealWorldDataset:
    """Geographic bounding-box dataset made of one or more named
    ScalarField layers. Distinct from the synthetic GeoDataset: it carries
    no road network/zones, only scalar fields for the morse_needed branch."""

    def __init__(
        self, name: str, bounds: tuple[float, float, float, float], scalar_fields: dict[str, ScalarField],
        layer_descriptions: dict[str, str] | None = None,
        layer_sources: dict[str, str | None] | None = None,
        elevation_source: str = "open_meteo",
        elevation_fallback_reason: str | None = None,
    ):
        # bounds = (lat_min, lat_max, lon_min, lon_max)
        self.name = name
        self.bounds = bounds
        self.is_geographic = True
        self.scalar_fields = scalar_fields
        # free-text, user-supplied explanation of what a custom (non-Open-Meteo)
        # layer actually measures -- passed to the LLM recommendation so it
        # can interpret a field it has no built-in knowledge of.
        self.layer_descriptions = layer_descriptions or {}
        # the layer_sources this was built from (None = live-fetched,
        # a path = user-uploaded) -- kept around so a caller that reuses a
        # cached dataset can still correctly report where the data came from.
        self.layer_sources = layer_sources or {}
        # which live source an online-fetched "elevation" layer actually
        # came from ("open_meteo" or "usgs") -- same reason as above.
        self.elevation_source = elevation_source
        # set only when elevation fell back to a LATER entry in the
        # source-attempt list (e.g. "auto" tried Open-Meteo first, it
        # failed, and USGS's much-more-limited grid cap was used instead) --
        # the earlier attempt's error, so the caller can explain WHY the
        # achieved grid is coarser than what was requested, rather than
        # that just silently showing up as a smaller grid with no reason.
        self.elevation_fallback_reason = elevation_fallback_reason


def _grid(bounds: tuple[float, float, float, float], resolution: int | tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    lat_min, lat_max, lon_min, lon_max = bounds
    nx, ny = _as_axis_counts(resolution)
    xs = np.linspace(lon_min, lon_max, nx)  # x = longitude
    ys = np.linspace(lat_min, lat_max, ny)  # y = latitude
    return xs, ys


def idw_interpolate(
    px: np.ndarray, py: np.ndarray, pv: np.ndarray,
    xs: np.ndarray, ys: np.ndarray, power: float = 2.0,
) -> np.ndarray:
    """Inverse-distance-weighted interpolation of scattered points (px, py, pv)
    onto a regular (len(ys), len(xs)) grid. A point that lands exactly on a
    grid node is returned exactly (avoids a division by zero)."""
    xx, yy = np.meshgrid(xs, ys)  # (ny, nx)
    values = np.zeros_like(xx)
    weight_sum = np.zeros_like(xx)
    for x0, y0, v0 in zip(px, py, pv):
        dist2 = (xx - x0) ** 2 + (yy - y0) ** 2
        on_point = dist2 < 1e-12
        if np.any(on_point):
            values[on_point] = v0
            weight_sum[on_point] = 1.0
            dist2 = np.where(on_point, np.inf, dist2)
        weight = 1.0 / (dist2 ** (power / 2))
        values += np.where(on_point, 0.0, weight * v0)
        weight_sum += np.where(on_point, 0.0, weight)
    return values / weight_sum


def load_csv_points(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads a CSV with header columns x,y,value (x=longitude, y=latitude
    for real-world layers)."""
    xs, ys, vs = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            vs.append(float(row["value"]))
    if not xs:
        raise ValueError(f"no data rows found in {path}")
    return np.array(xs), np.array(ys), np.array(vs)


def load_shapefile_points(path: str, value_field: str = "value") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads point samples from an ESRI shapefile (a .zip bundling
    .shp/.shx/.dbf, or a bare .shp alongside its sidecar files) via pyshp.
    Requires a POINT shape type and an attribute field named `value`
    (case-insensitive) holding the scalar sample value."""
    import shapefile

    reader = shapefile.Reader(path)
    if reader.shapeType not in (shapefile.POINT, shapefile.POINTZ, shapefile.POINTM):
        raise ValueError(
            f"shapefile '{path}' has shape type {reader.shapeType!r}, expected POINT "
            "(scattered value samples, same role as the CSV x,y,value format)"
        )
    field_names = [f[0] for f in reader.fields[1:]]  # skip the DeletionFlag pseudo-field
    matches = [f for f in field_names if f.lower() == value_field.lower()]
    if not matches:
        raise ValueError(
            f"shapefile '{path}' has no attribute field named '{value_field}' "
            f"(found: {field_names}); add one holding the scalar sample value"
        )
    field = matches[0]

    xs, ys, vs = [], [], []
    for shape_record in reader.shapeRecords():
        x, y = shape_record.shape.points[0]
        xs.append(x)
        ys.append(y)
        vs.append(float(shape_record.record[field]))
    if not xs:
        raise ValueError(f"no point features found in shapefile '{path}'")
    return np.array(xs), np.array(ys), np.array(vs)


def load_layer_points(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatches to the CSV or shapefile loader based on file extension
    (.zip/.shp -> shapefile, everything else -> CSV)."""
    lower = path.lower()
    if lower.endswith(".zip") or lower.endswith(".shp"):
        return load_shapefile_points(path)
    return load_csv_points(path)


def scalar_field_from_upload(
    name: str, path: str, bounds: tuple[float, float, float, float],
    resolution: int | tuple[int, int] = GRID_RESOLUTION, power: float = 2.0,
) -> ScalarField:
    """Builds a ScalarField for `name` from a user-uploaded file over `bounds`.

    A GeoTIFF (.tif/.tiff) -- e.g. a user's own DEM -- is clipped,
    reprojected, and resampled directly onto the analysis grid via GDAL
    (dem_cog_source.fetch_local_raster_field, the same QGIS-bundled-GDAL
    subprocess bridge the live USGS DEM COG source uses, since this
    project's main environment has no GDAL bindings). Anything else (CSV,
    or a shapefile .zip/.shp of POINT features) is treated as scattered
    point samples and IDW-interpolated onto the grid, as before.
    """
    lower = path.lower()
    if lower.endswith(".tif") or lower.endswith(".tiff"):
        from geoai_agent.dem_cog_source import fetch_local_raster_field  # local import: avoid a hard GDAL/QGIS dependency for callers that never upload a raster
        return fetch_local_raster_field(name, path, bounds, resolution)
    px, py, pv = load_layer_points(path)
    xs, ys = _grid(bounds, resolution)
    values = idw_interpolate(px, py, pv, xs, ys, power=power)
    return ScalarField(name=name, xs=xs, ys=ys, values=values)


# kept for existing callers -- CSV was the only supported upload format
# before shapefile support was added.
scalar_field_from_csv = scalar_field_from_upload


def fetch_open_meteo_field(
    variable: str, bounds: tuple[float, float, float, float], resolution: int | tuple[int, int] = GRID_RESOLUTION,
) -> ScalarField:
    """Live-fetches `variable` (a key in OPEN_METEO_VARIABLES) for every
    grid node over `bounds` from Open-Meteo, batched at a conservative
    coordinate-pair-per-request limit. Always a fresh HTTP request at call
    time -- nothing is downloaded ahead of time or cached to disk."""
    if variable not in OPEN_METEO_VARIABLES:
        raise ValueError(f"unknown Open-Meteo variable '{variable}', available: {list(OPEN_METEO_VARIABLES)}")

    xs, ys = _grid(bounds, resolution)
    xx, yy = np.meshgrid(xs, ys)
    lons = xx.ravel()
    lats = yy.ravel()
    out = np.empty(lons.shape[0], dtype=float)

    num_batches = -(-lons.shape[0] // _FETCH_BATCH)  # ceil division
    for batch_idx, start in enumerate(range(0, lons.shape[0], _FETCH_BATCH)):
        # Paced BEFORE every batch but the first -- observed live, issuing
        # batches back-to-back with no delay tripped the per-minute limit
        # as early as the 7th request. Retrying-after-429 alone isn't
        # enough: once the wait is over, resuming a rapid-fire burst on the
        # REMAINING batches immediately re-trips the same limit. Spreading
        # requests out from the start avoids ever bursting past it.
        if batch_idx > 0:
            time.sleep(_OPEN_METEO_BATCH_PACING_SECONDS)
        end = start + _FETCH_BATCH
        lat_param = ",".join(f"{v:.6f}" for v in lats[start:end])
        lon_param = ",".join(f"{v:.6f}" for v in lons[start:end])

        if variable == "elevation":
            url = f"https://api.open-meteo.com/v1/elevation?{urllib.parse.urlencode({'latitude': lat_param, 'longitude': lon_param})}"
        else:
            url = (
                "https://api.open-meteo.com/v1/forecast?"
                f"{urllib.parse.urlencode({'latitude': lat_param, 'longitude': lon_param, 'current': variable})}"
            )
        payload = None
        for attempt in range(_OPEN_METEO_MINUTELY_MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    payload = json.loads(resp.read())
                break
            except urllib.error.HTTPError as exc:
                reason = exc.read().decode("utf-8", errors="replace")
                try:
                    reason = json.loads(reason).get("reason", reason)
                except json.JSONDecodeError:
                    pass
                # only the per-minute limit is worth waiting out -- a daily
                # limit or any other error won't be fixed by retrying now.
                if exc.code == 429 and "minutely" in reason.lower() and attempt < _OPEN_METEO_MINUTELY_MAX_RETRIES:
                    time.sleep(_OPEN_METEO_MINUTELY_RETRY_WAIT_SECONDS)
                    continue
                raise RuntimeError(
                    f"Open-Meteo request failed on batch {batch_idx + 1}/{num_batches} "
                    f"({exc.code} {exc.reason}): {reason}. Try a lower resolution or a smaller "
                    "bounding box, or wait a moment before retrying."
                ) from exc

        if variable == "elevation":
            out[start:end] = payload["elevation"]
        else:
            out[start:end] = [entry["current"][variable] for entry in payload]

    values = out.reshape(xx.shape)
    return ScalarField(name=variable, xs=xs, ys=ys, values=values)


def is_in_continental_us(bounds: tuple[float, float, float, float]) -> bool:
    """Whether the center of `bounds` falls within the (conservative,
    continental-only) US bounding box USGS 3DEP coverage is assumed
    available in."""
    lat_min, lat_max, lon_min, lon_max = bounds
    us_lat_min, us_lat_max, us_lon_min, us_lon_max = _CONTINENTAL_US_BOUNDS
    mid_lat, mid_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    return us_lat_min <= mid_lat <= us_lat_max and us_lon_min <= mid_lon <= us_lon_max


def fetch_usgs_elevation_field(
    bounds: tuple[float, float, float, float], resolution: int | tuple[int, int] = GRID_RESOLUTION,
) -> ScalarField:
    """Live-fetches real elevation from USGS's 3DEP Elevation Point Query
    Service (EPQS) -- genuinely LiDAR-derived data (the service reports the
    native resolution of the raster each point came from; often 1m in
    LiDAR-covered parts of the continental US), unlike Open-Meteo's fixed
    ~90m SRTM data. Unlike Open-Meteo there is no batch endpoint: every
    point is its own HTTP request, issued concurrently
    (_USGS_CONCURRENCY workers) with retries; a request that still fails
    after retries is filled from its nearest successfully-fetched
    neighbor rather than left as a gap, since observed error rates under
    concurrent load are non-trivial (~10%) but the analysis grid needs a
    complete array. Always a fresh request at call time."""
    xs, ys = _grid(bounds, resolution)
    xx, yy = np.meshgrid(xs, ys)
    lons = xx.ravel()
    lats = yy.ravel()
    values = np.full(lons.shape[0], np.nan)

    def fetch_one(i: int):
        query = urllib.parse.urlencode({
            "x": f"{lons[i]:.6f}", "y": f"{lats[i]:.6f}", "units": "Meters", "wkid": 4326, "includeDate": "false",
        })
        url = f"https://epqs.nationalmap.gov/v1/json?{query}"
        for attempt in range(_USGS_MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(url, timeout=15) as resp:
                    payload = json.loads(resp.read())
                return i, float(payload["value"])
            except Exception:
                if attempt == _USGS_MAX_RETRIES:
                    return i, None
        return i, None

    with ThreadPoolExecutor(max_workers=_USGS_CONCURRENCY) as executor:
        for i, value in executor.map(fetch_one, range(lons.shape[0])):
            if value is not None:
                values[i] = value

    missing = np.isnan(values)
    if missing.all():
        raise RuntimeError(
            "USGS elevation service returned no valid data for this area "
            "(it may be outside 3DEP coverage) -- try Open-Meteo instead."
        )
    if missing.any():
        valid_idx = np.where(~missing)[0]
        valid_points = np.column_stack([lons[valid_idx], lats[valid_idx]])
        for mi in np.where(missing)[0]:
            dist2 = (valid_points[:, 0] - lons[mi]) ** 2 + (valid_points[:, 1] - lats[mi]) ** 2
            values[mi] = values[valid_idx[np.argmin(dist2)]]

    return ScalarField(name="elevation", xs=xs, ys=ys, values=values.reshape(xx.shape))


def fetch_elevation_field(
    bounds: tuple[float, float, float, float], resolution: int | tuple[int, int] = GRID_RESOLUTION,
) -> ScalarField:
    """Fetches real terrain elevation -- thin wrapper over
    fetch_open_meteo_field kept for existing callers (real_world_example.py)."""
    return fetch_open_meteo_field("elevation", bounds, resolution)


_CENSUS_BLOCKGROUPS_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "USA_Census_BlockGroups/FeatureServer/0/query"
)
_SQ_MILES_PER_SQ_KM = 0.386102  # 1 km2 = 0.386102 mi2 (1 mi2 = 2.58999 km2)


def fetch_population_exposure_field(
    bounds: tuple[float, float, float, float], resolution: int | tuple[int, int] = GRID_RESOLUTION,
) -> ScalarField:
    """Live-fetches US Census block-group population density (people per
    km2, 2020-Census-based) over `bounds` from Esri's ArcGIS Living Atlas
    `USA_Census_BlockGroups` FeatureServer, and rasterizes it onto the
    analysis grid -- a genuine, automatic exposure layer, rather than the
    entirely-optional user-upload this project relied on exclusively before
    (see flood_tools.rank_flood_basins's `exposure_field` parameter, and
    WEIGHT_PRESETS["life_safety"], which was previously only ever usable
    when a user happened to supply their own file).

    This Esri-hosted mirror is used instead of the Census Bureau's own data
    API deliberately: the Bureau's API now requires a free signup key,
    which would break this project's established no-key live-fetch
    philosophy (every other source here -- USGS, Open-Meteo, FEMA NFHL,
    Overpass, NOAA Atlas 14 -- needs none). The Living Atlas layer needs no
    key, is bbox-queryable, and already carries a precomputed POP_SQMI
    (population per square mile) field per block group, sourced from the
    same underlying Census data.

    Raises RuntimeError if the service is unreachable or returns no block
    groups for this area (this project has no live US Census coverage
    outside the US, obviously, but also no data for water/unpopulated areas
    with no block-group polygon at all) -- callers should catch this and
    treat it as "no exposure data for this run", the same way build_real_
    world_dataset callers already handle any other unavailable live source
    (e.g. web_app's agent flow: an exposure-fetch failure just means that
    scoring factor falls back to flood_tools' neutral 0.5 default, not a
    failed run).

    US-only, block-group scale (typically a few hundred to a few thousand
    people per polygon in an urban area) -- not building-level, but a
    material improvement over "no exposure signal unless the user happens
    to have their own file"."""
    from shapely.geometry import Point, Polygon
    from shapely.strtree import STRtree

    lat_min, lat_max, lon_min, lon_max = bounds
    query = urllib.parse.urlencode({
        "geometry": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "POPULATION,POP_SQMI,SQMI,FIPS",
        "returnGeometry": "true",
        "f": "json",
    })
    try:
        with urllib.request.urlopen(f"{_CENSUS_BLOCKGROUPS_URL}?{query}", timeout=20) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        raise RuntimeError(f"US Census block-group service request failed: {exc}") from exc

    features = payload.get("features") or []
    if not features:
        raise RuntimeError(
            "no US Census block groups returned for this area (outside US coverage, "
            "or an area with no block-group polygons at all)"
        )

    polygons = []
    densities_per_km2 = []
    for feature in features:
        attrs = feature.get("attributes", {})
        pop_sqmi = attrs.get("POP_SQMI")
        rings = (feature.get("geometry") or {}).get("rings")
        if pop_sqmi is None or not rings:
            continue
        try:
            polygon = Polygon(rings[0], holes=rings[1:]) if len(rings) > 1 else Polygon(rings[0])
        except Exception:
            continue
        if not polygon.is_valid or polygon.is_empty:
            continue
        polygons.append(polygon)
        # POP_SQMI is people per SQUARE MILE; a km2 is smaller than a mi2,
        # so people-per-km2 is the SMALLER number -- multiply by the
        # mi2->km2 area ratio, don't divide (dividing would inflate density
        # by ~6.7x, e.g. turning a plausible 8,000/km2 DC block group into
        # an implausible 54,000/km2).
        densities_per_km2.append(pop_sqmi * _SQ_MILES_PER_SQ_KM)

    if not polygons:
        raise RuntimeError("US Census block groups were returned but none had usable population/geometry data")

    tree = STRtree(polygons)
    xs, ys = _grid(bounds, resolution)
    xx, yy = np.meshgrid(xs, ys)
    values = np.zeros(xx.shape, dtype=float)

    for row in range(xx.shape[0]):
        for col in range(xx.shape[1]):
            point = Point(float(xx[row, col]), float(yy[row, col]))
            # STRtree.query(geometry, predicate) tests predicate(geometry,
            # tree_geometry) -- i.e. "is this point WITHIN a tree polygon",
            # not "does this point CONTAIN a tree polygon" (which is never
            # true for a point query and would silently return nothing).
            candidates = tree.query(point, predicate="within")
            if len(candidates):
                values[row, col] = densities_per_km2[int(candidates[0])]
            # else: no block group at this node (e.g. open water) -- left at 0.0

    return ScalarField(name="exposure", xs=xs, ys=ys, values=values)


_WORLDPOP_IMAGE_SERVER_URL = "https://worldpop.arcgis.com/arcgis/rest/services/WorldPop_Population_Density_100m/ImageServer"
# WorldPop's own NoData sentinel for this service (the standard float32
# "negative max" convention) -- confirmed live (a real exportImage request
# over a mixed land/water bbox returned this exact value at water/no-
# coverage cells, not zero) rather than assumed from documentation.
_WORLDPOP_NODATA_SENTINEL = -3.402823e38
# This service's own time-enabled coverage tops out at 2020 (checked
# directly via its timeInfo.timeExtent) -- WorldPop's global product line
# hasn't been extended past 2020 in this Living Atlas item as of this
# writing, unlike NLCD/the Sentinel-2 land-cover service above which do
# track their own "latest available" via the same timeInfo mechanism. Fixed
# here rather than queried per-request since there's nothing to discover.
_WORLDPOP_TIME_MS = 1577836800000  # 2020-01-01 UTC


def fetch_global_population_density_field(
    bounds: tuple[float, float, float, float], resolution: int | tuple[int, int] = GRID_RESOLUTION,
    timeout: float = 30.0,
) -> ScalarField:
    """Live-fetches WorldPop's global 100m gridded population density
    (people/km2, 2020 estimate) over `bounds`, via its public ArcGIS Living
    Atlas ImageServer -- no API key (confirmed live, despite being a Living
    Atlas item), genuinely worldwide coverage (checked: extent spans the
    full -180..180 longitude range), same exportImage-resized-server-side-
    to-the-exact-grid pattern as geoai_agent.land_cover's NLCD/global-land-
    cover fetches, just returning a continuous float raster instead of
    classified codes (no nearest-neighbor-only resampling constraint here --
    population density is a genuine continuous quantity, unlike a land-cover
    class code).

    Meant as fetch_population_exposure_field's fallback outside the US
    (that function's own Census block-group source has no coverage there):
    coarser (100m vs. block-group-scale, which is finer in dense urban
    areas but coarser in sparse ones) and a fixed 2020 estimate rather than
    the Census's own vintage, but real, live, worldwide data instead of the
    flat neutral 0.5 default every caller would otherwise fall back to.

    Raises RuntimeError on any failure -- network error, malformed
    response, or a returned image whose every cell is NoData (e.g. entirely
    open ocean, which WorldPop has no population estimate for) -- same
    degrade-gracefully contract as every other optional source in this
    project; callers should catch this and fall back further themselves."""
    nx, ny = _as_axis_counts(resolution)
    lat_min, lat_max, lon_min, lon_max = bounds
    query = urllib.parse.urlencode({
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}", "bboxSR": "4326", "imageSR": "4326",
        "size": f"{nx},{ny}", "format": "tiff", "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation", "noData": str(_WORLDPOP_NODATA_SENTINEL),
        "time": str(_WORLDPOP_TIME_MS), "f": "json",
    })
    try:
        with urllib.request.urlopen(f"{_WORLDPOP_IMAGE_SERVER_URL}/exportImage?{query}", timeout=timeout) as resp:
            meta = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        raise RuntimeError(f"WorldPop exportImage request failed: {exc}") from exc

    href = meta.get("href")
    if not href:
        raise RuntimeError(f"WorldPop exportImage returned no image href for this area: {meta}")
    try:
        with urllib.request.urlopen(href, timeout=timeout) as resp:
            tiff_bytes = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Failed to download WorldPop exported image: {exc}") from exc

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(tiff_bytes))
        values = np.array(image, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Could not read WorldPop exported TIFF: {exc}") from exc

    if values.shape != (ny, nx):
        raise RuntimeError(f"WorldPop export returned a {values.shape} image, expected ({ny}, {nx}).")

    nodata_mask = values < -1e30  # generous threshold around the sentinel -- float32 roundtrip through a TIFF isn't exact
    if nodata_mask.all():
        raise RuntimeError("WorldPop returned no usable population data for this entire area (e.g. open ocean).")
    values = np.where(nodata_mask, 0.0, values)  # no coverage -> treat as uninhabited, same as fetch_population_exposure_field's implicit-zero convention for cells with no matching Census block group

    xs, ys = _grid(bounds, resolution)
    # Same north-to-south -> south-to-north row flip as land_cover.py's
    # exportImage-based fetches -- Esri image services return row 0 = north.
    return ScalarField(name="exposure", xs=xs, ys=ys, values=np.flipud(values))


def derive_slope_field(elevation: ScalarField) -> ScalarField:
    """Gradient magnitude of an elevation field -- a second, genuinely
    real layer derived at no extra data-fetch cost, useful for spotting
    steep terrain (ridgelines/cliffs) rather than just highs and lows."""
    dy, dx = np.gradient(elevation.values, elevation.ys, elevation.xs)
    slope = np.sqrt(dx ** 2 + dy ** 2)
    return ScalarField(name="slope", xs=elevation.xs, ys=elevation.ys, values=slope)


def build_real_world_dataset(
    name: str,
    bounds: tuple[float, float, float, float],
    layer_sources: dict[str, str | None] | None = None,
    resolution: int | tuple[int, int] = GRID_RESOLUTION,
    include_slope: bool = True,
    layer_descriptions: dict[str, str] | None = None,
    elevation_source: str = "auto",
) -> RealWorldDataset:
    """Builds a RealWorldDataset with one or more named layers.

    layer_sources maps layer name -> a CSV or shapefile (.zip/.shp) path,
    or -> None to mean "fetch online" (only supported for
    OPEN_METEO_VARIABLES entries). If omitted entirely, defaults to
    fetching elevation online.

    layer_descriptions optionally maps layer name -> a free-text
    explanation of what it measures, carried through on the returned
    dataset for the caller to pass to an LLM recommendation step (custom
    uploaded layers have no built-in meaning the way "elevation" does).

    elevation_source picks which live source an online-fetched "elevation"
    layer comes from:
    - "auto" (the default): within the continental US, prefer streaming the
      actual USGS 3DEP DEM raster (see fetch_usgs_dem_cog_field in
      dem_cog_source.py -- clipped and resampled directly from the source
      GeoTIFF COG via GDAL, no point-count cap, no per-point rate limit).
      If that bridge is unavailable (no QGIS install found -- see
      DEMSourceUnavailableError) or the fetch fails outright, falls back to
      USGS's Elevation Point Query Service (one HTTP request per grid
      point, so only usable up to MAX_USGS_GRID_RESOLUTION), then to
      Open-Meteo (global, ~90m, much higher point-count ceiling but lower
      precision) as the last resort. Outside the continental US, prefers
      the SAME streamed-COG technique against Copernicus GLO-30 instead
      (fetch_copernicus_dem_field -- also no point-count cap, no per-point
      rate limit, genuinely global 30m coverage) -- checked directly: a
      real non-US run's elevation fetch took Open-Meteo 156-520s depending
      on rate-limit retries, for a grid this streams in a handful of
      seconds instead. Falls back to Open-Meteo only if that bridge is
      unavailable, the fetch fails outright, or the specific area falls in
      the small subset of tiles GLO-30 Public hasn't released yet.
    - "usgs_dem_cog": the streamed-COG source only, no fallback -- fails
      outright if the QGIS/GDAL bridge is unavailable or the fetch fails.
    - "usgs": the Elevation Point Query Service only, no fallback -- fails
      outright if USGS does.
    - "open_meteo": Open-Meteo only, no fallback -- for when the higher
      precision isn't worth the wait.
    """
    layer_sources = layer_sources or {"elevation": None}
    fields: dict[str, ScalarField] = {}
    # may change below depending on what was actually reachable -- kept in
    # sync with what was ACTUALLY used, not just requested.
    actual_elevation_source = elevation_source
    elevation_fallback_reason: str | None = None

    for layer_name, source in layer_sources.items():
        if source is not None:
            fields[layer_name] = scalar_field_from_upload(layer_name, source, bounds, resolution)
        elif layer_name == "elevation":
            nx, ny = _as_axis_counts(resolution)
            usgs_capped_resolution = (min(nx, MAX_USGS_GRID_RESOLUTION), min(ny, MAX_USGS_GRID_RESOLUTION))
            usgs_ok = is_in_continental_us(bounds)
            usgs_can_hit_target = usgs_ok and nx <= MAX_USGS_GRID_RESOLUTION and ny <= MAX_USGS_GRID_RESOLUTION

            if elevation_source == "usgs_dem_cog":
                attempts = [("usgs_dem_cog", resolution)]
            elif elevation_source == "usgs":
                attempts = [("usgs", resolution)]
            elif elevation_source == "open_meteo":
                attempts = [("open_meteo", resolution)]
                if usgs_ok:
                    attempts.append(("usgs", usgs_capped_resolution))
            elif usgs_ok:
                attempts = [("usgs_dem_cog", resolution)]
                if usgs_can_hit_target:
                    attempts.append(("usgs", resolution))
                attempts.append(("open_meteo", resolution))
            else:
                attempts = [("copernicus_dem_cog", resolution), ("open_meteo", resolution)]

            def _fetch(source_name: str, attempt_resolution):
                if source_name == "usgs_dem_cog":
                    from geoai_agent.dem_cog_source import fetch_usgs_dem_cog_field
                    return fetch_usgs_dem_cog_field(bounds, attempt_resolution)
                if source_name == "copernicus_dem_cog":
                    from geoai_agent.dem_cog_source import fetch_copernicus_dem_field
                    return fetch_copernicus_dem_field(bounds, attempt_resolution)
                if source_name == "usgs":
                    return fetch_usgs_elevation_field(bounds, attempt_resolution)
                return fetch_open_meteo_field("elevation", bounds, attempt_resolution)

            first_error = None
            for attempt_idx, (source_name, attempt_resolution) in enumerate(attempts):
                try:
                    fields[layer_name] = _fetch(source_name, attempt_resolution)
                    actual_elevation_source = source_name
                    if attempt_idx > 0 and first_error is not None:
                        elevation_fallback_reason = str(first_error)
                    break
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
            else:
                raise first_error
        elif layer_name in OPEN_METEO_VARIABLES:
            fields[layer_name] = fetch_open_meteo_field(layer_name, bounds, resolution)
        else:
            raise ValueError(
                f"layer '{layer_name}' has no uploaded source and is not a live-fetchable "
                f"variable (available: {list(OPEN_METEO_VARIABLES)})"
            )

    if include_slope and "elevation" in fields and "slope" not in fields:
        fields["slope"] = derive_slope_field(fields["elevation"])

    return RealWorldDataset(
        name=name, bounds=bounds, scalar_fields=fields, layer_descriptions=layer_descriptions,
        layer_sources=layer_sources, elevation_source=actual_elevation_source,
        elevation_fallback_reason=elevation_fallback_reason,
    )
