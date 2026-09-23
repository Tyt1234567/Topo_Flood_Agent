"""
Live NLCD (National Land Cover Database) classified land-cover fetch.

USGS/MRLC's Annual NLCD Land Cover product is served as a live, keyless Esri
ImageServer -- no bulk national raster download needed, the same live-fetch-
by-bbox pattern this project's other raster-shaped source (USGS 3DEP DEM,
see dem_cog_source.py) uses, just via a different transport.

https://di-nlcd.img.arcgis.com/arcgis/rest/services/USA_NLCD_Annual_LandCover/ImageServer

Two gotchas discovered by testing this service directly, not documented
anywhere obvious:

1. This is a TIME-ENABLED mosaic dataset (`timeInfo.startTimeField`/
   `endTimeField` both "Year", one calendar-year raster per catalog item,
   1985-2023 initially, now extending into 2024+ as new years are added). A
   query with NO `time=` parameter can't pick a year and silently returns
   NoData for every pixel -- it looks like the service is broken or doesn't
   cover the queried area, when actually it just needs a timestamp. Fixed by
   fetching the service's own `timeInfo.timeExtent` (a [min_ms, max_ms] pair)
   and passing the max value straight through as `time=` -- confirmed live
   that the raw max end-time value resolves to the correct latest available
   year's catalog item (Annual_NLCD_LndCov_2024_CU_C1V1 in one test), no
   need to convert to/from a calendar year and guess at a specific day.
2. This is CLASSIFIED (categorical) data -- each pixel value is an NLCD
   class code (11=Open Water, 24=Developed High Intensity, 41=Deciduous
   Forest, ...), not a continuous quantity. `exportImage`'s default
   resampling is NOT safe here: bilinear/cubic interpolation between two
   different class codes produces a fractional, meaningless "class" (e.g.
   averaging 21 and 82 gives 51.5, not a real NLCD code). Every request
   below passes `interpolation=RSP_NearestNeighbor` explicitly.

Unlike the DEM pipeline (dem_cog_source.py), this does NOT need the QGIS-
bundled-GDAL subprocess bridge: `exportImage` is asked to resize server-side
to the EXACT analysis grid dimensions (`size={nx},{ny}`), so the returned
TIFF already IS the target grid -- no client-side reprojection/resampling
is needed at all. The returned TIFF is small (tens of KB for a typical
analysis grid) and reads cleanly with Pillow (already an installed
dependency -- a transitive one of matplotlib -- confirmed directly rather
than assumed), so no new dependency and no GDAL bridge either.

Any failure (network error, malformed response, no coverage for this area --
NLCD is CONUS + Alaska/Hawaii/Puerto Rico, so most of the world has none)
raises RuntimeError; callers should catch this and fall back to the flat
`runoff_coefficient` constant flood_tools.rank_flood_basins already accepts,
the same "optional enhancement, degrade gracefully" contract every other
live source in this project follows.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

_IMAGE_SERVER_URL = "https://di-nlcd.img.arcgis.com/arcgis/rest/services/USA_NLCD_Annual_LandCover/ImageServer"

# Fallback if the service's own timeInfo metadata call fails for any reason
# -- 2021-01-01 UTC, confirmed live to resolve to real classified data
# (NLCD class 24 for a College Park, MD test point) rather than NoData.
_FALLBACK_TIME_MS = 1609459200000


def _latest_available_time_ms(timeout: float) -> int:
    """The most recent `time=` value this service actually has classified
    data for, read from its own timeInfo.timeExtent metadata (a [min_ms,
    max_ms] pair) -- confirmed live that passing the raw max value straight
    through to identify/exportImage resolves correctly to the latest year's
    catalog item, no year-arithmetic guessing needed. Falls back to a known-
    good fixed timestamp if this metadata call itself fails, rather than
    letting that failure block the whole land-cover fetch."""
    try:
        query = urllib.parse.urlencode({"f": "json"})
        with urllib.request.urlopen(f"{_IMAGE_SERVER_URL}?{query}", timeout=timeout) as resp:
            meta = json.loads(resp.read())
        return int(meta["timeInfo"]["timeExtent"][1])
    except Exception:
        return _FALLBACK_TIME_MS


def fetch_nlcd_landcover_grid(
    bounds: tuple[float, float, float, float], resolution: int | tuple[int, int], year: int | None = None,
    timeout: float = 30.0,
) -> np.ndarray:
    """Live-fetches NLCD classified land-cover codes over `bounds`, resized
    server-side to exactly the analysis grid (`resolution`, an (nx, ny)
    tuple or a single int meaning nx == ny -- same convention real_data.py's
    `_as_axis_counts` normalizes elsewhere). Returns a (ny, nx) array of
    NLCD class codes (e.g. 11, 21..24, 41..43, ...) on this project's
    standard south-to-north row convention (row 0 = southern edge).

    `year`: an explicit calendar year to request (converted to a Jan-1 UTC
    timestamp); `None` (the default) asks the service for its own most
    recent available year via `_latest_available_time_ms` instead of
    guessing one that might have aged out.

    Raises RuntimeError on any failure -- network error, malformed
    response, corrupt/empty image, or simply no NLCD coverage for this area
    (NLCD covers the US + territories only). Never returns None; callers
    that want the "unavailable, fall back gracefully" behavior every other
    optional source in this project has should catch this themselves, the
    same way web_app/app.py already does for fetch_population_exposure_field."""
    from geoai_agent.real_data import _as_axis_counts  # local import: avoid a circular import at module load

    nx, ny = _as_axis_counts(resolution)
    lat_min, lat_max, lon_min, lon_max = bounds

    if year is not None:
        import datetime
        time_ms = int(datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    else:
        time_ms = _latest_available_time_ms(timeout)

    query = urllib.parse.urlencode({
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}", "bboxSR": "4326", "imageSR": "4326",
        "size": f"{nx},{ny}", "format": "tiff", "pixelType": "U8",
        "time": str(time_ms), "interpolation": "RSP_NearestNeighbor", "f": "json",
    })
    try:
        with urllib.request.urlopen(f"{_IMAGE_SERVER_URL}/exportImage?{query}", timeout=timeout) as resp:
            meta = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        raise RuntimeError(f"NLCD exportImage request failed: {exc}") from exc

    href = meta.get("href")
    if not href:
        raise RuntimeError(
            f"NLCD exportImage returned no image href for this area (likely outside NLCD's "
            f"US+territories coverage): {meta}"
        )
    try:
        with urllib.request.urlopen(href, timeout=timeout) as resp:
            tiff_bytes = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Failed to download NLCD exported image: {exc}") from exc

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(tiff_bytes))
        values = np.array(image)
    except Exception as exc:
        raise RuntimeError(f"Could not read NLCD exported TIFF: {exc}") from exc

    if values.shape != (ny, nx):
        raise RuntimeError(
            f"NLCD export returned a {values.shape} image, expected ({ny}, {nx}) -- "
            "the service ignored the requested size for this request."
        )

    # Esri image services return row 0 = north, same top-to-bottom convention
    # GDAL uses for the DEM COG source (see dem_cog_source.py) -- this
    # project's grids run south-to-north (ys = linspace(lat_min, lat_max,
    # ny)), so flip to match.
    return np.flipud(values)


# NLCD's own official legend (label + color), keyed by the same class codes
# CN_TABLE (flood_tools.py) already looks runoff numbers up by -- so a code
# this map layer can't color is also, not coincidentally, a code that
# contributes nothing to the runoff-coefficient calculation either.
NLCD_CLASS_INFO = {
    11: ("Open Water", "#476ba1"),
    12: ("Perennial Ice/Snow", "#d1def8"),
    21: ("Developed, Open Space", "#dec5c5"),
    22: ("Developed, Low Intensity", "#d99282"),
    23: ("Developed, Medium Intensity", "#eb0000"),
    24: ("Developed, High Intensity", "#ab0000"),
    31: ("Barren Land", "#b3ac9f"),
    41: ("Deciduous Forest", "#68ab5f"),
    42: ("Evergreen Forest", "#1c5f2c"),
    43: ("Mixed Forest", "#b5c98e"),
    52: ("Shrub/Scrub", "#ccb879"),
    71: ("Grassland/Herbaceous", "#dfdfc2"),
    81: ("Pasture/Hay", "#dcd939"),
    82: ("Cultivated Crops", "#ab6c28"),
    90: ("Woody Wetlands", "#b8d9eb"),
    95: ("Emergent Herbaceous Wetlands", "#6c9fb8"),
}


def land_cover_geojson(field, land_cover_grid: np.ndarray, class_info: dict = NLCD_CLASS_INFO) -> dict:
    """`land_cover_grid` (fetch_nlcd_landcover_grid's classified array,
    pixel-aligned to `field`'s own grid -- see web_app/app.py's
    resolution=elevation_field.shape[::-1] call) as a GeoJSON
    FeatureCollection of per-cell rectangles, one per grid node exactly like
    visualize.field_to_geojson's elevation heatmap, but colored by the
    source's own categorical class legend (`class_info` -- NLCD_CLASS_INFO
    by default, or GLOBAL_LANDCOVER_CLASS_INFO for fetch_global_landcover_
    grid's output) instead of a continuous ramp -- averaging or
    interpolating between two class codes would produce a meaningless
    fractional "class" (see this module's docstring), so every cell keeps
    its own single class color as-is. A class code with no entry in
    `class_info` (rare -- e.g. NoData at a clip edge, or a Clouds pixel
    from the global source) is simply skipped rather than guessed."""
    ny, nx = field.shape

    def _edges(coords: list[float]) -> list[float]:
        n = len(coords)
        e = [0.0] * (n + 1)
        e[0] = coords[0] - (coords[1] - coords[0]) / 2
        e[n] = coords[-1] + (coords[-1] - coords[-2]) / 2
        for k in range(1, n):
            e[k] = (coords[k - 1] + coords[k]) / 2
        return e

    x_edges, y_edges = _edges(list(field.xs)), _edges(list(field.ys))
    features = []
    for i in range(ny):
        for j in range(nx):
            info = class_info.get(int(land_cover_grid[i, j]))
            if info is None:
                continue
            label, color = info
            ring = [
                [x_edges[j], y_edges[i]], [x_edges[j + 1], y_edges[i]],
                [x_edges[j + 1], y_edges[i + 1]], [x_edges[j], y_edges[i + 1]],
                [x_edges[j], y_edges[i]],
            ]
            features.append({
                "type": "Feature",
                "properties": {"land_cover_class": int(land_cover_grid[i, j]), "label": label, "color": color},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })
    return {"type": "FeatureCollection", "features": features}


# -- Global land cover (outside NLCD's US+territories-only coverage) -------

_GLOBAL_IMAGE_SERVER_URL = "https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer"
# Confirmed live, no API key required despite this being an Esri Living
# Atlas item (some Living Atlas layers require an org/developer account;
# this one's raw ImageServer REST endpoint does not -- checked directly
# with an anonymous request, not assumed from documentation).
_GLOBAL_FALLBACK_TIME_MS = 1609459200000  # 2021-01-01 UTC, same fallback convention as NLCD above

# Esri/Impact Observatory/Microsoft's own Sentinel-2 10m Land Cover class
# codes -- deliberately non-contiguous (this is their scheme, not this
# project's invention; there is no class 3 or 6). "Clouds" (10) is a
# data-quality flag from the source imagery, not a real land-cover type --
# omitted here the same way NLCD_CLASS_INFO already omits Perennial Ice/
# Snow, so land_cover_geojson silently skips it and CN_TABLE_GLOBAL
# (flood_tools.py) has nothing to look it up against either. "Snow/Ice" (9)
# is omitted for the same reason NLCD's Perennial Ice/Snow class has no
# CN_TABLE row: not meaningfully assignable a pluvial runoff curve number.
GLOBAL_LANDCOVER_CLASS_INFO = {
    1: ("Water", "#419bdf"),
    2: ("Trees", "#397d49"),
    4: ("Flooded Vegetation", "#7a87c6"),
    5: ("Crops", "#e49635"),
    7: ("Built Area", "#c4281b"),
    8: ("Bare Ground", "#a59b8f"),
    11: ("Rangeland", "#e3e2c3"),
}


def fetch_global_landcover_grid(
    bounds: tuple[float, float, float, float], resolution: int | tuple[int, int], timeout: float = 30.0,
) -> np.ndarray:
    """Same exportImage/TIFF-over-Pillow live-fetch pattern as
    fetch_nlcd_landcover_grid above, against Esri/Impact Observatory/
    Microsoft's global Sentinel-2 10m Land Cover ImageServer instead of
    NLCD -- for study areas outside NLCD's US+territories-only coverage.
    Returns a (ny, nx) array of this source's own class codes (see
    GLOBAL_LANDCOVER_CLASS_INFO above), same south-to-north row convention
    as every other field in this project.

    Meant as fetch_nlcd_landcover_grid's fallback, not a first choice even
    for a US location: NLCD is a purpose-built, single-country land-cover
    product; this is a general-purpose global classifier, coarser-grained
    (11 broad classes vs. NLCD's 16) and without NLCD's decades of
    validation. Raises RuntimeError on any failure -- network error,
    malformed response, no coverage -- same degrade-gracefully contract as
    fetch_nlcd_landcover_grid; callers should catch it themselves."""
    from geoai_agent.real_data import _as_axis_counts  # local import: avoid a circular import at module load

    nx, ny = _as_axis_counts(resolution)
    lat_min, lat_max, lon_min, lon_max = bounds

    try:
        query = urllib.parse.urlencode({"f": "json"})
        with urllib.request.urlopen(f"{_GLOBAL_IMAGE_SERVER_URL}?{query}", timeout=timeout) as resp:
            meta = json.loads(resp.read())
        time_ms = int(meta["timeInfo"]["timeExtent"][1])
    except Exception:
        time_ms = _GLOBAL_FALLBACK_TIME_MS

    query = urllib.parse.urlencode({
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}", "bboxSR": "4326", "imageSR": "4326",
        "size": f"{nx},{ny}", "format": "tiff", "pixelType": "U8",
        "time": str(time_ms), "interpolation": "RSP_NearestNeighbor", "f": "json",
    })
    try:
        with urllib.request.urlopen(f"{_GLOBAL_IMAGE_SERVER_URL}/exportImage?{query}", timeout=timeout) as resp:
            meta = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Global land-cover exportImage request failed: {exc}") from exc

    href = meta.get("href")
    if not href:
        raise RuntimeError(f"Global land-cover exportImage returned no image href for this area: {meta}")
    try:
        with urllib.request.urlopen(href, timeout=timeout) as resp:
            tiff_bytes = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Failed to download global land-cover exported image: {exc}") from exc

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(tiff_bytes))
        values = np.array(image)
    except Exception as exc:
        raise RuntimeError(f"Could not read global land-cover exported TIFF: {exc}") from exc

    if values.shape != (ny, nx):
        raise RuntimeError(
            f"Global land-cover export returned a {values.shape} image, expected ({ny}, {nx})."
        )
    return np.flipud(values)
