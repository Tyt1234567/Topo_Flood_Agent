"""
Live FEMA National Flood Hazard Layer (NFHL) lookup.

FEMA hosts the NFHL as a public ArcGIS REST MapServer -- no API key, no
bulk shapefile download needed. Layer 28 ("Flood Hazard Zones") is queried
directly by bounding box at request time, the same live-fetch pattern
every other data source in this project uses (USGS elevation, Open-Meteo,
Nominatim boundaries) -- nothing is ever pre-downloaded or cached to disk.

https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28

FLD_ZONE conventions (FEMA's own, not this project's invention):
- A, AE, AH, AO, AR, A99, V, VE -- Special Flood Hazard Area (SFHA): the
  1%-annual-chance ("100-year") floodplain, or worse (V/VE = coastal
  high-hazard, wave action). Federally backed mortgages require flood
  insurance here.
- X, with ZONE_SUBTY "0.2 PCT ANNUAL CHANCE FLOOD HAZARD" -- moderate risk,
  the 0.2%-annual-chance ("500-year") floodplain (the "shaded X" zone).
- X, no subtype -- minimal risk, outside the mapped floodplain.
- D -- flood risk undetermined (no detailed study for this area).

This is a courtesy public ArcGIS instance with no documented SLA -- it
returns occasional 500s (observed directly, transient) even for a
well-formed request, so every caller must treat a failure as "no NFHL data
for this area" and fall back gracefully, the same way geocode_place_boundary
(boundary.py) treats a Nominatim failure.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

_NFHL_QUERY_URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"

# FEMA's own term for the mapped 1%-annual-chance floodplain (or worse) --
# the zones that trigger mandatory flood insurance under federally backed
# mortgages.
_SFHA_ZONES = {"A", "AE", "AH", "AO", "AR", "A99", "V", "VE"}

# Fill colors for the map layer, by risk tier -- reds for the two SFHA
# tiers (coastal a touch darker/more saturated than inland), amber for the
# moderate "shaded X" tier, and no fill at all for minimal/undetermined
# (those cover most of a typical bounding box; filling them would just be
# visual noise over the terrain hillshade underneath).
_ZONE_COLORS = {
    "sfha": "#c0392b",
    "sfha_coastal": "#7b241c",
    "moderate": "#e8ab5c",
}


def _severity_label(fld_zone: str | None, zone_subty: str | None) -> str:
    """Plain-language risk tier from FEMA's own FLD_ZONE/ZONE_SUBTY codes."""
    if fld_zone in _SFHA_ZONES:
        coastal = " (coastal, wave action)" if fld_zone in ("V", "VE") else ""
        return f"high -- Special Flood Hazard Area, zone {fld_zone}{coastal}"
    if fld_zone == "X" and zone_subty and "0.2 PCT" in zone_subty:
        return "moderate -- 0.2% annual chance (\"500-year\") floodplain"
    if fld_zone == "X":
        return "minimal -- outside the mapped floodplain"
    if fld_zone == "D":
        return "undetermined -- no detailed flood study for this area"
    return f"unknown FEMA zone '{fld_zone}'"


def _severity_tier(fld_zone: str | None) -> str | None:
    """Short key into _ZONE_COLORS, or None for a zone not worth drawing
    a filled overlay for (minimal/undetermined/unrecognized)."""
    if fld_zone in ("V", "VE"):
        return "sfha_coastal"
    if fld_zone in _SFHA_ZONES:
        return "sfha"
    return None  # moderate ("shaded X") is colored per-feature below, using ZONE_SUBTY


def fetch_fema_flood_zones(bounds: tuple[float, float, float, float], timeout: float = 15.0) -> dict | None:
    """Live query of FEMA's NFHL flood-zone polygons intersecting `bounds`
    (lat_min, lat_max, lon_min, lon_max). Returns a GeoJSON FeatureCollection
    -- each feature's properties carry FLD_ZONE, ZONE_SUBTY, STATIC_BFE (raw
    FEMA fields; -9999 is FEMA's own "not applicable" sentinel for
    STATIC_BFE), plus "severity" (plain-language, see _severity_label) and
    "color" (a fill color for the moderate/high tiers, "" for minimal/
    undetermined zones that aren't worth drawing) -- or None on any failure
    (network error, malformed response, no coverage for this area). Every
    caller must treat None as "no flood-zone data available" and continue
    without it, not as a fatal error."""
    lat_min, lat_max, lon_min, lon_max = bounds
    query = urllib.parse.urlencode({
        "geometry": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "outSR": "4326",
        "outFields": "FLD_ZONE,ZONE_SUBTY,STATIC_BFE",
        "returnGeometry": "true",
        # Flood-zone polygons are already a generalized cartographic product,
        # not surveyed to sub-meter precision -- simplifying/quantizing the
        # returned geometry (~11m tolerance, 5 decimal places) is visually
        # lossless for a web-map overlay, and cuts the response from tens of
        # MB to under 1MB for a town-scale bounding box. Observed directly:
        # a full-precision request for a ~7km box returned HTTP 500 (the
        # unsimplified geometry was apparently too large for this service to
        # serialize), while the exact same query with these two params
        # succeeded immediately.
        "geometryPrecision": "5",
        "maxAllowableOffset": "0.0001",
        "f": "geojson",
    })
    request = urllib.request.Request(f"{_NFHL_QUERY_URL}?{query}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
    features = data.get("features")
    if not features:
        return None

    for feature in features:
        props = feature.setdefault("properties", {})
        fld_zone = props.get("FLD_ZONE")
        zone_subty = props.get("ZONE_SUBTY")
        props["severity"] = _severity_label(fld_zone, zone_subty)
        tier = _severity_tier(fld_zone)
        if tier is None and fld_zone == "X" and zone_subty and "0.2 PCT" in zone_subty:
            tier = "moderate"
        props["color"] = _ZONE_COLORS.get(tier, "")

    return {"type": "FeatureCollection", "features": features}


def tag_hotspots_with_flood_zone(hotspots: list[dict], flood_zones_geojson: dict) -> None:
    """Mutates each hotspot dict in place, adding "fema_flood_zone" (the raw
    FLD_ZONE code, or None if the point doesn't fall inside any returned
    polygon -- e.g. a small gap between adjacent flood studies) and
    "fema_flood_severity" (the plain-language tier)."""
    from shapely.geometry import Point, shape

    polygons = [
        (shape(f["geometry"]), f["properties"])
        for f in flood_zones_geojson["features"]
        if f.get("geometry")
    ]
    for h in hotspots:
        x, y = h["coords"]
        point = Point(x, y)
        match = next((props for geom, props in polygons if geom.contains(point)), None)
        h["fema_flood_zone"] = match.get("FLD_ZONE") if match else None
        h["fema_flood_severity"] = match.get("severity") if match else "not mapped by FEMA NFHL"
