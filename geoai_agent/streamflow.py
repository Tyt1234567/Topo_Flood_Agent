"""
Live USGS NWIS river/stream gauge lookup -- the fluvial-risk complement to
this project's otherwise entirely pluvial (rainfall-runoff, topological
basin-filling) flood analysis.

USGS's National Water Information System (NWIS) publishes real-time gauge
readings through a public, keyless Instantaneous Values web service, queried
directly by bounding box -- the same live-fetch-by-bbox pattern this
project's other sources use (FEMA NFHL, Overpass OSM), no API key, no bulk
download.

https://waterservices.usgs.gov/nwis/iv/?format=json&bBox={lon_min},{lat_min},{lon_max},{lat_max}&parameterCd=00065,00060&siteStatus=active

Parameter codes are USGS's own: 00065 = gage height (feet), 00060 =
discharge (cubic feet/second). Both are requested together; a site reporting
either (or both) is included.

Why this matters for a topological/pluvial flood tool: every compound
(fluvial + pluvial) flooding study surveyed for this project converges on
the same point -- a basin whose contributing area drains toward an
already-high or fast-rising river can't actually drain, even if its own
terrain and rainfall look unremarkable in isolation. This module doesn't
attempt to model that interaction (no flow-routing between a basin and a
river, no flood-stage threshold comparison) -- it only adds a real,
CURRENT gauge reading near a basin as contextual risk signal, the same
"corroborating real data, not a new score dimension" role FEMA NFHL zones
already play in flood_tools' pipeline (see tag_hotspots_with_flood_zone).

Gauge coverage is sparse nationally -- most of the country has zero active
gauges within any reasonable distance of a given analysis area. That is the
overwhelmingly common, EXPECTED outcome here, not an error; every caller
must treat a None/empty result the same way they already treat "no FEMA
data for this area" or "no DC stormwater pipes" -- fall back gracefully,
not fail the run.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request

_NWIS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
_GAGE_HEIGHT_CODE = "00065"
_DISCHARGE_CODE = "00060"

_METERS_PER_DEGREE_LAT = 111_320.0


def fetch_nearby_river_gauges(bounds: tuple[float, float, float, float], timeout: float = 15.0) -> list[dict] | None:
    """Live query of active USGS NWIS gauges intersecting `bounds` (lat_min,
    lat_max, lon_min, lon_max). Returns a list of {"site_code", "site_name",
    "lat", "lon", "gage_height_ft", "gage_height_datetime", "discharge_cfs",
    "discharge_datetime"} (either reading may be None if that site doesn't
    report it), or None on any failure (network error, malformed response,
    no active gauges in range -- the common case for most of the country).
    Every caller must treat None as "no gauge data available", not a fatal
    error, the same contract as fema_flood.fetch_fema_flood_zones."""
    lat_min, lat_max, lon_min, lon_max = bounds
    query = urllib.parse.urlencode({
        "format": "json",
        "bBox": f"{lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f}",
        "parameterCd": f"{_GAGE_HEIGHT_CODE},{_DISCHARGE_CODE}",
        "siteStatus": "active",
    })
    request = urllib.request.Request(f"{_NWIS_IV_URL}?{query}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError):
        return None

    series = data.get("value", {}).get("timeSeries") or []
    if not series:
        return None

    gauges: dict[str, dict] = {}
    for ts in series:
        try:
            source_info = ts["sourceInfo"]
            site_code = source_info["siteCode"][0]["value"]
            variable_code = ts["variable"]["variableCode"][0]["value"]
            latest = ts["values"][0]["value"][0]
            reading = float(latest["value"])
            reading_time = latest.get("dateTime")
        except (KeyError, IndexError, ValueError, TypeError):
            continue  # a malformed single time series shouldn't drop the whole response

        gauge = gauges.setdefault(site_code, {
            "site_code": site_code,
            "site_name": source_info.get("siteName"),
            "lat": source_info.get("geoLocation", {}).get("geogLocation", {}).get("latitude"),
            "lon": source_info.get("geoLocation", {}).get("geogLocation", {}).get("longitude"),
            "gage_height_ft": None, "gage_height_datetime": None,
            "discharge_cfs": None, "discharge_datetime": None,
        })
        if variable_code == _GAGE_HEIGHT_CODE:
            gauge["gage_height_ft"] = reading
            gauge["gage_height_datetime"] = reading_time
        elif variable_code == _DISCHARGE_CODE:
            gauge["discharge_cfs"] = reading
            gauge["discharge_datetime"] = reading_time

    result = [g for g in gauges.values() if g["lat"] is not None and g["lon"] is not None]
    return result or None


def tag_hotspots_with_river_gauge(
    hotspots: list[dict], gauges: list[dict], search_radius_m: float = 2000.0,
) -> None:
    """Mutates each hotspot dict in place, adding "nearest_gauge": None if no
    gauge is within `search_radius_m`, else {"site_name", "distance_m",
    "gage_height_ft", "discharge_cfs"} for the closest one in range.

    2000m default -- much wider than drainage_db's 150m pipe-search radius,
    because stream gauges are a sparse national network (a handful per
    watershed, not a dense buried-pipe grid), so a tight radius would almost
    always come back empty even right next to the relevant waterway.
    Straight-line distance via a local equirectangular approximation (same
    simplification used throughout this project, e.g. drainage_db.
    tag_hotspots_with_drainage_capacity) -- fine at this scale, not a true
    geodesic distance."""
    if not gauges:
        for h in hotspots:
            h["nearest_gauge"] = None
        return

    for h in hotspots:
        lon, lat = h["coords"]
        meters_per_deg_lon = _METERS_PER_DEGREE_LAT * math.cos(math.radians(lat))
        best = None
        best_dist_m = None
        for g in gauges:
            dist_m = math.hypot(
                (lon - g["lon"]) * meters_per_deg_lon, (lat - g["lat"]) * _METERS_PER_DEGREE_LAT,
            )
            if dist_m <= search_radius_m and (best_dist_m is None or dist_m < best_dist_m):
                best, best_dist_m = g, dist_m
        h["nearest_gauge"] = None if best is None else {
            "site_name": best["site_name"],
            "distance_m": round(best_dist_m, 1),
            "gage_height_ft": best["gage_height_ft"],
            "discharge_cfs": best["discharge_cfs"],
        }


def gauges_to_geojson(gauges: list[dict]) -> dict:
    """`fetch_nearby_river_gauges`'s output as a GeoJSON FeatureCollection of
    Points, for drawing on a map -- same pattern as drainage_db.
    pipes_to_geojson and fema_flood's zone polygons."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "site_name": g["site_name"],
                    "gage_height_ft": g["gage_height_ft"],
                    "discharge_cfs": g["discharge_cfs"],
                },
                "geometry": {"type": "Point", "coordinates": [g["lon"], g["lat"]]},
            }
            for g in gauges
        ],
    }
