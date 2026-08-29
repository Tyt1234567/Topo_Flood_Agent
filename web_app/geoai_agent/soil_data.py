"""
Live USDA SSURGO hydrologic soil group lookup, via Soil Data Access (SDA).

SSURGO (the Soil Survey Geographic Database) is USDA NRCS's detailed soil
survey, exposed through a public, keyless SQL-over-HTTP service (Soil Data
Access) rather than a bulk shapefile/geodatabase download -- the same
live-fetch philosophy this project's other sources use, just via a POSTed
SQL query instead of a REST query-string.

https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest

Hydrologic soil group (A/B/C/D, USDA's own classification of how readily a
soil type lets water infiltrate -- A = well-drained sand/gravel, D = poorly
drained clay/urban land) is the second input (alongside land cover) the
standard SCS/NRCS Curve Number method needs -- see flood_tools.CN_TABLE and
runoff_coefficient_from_land_cover.

This module deliberately does NOT try to resolve soil group per grid cell.
A single SDA query already returns every soil map unit intersecting a whole
analysis bounding box in one request (SSURGO map units are typically tens
to hundreds of meters across, finer than this project's own analysis grid
resolution in most runs, so a per-cell query would mean one HTTP request
per grid point for essentially no extra real information) -- one POST per
analysis run, matching how rainfall.py fetches one precipitation-frequency
table per run at the bounds centroid rather than per grid cell. What comes
back is reduced to a single, run-wide DOMINANT hydrologic group: real
SSURGO data is messy at this scale -- a map unit is often a *complex* of
several soil components, each with its own group and only a percentage
share of the unit, and some components (e.g. "Urban land") report a null
hydrologic group entirely (undifferentiated/paved, no meaningful
infiltration classification) -- so this sums each group's area-weighted
share (`comppct_r`, the NRCS "representative component percent" field)
across every returned row and picks whichever group has the largest total,
silently excluding null-group rows from that tally rather than guessing.

Returns None (never raises) on any failure or when no usable soil data
exists for the area -- the same "treat as unavailable, fall back
gracefully" contract every other live source in this project follows
(fema_flood.fetch_fema_flood_zones, rainfall.fetch_noaa_atlas14_rainfall).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

_SDA_URL = "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest"
_HYDROLOGIC_GROUPS = ("A", "B", "C", "D")


def fetch_dominant_hydrologic_soil_group(bounds: tuple[float, float, float, float], timeout: float = 20.0) -> str | None:
    """The single, area-weighted-dominant hydrologic soil group ("A", "B",
    "C", or "D") for the whole `bounds` (lat_min, lat_max, lon_min,
    lon_max) box, live from USDA Soil Data Access, or None if the request
    fails, the area has no SSURGO coverage, or every returned soil
    component has a null hydrologic group (rare, but real -- observed in
    testing for a mostly-open-water or entirely-paved-urban-land area)."""
    lat_min, lat_max, lon_min, lon_max = bounds
    polygon_wkt = (
        f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, {lon_max} {lat_max}, "
        f"{lon_min} {lat_max}, {lon_min} {lat_min}))"
    )
    sql = (
        "SELECT mu.mukey, c.hydgrp, c.comppct_r FROM mapunit mu "
        "INNER JOIN component c ON c.mukey = mu.mukey "
        "WHERE mu.mukey IN (SELECT DISTINCT mukey FROM "
        f"SDA_Get_Mukey_from_intersection_with_WktWgs84('{polygon_wkt}')) "
        "ORDER BY mu.mukey, c.comppct_r DESC"
    )
    request = urllib.request.Request(
        _SDA_URL, data=json.dumps({"query": sql, "format": "JSON"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError):
        return None

    rows = payload.get("Table") or []
    if not rows:
        return None

    weighted_totals = {g: 0.0 for g in _HYDROLOGIC_GROUPS}
    for row in rows:
        try:
            _mukey, hydgrp, comppct_r = row
        except (ValueError, TypeError):
            continue
        if hydgrp not in _HYDROLOGIC_GROUPS or comppct_r is None:
            continue  # null/unrecognized group (e.g. undifferentiated "Urban land") -- excluded, not guessed
        try:
            weighted_totals[hydgrp] += float(comppct_r)
        except (TypeError, ValueError):
            continue

    if not any(weighted_totals.values()):
        return None
    return max(weighted_totals, key=weighted_totals.get)
