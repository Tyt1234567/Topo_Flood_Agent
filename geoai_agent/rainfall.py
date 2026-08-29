"""
Live NOAA Atlas 14 precipitation-frequency lookup.

NOAA's Hydrometeorological Design Studies Center (HDSC, now under the Office
of Water Prediction) publishes point precipitation-frequency estimates --
depth-duration-frequency tables -- through a public, keyless CGI endpoint
that returns plain CSV text for any lat/lon. No API key, no bulk shapefile/
grid download needed, the same live-fetch-by-point pattern this project's
other data sources use (USGS elevation, FEMA NFHL, Nominatim boundaries).

https://hdsc.nws.noaa.gov/pfds/ (the interactive tool this endpoint backs)
https://hdsc.nws.noaa.gov/cgi-bin/new/fe_text_mean.csv?lat={lat}&lon={lon}&data=depth&units=metric&series=pds

This replaces flood_tools.RAINFALL_SCENARIOS's illustrative, single-region
table with a genuine site-specific estimate wherever NOAA Atlas 14 has
coverage (the continental US, plus Alaska/Hawaii/Puerto Rico/Pacific islands
each via their own Atlas 14 volume -- the endpoint auto-selects the right
one from lat/lon, no separate handling needed here).

The response is a small multi-section text/CSV document, not JSON -- e.g.:

    PRECIPITATION FREQUENCY ESTIMATES
    by duration for ARI (years):, 1,2,5,10,25,50,100,200,500,1000
    5-min:, 9,11,13,14,16,17,19,20,22,23
    ...
    60-min:, 30,37,46,54,63,71,79,87,98,107
    ...

Only the "60-min" duration row is used here: its depth in millimeters is
numerically the same as a 1-hour intensity in mm/hr, matching this
project's existing "1-hour design storm" framing (see flood_tools.py).

Locations outside every Atlas 14 project area (i.e. outside the US and its
territories) do NOT get a normal HTTP error -- the service still answers
200 OK, just with a body like:

    result = 'none';
    ErrorMsg =  'Error 3.0: Selected location is not within a project area';

so "no coverage" has to be detected from the response CONTENT (no
"PRECIPITATION FREQUENCY ESTIMATES" section, or no parseable 60-min row),
not from the HTTP status code. Any failure mode -- network error, timeout,
malformed response, out-of-coverage -- returns None, the same "treat as
unavailable, fall back gracefully" contract every other live source in this
project follows (fema_flood.fetch_fema_flood_zones, boundary.
geocode_place_boundary).
"""
from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request

_ATLAS14_URL = "https://hdsc.nws.noaa.gov/cgi-bin/new/fe_text_mean.csv"

# The six return periods flood_tools.RAINFALL_SCENARIOS already exposes --
# NOAA Atlas 14 also reports 1/5/200/1000yr columns, but every caller in
# this project (the flood_analyze UI's dropdown, the one-box agent's fixed
# scenario choice, report.py) only knows this set, so a live table has to
# match it exactly to be a drop-in replacement rather than a second, subtly
# different rainfall-scenario vocabulary.
_RETURN_PERIODS = (2, 10, 25, 50, 100, 500)

_ARI_HEADER_RE = re.compile(r"by duration for ARI \(years\):\s*,(.+)")
_DURATION_ROW_RE = re.compile(r"^\s*60-min:\s*,(.+)$", re.MULTILINE)


def fetch_noaa_atlas14_rainfall(lat: float, lon: float, timeout: float = 15.0) -> dict | None:
    """Live NOAA Atlas 14 point precipitation-frequency estimate for
    (lat, lon), reshaped into exactly flood_tools.RAINFALL_SCENARIOS's shape
    -- {"2yr": {"label", "return_period_years", "intensity_mm_per_hr"}, ...}
    -- so it can be passed straight into rank_flood_basins's rainfall_table
    parameter as a genuine per-location replacement for the static table.

    Returns None (never raises) if this location has no Atlas 14 coverage,
    or the request/parse fails for any other reason -- callers should fall
    back to the static RAINFALL_SCENARIOS table exactly as they would for
    any other unavailable live source in this project."""
    query = urllib.parse.urlencode({
        "lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "data": "depth", "units": "metric", "series": "pds",
    })
    try:
        with urllib.request.urlopen(f"{_ATLAS14_URL}?{query}", timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None

    ari_match = _ARI_HEADER_RE.search(body)
    if ari_match is None:
        return None
    duration_match = _DURATION_ROW_RE.search(body)
    if duration_match is None:
        return None

    try:
        return_periods = [int(v.strip()) for v in ari_match.group(1).split(",") if v.strip()]
        depths_mm = [float(v.strip()) for v in duration_match.group(1).split(",") if v.strip()]
    except ValueError:
        return None
    if len(return_periods) != len(depths_mm) or not return_periods:
        return None

    depth_by_period = dict(zip(return_periods, depths_mm))
    if not all(rp in depth_by_period for rp in _RETURN_PERIODS):
        return None  # malformed/partial response -- don't return an incomplete scenario table

    return {
        f"{rp}yr": {
            "label": f"{rp}-year storm",
            "return_period_years": rp,
            # 60-min depth (mm) == 1-hour intensity (mm/hr) numerically.
            "intensity_mm_per_hr": depth_by_period[rp],
        }
        for rp in _RETURN_PERIODS
    }
