"""
Local PostgreSQL/PostGIS store for real stormwater pipe network data.

Unlike every other data source in this project (USGS elevation, Open-Meteo,
FEMA NFHL, Nominatim boundaries), there is no live, nationally-general API
for stormwater/storm-drain infrastructure -- confirmed directly: no federal
aggregator exists, and of several city open-data GIS portals checked, only
Washington DC's (hosted, oddly, on a USGS StreamStats ArcGIS server) turned
out to have a genuinely queryable REST endpoint with capacity-relevant pipe
attributes (diameter, material, slope, invert elevations). Bulk-download,
not live-query, is realistically how this class of data has to be handled
-- see load_dc_stormwater.py (project root), a one-time ETL script that
pulls a city's full pipe + catch-basin layers and loads them here. This
module is the read side: given a bounding box, return whatever pipes this
local database happens to have for that area (initially DC only -- see
that script's docstring for how to add another city later), or None if
nothing's been loaded for it. A None here is a normal, expected outcome
for most of the country, not an error -- callers should treat it the same
way they already treat "no FEMA data for this area" (fema_flood.py) or "no
OSM boundary for this place" (boundary.py): fall back gracefully, and (per
this project's established pattern for gaps in live data -- see the
elevation/exposure/boundary "Modify" upload flow in web_app/app.py) let
the user supply their own pipe data for areas this database doesn't cover
yet, rather than silently pretending the area was checked and came back
clean.

Capacity estimate (pipe_capacity_m3s) is a direct Manning's-equation
full-flow calculation from a pipe's own diameter and slope, only -- no
per-material roughness lookup, no falling back to a slope derived from
invert elevations when the pipe's own SLOPE field is missing. When a pipe
doesn't have both of those two numbers directly, its capacity is simply
unknown (None); the app still reports whether a local minimum has ANY
pipe nearby at all (tag_hotspots_with_drainage_capacity's
"nearby_pipe_count") independent of whether that pipe's capacity could be
computed, which is the primary, always-available signal -- the capacity
number is a bonus for the subset of pipes with usable data, not something
worth extra machinery to coax out of incomplete records.
"""
from __future__ import annotations

import math
import os

# A single generic Manning's roughness coefficient (a standard published
# mid-range value, e.g. FHWA HEC-22 -- close to concrete/clay pipe, the
# most common materials in DC's own data) rather than a per-material
# lookup table -- close enough for an order-of-magnitude capacity estimate
# without needing to trust the source data's MATRL field.
_MANNING_N = 0.013

# Slope values outside this range are treated as bad data (a unit/decimal
# error in the source, e.g. percent entered where a fraction was expected)
# rather than real pipe grades -- observed directly: a small fraction of
# DC's Pipe records have SLOPE values like 3.9 (390%), physically
# implausible for a gravity sewer.
_MIN_PLAUSIBLE_SLOPE = 0.0001   # 0.01%
_MAX_PLAUSIBLE_SLOPE = 0.20     # 20%

_METERS_PER_INCH = 0.0254


def pipe_capacity_m3s(diameter_in: float | None, slope: float | None) -> float | None:
    """Manning's-equation full-flow capacity (m^3/s) of a circular pipe
    from its own diameter and slope only, or None if either is missing or
    the slope is outside _MIN/MAX_PLAUSIBLE_SLOPE."""
    if not diameter_in or diameter_in <= 0 or not slope or slope <= 0:
        return None
    if not (_MIN_PLAUSIBLE_SLOPE <= slope <= _MAX_PLAUSIBLE_SLOPE):
        return None
    diameter_m = diameter_in * _METERS_PER_INCH
    area_m2 = math.pi / 4 * diameter_m ** 2
    hydraulic_radius_m = diameter_m / 4  # full circular pipe
    return (1.0 / _MANNING_N) * area_m2 * hydraulic_radius_m ** (2 / 3) * slope ** 0.5


def get_connection():
    """A psycopg2 connection to the local drainage database, or None if
    DATABASE_URL isn't set or the connection fails for any reason (the
    Postgres service isn't running, wrong credentials, etc.) -- same
    "treat as unavailable, don't crash the pipeline" contract as every
    other optional data source here."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return None
    try:
        import psycopg2
        return psycopg2.connect(database_url)
    except Exception:
        return None


def ensure_schema(conn) -> None:
    """Creates the pipes/catch_basins tables (+ spatial indexes) if they
    don't already exist -- safe to call every time before a load or query,
    not just once at initial setup."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pipes (
                id SERIAL PRIMARY KEY,
                source_city TEXT NOT NULL,
                source_objectid TEXT,
                geom GEOMETRY(LineString, 4326) NOT NULL,
                material TEXT,
                diameter_in DOUBLE PRECISION,
                shape TEXT,
                slope DOUBLE PRECISION,
                slope_source TEXT,
                upstream_invert_ft DOUBLE PRECISION,
                downstream_invert_ft DOUBLE PRECISION,
                length_ft DOUBLE PRECISION,
                capacity_m3s DOUBLE PRECISION,
                raw_attrs JSONB,
                loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (source_city, source_objectid)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS pipes_geom_idx ON pipes USING GIST (geom);")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS catch_basins (
                id SERIAL PRIMARY KEY,
                source_city TEXT NOT NULL,
                source_objectid TEXT,
                geom GEOMETRY(Point, 4326) NOT NULL,
                raw_attrs JSONB,
                loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (source_city, source_objectid)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS catch_basins_geom_idx ON catch_basins USING GIST (geom);")
    conn.commit()


def fetch_nearby_pipes(bounds: tuple[float, float, float, float]) -> list[dict] | None:
    """Pipes whose geometry intersects `bounds` (lat_min, lat_max, lon_min,
    lon_max), or None if the database is unavailable OR this area simply
    isn't covered by anything loaded yet (the overwhelmingly common case,
    outside DC -- not an error). Each dict has: coords ([[lon,lat], ...]),
    material, diameter_in, slope, capacity_m3s, source_city."""
    conn = get_connection()
    if conn is None:
        return None
    lat_min, lat_max, lon_min, lon_max = bounds
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_city, material, diameter_in, slope, capacity_m3s,
                       ST_AsGeoJSON(geom)
                FROM pipes
                WHERE geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
                """,
                (lon_min, lat_min, lon_max, lat_max),
            )
            rows = cur.fetchall()
    except Exception:
        return None
    finally:
        conn.close()
    if not rows:
        return None

    import json
    pipes = []
    for source_city, material, diameter_in, slope, capacity_m3s, geom_json in rows:
        geom = json.loads(geom_json)
        pipes.append({
            "source_city": source_city,
            "material": material,
            "diameter_in": diameter_in,
            "slope": slope,
            "capacity_m3s": capacity_m3s,
            "coords": geom.get("coordinates"),
        })
    return pipes


def pipes_to_geojson(pipes: list[dict]) -> dict:
    """`fetch_nearby_pipes`'s output as a GeoJSON FeatureCollection of
    LineStrings, for drawing on a map (web_app's persistent map and the
    downloadable report's embedded one both use this the same way they
    already do fema_flood.fetch_fema_flood_zones's GeoJSON)."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "material": p["material"],
                    "diameter_in": p["diameter_in"],
                    "capacity_m3s": p["capacity_m3s"],
                },
                "geometry": {"type": "LineString", "coordinates": p["coords"]},
            }
            for p in pipes
        ],
    }


def tag_hotspots_with_drainage_capacity(hotspots: list[dict], pipes: list[dict], search_radius_m: float = 150.0) -> None:
    """Mutates each hotspot dict in place, adding "nearby_pipe_capacity_m3s"
    (the largest Manning's-equation capacity among pipes within
    `search_radius_m` of the hotspot, or None if no pipe with a computable
    capacity is that close) and "nearby_pipe_count" (how many pipe segments
    were within range at all, computable capacity or not -- lets a caller
    distinguish "no pipes nearby" from "pipes nearby but not enough data to
    size them"). Straight-line distance in a local equirectangular
    approximation, same simplification report.py's other meter-conversion
    code already uses -- fine at the ~150m scale this is meant to work at."""
    from shapely.geometry import LineString, Point

    meters_per_deg_lat = 111_320.0

    def _to_line(coords):
        return LineString(coords) if len(coords) >= 2 else None

    lines = [(_to_line(p["coords"]), p) for p in pipes if p.get("coords")]
    lines = [(line, p) for line, p in lines if line is not None]

    for h in hotspots:
        lon, lat = h["coords"]
        meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(lat))
        point = Point(lon, lat)
        nearby = []
        for line, p in lines:
            nearest = line.interpolate(line.project(point))
            dist_m = math.hypot((point.x - nearest.x) * meters_per_deg_lon, (point.y - nearest.y) * meters_per_deg_lat)
            if dist_m <= search_radius_m:
                nearby.append(p)
        h["nearby_pipe_count"] = len(nearby)
        capacities = [p["capacity_m3s"] for p in nearby if p.get("capacity_m3s")]
        h["nearby_pipe_capacity_m3s"] = round(max(capacities), 3) if capacities else None
