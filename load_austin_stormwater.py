"""
One-time ETL: loads Austin, TX's drainage Pipe + Manhole layers into the
local PostgreSQL/PostGIS database. Same shape as load_dc_stormwater.py --
see that script's docstring for the overall pattern.

Source: maps.austintexas.gov/gis/rest/Shared/DrainageInfrastructure/MapServer
(layer 6 = Drainage Pipe, layer 5 = Manhole) -- a public ArcGIS REST
service, no API key. maxRecordCount=10000/request.

SLOPE is in PERCENT here (same as Fairfax County, confirmed directly by
cross-checking against UP_INVERT/DN_INVERT/LENGTH on sample rows: e.g.
(959.98-955.99)/273*100 = 1.46, matching the SLOPE field exactly) -- divided
by 100 before calling pipe_capacity_m3s, which expects a unitless fraction.

~104,942 pipe segments + ~11,714 manholes as of this writing.

Usage:
    python load_austin_stormwater.py
Requires DATABASE_URL to be set (see .env / geoai_agent/drainage_db.py).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

from dotenv import load_dotenv

load_dotenv()

from geoai_agent.drainage_db import ensure_schema, get_connection, pipe_capacity_m3s

_BASE_URL = "https://maps.austintexas.gov/gis/rest/Shared/DrainageInfrastructure/MapServer"
_PAGE_SIZE = 2000
_SOURCE_CITY = "austin_tx"

_PIPE_FIELDS = "OBJECTID,MATERIAL,SLOPE,WIDTH,HEIGHT,STRUCT_SHAPE,UP_INVERT,DN_INVERT,LENGTH"


def _query_page(layer_id: int, out_fields: str, offset: int) -> dict:
    params = {
        "where": "1=1",
        "outFields": out_fields,
        "returnGeometry": "true",
        "outSR": "4326",
        "geometryPrecision": "6",
        "resultOffset": str(offset),
        "resultRecordCount": str(_PAGE_SIZE),
        "f": "geojson",
    }
    url = f"{_BASE_URL}/{layer_id}/query?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())


def _fetch_all(layer_id: int, out_fields: str, label: str):
    offset = 0
    while True:
        page = _query_page(layer_id, out_fields, offset)
        features = page.get("features", [])
        if not features:
            return
        yield from features
        print(f"  {label}: fetched {offset + len(features)} so far...")
        if len(features) < _PAGE_SIZE:
            return
        offset += _PAGE_SIZE
        time.sleep(0.2)  # a courtesy public service -- don't hammer it


def load_pipes(conn) -> int:
    count = 0
    with conn.cursor() as cur:
        for feature in _fetch_all(6, _PIPE_FIELDS, "pipes"):
            geom = feature.get("geometry")
            props = feature.get("properties", {})
            if not geom or geom.get("type") != "LineString":
                continue
            diameter_in = props.get("WIDTH") or props.get("HEIGHT")
            slope_pct = props.get("SLOPE")
            slope = abs(slope_pct) / 100.0 if slope_pct is not None else None
            capacity = pipe_capacity_m3s(diameter_in, slope)
            cur.execute(
                """
                INSERT INTO pipes (source_city, source_objectid, geom, material, diameter_in, shape,
                                    slope, slope_source, upstream_invert_ft, downstream_invert_ft,
                                    length_ft, capacity_m3s, raw_attrs)
                VALUES (%s, %s, ST_Force2D(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_city, source_objectid) DO UPDATE SET
                    geom = EXCLUDED.geom, material = EXCLUDED.material, diameter_in = EXCLUDED.diameter_in,
                    shape = EXCLUDED.shape, slope = EXCLUDED.slope, slope_source = EXCLUDED.slope_source,
                    upstream_invert_ft = EXCLUDED.upstream_invert_ft, downstream_invert_ft = EXCLUDED.downstream_invert_ft,
                    length_ft = EXCLUDED.length_ft, capacity_m3s = EXCLUDED.capacity_m3s,
                    raw_attrs = EXCLUDED.raw_attrs, loaded_at = now()
                """,
                (
                    _SOURCE_CITY, str(props.get("OBJECTID")), json.dumps(geom), props.get("MATERIAL"),
                    diameter_in, props.get("STRUCT_SHAPE"), slope, "attribute (pct)" if slope is not None else None,
                    props.get("UP_INVERT"), props.get("DN_INVERT"), props.get("LENGTH"),
                    capacity, json.dumps(props),
                ),
            )
            count += 1
            if count % 2000 == 0:
                conn.commit()
    conn.commit()
    return count


def load_catch_basins(conn) -> int:
    count = 0
    with conn.cursor() as cur:
        for feature in _fetch_all(5, "OBJECTID", "manholes"):
            geom = feature.get("geometry")
            props = feature.get("properties", {})
            if not geom or geom.get("type") != "Point":
                continue
            cur.execute(
                """
                INSERT INTO catch_basins (source_city, source_objectid, geom, raw_attrs)
                VALUES (%s, %s, ST_Force2D(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)), %s)
                ON CONFLICT (source_city, source_objectid) DO UPDATE SET
                    geom = EXCLUDED.geom, raw_attrs = EXCLUDED.raw_attrs, loaded_at = now()
                """,
                (_SOURCE_CITY, str(props.get("OBJECTID")), json.dumps(geom), json.dumps(props)),
            )
            count += 1
            if count % 2000 == 0:
                conn.commit()
    conn.commit()
    return count


def main() -> None:
    conn = get_connection()
    if conn is None:
        raise SystemExit("Could not connect -- check DATABASE_URL (.env) and that PostgreSQL is running.")
    try:
        ensure_schema(conn)
        print("Loading Austin pipes...")
        n_pipes = load_pipes(conn)
        print(f"Loaded {n_pipes} pipes.")
        print("Loading Austin manholes...")
        n_basins = load_catch_basins(conn)
        print(f"Loaded {n_basins} manholes.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
