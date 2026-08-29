"""
One-time ETL: loads Loudoun County, VA's stormwater Pipe + Structure
layers into the local PostgreSQL/PostGIS database. Same shape as
load_dc_stormwater.py -- see that script's docstring for the overall
pattern.

Source: logis.loudoun.gov/gis/rest/services/COL/Utilities/MapServer
(layer 4 = Stormwater Pipes, layer 3 = Stormwater Structures) -- a public
ArcGIS REST service, no API key. maxRecordCount=2000/request.

Minimal schema compared to DC/Charlotte -- only Pipe_Diam (inches) and
Pipe_Type (material code, e.g. "RCP"), no slope or invert fields at all.
capacity_m3s will be NULL for every pipe here (pipe_capacity_m3s needs a
slope, which this source simply doesn't have) -- that's fine and expected,
per drainage_db.py's module docstring: presence (nearby_pipe_count) is the
primary, always-available signal regardless of whether capacity can be
computed, and diameter/material are still real, useful data on their own.

~66,501 pipe segments + ~78,132 structures as of this writing.

Usage:
    python load_loudoun_stormwater.py
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

_BASE_URL = "https://logis.loudoun.gov/gis/rest/services/COL/Utilities/MapServer"
_PAGE_SIZE = 2000
_SOURCE_CITY = "loudoun_county_va"


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
        for feature in _fetch_all(4, "OBJECTID,Pipe_Diam,Pipe_Type,Feature", "pipes"):
            geom = feature.get("geometry")
            props = feature.get("properties", {})
            if not geom or geom.get("type") != "LineString":
                continue
            diameter_in = props.get("Pipe_Diam")
            capacity = pipe_capacity_m3s(diameter_in, None)  # no slope field on this source -- always None
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
                    _SOURCE_CITY, str(props.get("OBJECTID")), json.dumps(geom), props.get("Pipe_Type"),
                    diameter_in, None, None, None, None, None, None,
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
        for feature in _fetch_all(3, "OBJECTID", "structures"):
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
        print("Loading Loudoun County pipes...")
        n_pipes = load_pipes(conn)
        print(f"Loaded {n_pipes} pipes.")
        print("Loading Loudoun County structures...")
        n_basins = load_catch_basins(conn)
        print(f"Loaded {n_basins} structures.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
