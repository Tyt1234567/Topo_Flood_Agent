"""
One-time ETL: loads Washington DC's full stormwater Pipe + Catch Basin
layers into the local PostgreSQL/PostGIS database (geoai_agent/
drainage_db.py's read side then serves them to the app, city-wide, without
any further live network calls at analysis time).

Source: gis.streamstats.usgs.gov/arcgis/rest/services/stormdrain/dc/MapServer
(layer 2 = Pipe, layer 0 = Catch Basin) -- a public ArcGIS REST service,
no API key. See drainage_db.py's module docstring for why this is a
bulk-load-once script rather than a live per-request fetch like every
other data source in this project: there's no live API with the coverage/
reliability to query per-analysis-run the way FEMA's NFHL or USGS
elevation can be, so the data has to live somewhere -- Postgres, loaded
once, is that somewhere.

City-wide: 80,550 pipe segments and 38,173 catch basins as of this
writing, paginated 1000 records/request (the service's own cap) -- run
this once (or re-run to refresh; upserts on (source_city, source_objectid)
so it's safe to run repeatedly), not per web request.

Usage:
    python load_dc_stormwater.py
Requires DATABASE_URL to be set (see .env / geoai_agent/drainage_db.py).

To add another city later: write a sibling script following this same
shape (fetch each layer's schema first via .../MapServer/<id>?f=json to
find the right field names -- they vary a lot between cities' GIS systems
-- then map that city's fields onto the same pipes/catch_basins columns
this script writes, tagging source_city with that city's name).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

from dotenv import load_dotenv

load_dotenv()

from geoai_agent.drainage_db import ensure_schema, get_connection, pipe_capacity_m3s

_BASE_URL = "https://gis.streamstats.usgs.gov/arcgis/rest/services/stormdrain/dc/MapServer"
_PAGE_SIZE = 1000
_SOURCE_CITY = "dc"


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
    fields = "OBJECTID,MATRL,DIAHT,WIDTH,XSECTSHP,SLOPE,USINVERT,DSINVERT,LENGTH"
    count = 0
    with conn.cursor() as cur:
        for feature in _fetch_all(2, fields, "pipes"):
            geom = feature.get("geometry")
            props = feature.get("properties", {})
            if not geom or geom.get("type") != "LineString":
                continue
            diameter_in = props.get("DIAHT") or props.get("WIDTH")
            # The pipe's own SLOPE field, used as-is -- no falling back to a
            # slope derived from invert elevations when it's missing (per
            # geoai_agent/drainage_db.py's simplified pipe_capacity_m3s: a
            # pipe either has a directly usable diameter+slope or its
            # capacity is just unknown, no extra derivation machinery).
            slope = props.get("SLOPE")
            slope = abs(slope) if slope is not None else None
            slope_source = "attribute" if slope is not None else None
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
                    _SOURCE_CITY, str(props.get("OBJECTID")), json.dumps(geom), props.get("MATRL"),
                    diameter_in, props.get("XSECTSHP"), slope, slope_source,
                    props.get("USINVERT"), props.get("DSINVERT"), props.get("LENGTH"),
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
        for feature in _fetch_all(0, "OBJECTID", "catch basins"):
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
        print("Loading DC pipes...")
        n_pipes = load_pipes(conn)
        print(f"Loaded {n_pipes} pipes.")
        print("Loading DC catch basins...")
        n_basins = load_catch_basins(conn)
        print(f"Loaded {n_basins} catch basins.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
