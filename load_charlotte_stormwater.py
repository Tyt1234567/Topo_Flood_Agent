"""
One-time ETL: loads Charlotte/Mecklenburg County, NC's stormwater Pipe +
Structure layers into the local PostgreSQL/PostGIS database. Same shape as
load_dc_stormwater.py -- see that script's docstring for the overall
pattern.

Source: gis.charlottenc.gov/arcgis/rest/services/STM/StormPipes/MapServer/0
(pipes) and .../STM/StormStructures/MapServer/0 (catch basins/manholes) --
a public ArcGIS REST service, no API key. maxRecordCount=5000/request.

Better field data than DC's: PACP_Height/PACP_Width (diameter, inches --
PACP is the industry-standard Pipeline Assessment Certification Program
scale) AND a direct SLOPE field (unlike Anne Arundel County, which has
none) -- so a meaningfully larger fraction of Charlotte's pipes get a real
capacity_m3s than DC's or Anne Arundel's. SLOPE can be negative here (a
flow-direction sign convention, not a real uphill grade) -- stored as
abs(SLOPE), same as this project already does for DC's own (rare)
negative-slope records.

~227,782 pipe segments as of this writing.

Usage:
    python load_charlotte_stormwater.py
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

_PIPE_URL = "https://gis.charlottenc.gov/arcgis/rest/services/STM/StormPipes/MapServer/0"
_STRUCTURE_URL = "https://gis.charlottenc.gov/arcgis/rest/services/STM/StormStructures/MapServer/0"
_PAGE_SIZE = 2000
_SOURCE_CITY = "charlotte_nc"

_PIPE_FIELDS = "OBJECTID,MATERIAL,PACP_Height,PACP_Width,PI_SHAPE,SLOPE,US_INVERT,DS_INVERT,measuredlength"


def _query_page(base_url: str, out_fields: str, offset: int) -> dict:
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
    url = f"{base_url}/query?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())


def _fetch_all(base_url: str, out_fields: str, label: str):
    offset = 0
    while True:
        page = _query_page(base_url, out_fields, offset)
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
        for feature in _fetch_all(_PIPE_URL, _PIPE_FIELDS, "pipes"):
            geom = feature.get("geometry")
            props = feature.get("properties", {})
            if not geom or geom.get("type") != "LineString":
                continue
            diameter_in = props.get("PACP_Height") or props.get("PACP_Width")
            slope = props.get("SLOPE")
            slope = abs(slope) if slope is not None else None
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
                    diameter_in, props.get("PI_SHAPE"), slope, "attribute" if slope is not None else None,
                    props.get("US_INVERT"), props.get("DS_INVERT"), props.get("measuredlength"),
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
        for feature in _fetch_all(_STRUCTURE_URL, "OBJECTID", "structures"):
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
        print("Loading Charlotte pipes...")
        n_pipes = load_pipes(conn)
        print(f"Loaded {n_pipes} pipes.")
        print("Loading Charlotte storm structures...")
        n_basins = load_catch_basins(conn)
        print(f"Loaded {n_basins} structures.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
