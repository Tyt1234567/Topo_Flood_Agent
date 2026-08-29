"""
One-time ETL: loads Prince George's County, Maryland's full stormwater
conveyance (pipe) + structure (catch basin) layers into the local
PostgreSQL/PostGIS database -- same shape as load_dc_stormwater.py, see
that script's docstring and geoai_agent/drainage_db.py's module docstring
for the overall why. This is the county College Park, MD sits in -- the
place named in this session's demo request has no city-level GIS pipe
data of its own, but the county's DPWT (Department of Public Works and
Transportation) storm-drain-infrastructure layer covers it directly.

Source: gis.princegeorgescountymd.gov/arcgis/rest/services/DPWT_SDI/DPWT_SDI/MapServer
(layer 1 = CONVEYANCE (pipes), layer 0 = STRUCTURES (catch basins)) -- a
public ArcGIS REST service, no API key.

Important difference from DC's data: this layer has NO diameter or
material field at all (checked directly against the live schema -- just
type/status/owner/invert elevations/length/plan_length) -- so
pipe_capacity_m3s(diameter_in, slope) will always return None for every
row here; capacity_m3s is stored as NULL for the whole county. That's
fine and expected, not a bug: per this session's simplified design (see
drainage_db.py), a hotspot's PRIMARY drainage-infrastructure signal is
just whether it has a real pipe nearby at all ("nearby_pipe_count"),
independent of whether that pipe's capacity happens to be computable --
this county's data is a legitimate contributor to that presence/absence
signal even without ever contributing a capacity number.

County-wide: 89,453 pipe segments and 100,193 structures as of this
writing, paginated 2000 records/request (this service's own cap, higher
than DC's 1000) -- run this once (or re-run to refresh; upserts on
(source_city, source_objectid) so it's safe to run repeatedly).

Usage:
    python load_pgcounty_stormwater.py
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

_BASE_URL = "https://gis.princegeorgescountymd.gov/arcgis/rest/services/DPWT_SDI/DPWT_SDI/MapServer"
_PAGE_SIZE = 1000  # this service times out more than DC's under the full 2000-record cap it advertises
_SOURCE_CITY = "pgcounty_md"


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
    # This service is noticeably flakier than DC's under a 2000-record page
    # with geometry (observed directly: repeated read timeouts even on the
    # very first page) -- retry with backoff rather than treating a
    # transient hiccup as fatal for the whole multi-thousand-record load.
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                return json.loads(resp.read())
        except (TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise last_error


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
    fields = "objectid,type,invert_in_elevation,invert_out_elevation,plan_length"
    count = 0
    with conn.cursor() as cur:
        for feature in _fetch_all(1, fields, "pipes"):
            geom = feature.get("geometry")
            props = feature.get("properties", {})
            if not geom or geom.get("type") != "LineString":
                continue
            # No diameter/material/slope field exists in this source at all
            # (verified against the live schema) -- capacity is simply
            # unknown for every row, by design, not a parsing gap.
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
                    _SOURCE_CITY, str(props.get("objectid")), json.dumps(geom), props.get("type"),
                    None, None, None, None,
                    props.get("invert_in_elevation"), props.get("invert_out_elevation"), props.get("plan_length"),
                    None, json.dumps(props),
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
        for feature in _fetch_all(0, "objectid", "catch basins"):
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
                (_SOURCE_CITY, str(props.get("objectid")), json.dumps(geom), json.dumps(props)),
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
        print("Loading Prince George's County pipes...")
        n_pipes = load_pipes(conn)
        print(f"Loaded {n_pipes} pipes.")
        print("Loading Prince George's County catch basins...")
        n_basins = load_catch_basins(conn)
        print(f"Loaded {n_basins} catch basins.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
