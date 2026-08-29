"""
One-time ETL: loads Anne Arundel County, MD's stormwater Pipe + Inlet
layers into the local PostgreSQL/PostGIS database. Same shape as
load_dc_stormwater.py -- see that script's docstring for the overall
pattern and drainage_db.py's module docstring for why this is a
bulk-load-once script rather than a live per-request fetch.

Source: gis.aacounty.org/arcgis/rest/services/OpenData/Utility_Stormwater_OpenData/MapServer
(layer 5 = Storm Pipes, layer 2 = Storm Inlets -- "Non County" variants,
layers 14/11, cover storm infrastructure not owned/maintained by the
county, e.g. private/state roads; loaded too, tagged the same source_city
since they're still within the county's own geographic area) -- a public
ArcGIS REST service, no API key. maxRecordCount=2000/request.

Unlike DC's Pipe layer, this service has no direct SLOPE field -- only
INVERT_IN/INVERT_OUT/LENGTH. Per this project's simplified
pipe_capacity_m3s (geoai_agent/drainage_db.py), slope is NOT derived from
those inverts -- capacity is only computed for the (here, presumably rare
or zero) records that happen to expose a usable slope some other way, so
most Anne Arundel pipes will have capacity_m3s = NULL. That's expected and
fine: diameter/material are still real, useful data, and
tag_hotspots_with_drainage_capacity's "nearby_pipe_count" (does a
hotspot have ANY pipe nearby) doesn't depend on capacity being computable.

~52,484 county pipes + ~10,700 non-county pipes; ~33,400 county inlets +
~2,300 non-county inlets, as of this writing.

Usage:
    python load_anne_arundel_stormwater.py
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

_BASE_URL = "https://gis.aacounty.org/arcgis/rest/services/OpenData/Utility_Stormwater_OpenData/MapServer"
_PAGE_SIZE = 2000
_SOURCE_CITY = "anne_arundel_county_md"

# (layer_id, label) pairs -- county-maintained pipes/inlets plus their
# "Non County" counterparts (private/state-maintained infrastructure still
# within the county), all merged into the same source_city.
_PIPE_LAYERS = [(5, "county pipes"), (14, "non-county pipes")]
_INLET_LAYERS = [(2, "county inlets"), (11, "non-county inlets")]

_PIPE_FIELDS = "OBJECTID,MATERIAL,VERT_DIAMETER,HORIZ_DIAMETER,BARREL_SHAPE,INVERT_IN,INVERT_OUT,LENGTH"


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
        for layer_id, label in _PIPE_LAYERS:
            for feature in _fetch_all(layer_id, _PIPE_FIELDS, label):
                geom = feature.get("geometry")
                props = feature.get("properties", {})
                if not geom or geom.get("type") != "LineString":
                    continue
                # Vertical/horizontal diameter are equal for the circular
                # pipes seen in samples (BARREL_SHAPE='C') -- fall back to
                # whichever is present for non-circular shapes.
                diameter_in = props.get("VERT_DIAMETER") or props.get("HORIZ_DIAMETER")
                # No SLOPE field on this service -- capacity stays
                # unknown for these records (see module docstring).
                capacity = pipe_capacity_m3s(diameter_in, None)
                object_id = f"{layer_id}:{props.get('OBJECTID')}"  # layer-prefixed: county vs non-county share OBJECTID space
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
                        _SOURCE_CITY, object_id, json.dumps(geom), props.get("MATERIAL"),
                        diameter_in, props.get("BARREL_SHAPE"), None, None,
                        props.get("INVERT_IN"), props.get("INVERT_OUT"), props.get("LENGTH"),
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
        for layer_id, label in _INLET_LAYERS:
            for feature in _fetch_all(layer_id, "OBJECTID", label):
                geom = feature.get("geometry")
                props = feature.get("properties", {})
                if not geom or geom.get("type") != "Point":
                    continue
                object_id = f"{layer_id}:{props.get('OBJECTID')}"
                cur.execute(
                    """
                    INSERT INTO catch_basins (source_city, source_objectid, geom, raw_attrs)
                    VALUES (%s, %s, ST_Force2D(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)), %s)
                    ON CONFLICT (source_city, source_objectid) DO UPDATE SET
                        geom = EXCLUDED.geom, raw_attrs = EXCLUDED.raw_attrs, loaded_at = now()
                    """,
                    (_SOURCE_CITY, object_id, json.dumps(geom), json.dumps(props)),
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
        print("Loading Anne Arundel County pipes...")
        n_pipes = load_pipes(conn)
        print(f"Loaded {n_pipes} pipes.")
        print("Loading Anne Arundel County inlets...")
        n_basins = load_catch_basins(conn)
        print(f"Loaded {n_basins} inlets.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
