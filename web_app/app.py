"""
Web UI for the GeoAI Agent: Flask backend + JS/Leaflet frontend.

One panel, backed by JSON API endpoints, over live real-world data: a
lat/lon bounding box and topic (elevation, temperature, precipitation,
wind, humidity, pressure -- live-fetched, never pre-downloaded), fillable
either by hand or from a free-text query via Smart Query's LLM-based
understanding step, plus any number of user-uploaded CSV/shapefile layers,
all over a real OpenStreetMap basemap. One bounds/topic state, one
"Fetch & Analyze" action -- Smart Query only ever fills in the fields this
endpoint reads.

This folder is self-contained: geoai_agent/ below is a standalone copy of
the analysis package (not an import of anything outside web_app/), and
examples/ holds its own sample data. Run with:
    python app.py
from inside web_app/, or:
    python web_app/app.py
from the project root -- both work, since Python always puts the launched
script's own directory on sys.path first.
"""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

from flask import Flask, jsonify, render_template, request

from geoai_agent import prompts
from geoai_agent.llm_client import LLMClient
from geoai_agent.morse_topology import analyze_scalar_field
from geoai_agent.real_data import (
    MAX_USGS_GRID_RESOLUTION,
    OPEN_METEO_VARIABLES,
    bounding_box_around_point,
    build_real_world_dataset,
    geocode_place,
    ground_spacing_meters,
    is_in_continental_us,
    resolution_for_target_spacing,
)
from geoai_agent.visualize import critical_points_to_geojson, field_to_geojson

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def _layer_summary(result: dict) -> dict:
    return {
        "value_range": result["value_range"],
        "persistence_threshold": result["persistence_threshold"],
        "num_significant_basins": result["num_significant_basins"],
        "num_significant_peak_basins": result["num_significant_peak_basins"],
        "euler_characteristic": result["euler_characteristic"],
    }


# -- Real-World Data: one bounding box, one topic, any number of extra ---
# user-supplied layers. Smart Query only ever fills in the bounds/topic
# fields these endpoints read, it never fetches on its own, so there is
# exactly one bounds/topic state in the UI.
#
# Two-phase, so raw data loads as soon as a location is confirmed rather
# than only at the very end of the pipeline:
#   /load_data -- fetches the raw scalar field(s) only (no critical-point
#     analysis, no recommendation) and caches the result server-side under
#     a dataset_id, so the map can show the real heatmap immediately after
#     a place is resolved.
#   /analyze -- runs the actual Morse analysis (+ optional recommendation).
#     If given a still-valid dataset_id it reuses that cached raw data
#     instead of re-fetching (avoiding a second round of live API calls,
#     and Open-Meteo's per-minute rate limit); otherwise it fetches fresh,
#     so it still works standalone (e.g. after editing bounds by hand).

_DATASET_CACHE_TTL_SECONDS = 15 * 60
_dataset_cache: dict[str, tuple[float, object]] = {}


def _cache_dataset(dataset) -> str:
    now = time.time()
    for key in [k for k, (ts, _) in _dataset_cache.items() if now - ts > _DATASET_CACHE_TTL_SECONDS]:
        del _dataset_cache[key]
    dataset_id = uuid.uuid4().hex
    _dataset_cache[dataset_id] = (now, dataset)
    return dataset_id


def _get_cached_dataset(dataset_id: str):
    entry = _dataset_cache.get(dataset_id)
    if entry is None:
        return None
    ts, dataset = entry
    if time.time() - ts > _DATASET_CACHE_TTL_SECONDS:
        del _dataset_cache[dataset_id]
        return None
    return dataset


def _parse_realworld_form():
    """Shared bounds/resolution/topic/custom-layer parsing for both
    /load_data and /analyze. Returns (bounds, resolution, primary_variable,
    layer_sources, layer_descriptions, tmp_paths, elevation_source) or an
    (error, status) tuple if the form was invalid."""
    bounds_raw = request.form.get("bounds", "")
    try:
        bounds = tuple(float(x) for x in bounds_raw.split(","))
        if len(bounds) != 4:
            raise ValueError
    except ValueError:
        return {"error": "bounds must be 'lat_min,lat_max,lon_min,lon_max'"}, 400

    # Resolution is specified as a physical ground distance (pixel size, in
    # meters) rather than a raw grid point count -- resolution_for_target_
    # spacing converts it to the (nx, ny) point count needed to hit that
    # spacing over this bounding box, each axis independently clamped to
    # [MIN_GRID_RESOLUTION, MAX_GRID_RESOLUTION]. 80m matches the frontend's
    # default and is a reasonable default pixel size for a typical
    # neighborhood-scale box.
    try:
        resolution_meters = float(request.form.get("resolution_meters", 80))
    except ValueError:
        return {"error": "resolution_meters must be a number"}, 400
    if resolution_meters <= 0:
        return {"error": "resolution_meters must be positive"}, 400
    resolution = resolution_for_target_spacing(bounds, resolution_meters)

    primary_variable = request.form.get("primary_variable", "elevation")
    if primary_variable not in OPEN_METEO_VARIABLES:
        return {"error": f"unknown topic '{primary_variable}'"}, 400

    # "auto" (the default): try USGS 3DEP first for elevation, falling back
    # to Open-Meteo automatically -- see build_real_world_dataset.
    elevation_source = request.form.get("elevation_source", "auto")
    if elevation_source not in ("auto", "open_meteo", "usgs"):
        elevation_source = "auto"
    if elevation_source == "usgs":
        nx, ny = resolution
        resolution = (min(nx, MAX_USGS_GRID_RESOLUTION), min(ny, MAX_USGS_GRID_RESOLUTION))

    layer_sources: dict[str, str | None] = {primary_variable: None}
    layer_descriptions: dict[str, str] = {}
    tmp_paths = []
    for name in request.form.getlist("layer_name"):
        name = name.strip()
        if not name:
            continue
        file = request.files.get(f"layer_file::{name}")
        if file and file.filename:
            suffix = Path(file.filename).suffix.lower() or ".csv"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            file.save(tmp.name)
            tmp.close()
            tmp_paths.append(tmp.name)
            layer_sources[name] = tmp.name  # a layer named e.g. "elevation" overrides the live fetch above
        description = request.form.get(f"layer_description::{name}", "").strip()
        if description:
            layer_descriptions[name] = description

    if request.form.get("use_sample_layer") == "true":
        layer_sources["hazard_survey"] = str(APP_DIR / "examples" / "sample_hazard_layer.csv")

    return bounds, resolution, primary_variable, layer_sources, layer_descriptions, tmp_paths, elevation_source


@app.route("/api/realworld/load_data", methods=["POST"])
def realworld_load_data():
    parsed = _parse_realworld_form()
    if isinstance(parsed[0], dict):  # error tuple
        return jsonify(parsed[0]), parsed[1]
    bounds, resolution, primary_variable, layer_sources, layer_descriptions, tmp_paths, elevation_source = parsed

    try:
        dataset = build_real_world_dataset(
            "web_request", bounds, layer_sources, resolution=resolution,
            layer_descriptions=layer_descriptions, elevation_source=elevation_source,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        for p in tmp_paths:
            Path(p).unlink(missing_ok=True)

    dataset_id = _cache_dataset(dataset)
    layers = {name: {"field": field_to_geojson(field)} for name, field in dataset.scalar_fields.items()}
    fell_back = primary_variable == "elevation" and elevation_source == "open_meteo" and dataset.elevation_source == "usgs"

    lat_min, lat_max, lon_min, lon_max = bounds
    return jsonify({
        "dataset_id": dataset_id,
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "layers": layers,
        "elevation_source": dataset.elevation_source,
        "elevation_source_fallback_note": (
            "Open-Meteo's elevation API failed (rate limit or outage), so this automatically "
            "used USGS 3DEP instead."
        ) if fell_back else None,
    })


@app.route("/api/realworld/analyze", methods=["POST"])
def realworld_analyze():
    dataset_id = request.form.get("dataset_id", "").strip()
    cached = _get_cached_dataset(dataset_id) if dataset_id else None

    if cached is not None:
        dataset = cached
        bounds = dataset.bounds
        primary_variable = request.form.get("primary_variable", "elevation")
        layer_sources = dataset.layer_sources
        layer_descriptions = dataset.layer_descriptions
        elevation_source = dataset.elevation_source
        tmp_paths = []
    else:
        parsed = _parse_realworld_form()
        if isinstance(parsed[0], dict):
            return jsonify(parsed[0]), parsed[1]
        bounds, resolution, primary_variable, layer_sources, layer_descriptions, tmp_paths, elevation_source = parsed

        try:
            dataset = build_real_world_dataset(
                "web_request", bounds, layer_sources, resolution=resolution,
                layer_descriptions=layer_descriptions, elevation_source=elevation_source,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        finally:
            for p in tmp_paths:
                Path(p).unlink(missing_ok=True)
        elevation_source = dataset.elevation_source  # may have been upgraded by an automatic fallback

    results = {name: analyze_scalar_field(field) for name, field in dataset.scalar_fields.items()}

    recommendation = None
    if request.form.get("recommend") == "true" and primary_variable in results:
        try:
            llm = LLMClient()
            if layer_sources.get(primary_variable):
                live_source = "a user-uploaded file"
            elif primary_variable == "elevation" and elevation_source == "usgs":
                live_source = "USGS's 3DEP Elevation Point Query Service (genuinely LiDAR-derived, not pre-downloaded or cached)"
            else:
                live_source = "the live Open-Meteo API (not pre-downloaded or cached)"
            context = (
                f"'{primary_variable}' data for this run came from {live_source}. Basins/peaks were "
                "computed by persistence-guided steepest-descent/ascent watershed directly on the raw "
                "field, with NO heuristic depression-filling or breaching preprocessing.\n"
                f"Morse analysis result: {json.dumps(results[primary_variable], ensure_ascii=False)}"
            )
            if layer_descriptions:
                context += (
                    "\nUser-supplied descriptions of other layers analyzed alongside this one "
                    f"(for context, not necessarily this result): {json.dumps(layer_descriptions, ensure_ascii=False)}"
                )
            feedback = request.form.get("feedback", "").strip()
            if feedback:
                context += "\n\n" + prompts.RERUN_CONTEXT_TEMPLATE.format(
                    feedback=feedback,
                    original_query=f"Analyze '{primary_variable}' over bounds {bounds}",
                )
            recommendation = llm.chat(prompts.MORSE_RECOMMENDATION_SYSTEM, context)
        except Exception as exc:
            recommendation = f"(recommendation unavailable: {exc})"

    layers = {}
    for name, field in dataset.scalar_fields.items():
        result = results[name]
        summary = _layer_summary(result)
        # the grid actually achieved -- may be coarser than what was
        # requested if a source (e.g. USGS after an Open-Meteo fallback)
        # couldn't deliver the full resolution, so the frontend can show
        # this rather than silently rendering a coarser result unexplained.
        summary["shape"] = list(field.shape)
        layers[name] = {
            "field": field_to_geojson(field),
            "points": critical_points_to_geojson(result),
            "summary": summary,
        }

    lat_min, lat_max, lon_min, lon_max = bounds
    return jsonify({
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "layers": layers,
        "recommendation": recommendation,
        "elevation_source": elevation_source,
    })


# -- Smart Query: understand a free-text query, don't fetch anything -----
#
# An LLM extraction call (prompts.REALWORLD_QUERY_EXTRACTION_SYSTEM) works
# out what place/coordinates and which live variable (elevation,
# temperature, precipitation, wind, humidity, pressure -- see
# geoai_agent.real_data.OPEN_METEO_VARIABLES) a free-text query is about --
# natural-language phrasing (prepositions, capitalization, trailing
# qualifier clauses like "with a resolution of 30 meters") varies too much
# for a hand-written pattern to keep up with reliably. A named place is
# then live-geocoded to coordinates. This endpoint only ever returns what
# was understood -- it never fetches scalar-field data itself. The actual
# fetch+analyze is a single, separate action: /api/realworld/analyze,
# which the frontend fills in from this response's bounds/variable rather
# than triggering here, so there is exactly one bounds/topic state and one
# "run" button in the UI.

@app.route("/api/realworld/smart_query", methods=["POST"])
def realworld_smart_query():
    query = request.form.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        extraction = LLMClient().chat_json(prompts.REALWORLD_QUERY_EXTRACTION_SYSTEM, query)
    except Exception as exc:
        return jsonify({"error": f"query understanding failed: {exc}"}), 502

    variable = extraction.get("variable")
    if variable not in OPEN_METEO_VARIABLES:
        variable = "elevation"

    explicit_bounds = extraction.get("explicit_bounds")
    resolved_place = None
    if explicit_bounds and len(explicit_bounds) == 4:
        bounds = tuple(float(x) for x in explicit_bounds)
    else:
        place_name = extraction.get("place_name")
        geocoded = geocode_place(place_name) if place_name else None
        if geocoded is None:
            return jsonify({
                "error": (
                    f"Couldn't find a real-world location in that text"
                    f"{f' (understood it as \"{place_name}\")' if place_name else ''}. "
                    "Try naming a place more explicitly (e.g. \"...in Boulder, Colorado\"), "
                    "an explicit lat/lon box like '40.96N-41.15N, 75.15W-74.95W', "
                    "or use the manual Bounds field below instead."
                )
            }), 400
        bounds = bounding_box_around_point(geocoded["latitude"], geocoded["longitude"])
        resolved_place = geocoded

    resolution = None
    achieved_meters = None  # (lon_spacing_m, lat_spacing_m) -- the two axes are no longer forced equal
    requested_meters = extraction.get("target_resolution_meters")
    if requested_meters:
        try:
            requested_meters = float(requested_meters)
            resolution = resolution_for_target_spacing(bounds, requested_meters)
            achieved_meters = list(ground_spacing_meters(bounds, resolution))
        except (TypeError, ValueError):
            requested_meters = None

    # Source selection: for elevation, always try USGS 3DEP first
    # (genuinely higher-precision, often ~1m LiDAR-derived), falling back
    # to Open-Meteo automatically if USGS fails or the location has no
    # 3DEP coverage -- see build_real_world_dataset's elevation_source="auto".
    elevation_source = "auto" if variable == "elevation" else None
    source_note = None
    if variable == "elevation":
        if is_in_continental_us(bounds):
            source_note = (
                "Elevation requests try USGS 3DEP first (genuinely LiDAR-derived, often ~1m "
                "precision) -- USGS has no batch endpoint, so this can take up to a minute or "
                "two, and automatically falls back to Open-Meteo if USGS is unavailable."
            )
        else:
            source_note = "Using Open-Meteo (USGS 3DEP only covers the continental US)."

    lat_min, lat_max, lon_min, lon_max = bounds
    return jsonify({
        "bounds": [lat_min, lat_max, lon_min, lon_max],
        "variable": variable,
        "available_variables": list(OPEN_METEO_VARIABLES),
        "query": query,
        "resolved_place": resolved_place,
        "resolution": resolution,
        "requested_resolution_meters": requested_meters,
        "achieved_resolution_meters": achieved_meters,
        "elevation_source": elevation_source,
        "source_note": source_note,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
