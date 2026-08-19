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

import datetime
import json
import tempfile
import time
import uuid
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

from flask import Flask, Response, jsonify, render_template, request

from geoai_agent import prompts
from geoai_agent.boundary import clip_field_to_polygon, geocode_place_boundary, tag_hotspots_within_boundary
from geoai_agent.flood_tools import RAINFALL_SCENARIOS, WEIGHT_PRESETS, rank_flood_basins
from geoai_agent.llm_client import LLMClient
from geoai_agent.report import all_minima_geojson, elevation_hillshade_geojson, render_flood_report_html, terrain_legend_stops
from geoai_agent.ttk_engine import analyze_scalar_field_ttk
from geoai_agent.real_data import (
    MAX_USGS_GRID_RESOLUTION,
    OPEN_METEO_VARIABLES,
    bounding_box_around_point,
    build_real_world_dataset,
    geocode_place,
    ground_spacing_meters,
    is_in_continental_us,
    resolution_for_target_spacing,
    resolution_for_target_spacing_scaled,
    scalar_field_from_upload,
)
from geoai_agent.visualize import critical_points_to_geojson, field_to_geojson, flood_hotspots_to_geojson

app = Flask(__name__)


class _StepTracker:
    """Records each stage of a request's pipeline (what ran, how long it
    took, and a short JSON-able detail payload) so the frontend can show
    the agent's analysis process step by step -- the same information the
    CLI agent (agent.py's `_log`) prints to the console, surfaced here as
    structured data instead of stdout text, since a web request has no
    terminal for the user to watch."""

    def __init__(self):
        self.steps: list[dict] = []

    def step(self, label: str, detail: dict | str, started_at: float) -> None:
        self.steps.append({
            "label": label,
            "detail": detail,
            "elapsed_seconds": round(time.time() - started_at, 2),
        })


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
    # [MIN_GRID_RESOLUTION, MAX_GRID_RESOLUTION]. 100m (one pixel = 100m x
    # 100m) matches the frontend's default and is a reasonable default pixel
    # size for a typical neighborhood-scale box.
    try:
        resolution_meters = float(request.form.get("resolution_meters", 100))
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
    if elevation_source not in ("auto", "open_meteo", "usgs", "usgs_dem_cog"):
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


def _describe_elevation_source(elevation_source: str) -> str:
    """Plain-language description of where 'elevation' data actually came
    from, for the LLM recommendation context -- shared between /analyze
    and the one-box agent flow so the same source gets the same wording."""
    if elevation_source == "usgs_dem_cog":
        return (
            "the USGS 3DEP 1/3 arc-second (~10m) DEM, streamed directly from its source GeoTIFF "
            "and clipped to this exact area via GDAL (not pre-downloaded, not point-sampled)"
        )
    if elevation_source == "usgs":
        return "USGS's 3DEP Elevation Point Query Service (genuinely LiDAR-derived, not pre-downloaded or cached)"
    return "the live Open-Meteo API (not pre-downloaded or cached)"


def _elevation_fallback_note(dataset, primary_variable: str) -> str | None:
    """Explains WHY the elevation source/grid actually used differs from
    what "auto" prefers: within the continental US, "auto" tries the
    streamed USGS 3DEP DEM COG first (dem_cog_source.py -- no point-count
    cap, no per-point rate limit); dataset.elevation_fallback_reason is
    only ever set (see build_real_world_dataset) when that first attempt
    failed and a later one in the chain was used instead, so gating on it
    directly also means an explicit single-source request (which has no
    fallback chain to begin with) never produces a note here."""
    if primary_variable != "elevation" or not dataset.elevation_fallback_reason:
        return None
    reason = f" ({dataset.elevation_fallback_reason})"
    if dataset.elevation_source == "usgs":
        return (
            f"The streamed USGS 3DEP DEM source failed{reason}, so this automatically fell back to "
            f"USGS's Elevation Point Query Service -- which caps at {MAX_USGS_GRID_RESOLUTION}x"
            f"{MAX_USGS_GRID_RESOLUTION} points (one HTTP request per point, no batch endpoint), so "
            "the grid actually returned may be smaller than the requested resolution."
        )
    if dataset.elevation_source == "open_meteo":
        return (
            f"The streamed USGS 3DEP DEM source failed{reason}, so this automatically fell back to "
            "the lower-precision, global Open-Meteo elevation API (~90m) instead."
        )
    return None


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
    layers = {
        name: {"field": field_to_geojson(field), "shape": list(field.shape)}
        for name, field in dataset.scalar_fields.items()
    }

    lat_min, lat_max, lon_min, lon_max = bounds
    return jsonify({
        "dataset_id": dataset_id,
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "layers": layers,
        "elevation_source": dataset.elevation_source,
        "elevation_source_fallback_note": _elevation_fallback_note(dataset, primary_variable),
    })


@app.route("/api/realworld/analyze", methods=["POST"])
def realworld_analyze():
    dataset_id = request.form.get("dataset_id", "").strip()
    cached = _get_cached_dataset(dataset_id) if dataset_id else None
    # what the user actually asked for -- kept separate from `elevation_source`
    # below, which gets overwritten with what was ACTUALLY used, so the
    # fallback note can still tell an explicit "usgs" choice (no surprise)
    # from an "auto"/"open_meteo" request that silently landed on usgs.
    requested_elevation_source = request.form.get("elevation_source", "auto")
    if requested_elevation_source not in ("auto", "open_meteo", "usgs", "usgs_dem_cog"):
        requested_elevation_source = "auto"

    tracker = _StepTracker()

    if cached is not None:
        dataset = cached
        bounds = dataset.bounds
        primary_variable = request.form.get("primary_variable", "elevation")
        layer_sources = dataset.layer_sources
        layer_descriptions = dataset.layer_descriptions
        elevation_source = dataset.elevation_source
        tmp_paths = []
        tracker.step("Fetch data", {
            "reused_cached_dataset": True, "elevation_source": elevation_source,
            "layers": {n: list(f.shape) for n, f in dataset.scalar_fields.items()},
        }, time.time())
    else:
        parsed = _parse_realworld_form()
        if isinstance(parsed[0], dict):
            return jsonify(parsed[0]), parsed[1]
        bounds, resolution, primary_variable, layer_sources, layer_descriptions, tmp_paths, elevation_source = parsed

        t_fetch = time.time()
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
        tracker.step("Fetch data", {
            "requested_source": requested_elevation_source, "actual_source": elevation_source,
            "fallback_reason": dataset.elevation_fallback_reason,
            "bounds": list(bounds), "layers": {n: list(f.shape) for n, f in dataset.scalar_fields.items()},
        }, t_fetch)

    t_analyze = time.time()
    results = {name: analyze_scalar_field_ttk(field) for name, field in dataset.scalar_fields.items()}
    tracker.step("Morse-Smale analysis (TTK)", {
        name: {
            "grid_shape": r["grid_shape"], "engine": r["engine"],
            "persistence_threshold": r["persistence_threshold"],
            "num_significant_basins": r["num_significant_basins"], "num_raw_minima": r["num_raw_minima"],
            "num_significant_peak_basins": r["num_significant_peak_basins"], "num_raw_maxima": r["num_raw_maxima"],
        }
        for name, r in results.items()
    }, t_analyze)

    recommendation = None
    if request.form.get("recommend") == "true" and primary_variable in results:
        t_recommend = time.time()
        try:
            llm = LLMClient()
            if layer_sources.get(primary_variable):
                live_source = "a user-uploaded file"
            elif primary_variable == "elevation":
                live_source = _describe_elevation_source(elevation_source)
            else:
                live_source = "the live Open-Meteo API (not pre-downloaded or cached)"
            context = (
                f"'{primary_variable}' data for this run came from {live_source}. Basins/peaks were "
                "computed by TTK (Topology ToolKit)'s Morse-Smale complex: persistence-based "
                "topological simplification at the reported threshold, then critical points and "
                "descending/ascending manifolds (basins/peak catchments) on the simplified field -- "
                "no heuristic depression-filling or breaching preprocessing.\n"
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
            tracker.step("LLM recommendation", {"model": llm.model, "source_described_as": live_source}, t_recommend)
        except Exception as exc:
            recommendation = f"(recommendation unavailable: {exc})"
            tracker.step("LLM recommendation", {"error": str(exc)}, t_recommend)

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
        "elevation_source_fallback_note": _elevation_fallback_note(dataset, primary_variable),
        "pipeline_steps": tracker.steps,
    })


# -- Flood & Drainage Analysis: HITL rainfall-scenario / weight-preset ---
# controls over the same elevation layer -- a Track-A-style domain module
# on top of the generic Morse analysis above: every significant basin
# (topological depression) becomes a candidate flood hotspot, ranked by a
# weighted risk score and given a Rational-Method drainage demand under
# the selected design storm. See geoai_agent/flood_tools.py.
#
# Reuses the same dataset_id cache as /analyze (a location loaded via
# /load_data or /analyze also serves this), so switching between the
# generic Morse view and this one doesn't re-fetch live data.

@app.route("/api/realworld/flood_analyze", methods=["POST"])
def realworld_flood_analyze():
    dataset_id = request.form.get("dataset_id", "").strip()
    cached = _get_cached_dataset(dataset_id) if dataset_id else None
    requested_elevation_source = request.form.get("elevation_source", "auto")
    if requested_elevation_source not in ("auto", "open_meteo", "usgs", "usgs_dem_cog"):
        requested_elevation_source = "auto"

    tracker = _StepTracker()

    if cached is not None:
        dataset = cached
        bounds = dataset.bounds
        tmp_paths = []
        tracker.step("Fetch data", {
            "reused_cached_dataset": True, "elevation_source": dataset.elevation_source,
            "layers": {n: list(f.shape) for n, f in dataset.scalar_fields.items()},
        }, time.time())
    else:
        parsed = _parse_realworld_form()
        if isinstance(parsed[0], dict):
            return jsonify(parsed[0]), parsed[1]
        bounds, resolution, primary_variable, layer_sources, layer_descriptions, tmp_paths, elevation_source = parsed
        layer_sources = {**layer_sources, "elevation": layer_sources.get("elevation") or None}
        t_fetch = time.time()
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
        tracker.step("Fetch data", {
            "requested_source": requested_elevation_source, "actual_source": dataset.elevation_source,
            "fallback_reason": dataset.elevation_fallback_reason,
            "bounds": list(bounds), "layers": {n: list(f.shape) for n, f in dataset.scalar_fields.items()},
        }, t_fetch)

    elevation_field = dataset.scalar_fields.get("elevation")
    if elevation_field is None:
        return jsonify({
            "error": "No elevation layer is loaded for this location -- flood analysis needs terrain "
                     "elevation. Set Topic to \"elevation\" and Fetch & Analyze first."
        }), 400

    rainfall_scenario = request.form.get("rainfall_scenario", "10yr")
    if rainfall_scenario not in RAINFALL_SCENARIOS:
        return jsonify({"error": f"unknown rainfall_scenario, available: {list(RAINFALL_SCENARIOS)}"}), 400
    weight_preset = request.form.get("weight_preset", "balanced")
    if weight_preset not in WEIGHT_PRESETS:
        return jsonify({"error": f"unknown weight_preset, available: {list(WEIGHT_PRESETS)}"}), 400
    try:
        runoff_coefficient = float(request.form.get("runoff_coefficient", 0.5))
    except ValueError:
        return jsonify({"error": "runoff_coefficient must be a number"}), 400

    slope_field = dataset.scalar_fields.get("slope")
    exposure_field = dataset.scalar_fields.get("exposure") or dataset.scalar_fields.get("population")

    t_analyze = time.time()
    morse_result = analyze_scalar_field_ttk(elevation_field)
    tracker.step("Morse-Smale analysis (TTK)", {
        "grid_shape": morse_result["grid_shape"], "persistence_threshold": morse_result["persistence_threshold"],
        "num_significant_basins": morse_result["num_significant_basins"], "num_raw_minima": morse_result["num_raw_minima"],
    }, t_analyze)

    t_rank = time.time()
    flood = rank_flood_basins(
        elevation_field, morse_result, rainfall_scenario=rainfall_scenario, weights=weight_preset,
        runoff_coefficient=runoff_coefficient, slope_field=slope_field, exposure_field=exposure_field,
    )
    tracker.step("Flood risk ranking", {
        "rainfall_scenario": rainfall_scenario, "weight_preset": weight_preset,
        "runoff_coefficient": runoff_coefficient, "num_hotspots": len(flood["hotspots"]),
        "slope_layer_available": slope_field is not None, "exposure_layer_available": exposure_field is not None,
    }, t_rank)

    recommendation = None
    if request.form.get("recommend") == "true":
        t_recommend = time.time()
        try:
            llm = LLMClient()
            context = (
                f"Weight preset used: '{weight_preset}'. Slope layer available: {slope_field is not None}. "
                f"Exposure layer available: {exposure_field is not None}.\n"
                f"Flood risk ranking result: {json.dumps(flood, ensure_ascii=False)}"
            )
            feedback = request.form.get("feedback", "").strip()
            if feedback:
                context += "\n\n" + prompts.RERUN_CONTEXT_TEMPLATE.format(
                    feedback=feedback,
                    original_query=f"Rank flood risk / drainage demand over bounds {bounds}",
                )
            recommendation = llm.chat(prompts.FLOOD_RECOMMENDATION_SYSTEM, context)
            tracker.step("LLM recommendation", {"model": llm.model}, t_recommend)
        except Exception as exc:
            recommendation = f"(recommendation unavailable: {exc})"
            tracker.step("LLM recommendation", {"error": str(exc)}, t_recommend)

    lat_min, lat_max, lon_min, lon_max = bounds
    return jsonify({
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "rainfall_scenario": flood["rainfall_scenario"],
        "weights": flood["weights"],
        "weight_preset": weight_preset,
        "runoff_coefficient": flood["runoff_coefficient"],
        "hotspots": flood["hotspots"],
        "hotspots_geojson": flood_hotspots_to_geojson(flood["hotspots"]),
        "slope_layer_available": slope_field is not None,
        "exposure_layer_available": exposure_field is not None,
        "recommendation": recommendation,
        "available_rainfall_scenarios": RAINFALL_SCENARIOS,
        "available_weight_presets": list(WEIGHT_PRESETS),
        "elevation_source": dataset.elevation_source,
        "elevation_source_fallback_note": _elevation_fallback_note(dataset, "elevation"),
        "elevation_grid_shape": list(elevation_field.shape),
        "pipeline_steps": tracker.steps,
    })


# -- Place search: name a place, get a bounding box -- no LLM call -------
#
# The primary way to set a location: a plain place name/address, resolved
# to a bounding box via the same live, keyless Open-Meteo geocoding API
# geocode_place() already uses -- unlike /smart_query, this makes no LLM
# call at all (no topic/resolution to extract, just a place name), so it's
# faster and doesn't touch the Gemini quota. The frontend calls this from
# a "Place" search box; the raw lat/lon Bounding box field stays available
# under Advanced options for anyone who wants to type exact coordinates
# instead of searching.

@app.route("/api/realworld/geocode_place", methods=["POST"])
def realworld_geocode_place():
    place_name = request.form.get("place", "").strip()
    if not place_name:
        return jsonify({"error": "place is required"}), 400
    try:
        geocoded = geocode_place(place_name)
    except Exception as exc:
        return jsonify({"error": f"geocoding request failed: {exc}"}), 502
    if geocoded is None:
        return jsonify({
            "error": f"Couldn't find \"{place_name}\" -- try a more specific name (e.g. add a state/country)."
        }), 400
    bounds = bounding_box_around_point(geocoded["latitude"], geocoded["longitude"])
    return jsonify({"bounds": list(bounds), "resolved_place": geocoded})


# -- One-box agent flow: name a place, get a finished HTML report --------
#
# The primary UI is now just a place name and (optionally) an exposure/
# population file: everything else -- geocoding, elevation source
# selection, resolution, rainfall scenario, risk weighting -- is a fixed,
# documented default the agent applies itself (see the constants below),
# rather than a form field the user has to understand first. Progress is
# streamed back as newline-delimited JSON (one `{"type": ...}` object per
# line, flushed as each pipeline stage finishes) so the frontend can show
# a live, step-by-step log instead of a single spinner -- fetch()'s
# streaming reader consumes this directly; plain EventSource isn't used
# because it can't POST a file upload for the exposure layer.
#
# Population/exposure is the one category with no live online source at
# all in this project (elevation always has a working source chain, slope
# is free from elevation) -- so instead of pretending to "look for" it and
# failing, the frontend asks upfront whether the user has a file for it.
# If not, flood_tools already treats a missing exposure layer as a neutral
# 0.5 placeholder rather than an error, so that factor is simply ignored.

# Flat 30m default regardless of place size. For anything bigger than
# roughly a 15km box, 30m would need more than _AGENT_MAX_GRID_RESOLUTION
# points on the longer axis -- resolution_for_target_spacing_scaled (real_
# data.py) handles that by scaling the EFFECTIVE target spacing up
# uniformly on both axes rather than clamping one axis independently, so
# ground pixels stay square (same real meters both directions) instead of
# distorting into a rectangle for a wide/tall box. A user who wants finer
# detail for a specific run asks for it in the report chat instead (see
# parseResolutionRequest in main.js) -- analyze_stream accepts an optional
# resolution_meters override for exactly that re-run.
_AGENT_TARGET_RESOLUTION_METERS = 30
_AGENT_MIN_RESOLUTION_METERS = 5
_AGENT_MAX_RESOLUTION_METERS = 2000
# Grid POINT-COUNT cap (per axis), separate from real_data.py's own
# MAX_GRID_RESOLUTION=80 -- that one is tuned for the slow per-point
# sources (Open-Meteo/USGS EPQS) other endpoints in this file still use.
# The agent flow always fetches elevation via usgs_dem_cog (GDAL-streamed,
# ~3.5s regardless of resolution -- see real_data.py), so a bigger grid
# only costs TTK analysis time and GeoJSON payload size, not fetch time --
# raised well past 80 so a finer resolution_meters request (via chat) can
# actually be honored instead of immediately hitting the old point cap.
# NOTE: the interactive map still renders one polygon per grid cell, so a
# grid anywhere near this cap (250k cells) produces a tens-of-MB GeoJSON
# payload that's slow/heavy for a browser to render -- this cap protects
# the backend pipeline's own runtime, not map responsiveness.
_AGENT_MAX_GRID_RESOLUTION = 500
_AGENT_RAINFALL_SCENARIO = "100yr"
_AGENT_WEIGHT_PRESET = "balanced"
_AGENT_RUNOFF_COEFFICIENT = 0.5
# ~0 rather than exactly 0 -- TTK's simplification step compares against
# this value, and a hair above zero avoids any edge case with exactly-equal
# persistence pairs while still keeping essentially every raw local minimum
# (see report.py's "All local minima" section for why this view exists
# alongside the persistence-significant one above).
_AGENT_ALL_MINIMA_THRESHOLD = 1e-6
# Cap on how many raw local minima the chat's qa_context includes (see
# "all_minima_full" below) -- keeps the per-question payload/prompt bounded
# for runs with an unusually large grid (thousands of raw minima). Ordinary
# runs (hundreds of minima) fit under this easily and get every one.
_QA_ALL_MINIMA_CAP = 500


@app.route("/api/agent/analyze_stream", methods=["POST"])
def agent_analyze_stream():
    place = request.form.get("place", "").strip()
    if not place:
        return jsonify({"error": "place is required"}), 400

    # Optional override of the flat 100m default -- set when the frontend
    # detects a resolution change request in the report chat and re-POSTs
    # here for the same place rather than silently reinterpreting a chat
    # message as a form field the user never saw.
    target_resolution_meters = _AGENT_TARGET_RESOLUTION_METERS
    resolution_meters_raw = request.form.get("resolution_meters", "").strip()
    if resolution_meters_raw:
        try:
            requested = float(resolution_meters_raw)
            if _AGENT_MIN_RESOLUTION_METERS <= requested <= _AGENT_MAX_RESOLUTION_METERS:
                target_resolution_meters = requested
        except ValueError:
            pass  # falls back to the default rather than erroring the whole run over a bad override

    exposure_file = request.files.get("exposure_file")
    exposure_tmp_path = None
    if exposure_file and exposure_file.filename:
        suffix = Path(exposure_file.filename).suffix.lower() or ".csv"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        exposure_file.save(tmp.name)
        tmp.close()
        exposure_tmp_path = tmp.name

    def generate():
        def line(obj: dict) -> str:
            return json.dumps(obj, ensure_ascii=False) + "\n"

        t_start = time.time()
        try:
            yield line({"type": "step", "label": f'Looking up "{place}"\'s real boundary'})
            boundary = None
            try:
                boundary = geocode_place_boundary(place)
            except Exception:
                boundary = None  # any failure here just falls back to the point+box path below

            if boundary is not None:
                bounds = boundary["bounds"]
                place_label = boundary["name"]
                yield line({"type": "step", "label": "Resolved real boundary (OpenStreetMap)", "detail": {
                    "place_label": place_label, "display_name": boundary["display_name"], "bounds": list(bounds),
                }})
            else:
                yield line({"type": "step", "label": "No boundary polygon available -- geocoding to a point instead"})
                try:
                    geocoded = geocode_place(place)
                except Exception as exc:
                    yield line({"type": "error", "message": f"Geocoding failed: {exc}"})
                    return
                if geocoded is None:
                    yield line({"type": "error", "message": f'Couldn\'t find "{place}" -- try a more specific name (e.g. add a state/country).'})
                    return
                bounds = bounding_box_around_point(geocoded["latitude"], geocoded["longitude"])
                place_label = geocoded["name"] + (f", {geocoded['admin1']}" if geocoded.get("admin1") else "")
                yield line({"type": "step", "label": "Resolved location (point + padded box)", "detail": {
                    "place_label": place_label, "bounds": list(bounds),
                }})

            resolution = resolution_for_target_spacing_scaled(
                bounds, target_resolution_meters, max_resolution=_AGENT_MAX_GRID_RESOLUTION,
            )
            yield line({"type": "step", "label": "Fetching elevation data", "detail": {
                "target_resolution_meters": target_resolution_meters, "grid": list(resolution),
            }})
            t0 = time.time()
            try:
                dataset = build_real_world_dataset(place_label, bounds, resolution=resolution, elevation_source="auto")
            except Exception as exc:
                yield line({"type": "error", "message": f"Could not fetch elevation data for this location: {exc}"})
                return
            elevation_field = dataset.scalar_fields["elevation"]
            slope_field = dataset.scalar_fields.get("slope")
            yield line({"type": "step", "label": "Fetched elevation data", "detail": {
                "elevation_source": dataset.elevation_source, "grid_shape": list(elevation_field.shape),
                "elapsed_seconds": round(time.time() - t0, 2),
            }})

            exposure_field = None
            if exposure_tmp_path:
                yield line({"type": "step", "label": "Loading exposure/population data"})
                try:
                    exposure_field = scalar_field_from_upload("exposure", exposure_tmp_path, bounds, resolution)
                    yield line({"type": "step", "label": "Exposure data loaded"})
                except Exception as exc:
                    yield line({"type": "step", "label": "Exposure data failed to load -- ignoring that factor", "detail": {"error": str(exc)}})
                finally:
                    Path(exposure_tmp_path).unlink(missing_ok=True)
            else:
                yield line({"type": "step", "label": "No exposure/population data supplied -- that factor will use a neutral default"})

            # Clip BEFORE analyzing, not after: cells outside the real
            # boundary are raised to "wall" ground (see clip_field_to_polygon)
            # so TTK's Morse-Smale complex -- which finds LOCAL MINIMA -- can
            # never place a basin outside the actual place. Only the field
            # fed to TTK is clipped; elevation_field itself (used for
            # rendering and for each hotspot's real elevation_m) stays the
            # genuine unclipped data.
            real_value_range = [float(elevation_field.values.min()), float(elevation_field.values.max())]
            if boundary is not None:
                yield line({"type": "step", "label": "Clipping DEM to the real boundary before analysis"})
                elevation_field_for_ttk = clip_field_to_polygon(elevation_field, boundary["polygon"])
            else:
                elevation_field_for_ttk = elevation_field

            yield line({"type": "step", "label": "Running Morse-Smale topology (TTK)"})
            t0 = time.time()
            try:
                morse_result = analyze_scalar_field_ttk(elevation_field_for_ttk)
            except Exception as exc:
                yield line({"type": "error", "message": f"Topology analysis failed: {exc}"})
                return
            if boundary is not None:
                # the clipped field's range includes the synthetic wall --
                # restore the real range so flood_tools' persistence-based
                # topological_score isn't normalized against a fake, much
                # larger span than the actual terrain has.
                morse_result["value_range"] = real_value_range
            yield line({"type": "step", "label": "Topology analysis complete", "detail": {
                "num_significant_basins": morse_result["num_significant_basins"],
                "num_raw_minima": morse_result["num_raw_minima"],
                "persistence_threshold": morse_result["persistence_threshold"],
                "elapsed_seconds": round(time.time() - t0, 2),
            }})

            yield line({"type": "step", "label": "Ranking flood risk & drainage demand", "detail": {
                "rainfall_scenario": _AGENT_RAINFALL_SCENARIO, "weight_preset": _AGENT_WEIGHT_PRESET,
            }})
            flood_result = rank_flood_basins(
                elevation_field, morse_result, rainfall_scenario=_AGENT_RAINFALL_SCENARIO,
                weights=_AGENT_WEIGHT_PRESET, runoff_coefficient=_AGENT_RUNOFF_COEFFICIENT,
                slope_field=slope_field, exposure_field=exposure_field,
            )
            if boundary is not None:
                tag_hotspots_within_boundary(flood_result["hotspots"], boundary["polygon"])
            yield line({"type": "step", "label": "Flood ranking complete", "detail": {"num_hotspots": len(flood_result["hotspots"])}})

            yield line({"type": "step", "label": "Computing drainage demand for every local minimum"})
            t0 = time.time()
            all_minima_hotspots = None
            try:
                morse_result_all = analyze_scalar_field_ttk(elevation_field_for_ttk, persistence_threshold=_AGENT_ALL_MINIMA_THRESHOLD)
                if boundary is not None:
                    morse_result_all["value_range"] = real_value_range
                flood_result_all = rank_flood_basins(
                    elevation_field, morse_result_all, rainfall_scenario=_AGENT_RAINFALL_SCENARIO,
                    weights=_AGENT_WEIGHT_PRESET, runoff_coefficient=_AGENT_RUNOFF_COEFFICIENT,
                    slope_field=slope_field, exposure_field=exposure_field,
                )
                all_minima_hotspots = flood_result_all["hotspots"]
                if boundary is not None:
                    tag_hotspots_within_boundary(all_minima_hotspots, boundary["polygon"])
                yield line({"type": "step", "label": "All local minima ranked", "detail": {
                    "num_local_minima": len(all_minima_hotspots), "elapsed_seconds": round(time.time() - t0, 2),
                }})
            except Exception as exc:
                yield line({"type": "step", "label": "Skipping all-local-minima view", "detail": {"error": str(exc)}})

            recommendation = None
            system_prompt = prompts.FLOOD_RECOMMENDATION_SYSTEM
            model_name = "unavailable"
            live_source = _describe_elevation_source(dataset.elevation_source)
            user_context = (
                f"'elevation' data for this run came from {live_source}. Basins/peaks were computed by "
                "TTK (Topology ToolKit)'s Morse-Smale complex: persistence-based topological simplification "
                "at the reported threshold, then critical points and descending manifolds (basins) on the "
                "simplified field -- no heuristic depression-filling or breaching preprocessing.\n"
                f"Weight preset used: '{_AGENT_WEIGHT_PRESET}'. Slope layer available: {slope_field is not None}. "
                f"Exposure layer available: {exposure_field is not None}.\n"
                f"Flood risk ranking result: {json.dumps(flood_result, ensure_ascii=False)}"
            )
            yield line({"type": "step", "label": "Generating recommendation (LLM)"})
            t0 = time.time()
            try:
                llm = LLMClient()
                model_name = llm.model
                recommendation = llm.chat(system_prompt, user_context)
                yield line({"type": "step", "label": "Recommendation generated", "detail": {
                    "model": model_name, "elapsed_seconds": round(time.time() - t0, 2),
                }})
            except Exception as exc:
                yield line({"type": "step", "label": "Recommendation unavailable", "detail": {"error": str(exc)}})

            # A compact, LLM-sized summary of this run -- NOT the full
            # map_data/report HTML -- sent back to the frontend so a later
            # /api/agent/ask call can ground its answer in the same numbers
            # this report used, without the server needing to remember
            # anything about this request (stateless, matching the rest of
            # this app -- the frontend just resends this context each time).
            #
            # Earlier this only included the top 10 raw minima by drainage
            # demand, which made the chat unable to answer basic filter/
            # count questions ("basins with persistence > 1m") about the
            # other 260+ raw minima it never saw. Send every raw minimum (up
            # to a cap, for runs with an unusually large grid) instead, so
            # the agent can actually filter/aggregate over the real data
            # instead of guessing from a 10-row sample. Sorted by persistence
            # descending (not drainage demand) because that ordering makes
            # truncation self-describing: every entry cut off by the cap has
            # LESS persistence than every included one, so a "persistence >
            # X" question is still answerable in full as long as X is at or
            # above the reported cutoff -- the agent doesn't have to guess
            # whether it's missing something.
            def _persistence_sort_key(h):
                p = h.get("persistence_m")
                return (0, 0.0) if p is None else (1, -p)  # None = global sink = sorts first

            all_minima_sorted = sorted(all_minima_hotspots or [], key=_persistence_sort_key)
            all_minima_included = all_minima_sorted[:_QA_ALL_MINIMA_CAP]
            all_minima_truncated = len(all_minima_sorted) > len(all_minima_included)
            cutoff_persistence_m = all_minima_included[-1]["persistence_m"] if all_minima_truncated else None
            cutoff_note = (
                "Every raw local minimum from this run is included below -- nothing was omitted."
                if not all_minima_truncated else (
                    f"Only the {len(all_minima_included)} most persistent raw minima (of "
                    f"{len(all_minima_sorted)} total) are included below; every omitted one has "
                    f"persistence_m <= {cutoff_persistence_m:.3f}, so a 'persistence greater than Xm' "
                    f"filter is still complete as long as X >= {cutoff_persistence_m:.3f}."
                )
            )
            qa_context = {
                "place_label": place_label,
                "boundary_display_name": boundary["display_name"] if boundary else None,
                "elevation_stats": {
                    "grid_shape": list(elevation_field.shape),
                    "min_m": float(elevation_field.values.min()),
                    "max_m": float(elevation_field.values.max()),
                },
                "rainfall_scenario": flood_result["rainfall_scenario"],
                "weights": flood_result["weights"],
                "runoff_coefficient": flood_result["runoff_coefficient"],
                "hotspots": flood_result["hotspots"],
                "all_minima_full": {
                    "total_raw_minima": len(all_minima_sorted),
                    "included_count": len(all_minima_included),
                    "sorted_by": "persistence_m descending (a global sink, persistence_m=null, sorts first as maximally significant)",
                    "truncated": all_minima_truncated,
                    "cutoff_note": cutoff_note,
                    "entries": [
                        {
                            "rank": h["rank"], "coords": h["coords"], "persistence_m": h["persistence_m"],
                            "elevation_m": h["elevation_m"], "contributing_area_km2": h["contributing_area_km2"],
                            "estimated_peak_runoff_m3s": h["estimated_peak_runoff_m3s"],
                            "risk_score": h["risk_score"], "catchment_id": h.get("catchment_id"),
                            "within_boundary": h.get("within_boundary"),
                        }
                        for h in all_minima_included
                    ],
                },
                "exposure_data_supplied": exposure_field is not None,
                "recommendation": recommendation,
            }

            yield line({"type": "step", "label": "Building report"})
            # catchment_grid ties each terrain cell to the minimum whose
            # basin it drains into (ttk_engine's descending_manifold, from
            # the SAME low-threshold pass all_minima_hotspots came from --
            # not the significant-only pass, since every raw minimum on the
            # map needs a matching catchment) -- lets a viewer highlight one
            # minimum's contributing area on click with no extra geometry
            # pass of its own. Used by both the live persistent map
            # (map_data below) and the downloadable report's own embedded
            # map (via report_kwargs), so both offer the same click.
            catchment_grid = morse_result_all.get("descending_manifold") if all_minima_hotspots else None
            report_kwargs = dict(
                place_label=place_label, bounds=bounds, elevation=elevation_field, slope=slope_field,
                elevation_source=dataset.elevation_source, morse_result=morse_result, flood_result=flood_result,
                recommendation=recommendation, system_prompt=system_prompt, user_context=user_context,
                model_name=model_name, exposure_provided=exposure_field is not None,
                generated_at=datetime.date.today().isoformat(),
                boundary_polygon=boundary["polygon"] if boundary else None,
                boundary_geojson=boundary["geojson"] if boundary else None,
                boundary_display_name=boundary["display_name"] if boundary else None,
                all_minima_hotspots=all_minima_hotspots,
                catchment_grid=catchment_grid,
            )
            # Standalone version (embedded interactive map) is what the
            # download link/.html file actually is -- it has to stay a
            # complete, self-contained artifact once saved. The in-page
            # version skips that embedded map since the live UI renders its
            # own persistent map in a separate panel (map_data below) rather
            # than duplicating a second Leaflet instance inside an iframe.
            report_html = render_flood_report_html(**report_kwargs, embed_interactive_map=True)
            report_body_html = render_flood_report_html(**report_kwargs, embed_interactive_map=False)

            lat_min, lat_max, lon_min, lon_max = bounds
            map_data = {
                "bounds": [lat_min, lat_max, lon_min, lon_max],
                "field_geojson": elevation_hillshade_geojson(elevation_field, catchment_grid=catchment_grid),
                "elevation_legend": terrain_legend_stops(elevation_field),
                "all_minima_geojson": all_minima_geojson(all_minima_hotspots) if all_minima_hotspots else None,
                "boundary_geojson": boundary["geojson"] if boundary else None,
            }
            yield line({"type": "done", "report_html": report_html, "report_body_html": report_body_html,
                        "map_data": map_data, "qa_context": qa_context, "place_label": place_label,
                        "total_elapsed_seconds": round(time.time() - t_start, 2)})
        except Exception as exc:
            yield line({"type": "error", "message": f"Unexpected error: {exc}"})

    return Response(generate(), mimetype="application/x-ndjson")


# -- Report Q&A chat: the LLM as the front-facing agent, not just the ----
# backend step that writes the recommendation paragraph once and disappears.
# The frontend keeps the "qa_context" object from analyze_stream's "done"
# event (a compact summary of the run -- ranked hotspots, elevation stats,
# rainfall/weights, the recommendation text -- NOT the full map_data/report
# HTML) and resends it with every question, along with the running chat
# history, so this endpoint stays stateless like the rest of the app -- no
# server-side session to expire or leak between users.

_QA_HISTORY_TURNS_KEPT = 6  # most-recent turns resent for context, not the whole thread


def _friendly_llm_error(exc: Exception) -> str:
    """The google-genai SDK's raw exception text is a multi-hundred-
    character dump of the API's JSON error body -- fine buried in a log
    line, unreadable pasted verbatim into a chat bubble (which is exactly
    where agent_ask's caller puts it). The free-tier daily quota (20
    requests/day for gemini-3.5-flash, see llm_client.py) is the single
    most common failure during normal use of this project, so it gets its
    own plain-language message; anything else still shows the real reason."""
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (
            "Gemini's free-tier daily quota (20 requests/day) is used up for now -- "
            "this resets on its own, try again later."
        )
    return f"LLM request failed: {text}"


@app.route("/api/agent/ask", methods=["POST"])
def agent_ask():
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        context = json.loads(request.form.get("context", "{}"))
    except json.JSONDecodeError:
        return jsonify({"error": "invalid context JSON"}), 400
    try:
        history = json.loads(request.form.get("history", "[]"))
    except json.JSONDecodeError:
        history = []

    history_text = "".join(
        f"\n{turn.get('role', 'user').upper()}: {turn.get('content', '')}"
        for turn in history[-_QA_HISTORY_TURNS_KEPT:]
        if turn.get("content")
    )
    user_prompt = (
        f"Analysis context for this run (JSON): {json.dumps(context, ensure_ascii=False)}"
        + (f"\n\nPrior conversation in this thread:{history_text}" if history_text else "")
        + f"\n\nNEW QUESTION: {question}"
    )

    try:
        llm = LLMClient()
        answer = llm.chat(prompts.FLOOD_QA_SYSTEM, user_prompt)
    except Exception as exc:
        return jsonify({"error": _friendly_llm_error(exc)}), 502

    return jsonify({"answer": answer})


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
    # threaded=True: /api/agent/analyze_stream holds its connection open for
    # the whole pipeline run (a minute or more) -- without this, the dev
    # server can't serve anything else (even static assets) until it finishes.
    app.run(debug=True, port=5000, threaded=True)
