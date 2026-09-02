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

import concurrent.futures
import datetime
import json
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent

from dotenv import load_dotenv

load_dotenv(override=True)  # .env (see .env.example) is the documented place to set these --
# override=True so it actually wins over a stray same-named system/shell env var instead of
# being silently ignored (python-dotenv's default is override=False)

from flask import Flask, Response, jsonify, render_template, request

from geoai_agent import prompts, rainfall
from geoai_agent.agent_decisions import (
    build_persistence_candidates,
    compute_weight_sensitivity,
    decide_persistence_threshold,
    reflect_on_recommendation,
)
from geoai_agent.boundary import (
    boundary_from_drawn_polygon,
    boundary_from_shapefile,
    clip_field_to_polygon,
    clip_geojson_to_polygon,
    geocode_place_boundary,
    grid_inside_polygon_mask,
    tag_hotspots_within_boundary,
)
from geoai_agent.drainage_db import fetch_nearby_pipes, pipes_to_geojson, tag_hotspots_with_drainage_capacity
from geoai_agent.fema_flood import fetch_fema_flood_zones, tag_hotspots_with_flood_zone
from geoai_agent.flood_tools import (
    RAINFALL_SCENARIOS,
    WEIGHT_PRESETS,
    compute_expected_annual_runoff,
    rank_flood_basins,
)
from geoai_agent.interjection import CONTINUE_SIGNAL, HOLD_SIGNAL, post_interjection, take_interjection
from geoai_agent.land_cover import fetch_nlcd_landcover_grid, land_cover_geojson
from geoai_agent.llm_client import LLMClient
from geoai_agent.report import all_minima_geojson, elevation_hillshade_geojson, render_flood_report_html, terrain_legend_stops
from geoai_agent.flow_field import flow_field_geojson
from geoai_agent.morse_validation import compare_morse_to_d8
from geoai_agent.soil_data import fetch_dominant_hydrologic_soil_group
from geoai_agent.spatial_analysis import (
    buildings_geojson as buildings_to_geojson,
    compute_building_exposure,
    compute_terrain_metrics,
    compute_topographic_wetness_index,
    fetch_osm_infrastructure,
    roads_geojson,
    tag_hotspots_with_nearest_infrastructure,
    water_geojson,
)
from geoai_agent.streamflow import fetch_nearby_river_gauges, gauges_to_geojson, tag_hotspots_with_river_gauge
from geoai_agent.ttk_engine import analyze_scalar_field_ttk
from geoai_agent.real_data import (
    MAX_USGS_GRID_RESOLUTION,
    OPEN_METEO_VARIABLES,
    bounding_box_around_point,
    build_real_world_dataset,
    fetch_population_exposure_field,
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
    if elevation_source == "user_upload":
        return "a user-uploaded file (not live-fetched, not cross-checked against USGS/Open-Meteo)"
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
    # "entropy" isn't a named policy preset in WEIGHT_PRESETS -- it's the
    # objective, data-driven alternative (flood_tools.entropy_weights),
    # recognized as a special string by rank_flood_basins itself.
    if weight_preset != "entropy" and weight_preset not in WEIGHT_PRESETS:
        return jsonify({"error": f"unknown weight_preset, available: {list(WEIGHT_PRESETS)} or 'entropy'"}), 400
    try:
        runoff_coefficient = float(request.form.get("runoff_coefficient", 0.5))
    except ValueError:
        return jsonify({"error": "runoff_coefficient must be a number"}), 400

    slope_field = dataset.scalar_fields.get("slope")
    exposure_field = dataset.scalar_fields.get("exposure") or dataset.scalar_fields.get("population")

    # Live per-location precipitation-frequency, same six named return
    # periods the dropdown above already offers -- this only replaces the
    # INTENSITIES behind whichever scenario the user picked, not the choice
    # itself. Any failure (no Atlas 14 coverage for this location, network
    # error) falls back to the static RAINFALL_SCENARIOS table exactly as
    # rank_flood_basins already does when rainfall_table=None.
    lat_min0, lat_max0, lon_min0, lon_max0 = bounds
    try:
        rainfall_table = rainfall.fetch_noaa_atlas14_rainfall((lat_min0 + lat_max0) / 2, (lon_min0 + lon_max0) / 2)
    except Exception:
        rainfall_table = None

    # Land cover + soil -> a per-basin runoff coefficient instead of the one
    # flat `runoff_coefficient` above applying to every basin -- same
    # graceful "None on any failure, ranking still works with the flat
    # constant" contract as every other optional layer here.
    try:
        land_cover_grid = fetch_nlcd_landcover_grid(bounds, resolution=elevation_field.shape[::-1])
    except Exception:
        land_cover_grid = None
    try:
        hydrologic_group = fetch_dominant_hydrologic_soil_group(bounds)
    except Exception:
        hydrologic_group = None

    t_analyze = time.time()
    morse_result = analyze_scalar_field_ttk(elevation_field)
    tracker.step("Morse-Smale analysis (TTK)", {
        "grid_shape": morse_result["grid_shape"], "persistence_threshold": morse_result["persistence_threshold"],
        "num_significant_basins": morse_result["num_significant_basins"], "num_raw_minima": morse_result["num_raw_minima"],
    }, t_analyze)

    t_rank = time.time()
    flood = rank_flood_basins(
        elevation_field, morse_result, rainfall_scenario=rainfall_scenario, rainfall_table=rainfall_table,
        weights=weight_preset, runoff_coefficient=runoff_coefficient, slope_field=slope_field, exposure_field=exposure_field,
        land_cover_grid=land_cover_grid, hydrologic_group=hydrologic_group,
    )
    tracker.step("Flood risk ranking", {
        "rainfall_scenario": rainfall_scenario, "rainfall_source": flood["rainfall_source"], "weight_preset": weight_preset,
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
        "rainfall_source": flood["rainfall_source"],
        "weights": flood["weights"],
        "weight_preset": weight_preset,
        "runoff_coefficient": flood["runoff_coefficient"],
        "hotspots": flood["hotspots"],
        "hotspots_geojson": flood_hotspots_to_geojson(flood["hotspots"]),
        "slope_layer_available": slope_field is not None,
        "exposure_layer_available": exposure_field is not None,
        "land_cover_data_available": land_cover_grid is not None,
        "soil_data_available": hydrologic_group is not None,
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

# How many seconds each of agent_analyze_stream's three checkpoints waits
# for a user interjection before continuing automatically. Fixed and short
# (not user-configurable) so a run that nobody is watching still finishes
# in bounded time -- interjecting is an opportunity, never a requirement.
_INTERJECTION_WAIT_SECONDS = 15
# Once the user clicks into the feedback box (HOLD_SIGNAL, posted by
# /api/agent/checkpoint_hold), checkpoint_wait stops enforcing the ~15s
# limit above -- they're actively composing a comment, not ignoring the
# run. Not truly unbounded, though: a much longer safety cap so an
# abandoned tab (focused the box, then walked away) can't pin a worker
# thread on this single-process dev server open forever.
_INTERJECTION_HOLD_MAX_SECONDS = 600


@app.route("/api/agent/interject", methods=["POST"])
def agent_interject():
    """Lets the frontend send free-text feedback for a still-in-progress
    run (see agent_analyze_stream's "run_id" event and its checkpoint_wait
    calls) -- e.g. "that threshold feels too aggressive" or "care more
    about the hospital area." Silently accepts a run_id nobody is (or is
    no longer) listening for -- the run may not have reached a checkpoint
    yet, or may already be past it -- same "don't fail over a late/missed
    message" discipline as every other optional input in this app; the
    caller has no way to know a checkpoint's exact timing from outside."""
    run_id = request.form.get("run_id", "").strip()
    message = request.form.get("message", "").strip()
    if not run_id or not message:
        return jsonify({"error": "run_id and message are both required"}), 400
    post_interjection(run_id, message)
    return jsonify({"ok": True})


@app.route("/api/agent/continue", methods=["POST"])
def agent_continue():
    """Lets the frontend's per-checkpoint Continue button skip the rest of
    a still-open checkpoint's wait immediately, instead of sitting out the
    remainder of the countdown. Posts CONTINUE_SIGNAL rather than a normal
    message so checkpoint_wait can tell "user said go ahead" apart from
    "user typed feedback" -- the former ends the wait with nothing to act
    on, the latter loops back. Same silent-no-op-if-nobody's-listening
    behavior as /api/agent/interject."""
    run_id = request.form.get("run_id", "").strip()
    if not run_id:
        return jsonify({"error": "run_id is required"}), 400
    post_interjection(run_id, CONTINUE_SIGNAL)
    return jsonify({"ok": True})


@app.route("/api/agent/checkpoint_hold", methods=["POST"])
def agent_checkpoint_hold():
    """Lets the frontend tell a still-open checkpoint "the user just clicked
    into the feedback box, hold off on auto-continuing" -- posts HOLD_SIGNAL,
    which checkpoint_wait swaps its ~15s timeout for a much longer safety
    cap (_INTERJECTION_HOLD_MAX_SECONDS) rather than continuing to count
    down while they're mid-thought. Same silent-no-op-if-nobody's-listening
    behavior as /api/agent/interject and /api/agent/continue."""
    run_id = request.form.get("run_id", "").strip()
    if not run_id:
        return jsonify({"error": "run_id is required"}), 400
    post_interjection(run_id, HOLD_SIGNAL)
    return jsonify({"ok": True})


@app.route("/api/agent/analyze_stream", methods=["POST"])
def agent_analyze_stream():
    place = request.form.get("place", "").strip()
    # Two ways the frontend can pick weights, via weighting_method:
    # - "custom" (the Modify-weights sliders, available once a run has
    #   completed): a hand-set {"topological", "contributing_area",
    #   "rainfall", "slope", "exposure"} dict in custom_weights (JSON),
    #   validated below and passed straight through to rank_flood_basins.
    # - anything else (including the field being omitted, which is every
    #   run now that the UI has no weighting-method choice left): flood_
    #   tools.entropy_weights -- computed from how much each factor
    #   actually varies among THIS run's own basins, no expert judgment
    #   needed, at the cost of the caveats that method's own docstring/the
    #   report's methodology section state plainly (rainfall always gets
    #   weight 0; weights vary run to run). This is the only weighting
    #   mode every analysis uses unless a user hand-sets their own via
    #   Modify-weights.
    weighting_method_field = request.form.get("weighting_method", "")
    if weighting_method_field == "custom":
        custom_weights_raw = request.form.get("custom_weights", "").strip()
        try:
            custom_weights = json.loads(custom_weights_raw) if custom_weights_raw else None
        except json.JSONDecodeError:
            return jsonify({"error": "custom_weights must be JSON"}), 400
        required_weight_keys = {"topological", "contributing_area", "rainfall", "slope", "exposure"}
        if not isinstance(custom_weights, dict) or set(custom_weights) != required_weight_keys:
            return jsonify({
                "error": f"custom_weights must be a JSON object with exactly the keys {sorted(required_weight_keys)}"
            }), 400
        weight_preset_used = custom_weights
    else:
        weight_preset_used = "entropy"
    # Optional free-text priority statement (e.g. "protecting people's
    # safety", "critical infrastructure") -- feeds generate()'s persistence-
    # threshold agent decision (see geoai_agent.agent_decisions.
    # decide_persistence_threshold) regardless of weighting mode.
    objective_text = request.form.get("objective_text", "").strip()
    # Alternative to naming a place: the frontend lets a user draw a polygon
    # directly on the persistent map (main.js's draw-mode) and sends its
    # vertices here instead -- [[lat, lon], ...] as JSON, Leaflet's own
    # click-event order. `place`, if also given, is just used as that
    # polygon's display label; it plays no role in geocoding when a polygon
    # is present.
    boundary_polygon_raw = request.form.get("boundary_polygon", "").strip()
    # A third way to set the analysis area, alongside naming a place or
    # drawing on the map: the terminal-panel's "Modify" button on the
    # boundary step lets a user upload a shapefile of their own exact
    # boundary (a parcel, a watershed delineation, anything not in
    # OpenStreetMap) -- see boundary_from_shapefile.
    boundary_shapefile = request.files.get("boundary_shapefile")
    has_boundary_shapefile = bool(boundary_shapefile and boundary_shapefile.filename)
    if not place and not boundary_polygon_raw and not has_boundary_shapefile:
        return jsonify({
            "error": "place is required (or draw a boundary on the map, or upload a boundary shapefile)"
        }), 400

    drawn_polygon_points = None
    if boundary_polygon_raw:
        try:
            drawn_polygon_points = json.loads(boundary_polygon_raw)
        except json.JSONDecodeError:
            return jsonify({"error": "boundary_polygon must be JSON"}), 400
        if not isinstance(drawn_polygon_points, list) or len(drawn_polygon_points) < 3:
            return jsonify({"error": "boundary_polygon must have at least 3 [lat, lon] points"}), 400

    boundary_shapefile_tmp_path = None
    if has_boundary_shapefile:
        suffix = Path(boundary_shapefile.filename).suffix.lower()
        if suffix not in (".zip", ".shp"):
            return jsonify({
                "error": "boundary_shapefile must be a .zip (bundling .shp/.shx/.dbf) or a .shp"
            }), 400
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        boundary_shapefile.save(tmp.name)
        tmp.close()
        boundary_shapefile_tmp_path = tmp.name

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

    # Optional replacement for the live-fetched DEM -- the frontend's
    # terminal-panel offers a "Modify" button on the elevation step once a
    # run has finished, letting the user upload their own GeoTIFF and re-run
    # with it instead of USGS/Open-Meteo. GeoTIFF only (not CSV/shapefile
    # point samples, unlike exposure below): a full DEM replacement should be
    # an actual georeferenced raster, not a handful of IDW-interpolated
    # samples -- _gdal_local_raster_worker.py separately rejects a .tif with
    # no CRS/geotransform once GDAL actually opens it.
    elevation_file = request.files.get("elevation_file")
    elevation_tmp_path = None
    if elevation_file and elevation_file.filename:
        suffix = Path(elevation_file.filename).suffix.lower()
        if suffix not in (".tif", ".tiff"):
            return jsonify({"error": "elevation_file must be a GeoTIFF (.tif/.tiff) with a coordinate system"}), 400
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        elevation_file.save(tmp.name)
        tmp.close()
        elevation_tmp_path = tmp.name

    def generate():
        # Both are assigned further down (the interjection checkpoints can
        # revise the stated objective and, from a "life safety"/"balanced"/
        # "infrastructure" keyword match, the weighting philosophy) --
        # without `nonlocal` here, that first assignment inside generate()
        # would make Python treat each name as local to generate() for its
        # ENTIRE body, breaking every earlier read of the original
        # request-derived value (observed directly: "cannot access local
        # variable" / "cannot access free variable" UnboundLocalErrors,
        # caught by actually running a live test, not just parsing the
        # file -- ast.parse has no scope analysis, so this class of bug is
        # invisible to a syntax-only check).
        nonlocal objective_text, weight_preset_used

        def line(obj: dict) -> str:
            return json.dumps(obj, ensure_ascii=False) + "\n"

        # Identifies THIS run to /api/agent/interject -- emitted as the very
        # first event so the frontend can attach a feedback box to it before
        # any checkpoint below is reached. A fresh uuid per run (not e.g. the
        # place name) so two browser tabs analyzing the same place can't
        # cross-deliver feedback to each other's run.
        run_id = uuid.uuid4().hex
        yield line({"type": "run_id", "run_id": run_id})

        def checkpoint_wait(checkpoint_label: str, result_holder: list):
            """Yields one "listening" step, then a checkpoint_open event
            (carries wait_seconds so the frontend can render its own 15s
            countdown next to a Continue button), then polls geoai_agent.
            interjection.take_interjection once a second so a message lands
            within ~1s of being sent, not just at the end of the window.
            Always ends with a checkpoint_close event, however the wait
            ended, so the frontend has one reliable signal to hide the
            countdown/button.

            Four ways the wait can end:
            - Real feedback arrives: appends it to result_holder (a plain
              list used as an out-parameter -- a generator that yields
              step-by-step can't also `return` a value the way a normal
              function would) and yields an agent_decision event surfacing
              exactly what was received.
            - The user clicks Continue (POST /api/agent/continue, which
              posts CONTINUE_SIGNAL): ends the wait immediately with
              nothing appended to result_holder and nothing logged -- this
              is "go ahead now", not feedback to act on.
            - The user clicks into the feedback box (POST /api/agent/
              checkpoint_hold, which posts HOLD_SIGNAL): the ~15s deadline
              (_INTERJECTION_WAIT_SECONDS) is swapped for a much longer
              safety cap (_INTERJECTION_HOLD_MAX_SECONDS) instead of
              continuing to run out from under someone mid-thought. Purely
              server-side state for the rest of this checkpoint's wait --
              the frontend's own countdown display can say "stopped" all it
              wants, but without this the backend has no idea and times the
              checkpoint out on its own regardless of what the UI shows.
            - Nobody responds before whichever deadline currently applies:
              result_holder stays empty and the caller proceeds with the
              run's existing plan -- interjecting is an opportunity, never a
              requirement, so silence must never block or fail the run.

            Drains the mailbox once, silently, before it starts actually
            listening -- the mailbox is keyed only by run_id, not by which
            checkpoint is open, so a message that missed an earlier
            checkpoint's own window (sent a moment too late) would otherwise
            sit there and get misattributed to THIS checkpoint the instant
            it starts polling, even though the user typed it in response to
            a different step entirely. Caught by live testing: a "prioritize
            life safety" comment aimed at the ranking checkpoint arrived a
            few seconds after that window closed, then got silently
            collected by the next checkpoint (the recommendation one) and
            mis-logged as feedback on the recommendation instead. A user
            should only ever be able to influence the checkpoint that is
            actually open when they send something."""
            take_interjection(run_id)  # discard anything stale from before this checkpoint opened
            yield line({
                "type": "step",
                "label": f"Checkpoint: {checkpoint_label} -- listening for feedback for {_INTERJECTION_WAIT_SECONDS}s "
                          "before continuing automatically",
                "detail": {"run_id": run_id},
            })
            yield line({
                "type": "checkpoint_open",
                "checkpoint": checkpoint_label,
                "run_id": run_id,
                "wait_seconds": _INTERJECTION_WAIT_SECONDS,
            })
            held = False
            waited = 0
            while True:
                message = take_interjection(run_id)
                if message == HOLD_SIGNAL:
                    held = True
                    message = None
                if message is not None:
                    if message != CONTINUE_SIGNAL:
                        result_holder.append(message)
                        yield line({
                            "type": "agent_decision", "decision": "user_interjection",
                            "label": f"User feedback received -- looping back ({checkpoint_label})",
                            "reasoning": message,
                            "detail": {"checkpoint": checkpoint_label},
                        })
                    break
                waited += 1
                if waited >= (_INTERJECTION_HOLD_MAX_SECONDS if held else _INTERJECTION_WAIT_SECONDS):
                    break
                time.sleep(1)
            yield line({"type": "checkpoint_close", "run_id": run_id})

        t_start = time.time()
        try:
            boundary = None
            if boundary_shapefile_tmp_path is not None:
                yield line({"type": "step", "label": "Using the uploaded shapefile boundary"})
                try:
                    boundary = boundary_from_shapefile(boundary_shapefile_tmp_path, name=place or "Uploaded boundary")
                except ValueError as exc:
                    yield line({"type": "error", "message": f"Invalid boundary shapefile: {exc}"})
                    return
                finally:
                    Path(boundary_shapefile_tmp_path).unlink(missing_ok=True)
                bounds = boundary["bounds"]
                place_label = boundary["name"]
                yield line({"type": "step", "label": "Boundary ready", "detail": {
                    "place_label": place_label, "bounds": list(bounds),
                }})
            elif drawn_polygon_points is not None:
                yield line({"type": "step", "label": "Using the boundary drawn on the map"})
                try:
                    boundary = boundary_from_drawn_polygon(drawn_polygon_points, name=place or "Custom drawn area")
                except ValueError as exc:
                    yield line({"type": "error", "message": f"Invalid drawn boundary: {exc}"})
                    return
                bounds = boundary["bounds"]
                place_label = boundary["name"]
                yield line({"type": "step", "label": "Boundary ready", "detail": {
                    "place_label": place_label, "bounds": list(bounds), "num_vertices": len(drawn_polygon_points),
                }})
            else:
                yield line({"type": "step", "label": f'Looking up "{place}"\'s real boundary'})
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

            # Every run uses entropy weighting unless the user hand-set
            # weights via Modify-weights (see weight_preset_used above) --
            # there is no more "which weighting preset fits the stated
            # objective" decision to make here.
            strategy_decision = None

            # -- Retrieve data: 8 sources, dispatched to a 4-worker thread
            # pool instead of one network round trip after another. Only
            # elevation is required for the run to continue; every other
            # source keeps this project's existing "any failure degrades to
            # a documented fallback, never aborts the run" contract --
            # fetching it concurrently only changes WHEN the call happens,
            # not whether a failure is tolerated (each worker function below
            # catches exactly what its old sequential try/except caught).
            # Land cover is the one source that can't join this first batch:
            # it needs elevation's ACTUAL achieved grid shape (a fallback
            # source can return a smaller grid than requested) to stay
            # pixel-aligned for runoff_coefficient_from_land_cover's
            # catchment-mask indexing, so it's dispatched into the SAME pool
            # as a follow-up task once elevation resolves, instead of
            # forcing the whole batch to wait for it up front. Pipes/gauges/
            # infrastructure are fetched here too (previously fetched much
            # later, right before they were tagged onto ranked hotspots) --
            # the raw fetch only ever needed `bounds`, never the ranking
            # that comes later; only the per-hotspot TAGGING has to wait.
            resolution = resolution_for_target_spacing_scaled(
                bounds, target_resolution_meters, max_resolution=_AGENT_MAX_GRID_RESOLUTION,
            )
            lat_min0, lat_max0, lon_min0, lon_max0 = bounds

            def _fetch_fema():
                try:
                    return fetch_fema_flood_zones(bounds)
                except Exception:
                    return None  # a courtesy public ArcGIS instance with no SLA -- treat any failure as "no data", not fatal

            def _fetch_rainfall():
                try:
                    return rainfall.fetch_noaa_atlas14_rainfall((lat_min0 + lat_max0) / 2, (lon_min0 + lon_max0) / 2)
                except Exception:
                    return None  # a courtesy public HDSC endpoint with no documented SLA -- same "treat as unavailable" contract as every other live source here

            def _fetch_elevation():
                try:
                    if elevation_tmp_path:
                        ds = build_real_world_dataset(
                            place_label, bounds, layer_sources={"elevation": elevation_tmp_path}, resolution=resolution,
                        )
                        ds.elevation_source = "user_upload"
                    else:
                        ds = build_real_world_dataset(place_label, bounds, resolution=resolution, elevation_source="auto")
                    return ds
                finally:
                    if elevation_tmp_path:
                        Path(elevation_tmp_path).unlink(missing_ok=True)

            def _fetch_soil():
                try:
                    return fetch_dominant_hydrologic_soil_group(bounds)
                except Exception:
                    return None

            def _fetch_exposure():
                if exposure_tmp_path:
                    try:
                        return "user_upload", scalar_field_from_upload("exposure", exposure_tmp_path, bounds, resolution), None
                    except Exception as exc:
                        return None, None, exc
                    finally:
                        Path(exposure_tmp_path).unlink(missing_ok=True)
                # No file uploaded -- try a live US Census block-group density
                # fetch before falling back to the neutral default, so this
                # factor works out of the box for any US location instead of
                # only ever working when a user happens to have their own
                # file (see real_data.fetch_population_exposure_field). Any
                # failure here (non-US location, service unavailable) must
                # degrade to the exact same neutral-default message this
                # branch already gave before this fetch existed -- never a
                # hard failure for the whole run.
                try:
                    return "census_live", fetch_population_exposure_field(bounds, resolution), None
                except Exception as exc:
                    return None, None, exc

            def _fetch_pipes():
                return fetch_nearby_pipes(bounds)  # a local database, not a courtesy external API -- a failure here is unexpected enough to surface, not swallow

            def _fetch_gauges():
                try:
                    return fetch_nearby_river_gauges(bounds)
                except Exception:
                    return None  # a standard USGS service, but same "treat any failure as unavailable" contract as every other live source here

            def _fetch_infra():
                try:
                    return fetch_osm_infrastructure(bounds)
                except Exception:
                    return None

            yield line({"type": "step", "label": "Retrieving data from 8 sources (4 concurrent workers)", "detail": {
                "target_resolution_meters": target_resolution_meters, "grid": list(resolution),
            }})
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                fut_fema = pool.submit(_fetch_fema)
                fut_rainfall = pool.submit(_fetch_rainfall)
                fut_elevation = pool.submit(_fetch_elevation)
                fut_soil = pool.submit(_fetch_soil)
                fut_exposure = pool.submit(_fetch_exposure)
                fut_pipes = pool.submit(_fetch_pipes)
                fut_gauges = pool.submit(_fetch_gauges)
                fut_infra = pool.submit(_fetch_infra)

                t0 = time.time()
                try:
                    dataset = fut_elevation.result()
                except Exception as exc:
                    failed_on = "load your uploaded elevation file" if elevation_tmp_path else "fetch elevation data for this location"
                    yield line({"type": "error", "message": f"Could not {failed_on}: {exc}"})
                    return
                elevation_field = dataset.scalar_fields["elevation"]
                slope_field = dataset.scalar_fields.get("slope")
                yield line({"type": "step", "label": "Fetched elevation data", "detail": {
                    "elevation_source": dataset.elevation_source, "grid_shape": list(elevation_field.shape),
                    "elapsed_seconds": round(time.time() - t0, 2),
                }})

                # Only dispatched now (see the comment above this block for
                # why) -- still overlaps with whatever of the batch above
                # hasn't finished yet, rather than running fully serial
                # after it.
                fut_landcover = pool.submit(fetch_nlcd_landcover_grid, bounds, resolution=elevation_field.shape[::-1])

                fema_flood_geojson = fut_fema.result()
                rainfall_table = fut_rainfall.result()
                hydrologic_group = fut_soil.result()
                exposure_source, exposure_field, exposure_exc = fut_exposure.result()
                nearby_pipes = fut_pipes.result()
                river_gauges = fut_gauges.result()
                osm_infra = fut_infra.result()
                try:
                    land_cover_grid = fut_landcover.result()
                except Exception as exc:
                    land_cover_grid = None
                    yield line({"type": "step", "label": "No land cover data available -- runoff coefficient will use the flat default", "detail": {"error": str(exc)}})
                else:
                    yield line({"type": "step", "label": "Land cover data loaded (NLCD)"})

            # Clipped to the real boundary, not left as the padded rectangular
            # bbox FEMA was actually queried over -- same reasoning as every
            # other map layer below (pipes, gauges, the elevation hillshade
            # itself): once a real boundary exists, nothing should visually
            # spill past it just because a different layer happens to.
            if fema_flood_geojson and boundary is not None:
                fema_flood_geojson = clip_geojson_to_polygon(fema_flood_geojson, boundary["polygon"])
                if not fema_flood_geojson["features"]:
                    fema_flood_geojson = None  # every returned zone was entirely outside the real boundary
            if fema_flood_geojson:
                yield line({"type": "step", "label": "FEMA flood zones found", "detail": {
                    "num_zones": len(fema_flood_geojson["features"]),
                }})
            else:
                yield line({"type": "step", "label": "No FEMA flood hazard data available for this area"})

            if rainfall_table:
                yield line({"type": "step", "label": "NOAA Atlas 14 data loaded", "detail": {
                    "100yr_intensity_mm_per_hr": rainfall_table["100yr"]["intensity_mm_per_hr"],
                }})
            else:
                yield line({"type": "step", "label": "No NOAA Atlas 14 coverage for this location -- using the illustrative fallback table"})

            if hydrologic_group:
                yield line({"type": "step", "label": "Soil data loaded (USDA SSURGO)", "detail": {"hydrologic_group": hydrologic_group}})
            else:
                yield line({"type": "step", "label": "No usable USDA SSURGO soil data for this area -- runoff coefficient will use the flat default"})

            if exposure_field is not None:
                yield line({"type": "step", "label": (
                    "Exposure data loaded" if exposure_source == "user_upload"
                    else "Population/exposure data loaded (US Census block groups)"
                )})
            elif exposure_tmp_path:
                yield line({"type": "step", "label": "Exposure data failed to load -- ignoring that factor", "detail": {"error": str(exposure_exc)}})
            else:
                yield line({"type": "step", "label": "No exposure/population data available -- that factor will use a neutral default", "detail": {"error": str(exposure_exc)}})

            if nearby_pipes:
                yield line({"type": "step", "label": "Stormwater pipe data found", "detail": {"num_pipes_in_area": len(nearby_pipes)}})
            else:
                yield line({"type": "step", "label": "No stormwater pipe data loaded for this area (currently DC-only -- see load_dc_stormwater.py)"})

            if river_gauges:
                yield line({"type": "step", "label": "River/stream gauge data found", "detail": {"num_gauges_in_area": len(river_gauges)}})
            else:
                yield line({"type": "step", "label": "No active USGS river/stream gauges near this area"})

            if osm_infra:
                yield line({"type": "step", "label": "OpenStreetMap infrastructure data found", "detail": {
                    "num_roads": len(osm_infra["roads"]), "num_buildings": len(osm_infra["buildings"]),
                    "num_water_features": len(osm_infra["water"]),
                }})
            else:
                yield line({"type": "step", "label": "No OpenStreetMap infrastructure data available for this area"})

            # Clip BEFORE analyzing, not after: cells outside the real
            # boundary are raised to "wall" ground (see clip_field_to_polygon)
            # so TTK's Morse-Smale complex -- which finds LOCAL MINIMA -- can
            # never place a basin outside the actual place. Only the field
            # fed to TTK is clipped; elevation_field itself (used for
            # rendering and for each hotspot's real elevation_m) stays the
            # genuine unclipped data.
            real_value_range = [float(elevation_field.values.min()), float(elevation_field.values.max())]
            # Computed once and reused for both clip_field_to_polygon (below)
            # and rank_flood_basins's boundary_mask (further down) -- without
            # this, a basin's contributing area would include synthetic
            # "wall" cells outside the real boundary (see clip_field_to_
            # polygon's docstring: a basin's catchment can legitimately
            # extend across the boundary edge into the wall region, since the
            # wall slopes back down toward the real terrain there).
            boundary_mask = grid_inside_polygon_mask(elevation_field.xs, elevation_field.ys, boundary["polygon"]) if boundary is not None else None
            if boundary is not None:
                yield line({"type": "step", "label": "Clipping DEM to the real boundary before analysis"})
                elevation_field_for_ttk = clip_field_to_polygon(elevation_field, boundary["polygon"], mask=boundary_mask)
            else:
                elevation_field_for_ttk = elevation_field

            # -- Agent decision 2/2: persistence threshold ------------------
            # A cheap near-zero-threshold TTK scan (same threshold the later
            # "every raw minimum" pass uses -- see _AGENT_ALL_MINIMA_THRESHOLD
            # below) gives every raw minimum's own persistence value, from
            # which build_persistence_candidates derives a handful of real
            # candidate significance thresholds FOR THIS TERRAIN, each with
            # the basin count it would actually produce. The agent picks one
            # of those instead of this project's previous fixed "5% of
            # elevation range, no matter the terrain" default. This is a
            # second, separate TTK subprocess call from the "significant"
            # one just below (TTK has no way to answer "how many basins at
            # threshold X" without actually re-simplifying at X) -- accepted
            # as a deliberate cost for a real, terrain-aware decision instead
            # of a rewrite of the pipeline's data flow to avoid it.
            chosen_threshold = None
            threshold_decision = None
            yield line({"type": "step", "label": "Agent scanning terrain to evaluate persistence-threshold candidates"})
            try:
                scan_result = analyze_scalar_field_ttk(elevation_field_for_ttk, persistence_threshold=_AGENT_ALL_MINIMA_THRESHOLD)
                if boundary is not None:
                    scan_result["value_range"] = real_value_range
                candidates = build_persistence_candidates(scan_result)
                if candidates:
                    yield line({"type": "step", "label": "Persistence-threshold candidates computed", "detail": {"candidates": candidates}})
                    threshold_decision = decide_persistence_threshold(LLMClient(), place_label, objective_text, candidates)
                    chosen_threshold = threshold_decision["chosen_threshold"]
                    yield line({
                        "type": "agent_decision", "decision": "persistence_threshold",
                        "label": f"Agent chose a {chosen_threshold:.2f}m persistence threshold",
                        "reasoning": threshold_decision["reasoning"],
                        "detail": {"candidates": candidates, "chosen_threshold": chosen_threshold},
                    })
                else:
                    yield line({"type": "step", "label": "No usable persistence candidates -- using the default auto threshold"})
            except Exception as exc:
                chosen_threshold = None
                threshold_decision = None
                yield line({
                    "type": "step",
                    "label": "Threshold evaluation unavailable -- using the default auto threshold",
                    "detail": {"error": str(exc)},
                })

            yield line({"type": "step", "label": "Running Morse-Smale topology (TTK)"})
            t0 = time.time()
            try:
                morse_result = analyze_scalar_field_ttk(elevation_field_for_ttk, persistence_threshold=chosen_threshold)
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
            # The candidate table's own basin counts (build_persistence_
            # candidates) come from filtering the SEPARATE near-zero-
            # threshold scan's persistence values -- an ESTIMATE, not a
            # guarantee: TTK's persistence-pairing simplification is itself
            # threshold-dependent (observed directly: re-running at a fixed
            # threshold is deterministic, but a borderline critical point's
            # persistence pairing near the cutoff can genuinely differ
            # between two separately-simplified runs), so the actually-
            # achieved count can differ from the candidate table's estimate
            # by a small amount. Recorded here and surfaced honestly in the
            # report rather than left as two silently-disagreeing numbers.
            if threshold_decision is not None:
                threshold_decision["actual_num_significant_basins"] = morse_result["num_significant_basins"]

            # Factored so BOTH interjection checkpoint 1 (a human's comment)
            # and the autonomous self-correction loop below (the agent's OWN
            # reflection step deciding the threshold itself looks wrong, not
            # just the wording describing it) can trigger the exact same
            # real re-decision + re-run, instead of a human-only code path
            # and a separate, easier-to-drift-out-of-sync agent-only one.
            def _redecide_threshold(note: str):
                nonlocal threshold_decision, chosen_threshold, morse_result
                revised_objective = f"{objective_text}\n{note}".strip()
                try:
                    threshold_decision = decide_persistence_threshold(
                        LLMClient(), place_label, revised_objective, threshold_decision["candidates"],
                    )
                    chosen_threshold = threshold_decision["chosen_threshold"]
                    yield line({
                        "type": "agent_decision", "decision": "persistence_threshold",
                        "label": f"Agent re-chose a {chosen_threshold:.2f}m persistence threshold",
                        "reasoning": threshold_decision["reasoning"],
                        "detail": {"candidates": threshold_decision["candidates"], "chosen_threshold": chosen_threshold},
                    })
                    yield line({"type": "step", "label": "Re-running Morse-Smale topology (TTK) with the revised threshold"})
                    t0 = time.time()
                    morse_result = analyze_scalar_field_ttk(elevation_field_for_ttk, persistence_threshold=chosen_threshold)
                    if boundary is not None:
                        morse_result["value_range"] = real_value_range
                    threshold_decision["actual_num_significant_basins"] = morse_result["num_significant_basins"]
                    yield line({"type": "step", "label": "Topology analysis complete", "detail": {
                        "num_significant_basins": morse_result["num_significant_basins"],
                        "num_raw_minima": morse_result["num_raw_minima"],
                        "persistence_threshold": morse_result["persistence_threshold"],
                        "elapsed_seconds": round(time.time() - t0, 2),
                    }})
                except Exception as exc:
                    yield line({
                        "type": "step",
                        "label": "Could not re-decide the threshold -- keeping the original choice",
                        "detail": {"error": str(exc)},
                    })

            # -- Interjection checkpoint 1/3: persistence threshold ---------
            # Only offered when a real decision was made in the first place
            # (threshold_decision is not None) -- nothing to re-decide if
            # this run fell back to the fixed auto default.
            if threshold_decision is not None:
                feedback_holder: list = []
                yield from checkpoint_wait("persistence threshold", feedback_holder)
                if feedback_holder:
                    note = (
                        f"User feedback after seeing {morse_result['num_significant_basins']} significant "
                        f"basins at {chosen_threshold:.2f}m: {feedback_holder[0]}"
                    )
                    yield from _redecide_threshold(note)

            # Ranking + both evidence checks factored into one nested
            # generator so interjection checkpoint 2 below (and, initially,
            # this normal forward flow) can run them either once or twice
            # without duplicating the bodies -- a loop-back to Step 5 must
            # redo Steps 6-7 too, since a re-ranking's D8/weight-sensitivity
            # results would otherwise silently go stale against the new
            # hotspot order. `nonlocal` because Python closures can read an
            # enclosing scope's variables but need an explicit nonlocal
            # declaration to REASSIGN them from a nested function -- which
            # in turn requires each name to already be a real local of
            # generate() itself, hence the explicit None seeding right here
            # (Python's compiler resolves `nonlocal` at compile time, before
            # any of this code actually runs; without an assignment to these
            # names directly in generate()'s own body, there is nothing for
            # _rank_and_check's `nonlocal` to bind to and the module fails to
            # import at all, not just at call time).
            flood_result = None
            morse_vs_d8 = None
            weight_sensitivity = None

            def _rank_and_check():
                nonlocal flood_result, morse_vs_d8, weight_sensitivity
                yield line({"type": "step", "label": "Ranking flood risk & drainage demand", "detail": {
                    "rainfall_scenario": _AGENT_RAINFALL_SCENARIO, "weight_preset": weight_preset_used,
                    "rainfall_source": "noaa_atlas14_live" if rainfall_table else "static_illustrative_table",
                }})
                flood_result = rank_flood_basins(
                    elevation_field, morse_result, rainfall_scenario=_AGENT_RAINFALL_SCENARIO, rainfall_table=rainfall_table,
                    weights=weight_preset_used, runoff_coefficient=_AGENT_RUNOFF_COEFFICIENT,
                    slope_field=slope_field, exposure_field=exposure_field, boundary_mask=boundary_mask,
                    land_cover_grid=land_cover_grid, hydrologic_group=hydrologic_group,
                )
                if boundary is not None:
                    tag_hotspots_within_boundary(flood_result["hotspots"], boundary["polygon"])
                if fema_flood_geojson:
                    tag_hotspots_with_flood_zone(flood_result["hotspots"], fema_flood_geojson)
                yield line({"type": "step", "label": "Flood ranking complete", "detail": {"num_hotspots": len(flood_result["hotspots"])}})

                # Evidence-chain check: does a Morse-persistence-significant
                # basin actually correspond to a real point of convergent
                # drainage under an INDEPENDENT raster-hydrology model (D8),
                # or is "significant" purely a property of the Morse-Smale
                # complex's own topology? Cheap (D8 is pure numpy, no
                # subprocess -- see flow_field.compute_d8_flow, already
                # called once elsewhere for the map's arrow layer). See
                # geoai_agent.morse_validation.compare_morse_to_d8.
                yield line({"type": "step", "label": "Cross-checking Morse basins against an independent D8 flow model"})
                try:
                    # elevation_field_for_ttk, NOT elevation_field -- Morse's
                    # own basins were computed on the boundary-clipped/walled
                    # field (see "Clipping DEM to the real boundary" above),
                    # so D8 must be computed on that SAME field for an
                    # apples-to-apples comparison. Using the unclipped
                    # elevation_field here was a real bug caught by testing:
                    # it made D8 and Morse look like they barely agreed near
                    # the boundary, when that disagreement was actually just
                    # two different elevation fields, not a genuine
                    # Morse-vs-D8 methodological difference.
                    morse_vs_d8 = compare_morse_to_d8(
                        elevation_field_for_ttk, flood_result["hotspots"], morse_result["num_raw_minima"], boundary_mask=boundary_mask,
                    )
                    yield line({
                        "type": "agent_decision", "decision": "morse_vs_d8_validation",
                        "label": f"D8 cross-check: {morse_vs_d8['num_d8_sinks']} raw D8 sinks vs {morse_vs_d8['num_significant_morse_basins']} Morse-significant basins",
                        "reasoning": (
                            f"{morse_vs_d8['d8_sink_agreement_fraction']:.0%} of Morse-significant basins' pour points are also D8 sinks "
                            f"(sanity check -- should be near 100%). "
                            + (
                                f"Morse contributing area correlates with D8 flow accumulation at r={morse_vs_d8['contributing_area_correlation']:.2f} "
                                "across the significant basins."
                                if morse_vs_d8["contributing_area_correlation"] is not None
                                else "Too few significant basins this run to compute a meaningful area correlation."
                            )
                            if morse_vs_d8["d8_sink_agreement_fraction"] is not None
                            else "No significant basins this run to cross-check."
                        ),
                        "detail": morse_vs_d8,
                    })
                except Exception as exc:
                    morse_vs_d8 = None
                    yield line({"type": "step", "label": "Morse-vs-D8 cross-check unavailable", "detail": {"error": str(exc)}})

                # Agent uncertainty check: not an LLM decision, a real re-
                # computation -- re-scores these SAME significant basins
                # under every named weighting scheme (pure Python re-scoring
                # on data already in hand, no new TTK/data-fetch cost) to
                # check whether the #1-ranked basin is robust to a
                # reasonable choice of weighting, or an artifact of whichever
                # one this run happened to use. See geoai_agent.
                # agent_decisions.compute_weight_sensitivity.
                yield line({"type": "step", "label": "Agent checking whether the top-ranked basin is robust to different weighting schemes"})
                try:
                    weight_sensitivity = compute_weight_sensitivity(
                        elevation_field, morse_result, rainfall_scenario=_AGENT_RAINFALL_SCENARIO,
                        rainfall_table=rainfall_table, runoff_coefficient=_AGENT_RUNOFF_COEFFICIENT,
                        slope_field=slope_field, exposure_field=exposure_field, boundary_mask=boundary_mask,
                        land_cover_grid=land_cover_grid, hydrologic_group=hydrologic_group,
                    )
                    yield line({
                        "type": "agent_decision", "decision": "weight_sensitivity",
                        "label": (
                            "Top-ranked basin is stable across weighting schemes" if weight_sensitivity["rank1_stable_across_schemes"]
                            else "Top-ranked basin CHANGES depending on weighting scheme"
                        ),
                        "reasoning": (
                            "The same basin ranks #1 under every weighting scheme tested (balanced, life-safety, "
                            "infrastructure-protection, entropy) -- this finding does not depend on which "
                            "prioritization philosophy was used." if weight_sensitivity["rank1_stable_across_schemes"]
                            else "Different weighting schemes disagree on which basin ranks #1 -- treat the top "
                            "pick as sensitive to which prioritization philosophy is used, not a settled finding."
                        ),
                        "detail": weight_sensitivity,
                    })
                except Exception as exc:
                    weight_sensitivity = None
                    yield line({"type": "step", "label": "Weight-sensitivity check unavailable", "detail": {"error": str(exc)}})

            # Multi-return-period aggregate: how each basin ranks once ALL
            # six return periods are combined (probability-weighted), not
            # just the single 100yr storm the ranking above used -- see
            # flood_tools.compute_expected_annual_runoff's docstring for the
            # EAD methodology and its explicit truncation caveat. Run only
            # on the significant basins (not the hundreds of raw minima
            # below) -- 6x the ranking compute per basin is unnecessary for
            # an all-minima view nobody will read as a ranked list.
            #
            # Factored (like _rank_and_check above) and defined BEFORE the
            # first ranking call below so every loop-back that redoes the
            # ranking -- checkpoint 2's human feedback, or the autonomous
            # self-correction loop further down -- redoes this too;
            # otherwise a re-ranked flood_result would carry EAD numbers
            # computed against the PREVIOUS ranking's weights/basins.
            ead_result = None

            def _aggregate_ead():
                nonlocal ead_result
                yield line({"type": "step", "label": "Aggregating risk across all return periods (expected annual runoff)"})
                try:
                    ead_result = compute_expected_annual_runoff(
                        elevation_field, morse_result, rainfall_table=rainfall_table, weights=weight_preset_used,
                        runoff_coefficient=_AGENT_RUNOFF_COEFFICIENT, slope_field=slope_field, exposure_field=exposure_field,
                        boundary_mask=boundary_mask, land_cover_grid=land_cover_grid, hydrologic_group=hydrologic_group,
                    )
                    ead_by_catchment = {b["catchment_id"]: b["expected_annual_runoff_m3s"] for b in ead_result["basins"]}
                    for h in flood_result["hotspots"]:
                        h["expected_annual_runoff_m3s"] = ead_by_catchment.get(h.get("catchment_id"))
                    yield line({"type": "step", "label": "Expected-annual-runoff aggregation complete"})
                except Exception as exc:
                    ead_result = None
                    for h in flood_result["hotspots"]:
                        h["expected_annual_runoff_m3s"] = None
                    yield line({"type": "step", "label": "Skipping expected-annual-runoff aggregation", "detail": {"error": str(exc)}})

            def _consensus_weight_scheme():
                """Used only by the autonomous self-correction "ranking"
                branch further down: when weight_sensitivity shows the #1
                basin isn't stable across schemes, picks whichever named
                scheme's own #1 pick matches the MAJORITY of schemes' #1
                picks -- a real, evidence-driven alternative to whatever
                weight_preset_used currently is, computed for free from
                weight_sensitivity's already-computed per-scheme rankings
                (no new LLM call, no new TTK pass). Falls back to leaving
                weight_preset_used unchanged if weight_sensitivity is
                unavailable or every scheme failed to score."""
                if not weight_sensitivity or not weight_sensitivity.get("per_scheme"):
                    return weight_preset_used
                top1_by_scheme = {
                    name: v["top_3_catchment_ids"][0] for name, v in weight_sensitivity["per_scheme"].items()
                    if v.get("top_3_catchment_ids") and v["top_3_catchment_ids"][0] is not None
                }
                if not top1_by_scheme:
                    return weight_preset_used
                majority_id = Counter(top1_by_scheme.values()).most_common(1)[0][0]
                # If the CURRENT scheme is already part of the majority
                # group, leave it alone -- picking a different name that
                # happens to agree with it would be a cosmetic "switch" with
                # no actual change in the resulting top pick. (weight_
                # preset_used can also be a custom weights dict rather than
                # one of the named schemes -- .get() on a dict key simply
                # returns None for that, which never equals a real
                # catchment_id, so custom weights always fall through to
                # picking a real majority scheme below.)
                if top1_by_scheme.get(weight_preset_used) == majority_id:
                    return weight_preset_used
                for name, catchment_id in top1_by_scheme.items():
                    if catchment_id == majority_id:
                        return name
                return weight_preset_used  # unreachable (majority_id always comes from this same dict's values)

            yield from _rank_and_check()

            # -- Interjection checkpoint 2/3: ranking & prioritization ------
            # Sits BEFORE the return-period aggregate (Step 8, below) by
            # design -- see Figure 1 -- so feedback on the ranking/weighting
            # itself lands before spending compute aggregating EAD against a
            # ranking that's about to be redone. Loops back to Step 5 and
            # cascades through Steps 6-7 again (see _rank_and_check above) if
            # the user comments on prioritization. A simple keyword match
            # against the three named presets handles the common case of a
            # user naming a philosophy in their own words ("life safety",
            # "infrastructure"); anything else still flows forward as extra
            # context for the final recommendation even when it doesn't
            # change which weights are used.
            feedback_holder: list = []
            yield from checkpoint_wait("ranking & prioritization", feedback_holder)
            if feedback_holder:
                comment = feedback_holder[0]
                lowered = comment.lower()
                if "life" in lowered and "safety" in lowered:
                    weight_preset_used = "life_safety"
                elif "infrastructure" in lowered:
                    weight_preset_used = "infrastructure_protection"
                elif "balanced" in lowered:
                    weight_preset_used = "balanced"
                objective_text = f"{objective_text}\nUser feedback after seeing the ranking: {comment}".strip()
                yield line({"type": "step", "label": "Re-ranking with the requested prioritization"})
                yield from _rank_and_check()

            yield from _aggregate_ead()

            # -- Spatial analysis add-ons: OSM infrastructure proximity, real
            # pipe-capacity data (where loaded -- see load_dc_stormwater.py),
            # terrain metrics, and spatial autocorrelation. Applied only to
            # the significant hotspots (not all_minima_hotspots, hundreds of
            # entries) -- these are all extra per-hotspot analysis on top of
            # an already multi-step pipeline, so bounding scope to the
            # actionable set keeps runtime reasonable.
            #
            # nearby_pipes/river_gauges/osm_infra were already FETCHED way
            # up in the initial parallel batch (alongside elevation/rainfall/
            # etc) -- the raw fetch only ever needed `bounds`. Tagging is the
            # only thing that had to wait until now, since it needs the
            # RANKED hotspots that didn't exist yet back then.
            # GeoJSON built once here (not re-built from nearby_pipes at each
            # of the report_kwargs/map_data call sites below) and, once a
            # boundary exists, clipped to it -- for MAP DISPLAY only. The raw
            # `nearby_pipes` list itself stays UNCLIPPED and is what
            # tag_hotspots_with_drainage_capacity (inside _enrich_hotspots
            # below) uses: a pipe just outside the drawn boundary line can
            # still legitimately be the nearest one to a hotspot sitting
            # close to that edge, so clipping the data BEFORE that proximity
            # search would silently drop a real nearby pipe just for being a
            # few meters over the line -- only the drawn shape should stop
            # at the boundary, not the underlying nearest-feature search.
            pipes_geojson_data = pipes_to_geojson(nearby_pipes) if nearby_pipes else None
            if pipes_geojson_data and boundary is not None:
                pipes_geojson_data = clip_geojson_to_polygon(pipes_geojson_data, boundary["polygon"])

            # Same map-display-only clipping as pipes above -- tag_hotspots_
            # with_river_gauge (inside _enrich_hotspots below) still searches
            # the UNCLIPPED river_gauges list, so a gauge just outside the
            # boundary can still be found as the nearest one to an
            # edge-adjacent basin.
            gauges_geojson_data = gauges_to_geojson(river_gauges) if river_gauges else None
            if gauges_geojson_data and boundary is not None:
                gauges_geojson_data = clip_geojson_to_polygon(gauges_geojson_data, boundary["polygon"])

            # -- Per-hotspot enrichment: pipe/gauge/OSM-infrastructure
            # proximity tagging, terrain metrics, TWI, and building exposure
            # -- factored into a function (not run inline once) because the
            # autonomous self-correction loop further down can rebuild
            # flood_result from scratch (re-deciding the threshold or
            # re-ranking) and must re-apply ALL of this to the NEW hotspots.
            # A real bug caught by review: self-correction used to rebuild
            # flood_result and skip straight to regenerating the
            # recommendation, silently leaving the new hotspots with no
            # pipe/gauge/infrastructure tags, no terrain metrics, and no
            # map_catchment_id -- fields report.py assumes are always
            # present. significant_ways_by_catchment is seeded before this
            # def (assigned via `nonlocal`, re-set on every call); nearby_
            # pipes/river_gauges/osm_infra are NOT re-fetched here -- they
            # were fetched once, up in the initial parallel batch, and stay
            # valid regardless of how many times ranking is redone.
            significant_ways_by_catchment = {}

            def _enrich_hotspots():
                nonlocal significant_ways_by_catchment
                if nearby_pipes:
                    tag_hotspots_with_drainage_capacity(flood_result["hotspots"], nearby_pipes)
                if river_gauges:
                    tag_hotspots_with_river_gauge(flood_result["hotspots"], river_gauges)
                if osm_infra:
                    tag_hotspots_with_nearest_infrastructure(flood_result["hotspots"], osm_infra)
                compute_terrain_metrics(flood_result["hotspots"], elevation_field, morse_result.get("descending_manifold"))
                compute_topographic_wetness_index(flood_result["hotspots"], elevation_field, slope_field)
                significant_ways_by_catchment = compute_building_exposure(
                    flood_result["hotspots"], osm_infra["buildings"] if osm_infra else [],
                    morse_result.get("descending_manifold"), elevation_field, boundary_mask=boundary_mask,
                )

            # -- map_catchment_id reconciliation: maps each SIGNIFICANT
            # hotspot's own catchment_id (from its own higher-persistence-
            # threshold TTK run, `morse_result`) onto the numbering the
            # map's catchment_grid actually uses (the fixed near-zero-
            # threshold "all minima" run, `catchment_arr_all` below) --
            # without this, clicking a hotspot in the report table would
            # highlight whatever unrelated basin happens to share that same
            # integer in a completely different segmentation. Also factored
            # out for the same reason as _enrich_hotspots above: self-
            # correction rebuilds flood_result (and, for the "threshold"
            # target, `morse_result` too), so this must re-run against the
            # NEW hotspots. catchment_arr_all itself is NOT recomputed here
            # -- it comes from a fixed near-zero threshold, independent of
            # which SIGNIFICANT-basin threshold self-correction picks -- so
            # re-decided runs reuse the SAME reconciliation array, just
            # re-applied to the new hotspots' (row, col) positions.
            catchment_arr_all = None
            buildings_geojson_data = None

            def _reconcile_map_catchment_ids():
                nonlocal buildings_geojson_data
                buildings_geojson_data = None
                if catchment_arr_all is None:
                    return
                for h in flood_result["hotspots"]:
                    row, col = h["grid_index"]
                    h["map_catchment_id"] = int(catchment_arr_all[row, col])
                # Buildings were tallied per SIGNIFICANT catchment_id above
                # (significant_ways_by_catchment) -- re-key by the now-
                # reconciled map_catchment_id so buildings_to_geojson's
                # "catchment_id" property matches what the map's own
                # highlightCatchment() actually filters field_geojson by.
                remapped_ways = {}
                for h in flood_result["hotspots"]:
                    ways = significant_ways_by_catchment.get(h.get("catchment_id"))
                    if ways and h.get("map_catchment_id") is not None:
                        remapped_ways[h["map_catchment_id"]] = ways
                if remapped_ways:
                    buildings_geojson_data = buildings_to_geojson(remapped_ways)
                    if boundary is not None:
                        # compute_building_exposure already excluded a
                        # building whose CENTROID falls outside the
                        # boundary (see boundary_mask above) -- but a
                        # building straddling the edge, centroid just
                        # inside, can still have part of its own
                        # footprint poking past the exact boundary line.
                        # A real geometric clip (not just the centroid
                        # test) trims that sliver off too, for the same
                        # "nothing drawn extends past the boundary"
                        # consistency every other layer now has.
                        buildings_geojson_data = clip_geojson_to_polygon(buildings_geojson_data, boundary["polygon"])

            yield line({"type": "step", "label": "Applying stormwater/gauge/infrastructure proximity and terrain metrics (aspect, relief, flow length, TWI, building exposure) to ranked hotspots"})
            _enrich_hotspots()

            # Map-layer conversions for the data sources that don't already
            # produce one above: population/exposure density (same per-cell
            # heatmap shape as elevation -- reuses visualize.field_to_geojson
            # directly), NLCD land cover (categorical, not continuous, so it
            # needs its own coloring -- see land_cover.land_cover_geojson),
            # and OSM roads/water (fetched above alongside buildings for
            # proximity scoring, but not yet turned into their own layer).
            # Rainfall/soil are each a SINGLE value for the whole run (one
            # design storm, one dominant soil group -- nothing spatially
            # varying to draw as a coverage layer), so they're bundled into
            # one info marker at the study area's center instead, so every
            # data source fetched this run has SOMETHING to toggle on the
            # map, not just the ones that vary cell-to-cell.
            exposure_geojson_data = field_to_geojson(exposure_field) if exposure_field is not None else None
            if exposure_geojson_data and boundary is not None:
                exposure_geojson_data = clip_geojson_to_polygon(exposure_geojson_data, boundary["polygon"])

            land_cover_geojson_data = (
                land_cover_geojson(elevation_field, land_cover_grid) if land_cover_grid is not None else None
            )
            if land_cover_geojson_data and boundary is not None:
                land_cover_geojson_data = clip_geojson_to_polygon(land_cover_geojson_data, boundary["polygon"])

            roads_geojson_data = roads_geojson(osm_infra["roads"]) if osm_infra else None
            if roads_geojson_data and boundary is not None:
                roads_geojson_data = clip_geojson_to_polygon(roads_geojson_data, boundary["polygon"])

            water_geojson_data = water_geojson(osm_infra["water"]) if osm_infra else None
            if water_geojson_data and boundary is not None:
                water_geojson_data = clip_geojson_to_polygon(water_geojson_data, boundary["polygon"])

            scenario_used = flood_result["rainfall_scenario"]
            if boundary is not None:
                marker_centroid = boundary["polygon"].centroid
                marker_lat, marker_lon = marker_centroid.y, marker_centroid.x
            else:
                lat_min0, lat_max0, lon_min0, lon_max0 = bounds
                marker_lat, marker_lon = (lat_min0 + lat_max0) / 2, (lon_min0 + lon_max0) / 2
            rainfall_soil_props = {
                "design_storm_label": scenario_used["label"],
                "design_storm_intensity_mm_per_hr": scenario_used["intensity_mm_per_hr"],
                "rainfall_source": (
                    "NOAA Atlas 14 (live, site-specific)" if flood_result.get("rainfall_source") == "noaa_atlas14_live"
                    else "illustrative Mid-Atlantic fallback table"
                ),
            }
            if hydrologic_group:
                rainfall_soil_props["hydrologic_soil_group"] = hydrologic_group
                rainfall_soil_props["soil_source"] = "USDA SSURGO (Soil Data Access)"
            rainfall_soil_geojson_data = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature", "properties": rainfall_soil_props,
                    "geometry": {"type": "Point", "coordinates": [marker_lon, marker_lat]},
                }],
            }

            # Plain raster hydrology (D8 flow direction/accumulation), not a
            # TTK/persistence computation -- see geoai_agent.flow_field. This
            # can't meaningfully fail on data this run already has in hand
            # (it's pure math over elevation_field), but it's still wrapped
            # the same way every other optional layer here is: a bug in a
            # visualization extra shouldn't take down the whole run.
            try:
                flow_field_geojson_data = flow_field_geojson(elevation_field, boundary_mask=boundary_mask)
                yield line({"type": "step", "label": "Flow-direction field computed", "detail": {
                    "num_arrows": len(flow_field_geojson_data["features"]),
                }})
            except Exception as exc:
                flow_field_geojson_data = None
                yield line({"type": "step", "label": "Skipping flow-direction field", "detail": {"error": str(exc)}})

            yield line({"type": "step", "label": "Computing drainage demand for every local minimum"})
            t0 = time.time()
            all_minima_hotspots = None
            try:
                morse_result_all = analyze_scalar_field_ttk(elevation_field_for_ttk, persistence_threshold=_AGENT_ALL_MINIMA_THRESHOLD)
                if boundary is not None:
                    morse_result_all["value_range"] = real_value_range
                flood_result_all = rank_flood_basins(
                    elevation_field, morse_result_all, rainfall_scenario=_AGENT_RAINFALL_SCENARIO, rainfall_table=rainfall_table,
                    weights=weight_preset_used, runoff_coefficient=_AGENT_RUNOFF_COEFFICIENT,
                    slope_field=slope_field, exposure_field=exposure_field, boundary_mask=boundary_mask,
                    land_cover_grid=land_cover_grid, hydrologic_group=hydrologic_group,
                )
                all_minima_hotspots = flood_result_all["hotspots"]
                if boundary is not None:
                    tag_hotspots_within_boundary(all_minima_hotspots, boundary["polygon"])
                if fema_flood_geojson:
                    tag_hotspots_with_flood_zone(all_minima_hotspots, fema_flood_geojson)
                if river_gauges:
                    tag_hotspots_with_river_gauge(all_minima_hotspots, river_gauges)
                compute_topographic_wetness_index(all_minima_hotspots, elevation_field, slope_field)
                compute_building_exposure(
                    all_minima_hotspots, osm_infra["buildings"] if osm_infra else [],
                    morse_result_all.get("descending_manifold"), elevation_field, boundary_mask=boundary_mask,
                )
                # np.array(None) is a 0-d OBJECT array whose .size is 1 (an
                # array with one element -- the None itself), not 0 -- a
                # real bug caught by review: on a perfectly flat area (TTK
                # finds zero critical points, descending_manifold is None),
                # the old `catchment_arr_all = np.array(...); if catchment_
                # arr_all.size:` let a 0-d array through, then
                # `catchment_arr_all[row, col]` raised "too many indices for
                # array: array is 0-dimensional" -- caught by the broad
                # except below, but silently mislabeled as "skipping" a
                # step that had actually already mostly succeeded. Checking
                # `is not None` BEFORE wrapping in np.array (the same
                # pattern flood_tools.py and spatial_analysis.py already use
                # elsewhere) avoids ever constructing that 0-d array.
                descending_manifold_all = morse_result_all.get("descending_manifold")
                catchment_arr_all = np.array(descending_manifold_all) if descending_manifold_all is not None else None
                _reconcile_map_catchment_ids()
                yield line({"type": "step", "label": "All local minima ranked", "detail": {
                    "num_local_minima": len(all_minima_hotspots), "elapsed_seconds": round(time.time() - t0, 2),
                }})
            except Exception as exc:
                yield line({"type": "step", "label": "Skipping all-local-minima view", "detail": {"error": str(exc)}})

            recommendation = None
            system_prompt = prompts.FLOOD_RECOMMENDATION_SYSTEM
            model_name = "unavailable"
            live_source = _describe_elevation_source(dataset.elevation_source)

            # Factored (pure string-building, no events to yield, so a
            # plain function rather than a generator) so the autonomous
            # self-correction loop further down can rebuild this AFTER
            # re-deciding the threshold or re-ranking -- flood_result/
            # threshold_decision/morse_vs_d8/weight_sensitivity are mutable
            # objects updated in place by _redecide_threshold/_rank_and_
            # check via `nonlocal`, but user_context itself is a plain
            # string baked from their CONTENTS at build time; rebuilding it
            # is the only way the next _generate_and_reflect() call
            # actually sees the corrected evidence instead of silently
            # re-writing from the same stale bundle that triggered the
            # self-correction in the first place.
            def _build_user_context():
                nonlocal user_context
                if morse_vs_d8 is None:
                    morse_vs_d8_note = "\nMorse-vs-D8 cross-check was not computed for this run."
                else:
                    correlation_text = (
                        f"contributing-area-to-D8-accumulation correlation r={morse_vs_d8['contributing_area_correlation']:.2f}"
                        if morse_vs_d8["contributing_area_correlation"] is not None
                        else "too few significant basins this run for a meaningful area correlation"
                    )
                    sink_agreement_text = (
                        f"{morse_vs_d8['d8_sink_agreement_fraction']:.0%} of significant basins' pour points are also D8 sinks"
                        if morse_vs_d8["d8_sink_agreement_fraction"] is not None
                        else "no significant basins to check"
                    )
                    morse_vs_d8_note = (
                        "\nMorse-vs-D8 evidence-chain cross-check (does a Morse-persistence-significant basin "
                        "correspond to real convergent drainage under an independent D8 flow model): "
                        f"{morse_vs_d8['num_d8_sinks']} raw D8 sinks found vs {morse_vs_d8['num_significant_morse_basins']} "
                        f"Morse-significant basins ({morse_vs_d8['num_raw_morse_minima']} raw Morse minima before "
                        f"persistence filtering); {sink_agreement_text}; {correlation_text}."
                    )
                user_context = (
                f"'elevation' data for this run came from {live_source}. Basins/peaks were computed by "
                "TTK (Topology ToolKit)'s Morse-Smale complex: persistence-based topological simplification "
                "at the reported threshold, then critical points and descending manifolds (basins) on the "
                "simplified field -- no heuristic depression-filling or breaching preprocessing.\n"
                f"Weighting used: "
                + (
                    "'entropy' (objective, Entropy Weight Method -- see the weighting_method/weights fields "
                    "below, not an expert-picked preset)" if weight_preset_used == "entropy"
                    else "'custom' (hand-set by the user via the Modify-weights sliders, not an expert-named "
                         "preset and not computed by entropy_weights -- see the weighting_method/weights "
                         "fields below for the exact numbers they chose)" if isinstance(weight_preset_used, dict)
                    else f"'{weight_preset_used}'"
                )
                + f". Slope layer available: {slope_field is not None}. "
                f"Exposure layer available: {exposure_field is not None} (source: {exposure_source}). "
                f"Land cover data available: {land_cover_grid is not None}. Soil data available: {hydrologic_group is not None}"
                f"{f' ({hydrologic_group})' if hydrologic_group else ''}.\n"
                f"Flood risk ranking result: {json.dumps(flood_result, ensure_ascii=False)}"
                + (f"\nExpected-annual-runoff aggregation (all six return periods combined): "
                   f"{json.dumps(ead_result, ensure_ascii=False)}" if ead_result else
                   "\nExpected-annual-runoff aggregation was not computed for this run.")
                + (f"\nAgent's own analysis-strategy decision (why this run used the "
                   f"'{strategy_decision['weight_preset']}' weighting preset): {strategy_decision['reasoning']}"
                   if strategy_decision else "")
                + (f"\nAgent's own persistence-threshold decision (why this run used a "
                   f"{threshold_decision['chosen_threshold']:.2f}m threshold out of the candidates "
                   f"{json.dumps(threshold_decision['candidates'], ensure_ascii=False)}): {threshold_decision['reasoning']}"
                   if threshold_decision else "")
                + (f"\nAgent's own weight-sensitivity check across balanced/life_safety/infrastructure_"
                   f"protection/entropy weighting schemes: "
                   f"{'the #1-ranked basin is STABLE across every scheme tested' if weight_sensitivity['rank1_stable_across_schemes'] else 'the #1-ranked basin CHANGES depending on which scheme is used -- treat the top pick as sensitive to prioritization philosophy'}. "
                   f"Full per-scheme result: {json.dumps(weight_sensitivity['per_scheme'], ensure_ascii=False)}"
                   if weight_sensitivity else "\nWeight-sensitivity check was not computed for this run.")
                + morse_vs_d8_note
                + ("\nWhen writing the recommendation below, reference the agent's own strategy/threshold "
                   "decisions, the weight-sensitivity finding, and the Morse-vs-D8 cross-check directly where "
                   "relevant -- explain WHY this run made the choices it did and whether independent evidence "
                   "supports treating these basins as real drainage features, not just what the final numbers are."
                   if (strategy_decision or threshold_decision or weight_sensitivity or morse_vs_d8) else "")
                )

            user_context = None
            _build_user_context()

            # Steps 9-10 (execute -> reflect) factored the same way Steps
            # 5-7 were above, so interjection checkpoint 3 below can re-run
            # both -- a fresh recommendation written with the user's late
            # feedback folded into user_context still needs its own
            # self-review, not a reuse of whatever the first draft's review
            # found.
            recommendation = None
            reflection = None
            llm = None

            def _generate_and_reflect():
                nonlocal llm, model_name, recommendation, reflection
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

                # -- Reflection pattern: execute (above) -> critique -> refine
                # -- a SECOND LLM call re-reads the same evidence bundle the
                # writer saw (user_context) alongside the recommendation it
                # just wrote, specifically checking for claims the evidence
                # doesn't support -- not a free-form "make it better"
                # rewrite. Only ever replaces `recommendation` when the
                # reviewer both flags a real issue AND returns a usable
                # revision (see reflect_on_recommendation's own "revised_
                # applied" guard) -- a failed or malformed reflection call
                # falls back to the un-reviewed recommendation, never to
                # nothing, same degrade-gracefully contract as every other
                # agent decision in this pipeline.
                reflection = None
                if llm is not None and recommendation and not recommendation.startswith("(recommendation unavailable"):
                    yield line({"type": "step", "label": "Agent reviewing its own recommendation against the evidence"})
                    t0 = time.time()
                    try:
                        reflection = reflect_on_recommendation(llm, recommendation, user_context)
                        if reflection["revision_applied"]:
                            recommendation = reflection["final_recommendation"]
                        # Three distinct outcomes, not two -- issues_found
                        # and revision_applied can disagree (the model can
                        # flag a real issue but return an empty/malformed
                        # revision), and collapsing them into one "issues_
                        # found" field used to make that case print "found
                        # no unsupported claims" directly above a critique
                        # describing a real problem (a real bug caught by
                        # review).
                        if reflection["revision_applied"]:
                            label = "Agent revised its own recommendation after self-review"
                        elif reflection["issues_found"]:
                            label = "Agent's self-review flagged an issue but returned no usable revision -- keeping the original recommendation"
                        else:
                            label = "Agent's self-review found no unsupported claims"
                        yield line({
                            "type": "agent_decision", "decision": "recommendation_reflection",
                            "label": label,
                            "reasoning": reflection["critique"],
                            "detail": {
                                "issues_found": reflection["issues_found"], "revision_applied": reflection["revision_applied"],
                                "elapsed_seconds": round(time.time() - t0, 2),
                            },
                        })
                    except Exception as exc:
                        reflection = None
                        yield line({"type": "step", "label": "Self-review unavailable -- keeping the un-reviewed recommendation", "detail": {"error": str(exc)}})

            yield from _generate_and_reflect()

            # -- Autonomous self-correction (no human involved) -------------
            # The agent's OWN self-review can conclude that an earlier
            # ANALYSIS decision -- not just this text -- needs to be redone
            # (see agent_decisions.reflect_on_recommendation's "reanalysis_
            # target": the D8 cross-check or weight-sensitivity evidence
            # looking internally inconsistent with what the draft claims).
            # This is a genuine loop-back triggered by the agent's own
            # judgment, distinct from -- and independent of -- the three
            # human-interjection checkpoints elsewhere in this pipeline.
            # Bounded to firing at most once (no re-check of the SECOND
            # reflection's own reanalysis_target): otherwise a terrain
            # whose D8 agreement is genuinely, persistently weak (a real,
            # reportable finding in its own right -- see Section 3 of this
            # project's own methodology writeup) would loop forever instead
            # of ever reaching the user.
            if reflection and reflection.get("reanalysis_target") == "threshold" and threshold_decision is not None:
                yield line({
                    "type": "agent_decision", "decision": "self_correction",
                    "label": "Self-review flagged the persistence threshold itself, not just the wording -- re-deciding it autonomously",
                    "reasoning": reflection["critique"],
                    "detail": {"reanalysis_target": "threshold"},
                })
                yield from _redecide_threshold(
                    f"Agent's own self-review of the previous draft found a deeper issue than wording: {reflection['critique']}"
                )
                yield from _rank_and_check()
                yield from _aggregate_ead()
                _enrich_hotspots()
                _reconcile_map_catchment_ids()
                _build_user_context()
                yield line({"type": "step", "label": "Regenerating the recommendation after self-correction"})
                yield from _generate_and_reflect()
            elif reflection and reflection.get("reanalysis_target") == "ranking":
                # Unlike the "threshold" branch above (which genuinely asks
                # the LLM to re-decide via _redecide_threshold), just calling
                # _rank_and_check() again here with nothing changed would be
                # a no-op: rank_flood_basins/entropy_weights are
                # deterministic, so identical inputs reproduce the exact
                # same ranking -- a real bug caught by review (confirmed
                # independently by 3 review angles). If weight_sensitivity
                # shows the #1 basin ISN'T stable across schemes -- the
                # usual reason this target fires -- switch weight_preset_
                # used to whichever named scheme's own #1 pick matches the
                # MAJORITY of schemes' #1 picks, a real evidence-driven
                # change that costs no new LLM call or TTK pass (weight_
                # sensitivity already computed every scheme's ranking).
                new_scheme = _consensus_weight_scheme()
                scheme_changed = weight_sensitivity is not None and new_scheme != weight_preset_used
                if scheme_changed:
                    weight_preset_used = new_scheme
                yield line({
                    "type": "agent_decision", "decision": "self_correction",
                    "label": (
                        f"Self-review flagged the ranking itself -- switching to the \"{new_scheme}\" weighting scheme "
                        "(the majority pick across weight-sensitivity schemes) and re-ranking autonomously" if scheme_changed
                        else "Self-review flagged the ranking itself, not just the wording -- re-ranking autonomously"
                    ),
                    "reasoning": reflection["critique"],
                    "detail": {"reanalysis_target": "ranking", "weight_preset_used": weight_preset_used, "scheme_changed": scheme_changed},
                })
                yield from _rank_and_check()
                yield from _aggregate_ead()
                _enrich_hotspots()
                _reconcile_map_catchment_ids()
                _build_user_context()
                yield line({"type": "step", "label": "Regenerating the recommendation after self-correction"})
                yield from _generate_and_reflect()

            # -- Interjection checkpoint 3/3: the recommendation itself -----
            # Loops back to Step 9 (re-generate) and cascades through Step
            # 10 (re-reflect) if the user reacts to the draft -- e.g. "you're
            # wrong about the exposure at #2" or "explain the D8 disagreement
            # more." The feedback is folded into user_context, the SAME
            # evidence bundle the writer and the reviewer both already read,
            # so the next draft (and its own fresh self-review) is grounded
            # in the real numbers plus the user's comment, not the comment
            # alone.
            feedback_holder: list = []
            yield from checkpoint_wait("the recommendation", feedback_holder)
            if feedback_holder:
                user_context = f"{user_context}\nUser feedback on the draft recommendation above: {feedback_holder[0]}"
                yield line({"type": "step", "label": "Regenerating the recommendation with that feedback"})
                yield from _generate_and_reflect()

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
            # cutoff_persistence_m can itself be None: _persistence_sort_key
            # sorts None-persistence entries (global sinks) FIRST, so if
            # truncation lands exactly on a run of >= _QA_ALL_MINIMA_CAP such
            # entries, the last INCLUDED one is itself None-persistence -- a
            # real crash path caught by review (an f-string ":.3f" on None
            # raises TypeError). Worded without a numeric cutoff in that case
            # rather than guessing a fake threshold.
            if not all_minima_truncated:
                cutoff_note = "Every raw local minimum from this run is included below -- nothing was omitted."
            elif cutoff_persistence_m is None:
                cutoff_note = (
                    f"Only the {len(all_minima_included)} most persistent raw minima (of "
                    f"{len(all_minima_sorted)} total) are included below; the omitted ones are all "
                    "lower-persistence than the global-sink entries included here."
                )
            else:
                cutoff_note = (
                    f"Only the {len(all_minima_included)} most persistent raw minima (of "
                    f"{len(all_minima_sorted)} total) are included below; every omitted one has "
                    f"persistence_m <= {cutoff_persistence_m:.3f}, so a 'persistence greater than Xm' "
                    f"filter is still complete as long as X >= {cutoff_persistence_m:.3f}."
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
                "rainfall_source": flood_result["rainfall_source"],
                "weights": flood_result["weights"],
                "weighting_method": flood_result.get("weighting_method", "preset"),
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
                            "fema_flood_zone": h.get("fema_flood_zone"),
                            "fema_flood_severity": h.get("fema_flood_severity"),
                            "nearest_gauge": h.get("nearest_gauge"),
                            "twi": h.get("twi"),
                            "runoff_coefficient": h.get("runoff_coefficient"),
                            "runoff_coefficient_source": h.get("runoff_coefficient_source"),
                            "buildings_in_catchment": h.get("buildings_in_catchment"),
                            "building_density_per_km2": h.get("building_density_per_km2"),
                        }
                        for h in all_minima_included
                    ],
                },
                "exposure_data_supplied": exposure_field is not None,
                "exposure_source": exposure_source,
                "fema_flood_data_available": fema_flood_geojson is not None,
                "drainage_pipe_data_available": nearby_pipes is not None,
                "osm_infrastructure_data_available": osm_infra is not None,
                "river_gauge_data_available": bool(river_gauges),
                "land_cover_data_available": land_cover_grid is not None,
                "soil_data_available": hydrologic_group is not None,
                "expected_annual_runoff_available": ead_result is not None,
                "expected_annual_runoff_note": ead_result["note"] if ead_result else None,
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
                place_label=place_label, bounds=bounds, elevation=elevation_field,
                elevation_source=dataset.elevation_source, morse_result=morse_result, flood_result=flood_result,
                recommendation=recommendation, system_prompt=system_prompt, user_context=user_context,
                reflection=reflection,
                model_name=model_name, exposure_provided=exposure_field is not None,
                generated_at=datetime.date.today().isoformat(),
                boundary_polygon=boundary["polygon"] if boundary else None,
                boundary_geojson=boundary["geojson"] if boundary else None,
                boundary_display_name=boundary["display_name"] if boundary else None,
                boundary_mask=boundary_mask,
                all_minima_hotspots=all_minima_hotspots,
                catchment_grid=catchment_grid,
                fema_flood_geojson=fema_flood_geojson,
                pipes_geojson=pipes_geojson_data,
                gauges_geojson=gauges_geojson_data,
                buildings_geojson=buildings_geojson_data,
                flow_field_geojson=flow_field_geojson_data,
                exposure_geojson=exposure_geojson_data,
                land_cover_geojson_data=land_cover_geojson_data,
                roads_geojson=roads_geojson_data,
                water_geojson=water_geojson_data,
                rainfall_soil_geojson=rainfall_soil_geojson_data,
                drainage_pipe_data_available=nearby_pipes is not None,
                osm_infrastructure_data_available=osm_infra is not None,
                river_gauge_data_available=bool(river_gauges),
                exposure_source=exposure_source,
                land_cover_data_available=land_cover_grid is not None,
                soil_data_available=hydrologic_group is not None,
                expected_annual_runoff=ead_result,
                strategy_decision=strategy_decision,
                threshold_decision=threshold_decision,
                weight_sensitivity=weight_sensitivity,
                morse_vs_d8=morse_vs_d8,
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
                "field_geojson": elevation_hillshade_geojson(elevation_field, catchment_grid=catchment_grid, boundary_mask=boundary_mask),
                "elevation_legend": terrain_legend_stops(elevation_field),
                "all_minima_geojson": all_minima_geojson(all_minima_hotspots) if all_minima_hotspots else None,
                "boundary_geojson": boundary["geojson"] if boundary else None,
                "fema_flood_geojson": fema_flood_geojson,
                "pipes_geojson": pipes_geojson_data,
                "gauges_geojson": gauges_geojson_data,
                "buildings_geojson": buildings_geojson_data,
                "flow_field_geojson": flow_field_geojson_data,
                "exposure_geojson": exposure_geojson_data,
                "land_cover_geojson": land_cover_geojson_data,
                "roads_geojson": roads_geojson_data,
                "water_geojson": water_geojson_data,
                "rainfall_soil_geojson": rainfall_soil_geojson_data,
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
# Default row count for "all_minima_full.entries" when a question doesn't
# need row-level enumeration (see _question_needs_minima_enumeration below)
# -- small enough to keep the common-case prompt light, big enough that the
# model still has real per-basin data to point to if it wants an example.
_QA_MINIMA_SAMPLE_DEFAULT = 25

# Words/phrases that signal a question actually needs to filter, count, or
# list over the full raw-minima table rather than just reasoning about the
# already-short "hotspots" shortlist. Deliberately broad/over-inclusive --
# a false positive here just means an unnecessarily large prompt (the
# previous, always-on behavior), while a false negative means a genuine
# "list/count" question silently gets an incomplete answer, which is worse.
_MINIMA_ENUMERATION_HINTS = (
    "how many", "count", "list", "enumerate", "every basin", "every minim",
    "all basins", "all minima", "all local minima", "which basins", "which minima",
    "filter", "greater than", "less than", "more than", "at least", "at most",
    "over ", "under ", "above ", "below ", "between ",
)


def _question_needs_minima_enumeration(question: str) -> bool:
    q = question.lower()
    return any(hint in q for hint in _MINIMA_ENUMERATION_HINTS)


def _trim_minima_for_question(context: dict, question: str) -> None:
    """Mutates context in place: unless the question looks like it needs
    to filter/count/list over the raw-minima table, shrink
    "all_minima_full.entries" down to a small sample instead of resending
    every raw minimum (up to _QA_ALL_MINIMA_CAP=500 rows) on every single
    question.

    Root-caused a live 504 DEADLINE_EXCEEDED on a plain "which hotspot
    needs attention first, and why?" question -- a question the system
    prompt itself says should be answered from "hotspots" (already a short,
    ranked list), not "all_minima_full". The full 500-row dump was being
    attached regardless, making an unrelated question's prompt large enough
    to occasionally exceed Gemini's own generation deadline. The already-
    documented "truncated"/"cutoff_note" fields exist precisely to tell the
    model a list is incomplete, so reuse them here rather than inventing a
    parallel mechanism.
    """
    full = context.get("all_minima_full")
    if not isinstance(full, dict) or full.get("truncated"):
        return  # already short, or already marked incomplete -- leave as-is
    entries = full.get("entries")
    if not isinstance(entries, list) or len(entries) <= _QA_MINIMA_SAMPLE_DEFAULT:
        return
    if _question_needs_minima_enumeration(question):
        return

    total = full.get("total_raw_minima", len(entries))
    sample = entries[:_QA_MINIMA_SAMPLE_DEFAULT]
    full["entries"] = sample
    full["included_count"] = len(sample)
    full["truncated"] = True
    full["cutoff_note"] = (
        f"Only the {len(sample)} most persistent raw minima (of {total} total) are included below -- "
        "trimmed for this question since it didn't ask to filter/count/list basins. If the reviewer's "
        "question actually needs the complete raw-minima table, say so plainly rather than presenting "
        "this sample as the whole picture; a follow-up question phrased as a filter/count/list will "
        "receive the full table."
    )


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
    if "DEADLINE_EXCEEDED" in text or "504" in text:
        return (
            "Gemini took too long to respond (it hit its own generation deadline) -- "
            "this is usually transient, please try asking again."
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

    _trim_minima_for_question(context, question)

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
