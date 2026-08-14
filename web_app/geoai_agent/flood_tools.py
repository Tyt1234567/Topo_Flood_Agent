"""
Flood risk and drainage-demand domain engine.

Built on top of a TTK Morse-Smale complex of a terrain elevation field
(geoai_agent.ttk_engine.analyze_scalar_field_ttk): every significant
topological depression (a "basin" in that result, i.e. the descending
manifold of a persistent local minimum) is treated as a candidate
flood-prone sink, and scored/ranked using

- Topological susceptibility: the basin minimum's persistence, i.e. how
  deep/well-defined the depression is against the surrounding terrain
  (not just its raw elevation) -- this is the actual point of building
  flood analysis on Morse theory instead of a flat "low elevation = risk"
  heuristic.
- Contributing area: the basin's cell count, converted to real ground
  area from the field's lat/lon grid spacing -- how much surface can
  actually drain into this sink.
- Slope: flatter ground around the sink ponds more readily than a steep
  channel passing through the same low point.
- Rainfall intensity: from RAINFALL_SCENARIOS, an illustrative design
  storm the caller selects (this is the HITL "which storm are we
  planning for" control).
- Exposure (optional): a population/asset density layer sampled at the
  basin, if the caller has one -- weight 0 (excluded) if not supplied,
  rather than inventing a number.

All five factor weights are caller-adjustable (WEIGHT_PRESETS gives named
starting points) -- this is the HITL "life-safety vs. economic recovery"
control from the project brief: an emergency planner can shift weight
toward exposure/topological susceptibility, or toward contributing
area/rainfall if the goal is engineering capacity planning instead.

Drainage demand per basin is the design peak runoff (Rational Method,
Q = C*I*A/3.6 with I in mm/hr and A in km2, giving Q in m3/s) that would
need to be carried away during the selected storm -- NOT compared against
an "existing capacity", since no real stormwater infrastructure dataset is
plugged in here; a caller with one can subtract it from
estimated_peak_runoff_m3s themselves.
"""

from __future__ import annotations

import math

from geoai_agent.morse_topology import ScalarField

# Illustrative 1-hour-duration design-storm intensities for the College
# Park, MD / Mid-Atlantic area, approximating NOAA Atlas 14 Volume 2 point
# precipitation frequency estimates -- NOT queried live from NOAA. For a
# real engineering study, replace these with exact point values from
# https://hdsc.nws.noaa.gov/pfds/ for the site in question; a different
# region entirely would need a different table.
RAINFALL_SCENARIOS = {
    "2yr": {"label": "2-year storm", "return_period_years": 2, "intensity_mm_per_hr": 36.0},
    "10yr": {"label": "10-year storm", "return_period_years": 10, "intensity_mm_per_hr": 51.0},
    "25yr": {"label": "25-year storm", "return_period_years": 25, "intensity_mm_per_hr": 58.0},
    "50yr": {"label": "50-year storm", "return_period_years": 50, "intensity_mm_per_hr": 66.0},
    "100yr": {"label": "100-year storm", "return_period_years": 100, "intensity_mm_per_hr": 76.0},
    "500yr": {"label": "500-year storm", "return_period_years": 500, "intensity_mm_per_hr": 102.0},
}
_MAX_RAINFALL_INTENSITY = max(s["intensity_mm_per_hr"] for s in RAINFALL_SCENARIOS.values())

_DEFAULT_WEIGHTS = {
    "topological": 0.30, "contributing_area": 0.25, "rainfall": 0.20, "slope": 0.15, "exposure": 0.10,
}
# Named HITL starting points -- emergency planners/community reps pick one
# of these (or hand-tune _DEFAULT_WEIGHTS-shaped overrides) to shift what
# "highest risk" means, per the project brief's life-safety-vs-recovery
# trade-off control.
WEIGHT_PRESETS = {
    "balanced": _DEFAULT_WEIGHTS,
    "life_safety": {
        "topological": 0.25, "contributing_area": 0.15, "rainfall": 0.15, "slope": 0.10, "exposure": 0.35,
    },
    "infrastructure_protection": {
        "topological": 0.30, "contributing_area": 0.30, "rainfall": 0.25, "slope": 0.15, "exposure": 0.0,
    },
}

_METERS_PER_DEGREE_LAT = 111_320


def _cell_area_km2(field: ScalarField) -> float:
    """Ground area of one grid cell, from the field's own lat/lon spacing
    (assumes a geographic field, i.e. a RealWorldDataset layer -- xs are
    longitude, ys are latitude)."""
    if len(field.xs) < 2 or len(field.ys) < 2:
        raise ValueError("flood_tools needs a grid with at least 2 points on each axis")
    lon_spacing_deg = abs(float(field.xs[1] - field.xs[0]))
    lat_spacing_deg = abs(float(field.ys[1] - field.ys[0]))
    meters_per_deg_lon = _METERS_PER_DEGREE_LAT * math.cos(math.radians(float(field.ys.mean())))
    dx_m = lon_spacing_deg * meters_per_deg_lon
    dy_m = lat_spacing_deg * _METERS_PER_DEGREE_LAT
    return (dx_m * dy_m) / 1_000_000.0


def rational_method_runoff(runoff_coefficient: float, intensity_mm_per_hr: float, area_km2: float) -> float:
    """Peak runoff (m3/s) via the metric Rational Method: Q = C*I*A/3.6,
    I in mm/hr, A in km2. A standard first-cut design-storm sizing method
    for small watersheds (valid roughly up to a few km2 / short times of
    concentration -- the basin sizes this operates on, individual
    Morse-basin depressions, are comfortably within that range)."""
    return runoff_coefficient * intensity_mm_per_hr * area_km2 / 3.6


def _normalized_weights(weights: dict[str, float] | None) -> dict[str, float]:
    w = dict(_DEFAULT_WEIGHTS if weights is None else weights)
    missing = set(_DEFAULT_WEIGHTS) - set(w)
    if missing:
        raise ValueError(f"weights missing required keys: {sorted(missing)}")
    total = sum(w.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return {k: v / total for k, v in w.items()}


def rank_flood_basins(
    elevation_field: ScalarField,
    morse_result: dict,
    rainfall_scenario: str = "10yr",
    weights: dict[str, float] | str | None = None,
    runoff_coefficient: float = 0.5,
    slope_field: ScalarField | None = None,
    exposure_field: ScalarField | None = None,
) -> dict:
    """Ranks every significant basin in `morse_result` (a TTK
    analyze_scalar_field_ttk result for `elevation_field`) by flood risk,
    and reports each one's Rational-Method drainage demand under the
    selected rainfall scenario.

    weights: a WEIGHT_PRESETS name, a custom {"topological",
    "contributing_area", "rainfall", "slope", "exposure"} dict (need not
    sum to 1 -- renormalized), or None for the "balanced" preset.
    """
    if rainfall_scenario not in RAINFALL_SCENARIOS:
        raise ValueError(
            f"unknown rainfall_scenario '{rainfall_scenario}', available: {list(RAINFALL_SCENARIOS)}"
        )
    if isinstance(weights, str):
        if weights not in WEIGHT_PRESETS:
            raise ValueError(f"unknown weight preset '{weights}', available: {list(WEIGHT_PRESETS)}")
        weights = WEIGHT_PRESETS[weights]
    w = _normalized_weights(weights)

    scenario = RAINFALL_SCENARIOS[rainfall_scenario]
    cell_area_km2 = _cell_area_km2(elevation_field)
    value_min, value_max = morse_result["value_range"]
    value_range = (value_max - value_min) or 1.0
    rainfall_score = scenario["intensity_mm_per_hr"] / _MAX_RAINFALL_INTENSITY

    basins = morse_result.get("basins", [])
    if not basins:
        return {
            "rainfall_scenario": scenario, "weights": w, "runoff_coefficient": runoff_coefficient, "hotspots": [],
        }
    max_area_km2 = max(b["size_cells"] for b in basins) * cell_area_km2

    hotspots = []
    for basin in basins:
        minimum = basin["minimum"]
        i, j = minimum["grid_index"]
        contributing_area_km2 = basin["size_cells"] * cell_area_km2

        persistence = minimum["persistence"]
        topological_score = 1.0 if persistence is None else min(1.0, persistence / value_range)
        area_score = contributing_area_km2 / max_area_km2 if max_area_km2 > 0 else 0.0

        if slope_field is not None:
            slope_here = float(slope_field.values[i, j])
            slope_range = float(slope_field.values.max() - slope_field.values.min()) or 1.0
            # flatter ground ponds more: invert the normalized slope
            slope_score = 1.0 - min(1.0, max(0.0, slope_here / slope_range))
        else:
            slope_score = 0.5  # neutral -- no slope layer supplied for this run

        if exposure_field is not None:
            exp_here = float(exposure_field.values[i, j])
            exp_min, exp_max = float(exposure_field.values.min()), float(exposure_field.values.max())
            exp_range = (exp_max - exp_min) or 1.0
            exposure_score = (exp_here - exp_min) / exp_range
        else:
            exposure_score = 0.5  # neutral -- no exposure layer supplied for this run

        risk_score = (
            w["topological"] * topological_score
            + w["contributing_area"] * area_score
            + w["rainfall"] * rainfall_score
            + w["slope"] * slope_score
            + w["exposure"] * exposure_score
        )

        hotspots.append({
            "grid_index": [i, j],
            "coords": minimum["coords"],
            "elevation_m": minimum["value"],
            "persistence_m": persistence,
            "contributing_area_km2": contributing_area_km2,
            "topological_score": topological_score,
            "contributing_area_score": area_score,
            "slope_score": slope_score,
            "rainfall_score": rainfall_score,
            "exposure_score": exposure_score,
            "exposure_layer_available": exposure_field is not None,
            "slope_layer_available": slope_field is not None,
            "risk_score": risk_score,
            "estimated_peak_runoff_m3s": rational_method_runoff(
                runoff_coefficient, scenario["intensity_mm_per_hr"], contributing_area_km2
            ),
        })

    hotspots.sort(key=lambda h: h["risk_score"], reverse=True)
    for rank, h in enumerate(hotspots, start=1):
        h["rank"] = rank

    return {
        "rainfall_scenario": scenario, "weights": w, "runoff_coefficient": runoff_coefficient, "hotspots": hotspots,
    }
