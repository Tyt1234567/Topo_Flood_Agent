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
- Rainfall intensity: a design storm the caller selects (this is the HITL
  "which storm are we planning for" control), preferably a genuine
  site-specific NOAA Atlas 14 estimate live-fetched via rainfall.
  fetch_noaa_atlas14_rainfall and passed in as `rainfall_table`, falling
  back to the illustrative Mid-Atlantic RAINFALL_SCENARIOS table when no
  live estimate is available for the site.
- Exposure (optional): a population/asset density layer sampled at the
  basin, if the caller has one -- weight 0 (excluded) if not supplied,
  rather than inventing a number.
- Poverty (optional): US Census-tract poverty rate sampled at the basin
  (geoai_agent.real_data.fetch_poverty_rate_field), same neutral-0.5
  fallback as exposure when not supplied. A genuinely different dimension
  from exposure/population density -- exposure measures HOW MANY people/
  structures are there, poverty measures how financially able the people
  who ARE there are to absorb, evacuate from, or recover from a flood
  (the classic physical-exposure vs. social-vulnerability split in
  disaster-risk literature, e.g. Cutter et al.'s hazards-of-place model,
  and the same distinction CDC's own Social Vulnerability Index -- already
  a separate, non-scored context layer elsewhere in this project -- draws
  between its "exposure" and "socioeconomic status" themes).

Runoff coefficient itself can also come from real data now: pass
`land_cover_grid` (geoai_agent.land_cover.fetch_nlcd_landcover_grid) and
`hydrologic_group` (geoai_agent.soil_data.fetch_dominant_hydrologic_soil_
group) together, and each basin gets its OWN coefficient derived from the
actual NLCD land cover + SSURGO soil group across its catchment via the
real SCS Curve Number runoff equation (see runoff_coefficient_from_land_
cover below), instead of one flat constant applied to every basin in the
whole run. A basin with no usable land-cover data (or if either input is
missing) falls back to the flat `runoff_coefficient` parameter exactly as
before -- see each hotspot's own "runoff_coefficient_source".

All six factor weights are caller-adjustable (WEIGHT_PRESETS gives named
starting points, entropy_weights below gives an objective data-driven
alternative, and a caller/the frontend's Modify-weights sliders can pass
any custom {"topological", "contributing_area", "rainfall", "slope",
"exposure", "poverty"} dict directly) -- this is the HITL "life-safety vs.
economic recovery" control from the project brief: an emergency planner
can shift weight toward exposure/poverty/topological susceptibility, or
toward contributing area/rainfall if the goal is engineering capacity
planning instead.
_DEFAULT_WEIGHTS and every WEIGHT_PRESETS entry are an expert-defined
PRIORITIZATION scheme, not weights fit or validated against historical
flood-outcome data -- risk_score is a transparent, reproducible composite
for RANKING candidate basins relative to each other, not a calibrated
probability of flooding. See report.py's "How risk_score is computed"
section for the exact per-factor formulas and this same caveat, surfaced
to end users.

_DEFAULT_WEIGHTS's first five factors are not pulled out of thin air:
grouping this project's factors against the nine physical factors used in
a real published AHP flood-hazard-index study -- Mohammed, Z.T., Hussein,
L.Y. & Abood, M.H., "Potential Flood Hazard Mapping Based on GIS and
Analytical Hierarchy Process," Journal of Water Management Modeling 32,
C528 (2024), https://doi.org/10.14796/JWMM.C528, a Diyala governorate,
Iraq case study -- lands close to that paper's own AHP-derived weights:
(distance-to-river 16.0% + elevation 13.5%) = 29.5% maps to this
project's "topological" susceptibility (27%); (TWI 15.2% + drainage
density 10.6%) = 25.8% maps to "contributing_area" (22%); their
"rainfall" 15.1% is in the same range as this project's 18%; "slope"
11.1% is in the same range as this project's 13%; (LULC 7.0% + NDVI
6.2%) = 13.2% is in the same range as "exposure" 10% (though LULC/NDVI
there measure ground cover's effect on runoff, not population/building
exposure -- a genuinely different concept this project's exposure_score
actually measures). This is a real, checkable anchor point for "is this
ballpark reasonable," not proof this project's specific weights are
correct -- a single study from one river basin doesn't generalize, the
factor groupings above are an approximate correspondence rather than an
exact match (this project's factors come from Morse-theory basin
topology, not the raw GIS layers AHP studies pair-compare directly), and
other published AHP flood studies put rainfall's weight anywhere from
roughly 8% to 55% depending on how much precipitation actually varies
across THAT study's own area -- there is no single literature-consensus
number to transplant here even if the factor definitions matched exactly.
"poverty" (10% in _DEFAULT_WEIGHTS) has no anchor in that AHP study at
all -- it doesn't model social vulnerability as a factor -- so its weight
here is this project's own judgment call, not backed by that citation the
way the other five's ballpark is; see the class docstring's "Poverty"
bullet above for the (different) literature this factor draws on instead.

A separate, and more directly decisive, fact about the rainfall weight
specifically: RAINFALL_SCENARIOS applies ONE selected storm uniformly to
the whole study area, so rainfall_score is IDENTICAL for every basin in a
single run. Adding the same w["rainfall"] * rainfall_score term to every
basin's risk_score shifts every basin's score by the same amount, which
cannot change their relative ORDER -- verified by hand: re-ranking the
same basins with rainfall's raw weight set to 0%, 20%, and 55% (holding
the other five weights' relative proportions fixed) produces the exact
same ranking every time. So in preset/custom mode, rainfall's displayed
weight percentage is real (it changes risk_score's absolute magnitude,
and DOES matter if risk_score is ever compared ACROSS different storm
scenarios, e.g. compute_expected_annual_runoff's return-period sweep) but
has NO effect on which basin ranks highest within one scenario -- exactly
the same conclusion entropy_weights reaches mathematically on its own
(weight 0, not 20%), just for a reason readers shouldn't have to take on
faith.

Drainage demand per basin is the design peak runoff (Rational Method,
Q = C*I*A/3.6 with I in mm/hr and A in km2, giving Q in m3/s) that would
need to be carried away during the selected storm -- NOT compared against
an "existing capacity", since no real stormwater infrastructure dataset is
plugged in here; a caller with one can subtract it from
estimated_peak_runoff_m3s themselves.
"""

from __future__ import annotations

import math

import numpy as np

from geoai_agent.morse_topology import ScalarField

# FALLBACK table only: illustrative 1-hour-duration design-storm intensities
# for the College Park, MD / Mid-Atlantic area, approximating NOAA Atlas 14
# Volume 2 point precipitation frequency estimates. Used whenever a genuine
# per-location live fetch (rainfall.fetch_noaa_atlas14_rainfall -- exact
# point values from https://hdsc.nws.noaa.gov/pfds/) isn't available for the
# site in question (outside NOAA Atlas 14's US-and-territories coverage, or
# the request failed) -- see rank_flood_basins's `rainfall_table` parameter
# below. A different region entirely would need a different table if this
# fallback is ever hit for it; the live fetch, when it succeeds, needs no
# such caveat.
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
    "topological": 0.27, "contributing_area": 0.22, "rainfall": 0.18, "slope": 0.13, "exposure": 0.10,
    "poverty": 0.10,
}
# Named HITL starting points -- emergency planners/community reps pick one
# of these (or hand-tune _DEFAULT_WEIGHTS-shaped overrides) to shift what
# "highest risk" means, per the project brief's life-safety-vs-recovery
# trade-off control. life_safety splits its exposure-heavy weight between
# exposure (how many people/structures) and poverty (how able those people
# are to evacuate/recover) rather than folding poverty into exposure's own
# number -- infrastructure_protection zeroes BOTH out, same as it already
# zeroed exposure alone before poverty existed, since neither is relevant
# to an engineering-capacity-planning objective.
WEIGHT_PRESETS = {
    "balanced": _DEFAULT_WEIGHTS,
    "life_safety": {
        "topological": 0.25, "contributing_area": 0.15, "rainfall": 0.15, "slope": 0.10, "exposure": 0.20,
        "poverty": 0.15,
    },
    "infrastructure_protection": {
        "topological": 0.30, "contributing_area": 0.30, "rainfall": 0.25, "slope": 0.15, "exposure": 0.0,
        "poverty": 0.0,
    },
}

_METERS_PER_DEGREE_LAT = 111_320

# NLCD class -> Curve Number by hydrologic soil group (A/B/C/D), standard
# NRCS TR-55 urban/agricultural runoff-curve-number tables. Verbatim from
# WikiWatershed/tr-55 (github.com/WikiWatershed/tr-55, tr55/tables.py), an
# actively maintained open-source NRCS-TR-55 runoff modeling tool -- real,
# citable reference values, not invented ones. NLCD classes with no listed
# row (e.g. rarer codes like 12 Perennial Ice/Snow, 51 Dwarf Scrub, 72/73/74
# Alpine tundra/lichen/ice) are simply excluded from a catchment's average
# by runoff_coefficient_from_land_cover below, the same "skip what isn't
# confidently classifiable rather than guess" discipline drainage_db.py
# already applies to implausible pipe-slope values.
CN_TABLE = {
    11: {"A": 100, "B": 100, "C": 100, "D": 100},  # Open Water
    21: {"A": 59, "B": 75, "C": 83, "D": 87},       # Developed, Open Space
    22: {"A": 68, "B": 80, "C": 86, "D": 89},       # Developed, Low Intensity
    23: {"A": 81, "B": 88, "C": 91, "D": 93},       # Developed, Medium Intensity
    24: {"A": 91, "B": 94, "C": 95, "D": 96},       # Developed, High Intensity
    31: {"A": 77, "B": 86, "C": 91, "D": 94},       # Barren Land
    41: {"A": 30, "B": 55, "C": 70, "D": 77},       # Deciduous Forest
    42: {"A": 30, "B": 55, "C": 70, "D": 77},       # Evergreen Forest
    43: {"A": 30, "B": 55, "C": 70, "D": 77},       # Mixed Forest
    52: {"A": 35, "B": 56, "C": 70, "D": 77},       # Shrub/Scrub
    71: {"A": 30, "B": 58, "C": 71, "D": 78},       # Grassland/Herbaceous
    81: {"A": 39, "B": 61, "C": 74, "D": 80},       # Pasture/Hay
    82: {"A": 67, "B": 78, "C": 85, "D": 89},       # Cultivated Crops
    90: {"A": 30, "B": 30, "C": 30, "D": 30},       # Woody Wetlands
    95: {"A": 30, "B": 30, "C": 30, "D": 30},       # Emergent Herbaceous Wetlands
}

# Same idea as CN_TABLE above, for geoai_agent.land_cover.
# fetch_global_landcover_grid's class codes (Esri/Impact Observatory's
# global Sentinel-2 10m Land Cover, used outside NLCD's US+territories-only
# coverage) -- a DIFFERENT, coarser 11-class scheme whose codes partially
# COLLIDE with NLCD's own (e.g. both schemes use 11), so this must never be
# looked up interchangeably with CN_TABLE; callers pass whichever table
# actually matches the land_cover_grid's own source (see land_cover.
# GLOBAL_LANDCOVER_CLASS_INFO / NLCD_CLASS_INFO). Values are this project's
# own best analogues from the same NRCS TR-55 table CN_TABLE cites, matched
# by land-cover description rather than copied from a global-scheme-
# specific published source (none as authoritative as TR-55 exists for this
# 11-class scheme) -- e.g. "Trees" -> TR-55's forest row, "Built Area" (this
# scheme has no separate intensity tiers the way NLCD's four Developed
# classes do) -> TR-55's Developed/Medium Intensity row, as a reasonable
# single middle estimate. Treat this table as a coarser approximation than
# CN_TABLE, not an equally-authoritative one -- flagged as such wherever a
# global-source-derived runoff coefficient is reported (see web_app/app.py).
CN_TABLE_GLOBAL = {
    1: {"A": 100, "B": 100, "C": 100, "D": 100},    # Water
    2: {"A": 30, "B": 55, "C": 70, "D": 77},        # Trees (-> Forest)
    4: {"A": 30, "B": 30, "C": 30, "D": 30},        # Flooded Vegetation (-> Wetlands)
    5: {"A": 67, "B": 78, "C": 85, "D": 89},        # Crops (-> Cultivated Crops)
    7: {"A": 81, "B": 88, "C": 91, "D": 93},        # Built Area (-> Developed, Medium Intensity)
    8: {"A": 77, "B": 86, "C": 91, "D": 94},        # Bare Ground (-> Barren Land)
    11: {"A": 30, "B": 58, "C": 71, "D": 78},       # Rangeland (-> Grassland/Herbaceous)
}


def runoff_coefficient_from_land_cover(
    land_cover_grid: np.ndarray, hydrologic_group: str, catchment_mask: np.ndarray, storm_depth_mm: float,
    cn_table: dict = CN_TABLE,
) -> float | None:
    """A basin-specific runoff coefficient derived from real land cover
    (NLCD, or geoai_agent.land_cover.fetch_global_landcover_grid's output
    outside NLCD's US-only coverage -- pass the matching `cn_table` either
    way, see CN_TABLE/CN_TABLE_GLOBAL above) + hydrologic soil group data,
    via the actual SCS Curve Number runoff equation -- not a fixed
    CN-to-coefficient lookup, which would ignore a real hydrologic fact:
    heavier storms produce proportionally MORE runoff from the same
    ground, so a single fixed mapping from CN alone can't be storm-depth-
    aware the way this project's design-storm-driven analysis needs.

    For every cell where `catchment_mask` is True (a basin's own catchment
    membership -- same boolean-mask shape rank_flood_basins's own boundary_
    mask/descending-manifold lookups use), looks up that cell's Curve
    Number from cn_table[land_cover_grid[cell]][hydrologic_group] (silently
    skipping a class code cn_table has no row for), averages CN across the
    catchment's usable cells, then applies the standard metric SCS-CN
    runoff equation at `storm_depth_mm` (the selected design storm's own
    rainfall depth, e.g. RAINFALL_SCENARIOS[...]["intensity_mm_per_hr"] --
    numerically a depth in mm for this project's 1-hour-duration framing,
    see rainfall.py):

        S  = 25400 / CN - 254                          (potential max retention, mm)
        Q  = (P - 0.2*S)^2 / (P + 0.8*S)  if P > 0.2*S  else  0
        C  = Q / P                                       (effective runoff coefficient)

    Returns None if the catchment has zero usable cells (no land-cover data
    reaches this basin at all -- e.g. it sits right at the edge of the
    fetched bounding box) or `hydrologic_group`/`storm_depth_mm` isn't
    usable (storm_depth_mm <= 0) -- callers must fall back to the flat
    `runoff_coefficient` constant in that case, the same way every other
    optional real-data enhancement in this project degrades gracefully
    rather than failing the whole analysis."""
    if not hydrologic_group or storm_depth_mm is None or storm_depth_mm <= 0:
        return None
    cell_values = land_cover_grid[catchment_mask]
    cns = [
        cn_table[int(code)][hydrologic_group]
        for code in cell_values
        if int(code) in cn_table
    ]
    if not cns:
        return None
    cn = sum(cns) / len(cns)
    s_mm = 25400.0 / cn - 254.0
    if storm_depth_mm <= 0.2 * s_mm:
        return 0.0
    q_mm = (storm_depth_mm - 0.2 * s_mm) ** 2 / (storm_depth_mm + 0.8 * s_mm)
    return max(0.0, min(1.0, q_mm / storm_depth_mm))


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


_ENTROPY_FACTOR_KEYS = ("topological", "contributing_area", "rainfall", "slope", "exposure", "poverty")


def entropy_weights(score_matrix: dict[str, list[float]]) -> dict[str, float]:
    """Objective, data-driven risk-factor weights via the Entropy Weight
    Method (Shannon entropy applied to each factor's own variation across
    the samples being ranked -- see Zhu, Hu, Feng & Wang, "Failure Mode
    and Effects Analysis Method Based on the Combination of Entropy Weight
    and Analytic Hierarchy Process" / the same entropy-weighting
    formulation described in Zhu et al., "Effectiveness of the Entropy
    Weight Method in Decision-Making," Mathematical Problems in
    Engineering, 2020; applied to urban flood risk specifically e.g. in
    Wang et al.'s Beijing FAHP-EWM flooding study). Unlike WEIGHT_PRESETS
    (an expert-defined POLICY choice -- life-safety vs. engineering-cost
    priorities), this needs no expert judgment and no historical
    ground-truth data at all: a factor that varies MORE across this run's
    own basins carries MORE discriminating information about which basin
    stands out, and gets more weight; a factor that's nearly the same for
    every basin carries little information for telling them apart, and
    gets little weight.

    `score_matrix` maps each factor name (this file's five:
    "topological"/"contributing_area"/"rainfall"/"slope"/"exposure") to
    the list of that factor's raw 0-1 score across every significant
    basin in ONE run (i.e. rank_flood_basins's own topological_score/
    contributing_area_score/rainfall_score/slope_score/exposure_score,
    collected across all basins before weighting). Returns weights
    summing to 1.0.

    Method, per factor j across n basins:
      1. min-max normalize: Z_ij = (x_ij - min_j) / (max_j - min_j)
      2. convert to a proportion: P_ij = Z_ij / sum_i(Z_ij)
      3. entropy: H_j = -(1/ln n) * sum_i( P_ij * ln(P_ij) )  (0 when P_ij=0)
      4. redundancy (information content): d_j = 1 - H_j
      5. weight: w_j = d_j / sum_j(d_j)

    Three honest caveats, not glossed over:
    - This measures which factor best DISTINGUISHES these specific basins
      from each other, NOT which factor best PREDICTS real flood
      occurrence -- it is objective, but it is not the same claim as a
      model calibrated against historical flood outcomes.
    - Weights are computed FROM this run's own basins, so the SAME basin
      analyzed as part of a different set of basins (a different place, a
      different bounding box) can get a different weight and therefore a
      different risk_score -- a real reproducibility trade-off against the
      fixed, comparable-across-runs WEIGHT_PRESETS.
    - `rainfall` is IDENTICAL across every basin in a single run (one
      selected design storm applies uniformly across the whole study
      area) -- a factor with zero variation has maximum entropy (H_j = 1,
      so d_j = 0), so under this method rainfall ALWAYS gets weight 0.
      That is mathematically correct, not a bug, but it means an
      entropy-weighted risk_score completely ignores rainfall intensity
      -- callers/readers should be told this plainly if this weighting is
      used, not left to notice it silently.

    Falls back to equal weights (not a crash) when there's nothing
    meaningful to compute from: fewer than 2 basins (entropy needs
    variation ACROSS samples; a single basin has none to measure), or
    every factor happens to be perfectly constant across all basins
    (every d_j is 0, so the normalization in step 5 would divide by
    zero)."""
    factors = list(score_matrix)
    n = len(next(iter(score_matrix.values()), []))
    if n < 2:
        return {k: 1.0 / len(factors) for k in factors}

    redundancy = {}
    for factor in factors:
        values = score_matrix[factor]
        vmin, vmax = min(values), max(values)
        vrange = (vmax - vmin) or 1e-12
        z = [(v - vmin) / vrange for v in values]
        total_z = sum(z)
        if total_z <= 0:
            redundancy[factor] = 0.0  # every basin scored exactly the SAME (post-normalization 0) on this factor
            continue
        p = [zi / total_z for zi in z]
        entropy = -sum((pi * math.log(pi) if pi > 0 else 0.0) for pi in p) / math.log(n)
        redundancy[factor] = 1.0 - entropy

    total_redundancy = sum(redundancy.values())
    if total_redundancy <= 0:
        return {k: 1.0 / len(factors) for k in factors}
    return {k: v / total_redundancy for k, v in redundancy.items()}


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
    rainfall_table: dict | None = None,
    weights: dict[str, float] | str | None = None,
    runoff_coefficient: float = 0.5,
    slope_field: ScalarField | None = None,
    exposure_field: ScalarField | None = None,
    poverty_field: ScalarField | None = None,
    boundary_mask: np.ndarray | None = None,
    land_cover_grid: np.ndarray | None = None,
    hydrologic_group: str | None = None,
    land_cover_cn_table: dict = CN_TABLE,
) -> dict:
    """Ranks every significant basin in `morse_result` (a TTK
    analyze_scalar_field_ttk result for `elevation_field`) by flood risk,
    and reports each one's Rational-Method drainage demand under the
    selected rainfall scenario.

    rainfall_table: None (the default) means "use the static RAINFALL_
    SCENARIOS fallback table", exactly as before this parameter existed --
    every existing caller that never passes it gets byte-identical numbers.
    Given a dict shaped like RAINFALL_SCENARIOS (e.g. the output of
    rainfall.fetch_noaa_atlas14_rainfall, a genuine live per-location
    estimate), `rainfall_scenario` is looked up in THAT table instead, and
    the rainfall-score normalization (against the max intensity across the
    table's own return periods) is computed from that table too -- a live
    table's absolute intensities can differ from the static table's range,
    so reusing the static table's own max would silently mis-normalize.

    weights: a WEIGHT_PRESETS name, a custom {"topological",
    "contributing_area", "rainfall", "slope", "exposure", "poverty"} dict
    (need not sum to 1 -- renormalized), None for the "balanced" preset, or the
    literal string "entropy" to use entropy_weights (above) instead of any
    named policy preset -- weights computed OBJECTIVELY from how much each
    factor actually varies across this run's own significant basins, no
    expert judgment or historical data needed, at the cost of the
    caveats entropy_weights' own docstring states plainly (rainfall
    always gets weight 0; weights vary run to run). The returned dict's
    "weights" key reports whichever weights were ACTUALLY used, so a
    caller/report always shows the real numbers regardless of which mode
    produced them. The returned "weighting_method" key records WHY those
    numbers were used -- "preset" (a named WEIGHT_PRESETS entry or the
    None/balanced default), "entropy" (entropy_weights), or "custom" (a
    caller-supplied dict that is neither of those -- e.g. the frontend's
    Modify-weights sliders, a person's own hand-picked numbers, not backed
    by either an expert-named policy or the entropy calculation).

    poverty_field: a US Census-tract poverty-rate layer (geoai_agent.
    real_data.fetch_poverty_rate_field) sampled at each basin, same
    min-max-normalized-across-this-run's-basins treatment as exposure_
    field. Left out (None, the default), poverty_score is the same neutral
    0.5 every unsupplied optional factor here falls back to.

    boundary_mask: a boolean (ny, nx) grid (see boundary.
    grid_inside_polygon_mask), True where a cell is actually inside the
    resolved place boundary. When given (together with
    morse_result["descending_manifold"]), each basin's contributing_area_km2
    only counts cells that are BOTH part of its catchment AND inside the
    boundary -- otherwise `basin["size_cells"]` is used as-is, which can
    include synthetic "wall" cells outside the real boundary (see
    boundary.clip_field_to_polygon: cells outside the polygon are raised
    above the real terrain's max so TTK can't place a MINIMUM there, but a
    basin's descending manifold -- everything that drains INTO it -- can
    still legitimately extend across the boundary edge into that wall
    region, since the wall slopes back down toward the real terrain at the
    edge). Left out entirely (None), every basin's full catchment counts,
    same as before this parameter existed -- existing callers are
    unaffected.

    land_cover_grid / hydrologic_group: when BOTH are given (see geoai_
    agent.land_cover.fetch_nlcd_landcover_grid and geoai_agent.soil_data.
    fetch_dominant_hydrologic_soil_group), each basin's own runoff_
    coefficient is derived from real land-cover + soil data across its
    catchment via runoff_coefficient_from_land_cover, instead of every
    basin sharing the flat `runoff_coefficient` parameter. A basin whose
    catchment has no usable land-cover data still falls back to the flat
    constant. Left out (either is None, the default), every basin uses the
    flat constant exactly as before these parameters existed -- existing
    callers are unaffected."""
    table = rainfall_table if rainfall_table is not None else RAINFALL_SCENARIOS
    if rainfall_scenario not in table:
        raise ValueError(
            f"unknown rainfall_scenario '{rainfall_scenario}', available: {list(table)}"
        )
    use_entropy_weights = weights == "entropy"
    # Tracks WHY these particular weights are being used, independent of the
    # numbers themselves -- a raw dict can arrive either as a named preset
    # (resolved from WEIGHT_PRESETS just below, or the None/_DEFAULT_WEIGHTS
    # default) or as a caller-supplied custom dict (the frontend's
    # Modify-weights sliders), and report.py's methodology section needs to
    # tell those two apart even though both end up as an ordinary dict here.
    if use_entropy_weights:
        weight_source = "entropy"
    elif weights is None or isinstance(weights, str):
        if isinstance(weights, str) and weights not in WEIGHT_PRESETS:
            raise ValueError(
                f"unknown weight preset '{weights}', available: {list(WEIGHT_PRESETS)} or 'entropy'"
            )
        weight_source = "preset"
        if isinstance(weights, str):
            weights = WEIGHT_PRESETS[weights]
    else:
        weight_source = "custom"
    # Deferred when use_entropy_weights: entropy_weights needs every basin's
    # raw factor scores first (computed in the loop below), so `w` isn't
    # known yet -- risk_score for every basin is computed in a SECOND pass
    # further down, once `w` is actually resolved either way.
    w = None if use_entropy_weights else _normalized_weights(weights)

    scenario = table[rainfall_scenario]
    rainfall_source = "noaa_atlas14_live" if rainfall_table is not None else "static_illustrative_table"
    max_intensity = _MAX_RAINFALL_INTENSITY if rainfall_table is None else max(
        s["intensity_mm_per_hr"] for s in table.values()
    )
    cell_area_km2 = _cell_area_km2(elevation_field)
    value_min, value_max = morse_result["value_range"]
    value_range = (value_max - value_min) or 1.0
    rainfall_score = scenario["intensity_mm_per_hr"] / max_intensity

    basins = morse_result.get("basins", [])
    if not basins:
        # Nothing to compute entropy_weights FROM (no basins' scores exist) --
        # equal weights is the same degenerate fallback entropy_weights
        # itself uses for a too-small sample, so this stays consistent
        # rather than reporting `w` as None/undefined for an empty result.
        empty_weights = w if w is not None else {k: 1.0 / len(_ENTROPY_FACTOR_KEYS) for k in _ENTROPY_FACTOR_KEYS}
        return {
            "rainfall_scenario": scenario, "rainfall_source": rainfall_source, "weights": empty_weights,
            "weighting_method": weight_source,
            "runoff_coefficient": runoff_coefficient, "hotspots": [],
        }

    descending_manifold = morse_result.get("descending_manifold")
    if descending_manifold is not None:
        descending_manifold = np.asarray(descending_manifold)
    use_boundary_clip = boundary_mask is not None and descending_manifold is not None
    use_land_cover = land_cover_grid is not None and hydrologic_group is not None and descending_manifold is not None

    def _basin_area_km2(basin: dict) -> float:
        if use_boundary_clip:
            catchment_id = basin.get("catchment_id")
            if catchment_id is None:
                return 0.0
            cells_in_boundary = int(np.sum((descending_manifold == catchment_id) & boundary_mask))
            return cells_in_boundary * cell_area_km2
        return basin["size_cells"] * cell_area_km2

    def _basin_catchment_mask(basin: dict) -> np.ndarray | None:
        """Boolean (ny, nx) mask of this basin's own catchment cells (same
        descending-manifold-membership test _basin_area_km2 uses), ANDed
        with boundary_mask when one was given -- so the land-cover-derived
        coefficient is computed over the same real-boundary-respecting
        catchment as contributing_area_km2 itself, not a differently-scoped
        one. None if this basin has no catchment_id (degenerate case)."""
        catchment_id = basin.get("catchment_id")
        if catchment_id is None:
            return None
        mask = descending_manifold == catchment_id
        if boundary_mask is not None:
            mask = mask & boundary_mask
        return mask

    # Two passes: contributing_area_km2 must be known for every basin
    # before max_area_km2 (needed to normalize area_score) can be computed.
    basin_areas_km2 = [_basin_area_km2(basin) for basin in basins]
    max_area_km2 = max(basin_areas_km2)

    hotspots = []
    for basin, contributing_area_km2 in zip(basins, basin_areas_km2):
        minimum = basin["minimum"]
        i, j = minimum["grid_index"]

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

        if poverty_field is not None:
            pov_here = float(poverty_field.values[i, j])
            pov_min, pov_max = float(poverty_field.values.min()), float(poverty_field.values.max())
            pov_range = (pov_max - pov_min) or 1.0
            poverty_score = (pov_here - pov_min) / pov_range
        else:
            poverty_score = 0.5  # neutral -- no poverty layer supplied for this run

        basin_runoff_coefficient = runoff_coefficient
        runoff_coefficient_source = "constant"
        if use_land_cover:
            mask = _basin_catchment_mask(basin)
            if mask is not None and mask.any():
                derived = runoff_coefficient_from_land_cover(
                    land_cover_grid, hydrologic_group, mask, scenario["intensity_mm_per_hr"],
                    cn_table=land_cover_cn_table,
                )
                if derived is not None:
                    basin_runoff_coefficient = derived
                    runoff_coefficient_source = "land_cover_derived"

        # risk_score isn't computed here -- `w` may not be known yet (entropy
        # mode needs every basin's raw scores below, first). Filled in by the
        # second pass right after this loop, once `w` is actually resolved
        # either way.
        hotspots.append({
            "grid_index": [i, j],
            "coords": minimum["coords"],
            "elevation_m": minimum["value"],
            "persistence_m": persistence,
            "catchment_id": basin.get("catchment_id"),
            "contributing_area_km2": contributing_area_km2,
            "topological_score": topological_score,
            "contributing_area_score": area_score,
            "slope_score": slope_score,
            "rainfall_score": rainfall_score,
            "exposure_score": exposure_score,
            "exposure_layer_available": exposure_field is not None,
            "poverty_score": poverty_score,
            "poverty_layer_available": poverty_field is not None,
            "slope_layer_available": slope_field is not None,
            "runoff_coefficient": basin_runoff_coefficient,
            "runoff_coefficient_source": runoff_coefficient_source,
            "estimated_peak_runoff_m3s": rational_method_runoff(
                basin_runoff_coefficient, scenario["intensity_mm_per_hr"], contributing_area_km2
            ),
        })

    if w is None:  # use_entropy_weights -- now that every basin's raw scores exist, derive weights from their own spread
        w = entropy_weights({
            "topological": [h["topological_score"] for h in hotspots],
            "contributing_area": [h["contributing_area_score"] for h in hotspots],
            "rainfall": [h["rainfall_score"] for h in hotspots],
            "slope": [h["slope_score"] for h in hotspots],
            "exposure": [h["exposure_score"] for h in hotspots],
            "poverty": [h["poverty_score"] for h in hotspots],
        })

    for h in hotspots:
        h["risk_score"] = (
            w["topological"] * h["topological_score"]
            + w["contributing_area"] * h["contributing_area_score"]
            + w["rainfall"] * h["rainfall_score"]
            + w["slope"] * h["slope_score"]
            + w["exposure"] * h["exposure_score"]
            + w["poverty"] * h["poverty_score"]
        )

    hotspots.sort(key=lambda h: h["risk_score"], reverse=True)
    for rank, h in enumerate(hotspots, start=1):
        h["rank"] = rank

    return {
        "rainfall_scenario": scenario, "rainfall_source": rainfall_source, "weights": w,
        "weighting_method": weight_source,
        "runoff_coefficient": runoff_coefficient, "hotspots": hotspots,
    }


_EAD_TRUNCATION_NOTE = (
    "Expected annual runoff is a probability-weighted integral of runoff across ONLY the "
    "2yr-500yr return periods this project fetches (annual exceedance probability 0.5 down "
    "to 0.002), trapezoidally -- the same integration principle as the standard USACE/HEC "
    "Expected-Annual-Damage approach, applied here to runoff volume as a proxy since no "
    "dollar depth-damage function exists yet (the same honest-proxy framing this project "
    "already uses elsewhere, e.g. contributing_area_km2 standing in for population served "
    "before a real Census exposure layer existed). This is NOT a full annual expectation, "
    "and can legitimately come out SMALLER than even the most frequent (2yr) single-storm "
    "runoff value for the same basin -- that is expected, not a bug: it omits BOTH tails a "
    "true EAD would integrate over -- storms MORE frequent than a 2-year event (which this "
    "project has no design-storm data for at all, and which would pull a true annual "
    "expectation further down, since a small, frequent storm contributes little runoff "
    "weighted over a large share of yearly probability) and storms RARER than the 500-year "
    "event this project's data stops at (which would push a true annual expectation up, "
    "since an extreme event's outsized runoff can matter even at very low probability). "
    "Treat this as a RELATIVE ranking across basins -- which one carries the most "
    "probability-weighted runoff burden across the return periods this project actually has "
    "data for -- not as a literal expected cubic-meters-per-second-per-year figure directly "
    "comparable in magnitude to any single storm's own runoff number."
)


def compute_expected_annual_runoff(
    elevation_field: ScalarField,
    morse_result: dict,
    rainfall_table: dict | None = None,
    weights: dict[str, float] | str | None = None,
    runoff_coefficient: float = 0.5,
    slope_field: ScalarField | None = None,
    exposure_field: ScalarField | None = None,
    poverty_field: ScalarField | None = None,
    boundary_mask: np.ndarray | None = None,
    land_cover_grid: np.ndarray | None = None,
    hydrologic_group: str | None = None,
    land_cover_cn_table: dict = CN_TABLE,
) -> dict:
    """A multi-storm AGGREGATE risk metric, layered on top of rank_flood_
    basins rather than replacing it: rank_flood_basins scores basins against
    ONE selected design storm at a time (the HITL "which storm are we
    planning for" control), so a basin's single-scenario risk_score/
    estimated_peak_runoff_m3s depends on which one of the six return
    periods (2/10/25/50/100/500yr) happened to be picked. This runs rank_
    flood_basins once PER return period in the active table (`rainfall_
    table` or the static RAINFALL_SCENARIOS fallback -- same six periods
    either way), reusing the SAME `morse_result` for every call (the
    topology doesn't change across storms, only the rainfall-driven
    numbers do -- basin identity via "catchment_id"/"grid_index" is
    therefore guaranteed stable across all six calls, since it's the exact
    same input object read six times, never mutated by rank_flood_basins),
    then trapezoidally integrates each basin's estimated_peak_runoff_m3s
    across annual exceedance probability (AEP = 1/return_period_years) into
    a single "expected annual runoff" number -- the standard EAD-style way
    to combine multiple return periods into one ranking instead of picking
    just one storm and reporting only its result. See _EAD_TRUNCATION_NOTE
    for the explicit, deliberate truncation caveat (also returned as
    "note" below, meant to be surfaced to the end user, not buried).

    Every parameter besides `elevation_field`/`morse_result` is passed
    straight through to each of the six rank_flood_basins calls unchanged
    (same weights/runoff_coefficient/slope_field/exposure_field/
    poverty_field/boundary_mask/land_cover_grid/hydrologic_group/
    land_cover_cn_table every time) -- only `rainfall_scenario` varies call
    to call, driven by this function itself.

    Returns {"basins": [{"catchment_id", "grid_index", "coords",
    "expected_annual_runoff_m3s", "per_scenario": {"2yr": <m3s>, ...}},
    ...], "note": <the truncation caveat, user-facing text>}. This is a
    genuinely separate, additive analysis a caller opts into -- it does not
    modify rank_flood_basins's own return shape or get folded into
    risk_score."""
    table = rainfall_table if rainfall_table is not None else RAINFALL_SCENARIOS
    per_scenario_results = {}
    for scenario_key, scenario in table.items():
        per_scenario_results[scenario_key] = rank_flood_basins(
            elevation_field, morse_result, rainfall_scenario=scenario_key, rainfall_table=rainfall_table,
            weights=weights, runoff_coefficient=runoff_coefficient, slope_field=slope_field,
            exposure_field=exposure_field, poverty_field=poverty_field, boundary_mask=boundary_mask,
            land_cover_grid=land_cover_grid, hydrologic_group=hydrologic_group,
            land_cover_cn_table=land_cover_cn_table,
        )

    # AEP = 1 / return_period_years, sorted ascending return period ==
    # descending (most-frequent-first) exceedance probability -- the order
    # the trapezoidal integral below walks in.
    ordered_keys = sorted(table, key=lambda k: table[k]["return_period_years"])
    aeps = [1.0 / table[k]["return_period_years"] for k in ordered_keys]

    def _basin_identity(h: dict):
        return h.get("catchment_id") if h.get("catchment_id") is not None else tuple(h["grid_index"])

    first_key = ordered_keys[0]
    basins_out = []
    for h0 in per_scenario_results[first_key]["hotspots"]:
        identity = _basin_identity(h0)
        per_scenario_m3s = []
        for key in ordered_keys:
            match = next(
                (h for h in per_scenario_results[key]["hotspots"] if _basin_identity(h) == identity), None,
            )
            per_scenario_m3s.append(match["estimated_peak_runoff_m3s"] if match else 0.0)

        expected_annual_m3s = sum(
            0.5 * (per_scenario_m3s[i] + per_scenario_m3s[i + 1]) * (aeps[i] - aeps[i + 1])
            for i in range(len(aeps) - 1)
        )
        basins_out.append({
            "catchment_id": h0.get("catchment_id"),
            "grid_index": h0["grid_index"],
            "coords": h0["coords"],
            "expected_annual_runoff_m3s": expected_annual_m3s,
            "per_scenario": {key: per_scenario_m3s[idx] for idx, key in enumerate(ordered_keys)},
        })

    basins_out.sort(key=lambda b: b["expected_annual_runoff_m3s"], reverse=True)
    return {"basins": basins_out, "note": _EAD_TRUNCATION_NOTE}
