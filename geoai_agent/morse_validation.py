"""
Direct evidence-chain check for the central methodological question this
project's own critique kept coming back to: is a Morse-persistence-
significant basin actually a hydrologically meaningful drainage feature, or
just a topologically robust artifact of the Morse-Smale complex with no
independent correspondence to how water actually flows?

Rather than argue this in prose, this module compares each Morse-
significant basin against an INDEPENDENT raster-hydrology model computed
the standard way -- D8 flow direction/accumulation (O'Callaghan & Mark,
1984, the same algorithm geoai_agent.flow_field already implements for the
map's flow-direction arrows) -- and reports what actually agrees and what
doesn't, numerically, not as a claim to be taken on faith:

- Does each Morse-significant basin's own pour point ALSO come out as a
  local sink (no downhill D8 neighbor) under the independent D8 model?
  This should be very close to 1.0 by construction (a true local minimum
  of the DEM has no downhill neighbor under any reasonable discrete
  comparison rule) -- it's a correctness/sanity check more than a
  discovery, but a value noticeably below 1.0 would reveal a real
  discrepancy between the two methods' handling of flat regions/plateaus
  worth investigating, not something to assume away.
- Does Morse's own contributing_area_km2 (derived from the descending-
  manifold catchment segmentation) correlate with D8's flow accumulation
  at the same pour point (an entirely separately-computed measure of
  upstream drainage area, cell-by-cell rather than region-by-region)? A
  strong correlation is real, checkable evidence that Morse's basin-size
  concept tracks the same physical quantity D8 accumulation does, not
  merely an assertion.
- How does the raw count of local depressions compare between the two
  methods (Morse's own raw critical-point count vs. every D8 cell with no
  downhill neighbor)? This is where Morse's actual methodological
  advantage over plain D8 shows up numerically: D8 alone has no principled
  way to separate "a real depression" from "a single noisy pixel" --
  practical D8 pipelines bolt on an arbitrary heuristic (a fixed minimum
  drainage-area or "pit-filling" threshold) to do that filtering, whereas
  Morse persistence is a continuous, per-feature significance measure
  derived from the scalar field's own topology, letting a principled
  threshold (see geoai_agent.agent_decisions.decide_persistence_threshold)
  replace an arbitrary one.
"""
from __future__ import annotations

import numpy as np

from geoai_agent.flow_field import compute_d8_flow
from geoai_agent.morse_topology import ScalarField


def compare_morse_to_d8(
    field: ScalarField, hotspots: list[dict], num_raw_minima: int, boundary_mask: np.ndarray | None = None,
) -> dict:
    """Cross-checks `hotspots` (flood_tools.rank_flood_basins' significant-
    basin list -- each already carrying "grid_index" and
    "contributing_area_km2") against a freshly-computed D8 flow field over
    the SAME `field`. Computing D8 here is a second call to flow_field.
    compute_d8_flow (the map's flow-arrow layer already calls it once) --
    cheap, pure numpy, no subprocess, so a second call costs nothing
    meaningful. `num_raw_minima` is morse_result["num_raw_minima"] from the
    SAME TTK run `hotspots` came from, for the raw-depression-count
    comparison.

    Returns {"num_raw_morse_minima", "num_significant_morse_basins",
    "num_d8_sinks", "d8_sink_agreement_fraction",
    "contributing_area_correlation", "per_basin": [...]}.
    `d8_sink_agreement_fraction`/`contributing_area_correlation` are None
    when there are too few basins (<1 or <3 respectively) to make the
    figure meaningful, rather than a misleading number from a tiny sample."""
    flow = compute_d8_flow(field)
    has_downhill = flow["has_downhill"]
    accumulation_km2 = flow["accumulation_km2"]

    sink_mask = ~has_downhill
    num_d8_sinks = int(np.sum(sink_mask & boundary_mask)) if boundary_mask is not None else int(np.sum(sink_mask))

    per_basin = []
    for h in hotspots:
        row, col = h["grid_index"]
        per_basin.append({
            "catchment_id": h.get("catchment_id"),
            "rank": h.get("rank"),
            "coords": h.get("coords"),
            "is_d8_sink": bool(not has_downhill[row, col]),
            "d8_flow_accumulation_km2": round(float(accumulation_km2[row, col]), 5),
            "morse_contributing_area_km2": h["contributing_area_km2"],
        })

    d8_sink_agreement = (
        sum(1 for p in per_basin if p["is_d8_sink"]) / len(per_basin) if per_basin else None
    )

    correlation = None
    if len(per_basin) >= 3:
        morse_areas = np.array([p["morse_contributing_area_km2"] for p in per_basin])
        d8_areas = np.array([p["d8_flow_accumulation_km2"] for p in per_basin])
        if np.std(morse_areas) > 0 and np.std(d8_areas) > 0:
            correlation = float(np.corrcoef(morse_areas, d8_areas)[0, 1])

    return {
        "num_raw_morse_minima": num_raw_minima,
        "num_significant_morse_basins": len(hotspots),
        "num_d8_sinks": num_d8_sinks,
        "d8_sink_agreement_fraction": d8_sink_agreement,
        "contributing_area_correlation": correlation,
        "per_basin": per_basin,
    }
