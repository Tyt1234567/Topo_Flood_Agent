"""
Real, consequential agent decision points for the flood-risk pipeline --
places where an LLM call actually picks something that changes the
analysis's output, not just narrates a result that was already fixed.

- decide_persistence_threshold: given a small table of candidate
  persistence thresholds -- each with the basin count it would actually
  produce on THIS run's own terrain (see build_persistence_candidates) --
  picks which one to use for the "significant basins" TTK pass, instead of
  the fixed 5%-of-elevation-range default every run used before this
  existed. Bounded-choice design: the model picks one of a handful of
  real, precomputed candidates, it doesn't invent a raw meter value. On
  any failure (network, malformed JSON, missing GEMINI_API_KEY), the
  caller (web_app/app.py) catches the exception and falls back to this
  project's original fixed 5%-of-range auto threshold -- the same
  "optional enhancement degrades gracefully" contract every live data
  source in this project already follows.

- compute_weight_sensitivity is not an LLM decision at all -- it re-scores
  the SAME significant basins under every named weighting scheme (pure
  re-computation on already-fetched data, no new TTK/subprocess call) so
  the report can show whether the #1-ranked basin is robust to a
  reasonable choice of weighting scheme, or an artifact of whichever one
  happened to be used. Its result is handed to the final recommendation
  LLM call as evidence, the same way every other structured pipeline
  result is.

Every analysis uses entropy weighting (flood_tools.entropy_weights) unless
a user hand-sets their own weights via the Modify-weights sliders -- there
is no more per-run "which weighting philosophy" choice for an agent
decision to make (an earlier version of this project had a
decide_analysis_strategy step that picked among named WEIGHT_PRESETS based
on a stated objective; removed once the UI dropped the preset/entropy
choice in favor of always using entropy).

- reflect_on_recommendation implements the Reflection agent pattern
  (execute -> critique -> refine) around the final recommendation-writing
  call: a second LLM call re-reads the SAME evidence the writer saw
  alongside the recommendation it produced, checks specifically for claims
  the evidence doesn't support (overclaiming words, a finding that
  contradicts the weight-sensitivity/D8-cross-check evidence, an ignored
  caveat), and returns a corrected recommendation only when it actually
  finds something wrong -- not a free-form "improve this" rewrite. On any
  failure the caller keeps the original, un-reflected recommendation, same
  degrade-gracefully contract as every other agent decision here.
"""
from __future__ import annotations

import numpy as np

from geoai_agent.flood_tools import rank_flood_basins
from geoai_agent.llm_client import LLMClient
from geoai_agent.morse_topology import ScalarField

_THRESHOLD_SYSTEM_PROMPT = """You are the persistence-threshold-selection step of a Morse-theoretic terrain analysis agent.

Discrete Morse theory finds every local depression in a digital elevation model, then a "persistence" threshold (in meters of relief) decides which of those depressions count as significant enough to analyze versus noise from DEM artifacts or microtopography. A LOWER threshold keeps more, smaller basins (more sensitive to real small depressions, but more prone to noise); a HIGHER threshold keeps only the deepest, most well-defined basins (less noise, but can discard or merge real smaller depressions that could still matter). This choice CHANGES WHAT IS BEING ANALYZED -- going from 10 basins to 100 basins is not a minor parameter tweak, it is a different analysis object -- so treat it that seriously.

IMPORTANT constraints on how you may reason and what you may claim:
- There is no optimization objective, loss function, or validation data defining a mathematically "optimal" threshold here. NEVER use the words "optimal" or "the best choice" -- you are making a bounded, defensible SELECTION among real alternatives, not solving an optimization problem. Call it your "selected threshold," not an optimum.
- Your reasoning must cite SPECIFIC EVIDENCE FROM THE CANDIDATE TABLE ITSELF -- e.g. a sharp drop-off in basin count between two adjacent candidates (a natural break/gap in the persistence distribution), or a candidate count that seems implausibly large/small for a real coverage screening. Do NOT reason purely from the stated objective in the abstract (e.g. do not just say "safety requires high sensitivity, therefore threshold X" without pointing to what in the actual numbers below makes X specifically the right cut point, not some other candidate). If you cannot find distributional evidence favoring one candidate over its neighbors, say so explicitly and pick the default candidate.
- Do not claim your choice is what "Morse theory tells us" -- Morse theory computed the candidates; deciding which one is worth using for this objective is a policy choice you are making, and your reasoning should be honest that it's a policy choice grounded in (but not dictated by) the data.

You are choosing from a FIXED list of candidate thresholds below, each already computed from THIS run's own terrain data together with the exact number of basins it would produce -- you are not inventing a raw meter value.

Respond with ONLY a JSON object of this exact shape:
{"chosen_threshold": <one of the candidate threshold numbers, exactly>, "reasoning": "2-4 plain-language sentences: name the specific evidence in the candidate table that favors this one over its neighbors, then the tradeoff for the stated objective. Do not say 'optimal'."}"""


def build_persistence_candidates(scan_result: dict) -> list[dict]:
    """From a near-zero-threshold TTK scan (every raw minimum's own
    persistence already computed -- see web_app/app.py's
    _AGENT_ALL_MINIMA_THRESHOLD scan), builds a small set of candidate
    significance thresholds spanning THIS terrain's own actual distribution
    of depression depths, not arbitrary fixed meter values -- what counts
    as "a real depression, not noise" depends entirely on this specific
    study area's own relief, so a threshold that's meaningful in a
    mountainous area would keep almost nothing significant in a flat one
    and vice versa. Returns a list of {"threshold", "label",
    "num_significant_basins"}, sorted by threshold ascending, deduplicated
    (a percentile that happens to land on the same value as another
    candidate, or on 0, is dropped) -- empty if the scan found no minima
    with a finite persistence at all (e.g. a perfectly flat field).

    IMPORTANT: `num_significant_basins` here is an ESTIMATE (a simple count
    of this scan's own raw minima with persistence >= that threshold), not
    a guarantee of what a real TTK run AT that threshold will report.
    Observed directly: TTK's persistence-based simplification is itself
    threshold-dependent -- a borderline critical point's persistence
    pairing can genuinely differ between two separately-simplified runs at
    different thresholds, even though a fixed threshold reproduces
    deterministically across repeated runs. The web_app/app.py caller
    reconciles this after the real significant-threshold run completes
    (see its own comment there) and surfaces any discrepancy honestly in
    the report rather than leaving two silently-disagreeing basin counts."""
    persistences = sorted(
        p["persistence"] for p in scan_result["critical_points"]["minimum"]
        if p.get("persistence") is not None
    )
    if not persistences:
        return []
    value_min, value_max = scan_result["value_range"]
    auto_default = 0.05 * ((value_max - value_min) or 1.0)

    candidates: list[dict] = []
    seen_thresholds: set[float] = set()

    def _add(threshold: float, label: str) -> None:
        # Round FIRST, then count against the ROUNDED value -- counting
        # against the raw unrounded percentile and then rounding the
        # threshold for display/use is what caused a real, observed
        # off-by-one (a candidate labeled "69 basins" that actually produced
        # 70 once the rounded value was the one really passed to TTK,
        # because rounding down let one more borderline minimum's
        # persistence clear the threshold). The count shown here must be
        # for the EXACT value that will actually be used downstream.
        rounded = round(float(threshold), 3)
        if rounded <= 0 or rounded in seen_thresholds:
            return
        seen_thresholds.add(rounded)
        count = sum(1 for p in persistences if p >= rounded)
        candidates.append({"threshold": rounded, "label": label, "num_significant_basins": count})

    _add(auto_default, "default (5% of this run's elevation range)")
    # Denser percentile grid (was just 50/75/90/95) -- the system prompt
    # above specifically asks the model to point to "a sharp drop-off in
    # basin count between two adjacent candidates" as its evidence; a
    # coarser 4-point grid can straddle a real break in the distribution
    # without ever landing a candidate near it. More candidates costs
    # nothing extra here (build_persistence_candidates is pure numpy over
    # an already-computed persistence list, no new TTK/subprocess call),
    # and llm_client.py's multi-key rotation now absorbs the small prompt-
    # size increase this adds to the threshold-decision call.
    for pct in (25, 40, 50, 60, 70, 80, 90, 95, 99):
        _add(float(np.percentile(persistences, pct)), f"{pct}th percentile of this run's raw-minima persistence")

    candidates.sort(key=lambda c: c["threshold"])
    return candidates


def decide_persistence_threshold(
    llm: LLMClient, place: str, objective_text: str, candidates: list[dict],
) -> dict:
    """Returns {"chosen_threshold", "reasoning", "candidates"}. `chosen_
    threshold` is always one of `candidates`' own threshold values -- if the
    model's response names a value not in the list (including simple
    floating-point round-tripping drift through JSON), it's snapped to the
    nearest candidate rather than rejected outright. Raises ValueError if
    `candidates` is empty (nothing to choose from -- the caller should have
    already skipped calling this in that case) or on a genuine LLM call
    failure, same "caller catches and falls back to the fixed default"
    contract as decide_analysis_strategy."""
    if not candidates:
        raise ValueError("no persistence-threshold candidates to choose from")
    candidates_text = "\n".join(
        f"- {c['threshold']}m ({c['label']}) -> {c['num_significant_basins']} basins would be significant"
        for c in candidates
    )
    user_prompt = f"Place: {place}\nStated objective: {objective_text or '(none given)'}\n\nCandidates:\n{candidates_text}"
    result = llm.chat_json(_THRESHOLD_SYSTEM_PROMPT, user_prompt)
    chosen = result.get("chosen_threshold")
    valid_values = [c["threshold"] for c in candidates]
    if not isinstance(chosen, (int, float)) or round(float(chosen), 3) not in valid_values:
        chosen = min(valid_values, key=lambda v: abs(v - float(chosen))) if isinstance(chosen, (int, float)) else valid_values[0]
    reasoning = result.get("reasoning") or "No reasoning was returned; snapped to the nearest valid candidate."
    return {"chosen_threshold": float(chosen), "reasoning": reasoning, "candidates": candidates}


_SENSITIVITY_SCHEMES = ("balanced", "life_safety", "infrastructure_protection", "entropy")


def compute_weight_sensitivity(elevation_field: ScalarField, morse_result: dict, **rank_kwargs) -> dict:
    """Re-scores the SAME significant basins (`morse_result`, already
    computed once) under every named weighting scheme in _SENSITIVITY_
    SCHEMES -- pure re-computation on data already in hand, no new TTK/
    subprocess call, no new live data fetch -- to check whether the #1-
    ranked basin is a robust finding (the same basin wins under every
    reasonable weighting philosophy) or an artifact of whichever specific
    scheme this run happened to use. `rank_kwargs` are forwarded straight
    to flood_tools.rank_flood_basins (rainfall_scenario, rainfall_table,
    runoff_coefficient, slope_field, exposure_field, boundary_mask,
    land_cover_grid, hydrologic_group) -- the exact same inputs the run's
    own primary ranking already used, so this is an apples-to-apples
    re-weighting, not a re-analysis with different underlying data.

    Returns {"per_scheme": {scheme_name: {"weights", "top_3_catchment_ids",
    "top_1_coords"}}, "rank1_stable_across_schemes": bool} -- a scheme that
    fails to score (should not normally happen, since these are the same
    already-validated presets/entropy_weights every run already supports)
    is simply omitted from per_scheme rather than failing this whole
    computation, consistent with this project's "skip what didn't work,
    don't fail on it" discipline elsewhere (e.g. land_cover_geojson
    skipping an unrecognized NLCD class code)."""
    per_scheme: dict[str, dict] = {}
    for scheme in _SENSITIVITY_SCHEMES:
        try:
            result = rank_flood_basins(elevation_field, morse_result, weights=scheme, **rank_kwargs)
        except Exception:
            continue
        ranked = result["hotspots"]
        if not ranked:
            continue
        per_scheme[scheme] = {
            "weights": result["weights"],
            "top_3_catchment_ids": [h.get("catchment_id") for h in ranked[:3]],
            "top_1_coords": ranked[0]["coords"],
        }
    top1_ids = {
        v["top_3_catchment_ids"][0] for v in per_scheme.values()
        if v["top_3_catchment_ids"] and v["top_3_catchment_ids"][0] is not None
    }
    return {
        "per_scheme": per_scheme,
        "rank1_stable_across_schemes": len(top1_ids) <= 1 and len(per_scheme) > 1,
    }


_REFLECTION_SYSTEM_PROMPT = """You are the self-review step of a flood-risk analysis agent, implementing the Reflection pattern: execute, then critique, then refine.

Another instance of yourself was given a JSON evidence bundle (topology results, D8 cross-check, weight-sensitivity check, and other pipeline data) and wrote a recommendation from it. You will be shown BOTH. Your job is to critically check the recommendation against the evidence -- not to rewrite it for style, tone, or length.

Check specifically for:
- A claim (a number, a certainty, a causal statement) that the evidence bundle does not actually support.
- Overclaiming language ("optimal", "guaranteed", "will flood", "proven") where the evidence only supports a weaker, more conditional claim.
- A statement that CONTRADICTS evidence present in the bundle -- e.g. treating the #1-ranked basin as a settled finding when weight_sensitivity shows it changes under a different weighting scheme, or treating a Morse-significant basin as confirmed real drainage when the D8 cross-check evidence is weak (low sink-agreement fraction, weak or missing correlation), or citing the wrong threshold/candidate count.
- A caveat the evidence bundle explicitly flagged (an estimate-vs-actual basin-count discrepancy, a missing/unavailable data source) that the recommendation silently ignores.

Do not invent new critiques beyond what the evidence bundle actually supports checking; if the recommendation is accurate and appropriately hedged, say so and change nothing.

Separately from the wording check above, judge whether any issue you found is fixable by editing the TEXT alone, or whether it actually points to a problem with an earlier ANALYSIS decision that no amount of rewording can fix:
- If the D8 cross-check shows weak sink-agreement or a near-zero/negative contributing-area correlation for what should be real drainage basins, the persistence THRESHOLD chosen earlier may have been too permissive or too strict for this terrain -- set "reanalysis_target" to "threshold".
- If weight_sensitivity shows the #1-ranked basin changes across weighting schemes, or the ranking looks driven by one factor in a way that doesn't match the stated objective, the RANKING/weighting may need to be redone -- set "reanalysis_target" to "ranking".
- If the only problem is unsupported wording that a rewrite fixes (the usual case), set "reanalysis_target" to null -- do NOT ask for re-analysis just because a number looks surprising; only when the evidence itself looks internally inconsistent with what "significant" or "well-ranked" should mean.

Respond with ONLY a JSON object of this exact shape:
{"issues_found": <bool>, "critique": "1-3 plain-language sentences: what you checked, and specifically what was wrong (quote or paraphrase the unsupported claim) if issues_found is true, or what you verified as consistent with the evidence if false.", "revised_recommendation": <the full corrected recommendation text, same structure/length/formatting as the original but with the specific issues fixed, if issues_found is true -- otherwise null>, "reanalysis_target": <"threshold", "ranking", or null -- per the rule above, independent of whether issues_found is true>}"""


def reflect_on_recommendation(llm: LLMClient, recommendation: str, evidence_context: str) -> dict:
    """Reflection pattern around the recommendation-writing call: a second
    LLM call re-reads the exact `evidence_context` the writer saw (the same
    user_context string web_app/app.py already builds for the writing
    call) alongside the `recommendation` it produced, and checks the
    recommendation for claims the evidence doesn't support. Returns
    {"issues_found", "revision_applied", "critique", "final_recommendation",
    "reanalysis_target"}. `issues_found` is the model's own verdict,
    reported as-is -- do NOT read it as "the text changed." `revision_
    applied` is true only when `issues_found` was true AND a non-empty
    revision was actually returned; `final_recommendation` is the model's
    `revised_recommendation` only when `revision_applied` is true,
    otherwise `recommendation` unchanged, so a malformed or empty revision
    can never silently blank out a working recommendation. These two were
    collapsed into one field early on (`"issues_found": revised_applied`)
    -- a real bug caught by review: a model reporting `issues_found: true`
    with a real critique but an empty/malformed revision would report
    `issues_found: False` to the caller, contradicting its own critique --
    e.g. report.py's reflection section printing "found no unsupported
    claims" directly above a critique paragraph describing a real problem.
    Callers wanting "was the text actually changed" must read
    `revision_applied`, not `issues_found`.

    `reanalysis_target` ("threshold", "ranking", or None) is this
    function's answer to a harder question than "is the wording right":
    does the SELF-review think an earlier analysis decision (not just this
    text) needs to be redone? This is what lets web_app/app.py's pipeline
    loop backward on its OWN judgment -- not only when a human happens to
    say something at one of the interjection checkpoints (see
    agent_analyze_stream's checkpoint_wait) -- e.g. when the D8 cross-check
    or weight-sensitivity evidence looks internally inconsistent with what
    the recommendation is claiming. The caller is responsible for actually
    bounding how many times it acts on this (this function has no memory
    of previous rounds and would say "threshold" again forever if asked
    again with the same bad evidence).

    Raises on a genuine LLM call failure (network, missing API key) --
    same "caller catches and keeps the un-reflected recommendation instead
    of failing the run" contract as every other agent decision here."""
    user_prompt = (
        f"Evidence bundle this run collected:\n{evidence_context}\n\n"
        f"Recommendation written from that evidence:\n{recommendation}"
    )
    result = llm.chat_json(_REFLECTION_SYSTEM_PROMPT, user_prompt)
    issues_found = bool(result.get("issues_found"))
    critique = result.get("critique") or "No critique text was returned."
    revised = result.get("revised_recommendation")
    revision_applied = issues_found and isinstance(revised, str) and bool(revised.strip())
    reanalysis_target = result.get("reanalysis_target")
    if reanalysis_target not in ("threshold", "ranking"):
        reanalysis_target = None
    return {
        "issues_found": issues_found,
        "revision_applied": revision_applied,
        "critique": critique,
        "final_recommendation": revised if revision_applied else recommendation,
        "reanalysis_target": reanalysis_target,
    }
