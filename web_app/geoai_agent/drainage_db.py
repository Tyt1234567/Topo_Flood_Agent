"""
Live, per-request stormwater pipe network lookup -- no database.

Unlike every other data source in this project's original design, there is
still no single national or global stormwater/storm-drain API -- confirmed
directly: no federal aggregator exists. But the handful of city/county GIS
portals registered in _SOURCES below each turned out to be, on inspection,
ordinary public ArcGIS REST MapServer endpoints with a queryable geometry
filter -- exactly the same technology fema_flood.py's fetch_fema_flood_zones
already queries live, by bounding box, at request time. This module used to
bulk-ETL each of those into a local PostgreSQL/PostGIS database (see git
history / load_*_stormwater.py, now removed) and read from that; it now
queries every source directly, per request, the same live-fetch-no-key-no-
bulk-download pattern as everything else here -- no DATABASE_URL, no
one-time load step, no locally-hosted copy of anyone's GIS data to keep in
sync.

Field-name/unit quirks per source (percent vs. fraction slope, sentinel
nulls, which field is actually a usable diameter) are exactly what the old
loader scripts discovered by inspecting each service's live schema --
carried over unchanged into each source's `normalize` function below.

Coverage is still only the specific US regions in _SOURCES (there is no
national or global equivalent) -- a location outside all of them returns
None, same as "no FEMA data for this area." Each source also carries an
approximate bounding box so a request nowhere near any of them (the common
case, non-US locations especially) skips every network call rather than
firing several pointless requests.

Capacity estimate (pipe_capacity_m3s) is a direct Manning's-equation
full-flow calculation from a pipe's own diameter and slope, only -- no
per-material roughness lookup, no falling back to a slope derived from
invert elevations when the pipe's own SLOPE field is missing. When a pipe
doesn't have both of those two numbers directly, its capacity is simply
unknown (None); the app still reports whether a local minimum has ANY
pipe nearby at all (tag_hotspots_with_drainage_capacity's
"nearby_pipe_count") independent of whether that pipe's capacity could be
computed, which is the primary, always-available signal -- the capacity
number is a bonus for the subset of pipes with usable data, not something
worth extra machinery to coax out of incomplete records.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# A single generic Manning's roughness coefficient (a standard published
# mid-range value, e.g. FHWA HEC-22 -- close to concrete/clay pipe, the
# most common materials in this project's registered sources) rather than a
# per-material lookup table -- close enough for an order-of-magnitude
# capacity estimate without needing to trust each source's own MATRL field.
_MANNING_N = 0.013

# Slope values outside this range are treated as bad data (a unit/decimal
# error in the source, e.g. percent entered where a fraction was expected)
# rather than real pipe grades -- observed directly: a small fraction of
# DC's Pipe records have SLOPE values like 3.9 (390%), physically
# implausible for a gravity sewer.
_MIN_PLAUSIBLE_SLOPE = 0.0001   # 0.01%
_MAX_PLAUSIBLE_SLOPE = 0.20     # 20%

_METERS_PER_INCH = 0.0254

_SENTINELS = frozenset({-9998, -9999})


def pipe_capacity_m3s(diameter_in: float | None, slope: float | None) -> float | None:
    """Manning's-equation full-flow capacity (m^3/s) of a circular pipe
    from its own diameter and slope only, or None if either is missing or
    the slope is outside _MIN/MAX_PLAUSIBLE_SLOPE."""
    if not diameter_in or diameter_in <= 0 or not slope or slope <= 0:
        return None
    if not (_MIN_PLAUSIBLE_SLOPE <= slope <= _MAX_PLAUSIBLE_SLOPE):
        return None
    diameter_m = diameter_in * _METERS_PER_INCH
    area_m2 = math.pi / 4 * diameter_m ** 2
    hydraulic_radius_m = diameter_m / 4  # full circular pipe
    return (1.0 / _MANNING_N) * area_m2 * hydraulic_radius_m ** (2 / 3) * slope ** 0.5


def _clean(value):
    return None if value is None or value in _SENTINELS else value


def _dc_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    diameter_in = props.get("DIAHT") or props.get("WIDTH")
    slope = props.get("SLOPE")
    slope = abs(slope) if slope is not None else None
    return props.get("MATRL"), diameter_in, slope


def _austin_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    diameter_in = props.get("WIDTH") or props.get("HEIGHT")
    slope_pct = props.get("SLOPE")
    slope = abs(slope_pct) / 100.0 if slope_pct is not None else None
    return props.get("MATERIAL"), diameter_in, slope


def _fairfax_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    diameter_in = _clean(props.get("HEIGHT_DIA")) or _clean(props.get("WIDTH"))
    slope_pct = _clean(props.get("SLOPE"))
    slope = abs(slope_pct) / 100.0 if slope_pct is not None else None
    return props.get("MATERIAL"), diameter_in, slope


def _loudoun_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # No slope/invert fields on this source at all -- capacity is always
    # unknown here, by design, not a parsing gap.
    return props.get("Pipe_Type"), props.get("Pipe_Diam"), None


def _pgcounty_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # No diameter/material/slope field exists in this source at all
    # (verified against the live schema) -- capacity is simply unknown for
    # every row, by design, not a parsing gap.
    return props.get("type"), None, None


def _charlotte_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    diameter_in = props.get("PACP_Height") or props.get("PACP_Width")
    slope = props.get("SLOPE")
    slope = abs(slope) if slope is not None else None
    return props.get("MATERIAL"), diameter_in, slope


def _anne_arundel_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # Vertical/horizontal diameter are equal for the circular pipes seen in
    # samples (BARREL_SHAPE='C') -- fall back to whichever is present for
    # non-circular shapes. No SLOPE field on this service -- capacity stays
    # unknown for these records.
    diameter_in = props.get("VERT_DIAMETER") or props.get("HORIZ_DIAMETER")
    return props.get("MATERIAL"), diameter_in, None


def _san_diego_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # DIAMETER is 0 (not null) for box culverts/unsurveyed segments in this
    # source -- `or` treats that as "missing" the same as None, falling
    # through to WIDTH (box-culvert width, inches, same convention as
    # Austin/Charlotte's WIDTH/HEIGHT fields). SLOPE is in PERCENT here --
    # confirmed directly by cross-checking against FROM_ELEV/TO_ELEV/
    # ACT_LENGTH on sample rows (e.g. a SLOPE=1 record's own inverts/length
    # computed to 1.000% exactly), same as Austin/Fairfax/San Diego's own
    # peers -- divided by 100 before calling pipe_capacity_m3s.
    diameter_in = props.get("DIAMETER") or props.get("WIDTH")
    slope_pct = props.get("SLOPE")
    slope = abs(slope_pct) / 100.0 if slope_pct is not None else None
    return props.get("MATERIAL"), diameter_in, slope


def _portland_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # PIPESIZE (inches, confirmed via sample rows -- circular pipes tagged
    # PIPESHPE='CIRC' show plausible values like 30/48/54) or PIPEHEIGHT for
    # non-circular shapes. No direct SLOPE field on this source (FRM_ELEV/
    # TO_ELEV/SRVY_LEN exist but, per this module's own discipline, slope is
    # never derived from inverts) -- capacity stays unknown for every row.
    diameter_in = props.get("PIPESIZE") or props.get("PIPEHEIGHT")
    return props.get("MATERIAL"), diameter_in, None


def _san_antonio_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # Diameter_Inches is explicitly in inches (confirmed by the field's own
    # name and plausible sample values like 24/36/48); Width_Feet is in FEET
    # for the rarer non-circular segments -- converted to inches (*12) for a
    # consistent unit before falling back to it. No direct slope field
    # (Downstream/UpstreamInvert_Feet + Length_Feet exist but aren't used to
    # derive one, same as this module's other invert-only sources).
    diameter_in = props.get("Diameter_Inches") or None
    if not diameter_in:
        width_ft = props.get("Width_Feet")
        diameter_in = width_ft * 12 if width_ft else None
    return props.get("Material"), diameter_in, None


def _ncdot_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # NCDOT's statewide (not just one city) storm drainage pipe inventory --
    # PipeSize (inches) and PipeType are frequently "Unknown"/None in
    # practice (checked directly: sparse coverage even along inventoried
    # routes) but still contribute a real nearby_pipe_count presence signal
    # per this module's design. No slope field on this source.
    return props.get("PipeType"), props.get("PipeSize"), None


def _achd_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # Ada County Highway District (Boise, ID metro) -- Diameter (inches) and
    # Material both directly usable, confirmed via sample rows. No slope
    # field on this source.
    return props.get("Material"), props.get("Diameter"), None


def _ctdot_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # CTDOT's statewide closed-drainage-conveyance inventory -- Diameter
    # (inches, confirmed via sample rows e.g. 15/24) for circular pipes,
    # Width for the rarer non-circular ones. ElevationIn/ElevationOut/Length
    # exist but, per this module's discipline, aren't used to derive a
    # slope this source doesn't report directly.
    diameter_in = props.get("Diameter") or props.get("Width")
    return props.get("Material"), diameter_in, None


def _mdot_sha_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # Maryland State Highway Administration's statewide NPDES conveyance
    # inventory (highway-maintained drainage, distinct coverage from the
    # county-maintained pgcounty_md/anne_arundel_county_md sources already
    # registered). PI_PIPE_SIZE1 uses 9999 as an explicit "unknown" sentinel
    # (observed directly on real rows) -- treated as missing, same idea as
    # Fairfax's -9998/-9999. No slope field despite INVERT_IN/INVERT_OUT/
    # UNIT_LEN existing (same invert-not-derived-into-slope discipline).
    diameter_in = props.get("PI_PIPE_SIZE1")
    if diameter_in in (9999, 0):
        diameter_in = None
    return props.get("PI_MTRL_PIPE"), diameter_in, None


_LA_COUNTY_MATERIAL_CODES = {
    1: "Reinforced Concrete Pipe (RCP)", 2: "Plastic Pipe", 3: "Reinforced Concrete Box (RCB)",
    4: "Corrugated Metal Pipe (CMP)", 5: "Reinforced Concrete Arch (RCA)", 6: "Cast/Ductile Iron Pipe (CIP)",
    7: "Improved Channel", 8: "Corrugated Steel Pipe (CSP)", 9: "Concrete Pipe",
    10: "Acrylonitrile-Butadiene-Styrene (ABS)", 11: "Polyvinyl Chloride (PVC)", 12: "Steel Pipe",
    13: "Vitrified Clay Pipe (VCP)", 14: "Unreinforced Concrete Pipe", 15: "Asbestos Cement Pipe",
    16: "Polyethylene Liner", 17: "Techite", 18: "Dirt Channel", 19: "Dirt Swale", 20: "Brick",
    21: "Cured-In-Place Pipe Liner (CIPP)", 22: "High Density Polyethylene Pipe (HDPE)",
    23: "Reinforced Cement Concrete (RCC)", 24: "Truss Pipe", 25: "High Density Polypropylene Pipe (HDPP)",
}


def _la_county_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # LA County DPW's "Drains Completed" layer (part of a much larger multi-
    # layer storm drain network service, sdn/MapServer -- this is the one
    # real Feature Layer among several "Group Layer" siblings like the plain
    # "Drains" (33) folder). MATERIAL is a coded integer (verified against
    # the layer's own published domain, swDomainMaterial -- not guessed);
    # DIAMETER_HEIGHT (inches, confirmed via sample rows: 48/39/72) covers
    # circular pipes, WIDTH the rarer box culverts. No slope field on this
    # source -- capacity stays unknown for every row, same as most other
    # sources registered here.
    material_code = props.get("MATERIAL")
    material = _LA_COUNTY_MATERIAL_CODES.get(material_code) if material_code is not None else None
    diameter_in = props.get("DIAMETER_HEIGHT") or props.get("WIDTH")
    return material, diameter_in, None


def _riverside_ca_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # City of Riverside's Stormwater "Main" layer -- unusually rich schema
    # among this registry's sources: PipeDiameter (inches, confirmed via
    # sample rows: 12/18/24/36) AND a real Slope field. Slope here is
    # ALREADY a unitless fraction, not percent -- confirmed directly by
    # cross-checking against UpstreamInvert/DownstreamInvert/PipeLength on
    # real rows (e.g. a Slope=0.006 record's own inverts/length computed to
    # 0.0059, matching almost exactly) -- used as-is, no /100 division
    # (same convention as DC/Charlotte, unlike Austin/Fairfax/San Diego's
    # percent-based SLOPE fields).
    diameter_in = props.get("PipeDiameter") or props.get("PipeWidth")
    slope = props.get("Slope")
    slope = abs(slope) if slope is not None else None
    return props.get("PipeMaterial"), diameter_in, slope


def _tampa_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # City of Tampa's "Stormwater Gravity Mains" layer -- DIAMETER (inches)
    # is mostly clean municipal nominal sizes (12/15/18/24/30 seen directly
    # in samples) with an occasional non-round value (e.g. 12.18) that's a
    # real surveyed/as-built measurement, not a unit-conversion bug. SLOPE
    # is a unitless fraction, not percent -- confirmed by cross-checking
    # against DOWNELEV/UPELEV on real rows without a matching LENGTH field:
    # a SLOPE=0.006 row with a 0.80ft elevation drop implies a 133ft pipe
    # (plausible) if fraction, vs. a 2.5-MILE segment if treated as percent
    # (impossible) -- decisively fraction, same convention as DC/Charlotte/
    # Riverside.
    diameter_in = props.get("DIAMETER")
    slope = props.get("SLOPE")
    slope = abs(slope) if slope is not None else None
    return props.get("MATERIAL"), diameter_in, slope


def _hillsborough_county_fl_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # Hillsborough County, FL (Tampa's surrounding county)'s HC_STORMWATER
    # layer -- WIDTH equals HEIGHT for circular pipes in every sample row
    # checked (P_SHAPE='CIRC'), both in inches (12/15/18/24 seen directly)
    # -- WIDTH used as the single diameter value. No elevation/invert/
    # length field exists on this source at all -- capacity stays unknown
    # for every row, by design, not a parsing gap.
    return props.get("MATERIAL"), props.get("WIDTH"), None


def _las_vegas_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # City of Las Vegas "Storm_Arcs" layer -- no MATERIAL field exists on
    # this source at all (checked against the live schema). DIAMETER
    # (inches, confirmed via sample rows: consistently 36 in the test
    # sample) for circular pipes, WIDTH for box culverts (0 when unused,
    # `or` treats that as missing). SLOPE is in PERCENT -- confirmed by
    # cross-checking against US_INVERT/DS_INVERT/FAC_LENGTH on real rows
    # (e.g. SLOPE=10 rows computed to ~0.10-0.12 fraction from their own
    # inverts/length, matching the /100 interpretation; the fraction-as-is
    # interpretation would imply 1000% grades, physically impossible).
    diameter_in = props.get("DIAMETER") or props.get("WIDTH")
    slope_pct = props.get("SLOPE")
    slope = abs(slope_pct) / 100.0 if slope_pct is not None else None
    return None, diameter_in, slope


def _massdot_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # MassDOT's statewide stormwater conveyance inventory -- PIPESIZE is a
    # STRING here (unlike every other source registered), formatted like
    # '12"'/'10"' (confirmed via sample rows) -- stripped and parsed, falling
    # back to None on anything that doesn't match that pattern rather than
    # guessing. No slope field on this source.
    diameter_in = None
    raw_size = props.get("PIPESIZE")
    if raw_size:
        try:
            diameter_in = float(str(raw_size).strip().rstrip('"'))
        except ValueError:
            diameter_in = None
    return props.get("MATERIAL"), diameter_in, None


def _milwaukee_county_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # Milwaukee County DoT's storm system -- county-highway-maintained only
    # (sparse coverage confirmed directly: a downtown-Milwaukee-city bbox
    # returned nothing, but a full-county bbox does), not a city-wide
    # network. material is already a readable string ("Concrete"/"PVC"),
    # no domain decode needed. height doubles as diameter for circular
    # pipes (confirmed via sample rows: 12/36/48, plausible inches); width
    # is the box-culvert fallback. No slope field on this source.
    diameter_in = props.get("height") or props.get("width")
    return props.get("material"), diameter_in, None


def _oakland_ca_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # Oakland's citywide storm drain pipe layer. DIAMINCH is 0 (not null)
    # for many unsurveyed segments -- `or` treats that as missing, same
    # convention as San Diego. MAT_CODE is a coded integer but the schema
    # exposes no domain/codedValues for it, and the adjacent MAT_INSP text
    # field (checked directly) is blank on every sampled row -- left as
    # material=None rather than guessing what the codes mean. No slope
    # field on this source (FROMFLLINE/TOFLLINE flowline elevations exist
    # but, per this module's discipline, aren't used to derive one).
    diameter_in = props.get("DIAMINCH") or None
    return None, diameter_in, None


def _stark_county_oh_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # Stark County, OH's aggregated countywide stormwater pipe layer.
    # Diameter (inches, confirmed via sample rows: 18/15/24/20/12) and
    # Material (often null, but real text like "HDPE" when present) used
    # directly. Slope is in PERCENT here -- confirmed directly by cross-
    # checking against UpInvertElevation/DownInvertElevation/RecordLength
    # on real rows (e.g. Slope=1.0 matched a computed 1.000% exactly, and
    # several others matched to 3+ decimal places) -- divided by 100.
    diameter_in = props.get("Diameter")
    slope_pct = props.get("Slope")
    slope = abs(slope_pct) / 100.0 if slope_pct is not None else None
    return props.get("Material"), diameter_in, slope


_WSDOT_WIDTH_UNIT_TO_INCHES = {"Inches": 1.0, "Feet": 12.0, "Meters": 39.3701}


def _wsdot_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # WSDOT's (Washington State DOT) statewide highway drainage pipe
    # inventory -- unusually well-documented schema with an explicit units
    # field per dimension (PipeDiameterUnits, PipeWidthUnits) rather than
    # this registry's usual "guess from plausible values" approach.
    # PipeDiameter's own units field is consistently "Inches" across a
    # 200-row sample (checked directly, no conversion needed); PipeWidth's
    # units field is NOT consistent (mixes "Inches" and "Meters" in real
    # data, checked directly) so it IS converted here, unlike every other
    # box-culvert-width fallback in this registry. Slope has no units
    # field of its own, but its value distribution (30-row sample: mode of
    # exactly 0.5, matching the extremely common "0.5% minimum drainage
    # slope" civil-engineering convention, plus a cluster of values in the
    # 0.1-2.3 range) is decisively consistent with PERCENT, not a
    # unitless fraction (which would imply absurd 50%+ grades for the
    # modal value) -- divided by 100.
    diameter_in = props.get("PipeDiameter")
    if not diameter_in:
        width = props.get("PipeWidth")
        width_units = props.get("PipeWidthUnits")
        factor = _WSDOT_WIDTH_UNIT_TO_INCHES.get(width_units)
        diameter_in = width * factor if (width and factor) else None
    slope_pct = props.get("Slope")
    slope = abs(slope_pct) / 100.0 if slope_pct is not None else None
    return props.get("PipeMaterialType"), diameter_in, slope


def _lee_county_fl_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # Lee County, FL (Fort Myers/Cape Coral) drainage pipe layer -- Pipe
    # Diameter is a numeric STRING here (confirmed via sample rows: '24',
    # '36', '60', all plausible inches), parsed safely; RdPipeEquivalentDia
    # is already numeric and used as a fallback for the rows that have it
    # instead. TYPE doubles as the material field (RCP etc., confirmed via
    # samples). No slope field on this source.
    diameter_in = None
    raw = props.get("PipeDiameter")
    if raw:
        try:
            diameter_in = float(raw)
        except (TypeError, ValueError):
            diameter_in = None
    if not diameter_in:
        diameter_in = props.get("RdPipeEquivalentDia")
    return props.get("TYPE"), diameter_in, None


def _fmfcd_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # Fresno Metropolitan Flood Control District's own pipe network (a
    # dedicated flood-control agency, not the city's general GIS) -- real
    # MATERIAL/DIAMETER(inches)/SLOPE all present. SLOPE is ALREADY a
    # unitless fraction here -- confirmed directly by cross-checking against
    # FL_UP/FL_DN (flowline elevations) and Shape__Length on real rows (e.g.
    # a SLOPE=0.0234 record's own inverts/length computed to 0.02343,
    # matching almost exactly), same convention as DC/Charlotte/Riverside,
    # unlike Austin/Fairfax/San Diego's percent-based SLOPE fields.
    diameter_in = props.get("DIAMETER")
    slope = props.get("SLOPE")
    slope = abs(slope) if slope is not None else None
    return props.get("MATERIAL"), diameter_in, slope


def _elk_grove_ca_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # City of Elk Grove, CA (Sacramento metro) -- PIPE_SIZE is a numeric
    # STRING with its own explicit PIPE_SIZE_UOM field (checked directly:
    # always "INCH" across samples, no unit ambiguity), parsed safely. This
    # single "Pipeline" layer also carries other utility types, but
    # ASSET_ENTITY='DRAINAGE' identifies the stormwater subset -- every
    # sampled row (50/50) was already DRAINAGE, so no additional filtering
    # was added here, consistent with every other source in this registry
    # not doing server-side WHERE filtering. No slope field.
    diameter_in = None
    raw = props.get("PIPE_SIZE")
    if raw:
        try:
            diameter_in = float(raw)
        except (TypeError, ValueError):
            diameter_in = None
    return props.get("MATERIAL"), diameter_in, None


def _memphis_area_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # A Shelby County/Memphis, TN drainage-basin ("Allen") stormwater model
    # reconciliation dataset -- DIAMETER (inches, plausible samples: 15/18/
    # 21/36/42) and SLOPE both present. SLOPE is in PERCENT here -- checked
    # directly against FROM_ELE/TO_ELE/Shape__Length on real rows with
    # non-zero inverts (e.g. SLOPE=4 vs. computed 4.21%, SLOPE=1.45 vs.
    # computed 1.44%) -- divided by 100 before calling pipe_capacity_m3s.
    # MATERIAL is noisy in this source (a meaningful fraction of sampled
    # rows show "STRUCT_COMM", apparently a structure-comment placeholder
    # bleeding into the field rather than a real material) -- passed through
    # as-is rather than guessed at or filtered, same "don't silently
    # fabricate a cleaner answer than the source actually gives" discipline
    # this module applies everywhere else.
    diameter_in = props.get("DIAMETER")
    slope_pct = props.get("SLOPE")
    slope = abs(slope_pct) / 100.0 if slope_pct is not None else None
    return props.get("MATERIAL"), diameter_in, slope


def _pinellas_park_fl_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # City of Pinellas Park, FL (Tampa Bay area) closed-conveyance layer --
    # "PIPE" is mostly a real material description (HDPE/PVC/CORRUGATED
    # METAL PIPE/ELLIPTICAL REINFORCED CONCRETE PIPE, confirmed via sample
    # rows), though a minority of rows show a manufacturer name instead
    # ("ADVANCED DRAINAGE SYSTEMS") -- passed through as-is rather than
    # filtered, same "don't fabricate a cleaner answer than the source
    # gives" rule this module applies everywhere. HEIGHT (inches, plausible
    # samples: 12/15/18/24) is the diameter field for the circular majority;
    # WIDTH is populated only for the rarer non-circular (elliptical/box)
    # shapes. No slope field on this source.
    diameter_in = props.get("HEIGHT") or props.get("WIDTH")
    return props.get("PIPE"), diameter_in, None


def _bakersfield_ca_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # City of Bakersfield, CA storm water pipe layer. DIAMETER (inches,
    # plausible samples: 18/21/24/30) and MATERIAL direct. SLOPE_BOT is in
    # PERCENT -- confirmed directly by parsing the leading numeric value off
    # INVERT_IN/INVERT_OUT (stored as strings like "339.13 N", a value plus
    # a compass-direction suffix this module doesn't otherwise need) and
    # cross-checking against PIPE_LGTH (e.g. a SLOPE_BOT=0.4 record's own
    # inverts/length computed to 0.397%, matching closely) -- divided by 100
    # before calling pipe_capacity_m3s.
    diameter_in = props.get("DIAMETER")
    slope_pct = props.get("SLOPE_BOT")
    slope = abs(slope_pct) / 100.0 if slope_pct is not None else None
    return props.get("MATERIAL"), diameter_in, slope


def _rancho_cordova_ca_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # A "Drainage_Laterals" service covering Rancho Cordova, CA (Sacramento
    # County -- identified by its own LATITUDE/LONGITUDE field extent,
    # 38.53-38.64N/121.34-121.19W, since the item carries no agency name in
    # its metadata). DPDIA (inches, plausible samples: 10/15) is the
    # diameter field; DRNPMAT the material. SLOPE is ALREADY a unitless
    # fraction here -- confirmed directly by cross-checking against
    # UPSTRM_INVERT/DNSTRM_INVERT/Shape__Length on real rows (e.g.
    # SLOPE=0.0022 vs. computed 0.00347, same order of magnitude and no
    # ~100x mismatch the way a percent-vs-fraction error would show) -- used
    # as-is, no /100 division.
    diameter_in = props.get("DPDIA")
    slope = props.get("SLOPE")
    slope = abs(slope) if slope is not None else None
    return props.get("DRNPMAT"), diameter_in, slope


def _salinas_ca_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # City of Salinas, CA "StormGravityMain" layer -- DIAMETER (inches,
    # plausible samples: 24/30/78/84) is the diameter field; MATERIAL exists
    # in the schema but is null on every sampled row (passed through as-is,
    # not fabricated). SLOPE values observed (0.0017-0.005) sit squarely in
    # the plausible unitless-fraction range for gravity storm sewer grades
    # (0.17%-0.5%) -- this source's own UPELEV/DOWNELEV invert fields are
    # null on every sampled row, so a direct percent-vs-fraction cross-check
    # wasn't possible, but interpreting these values as already-percent
    # would imply a physically-nonsensical ~0.002% (nearly dead flat) grade,
    # and this module's own pipe_capacity_m3s plausibility filter
    # (_MIN/MAX_PLAUSIBLE_SLOPE) would silently reject the wrong
    # interpretation's implied fraction anyway -- used as-is, no /100.
    diameter_in = props.get("DIAMETER")
    slope = props.get("SLOPE")
    slope = abs(slope) if slope is not None else None
    return props.get("MATERIAL"), diameter_in, slope


def _pinellas_county_fl_normalize(props: dict) -> tuple[str | None, float | None, float | None]:
    # Pinellas County, FL "Stormwater Gravity Mains" layer -- MATERIAL is a
    # real string (RCP/ERCP seen directly); CUL_SIZE is a STRING, either a
    # plain number (circular pipes, inches -- 15/24/18/36 seen) or a
    # "WxH"-style elliptical/box dimension (e.g. "14X23") -- the larger of
    # the two numbers is used as the effective diameter for those. No
    # slope/invert field on this source -- capacity stays unknown for every
    # row, by design, not a parsing gap.
    diameter_in = None
    raw_size = props.get("CUL_SIZE")
    if raw_size:
        parts = str(raw_size).upper().split("X")
        try:
            diameter_in = max(float(p.strip()) for p in parts if p.strip())
        except ValueError:
            diameter_in = None
    return props.get("MATERIAL"), diameter_in, None


# Each entry: a public ArcGIS REST MapServer pipe layer (queried live, by
# bounding box, exactly like fema_flood.fetch_fema_flood_zones), the
# OBJECTID+attribute fields worth asking for, a `normalize` function mapping
# that source's own field names/units onto (material, diameter_in, slope),
# and an approximate bounding box (generous margins) used to skip a source
# entirely -- no network call at all -- when the requested area couldn't
# possibly be inside it.
_SOURCES = [
    {
        "name": "dc",
        "urls": ["https://gis.streamstats.usgs.gov/arcgis/rest/services/stormdrain/dc/MapServer/2/query"],
        "fields": "OBJECTID,MATRL,DIAHT,WIDTH,XSECTSHP,SLOPE,USINVERT,DSINVERT,LENGTH",
        "normalize": _dc_normalize,
        "bbox": (38.79, 39.00, -77.13, -76.90),
    },
    {
        "name": "austin_tx",
        "urls": ["https://maps.austintexas.gov/gis/rest/Shared/DrainageInfrastructure/MapServer/6/query"],
        "fields": "OBJECTID,MATERIAL,SLOPE,WIDTH,HEIGHT,STRUCT_SHAPE,UP_INVERT,DN_INVERT,LENGTH",
        "normalize": _austin_normalize,
        "bbox": (30.08, 30.53, -97.95, -97.55),
    },
    {
        "name": "fairfax_county_va",
        "urls": ["https://www.fairfaxcounty.gov/gispub1/rest/services/DPWES/StormNetServReq/MapServer/16/query"],
        "fields": "OBJECTID,MATERIAL,SLOPE,HEIGHT_DIA,WIDTH,IN_ELEV,OUT_ELEV,ASSETLENGTH",
        "normalize": _fairfax_normalize,
        "bbox": (38.60, 38.99, -77.53, -77.00),
    },
    {
        "name": "loudoun_county_va",
        "urls": ["https://logis.loudoun.gov/gis/rest/services/COL/Utilities/MapServer/4/query"],
        "fields": "OBJECTID,Pipe_Diam,Pipe_Type,Feature",
        "normalize": _loudoun_normalize,
        "bbox": (38.85, 39.33, -78.00, -77.35),
    },
    {
        "name": "pgcounty_md",
        "urls": ["https://gis.princegeorgescountymd.gov/arcgis/rest/services/DPWT_SDI/DPWT_SDI/MapServer/1/query"],
        "fields": "objectid,type,invert_in_elevation,invert_out_elevation,plan_length",
        "normalize": _pgcounty_normalize,
        "bbox": (38.68, 39.17, -77.02, -76.64),
    },
    {
        "name": "charlotte_nc",
        "urls": ["https://gis.charlottenc.gov/arcgis/rest/services/STM/StormPipes/MapServer/0/query"],
        "fields": "OBJECTID,MATERIAL,PACP_Height,PACP_Width,PI_SHAPE,SLOPE,US_INVERT,DS_INVERT,measuredlength",
        "normalize": _charlotte_normalize,
        "bbox": (35.00, 35.50, -81.10, -80.55),
    },
    {
        "name": "anne_arundel_county_md",
        "urls": [
            "https://gis.aacounty.org/arcgis/rest/services/OpenData/Utility_Stormwater_OpenData/MapServer/5/query",
            "https://gis.aacounty.org/arcgis/rest/services/OpenData/Utility_Stormwater_OpenData/MapServer/14/query",
        ],
        "fields": "OBJECTID,MATERIAL,VERT_DIAMETER,HORIZ_DIAMETER,BARREL_SHAPE,INVERT_IN,INVERT_OUT,LENGTH",
        "normalize": _anne_arundel_normalize,
        "bbox": (38.80, 39.23, -76.87, -76.42),
    },
    {
        "name": "san_diego_ca",
        "urls": ["https://webmaps.sandiego.gov/arcgis/rest/services/StormWater/SWD_Public/MapServer/6/query"],
        "fields": "OBJECTID,MATERIAL,DIAMETER,WIDTH,SLOPE,CONV_SHP",
        "normalize": _san_diego_normalize,
        "bbox": (32.53, 33.11, -117.32, -116.75),
    },
    {
        "name": "portland_or",
        "urls": ["https://www.portlandmaps.com/arcgis/rest/services/Public/Utilities_Sewer/MapServer/7/query"],
        "fields": "OBJECTID,MATERIAL,PIPESIZE,PIPEHEIGHT,PIPESHPE",
        "normalize": _portland_normalize,
        "bbox": (45.40, 45.65, -122.85, -122.45),
    },
    {
        "name": "san_antonio_tx",
        "urls": ["https://services.arcgis.com/g1fRTDLeMgspWrYp/arcgis/rest/services/StormwaterUnderground/FeatureServer/0/query"],
        "fields": "OBJECTID,Material,Diameter_Inches,Width_Feet,Height_Feet,StructureShape",
        "normalize": _san_antonio_normalize,
        "bbox": (29.17, 29.70, -98.80, -98.25),
    },
    {
        # Statewide -- NCDOT-maintained roadway storm drainage, not a
        # city-limited source like the rest of the registry.
        "name": "nc_statewide_dot",
        "urls": ["https://services.arcgis.com/NuWFvHYDMVmmxMeM/arcgis/rest/services/NCDOT_Storm_Drainage_Pipes_Public_View/FeatureServer/1/query"],
        "fields": "OBJECTID,PipeType,PipeSize,LengthFeet",
        "normalize": _ncdot_normalize,
        "bbox": (33.75, 36.60, -84.35, -75.40),
    },
    {
        "name": "ada_county_id",
        "urls": ["https://gis.achdidaho.org/server/rest/services/ArcGIS_Hub/Storm_Drain_Pipes/FeatureServer/0/query"],
        "fields": "OBJECTID,Material,Diameter,PipeShape",
        "normalize": _achd_normalize,
        "bbox": (43.35, 43.75, -116.45, -115.80),
    },
    {
        # Statewide -- CTDOT-maintained roadway closed drainage conveyance.
        "name": "ct_statewide_dot",
        "urls": ["https://services1.arcgis.com/FCaUeJ5SOVtImake/arcgis/rest/services/CTDOT_Drainage_Network/FeatureServer/1/query"],
        "fields": "OBJECTID,Material,Diameter,Width,Height",
        "normalize": _ctdot_normalize,
        "bbox": (40.95, 42.10, -73.75, -71.75),
    },
    {
        # Statewide -- MDOT SHA (highway-maintained) NPDES conveyance
        # inventory, distinct coverage from the county-maintained
        # pgcounty_md/anne_arundel_county_md sources above.
        "name": "md_statewide_sha",
        "urls": ["https://utility.arcgis.com/usrsvcs/servers/23ab9309817f4832810937c8a16239d1/rest/services/NPDES/NPDES_Simple_Conveyance_Maryland_MDOTSHA/FeatureServer/0/query"],
        "fields": "OBJECTID,PI_MTRL_PIPE,PI_PIPE_SIZE1,PI_XS_SHAPE",
        "normalize": _mdot_sha_normalize,
        "bbox": (37.90, 39.75, -79.50, -75.00),
    },
    {
        # Statewide -- MassDOT-maintained stormwater conveyance inventory.
        "name": "ma_statewide_dot",
        "urls": ["https://gis.massdot.state.ma.us/arcgis/rest/services/Assets/Stormwater/FeatureServer/5/query"],
        "fields": "OBJECTID,MATERIAL,PIPESIZE,LINETYPE",
        "normalize": _massdot_normalize,
        "bbox": (41.20, 42.90, -73.55, -69.85),
    },
    {
        # LA County DPW's countywide storm drain network (city of LA plus
        # unincorporated county area) -- "Drains Completed" (layer 34), the
        # one real feature layer among several "Group Layer" siblings under
        # the "Drains" folder (33) in this same multi-hundred-layer service.
        "name": "la_county_ca",
        "urls": ["https://dpw.gis.lacounty.gov/dpw/rest/services/sdn/MapServer/34/query"],
        "fields": "OBJECTID,MATERIAL,DIAMETER_HEIGHT,WIDTH,CROSS_SECTION_SHAPE",
        "normalize": _la_county_normalize,
        "bbox": (33.70, 34.35, -118.95, -117.60),
    },
    {
        "name": "riverside_ca",
        "urls": ["https://mapriverside.riversideca.gov/server/rest/services/Stormwater/Stormwater/MapServer/24/query"],
        "fields": "OBJECTID,PipeMaterial,PipeDiameter,PipeWidth,Slope",
        "normalize": _riverside_ca_normalize,
        "bbox": (33.85, 34.05, -117.50, -117.25),
    },
    {
        "name": "tampa_fl",
        "urls": ["https://arcgis.tampagov.net/arcgis/rest/services/OpenData/UtilityStorm/MapServer/4/query"],
        "fields": "OBJECTID,MATERIAL,DIAMETER,SLOPE",
        "normalize": _tampa_normalize,
        "bbox": (27.85, 28.10, -82.60, -82.30),
    },
    {
        "name": "hillsborough_county_fl",
        "urls": ["https://maps.hillsboroughcounty.org/arcgis/rest/services/CORPORATE/HC_STORMWATER/MapServer/0/query"],
        "fields": "OBJECTID,MATERIAL,WIDTH,HEIGHT,P_SHAPE",
        "normalize": _hillsborough_county_fl_normalize,
        "bbox": (27.65, 28.20, -82.75, -82.10),
    },
    {
        "name": "las_vegas_nv",
        "urls": ["https://mapdata.lasvegasnevada.gov/clvgis/rest/services/ExistingInfrastructure/Storm_Drains_Channels_Junctions/MapServer/1/query"],
        "fields": "OBJECTID,DIAMETER,WIDTH,SLOPE",
        "normalize": _las_vegas_normalize,
        "bbox": (36.00, 36.35, -115.35, -115.00),
    },
    {
        # County-highway-maintained only, not a full city network (see
        # _milwaukee_county_normalize) -- sparse but real.
        "name": "milwaukee_county_wi",
        "urls": ["https://services7.arcgis.com/BdlPm2SqreXhxZvT/arcgis/rest/services/Milwaukee_Co_DoT_Storm_System/FeatureServer/1/query"],
        "fields": "OBJECTID,material,height,width,pipeshape",
        "normalize": _milwaukee_county_normalize,
        "bbox": (42.85, 43.20, -88.20, -87.85),
    },
    {
        "name": "oakland_ca",
        "urls": ["https://services.arcgis.com/9tC74aDHuml0x5Yz/arcgis/rest/services/StormDrain/FeatureServer/1/query"],
        "fields": "OBJECTID_1,MAT_CODE,DIAMINCH",
        "normalize": _oakland_ca_normalize,
        "bbox": (37.70, 37.88, -122.35, -122.10),
    },
    {
        # Countywide (Canton, OH area) -- found via ArcGIS Online item
        # search, not a target metro but a real, well-attributed source.
        "name": "stark_county_oh",
        "urls": ["https://scgisa.starkcountyohio.gov/arcgis/rest/services/AggregateDatasets/StarkCountywideStormwaterSystems/MapServer/2/query"],
        "fields": "OBJECTID,Material,Diameter,Slope",
        "normalize": _stark_county_oh_normalize,
        "bbox": (40.65, 40.95, -81.55, -81.15),
    },
    {
        # Statewide -- WSDOT-maintained highway drainage pipe inventory
        # (covers Seattle/Tacoma/Spokane and every other WA metro along
        # state highways, not a city-limited source).
        "name": "wa_statewide_dot",
        "urls": ["https://data.wsdot.wa.gov/arcgis/rest/services/Shared/StormwaterFeaturesInventoryOffice/FeatureServer/7/query"],
        "fields": "OBJECTID,PipeMaterialType,PipeDiameter,PipeWidth,PipeWidthUnits,Slope",
        "normalize": _wsdot_normalize,
        "bbox": (45.50, 49.05, -124.85, -116.90),
    },
    {
        "name": "lee_county_fl",
        "urls": ["https://gismapserver.leegov.com/gisserver910/rest/services/Layers/Cartegraph_Lines/MapServer/5/query"],
        "fields": "OBJECTID,TYPE,PipeDiameter,RdPipeEquivalentDia",
        "normalize": _lee_county_fl_normalize,
        "bbox": (26.35, 26.75, -82.15, -81.55),
    },
    {
        "name": "fresno_ca",
        "urls": ["https://services.arcgis.com/Z7HrzLV55ji5SaOw/arcgis/rest/services/FMFCD_Layers/FeatureServer/6/query"],
        "fields": "OBJECTID,MATERIAL,DIAMETER,SLOPE",
        "normalize": _fmfcd_normalize,
        "bbox": (36.60, 36.95, -119.95, -119.60),
    },
    {
        "name": "elk_grove_ca",
        "urls": ["https://webmaps.elkgrove.gov/arcgis/rest/services/OPEN_DATA_PORTAL/Pipeline/MapServer/0/query"],
        "fields": "OBJECTID,MATERIAL,PIPE_SIZE,PIPE_SIZE_UOM,ASSET_ENTITY",
        "normalize": _elk_grove_ca_normalize,
        "bbox": (38.35, 38.48, -121.45, -121.30),
    },
    {
        "name": "memphis_area_tn",
        "urls": ["https://services7.arcgis.com/3EUdJEPlNgTfOO6w/arcgis/rest/services/Allen_SWMMRecon/FeatureServer/2/query"],
        "fields": "OBJECTID,MATERIAL,DIAMETER,SLOPE,FROM_ELE,TO_ELE",
        "normalize": _memphis_area_normalize,
        "bbox": (35.10, 35.30, -90.05, -89.85),
    },
    {
        "name": "rancho_cordova_ca",
        "urls": ["https://services1.arcgis.com/kSSEfQQl69HmZZw6/arcgis/rest/services/Drainage_Laterals/FeatureServer/0/query"],
        "fields": "OBJECTID,DPDIA,DRNPMAT,SLOPE,UPSTRM_INVERT,DNSTRM_INVERT",
        "normalize": _rancho_cordova_ca_normalize,
        "bbox": (38.50, 38.66, -121.36, -121.17),
    },
    {
        "name": "bakersfield_ca",
        "urls": ["https://gis.bakersfieldcity.us/webmaps/rest/services/General/StormWater/MapServer/4/query"],
        "fields": "OBJECTID,DIAMETER,MATERIAL,SLOPE_BOT,INVERT_IN,INVERT_OUT,PIPE_LGTH",
        "normalize": _bakersfield_ca_normalize,
        "bbox": (35.25, 35.50, -119.15, -118.85),
    },
    {
        "name": "pinellas_park_fl",
        "urls": ["https://services6.arcgis.com/fH2ZwfxOgb5eaBS4/arcgis/rest/services/Public_Works_2_WFL1/FeatureServer/18/query"],
        "fields": "OBJECTID,PIPE,WIDTH,HEIGHT",
        "normalize": _pinellas_park_fl_normalize,
        "bbox": (27.78, 27.88, -82.75, -82.65),
    },
    {
        "name": "salinas_ca",
        "urls": ["https://services6.arcgis.com/bxsVBCAOuftUsVxD/arcgis/rest/services/Salinas_Stormwater_Infrastructure_WFL1/FeatureServer/6/query"],
        "fields": "OBJECTID,MATERIAL,DIAMETER,SLOPE",
        "normalize": _salinas_ca_normalize,
        "bbox": (36.63, 36.72, -121.70, -121.60),
    },
    {
        "name": "pinellas_county_fl",
        "urls": ["https://egis.pinellas.gov/gis/rest/services/AGO/StormwaterData/MapServer/2/query"],
        "fields": "OBJECTID,MATERIAL,CUL_SIZE",
        "normalize": _pinellas_county_fl_normalize,
        "bbox": (27.83, 28.15, -82.90, -82.60),
    },
]

_TIMEOUT_SECONDS = 12.0
_PAGE_SIZE = 1000
# Bounds a pathologically large bounding box to at most 5000 pipes per
# source rather than unlimited pagination -- this is a per-request live
# call inside the analysis pipeline now, not a background one-time ETL job.
_MAX_PAGES_PER_SOURCE = 5


def _bbox_overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    a_lat_min, a_lat_max, a_lon_min, a_lon_max = a
    b_lat_min, b_lat_max, b_lon_min, b_lon_max = b
    return a_lat_min <= b_lat_max and a_lat_max >= b_lat_min and a_lon_min <= b_lon_max and a_lon_max >= b_lon_min


_REQUEST_HEADERS = {
    # Riverside's server (found live, not documented anywhere) 403s any
    # request carrying Python's default "Python-urllib/3.x" User-Agent
    # specifically -- confirmed directly: the exact same query succeeds
    # with curl's default UA, an empty UA, or any self-identifying one, and
    # fails only with urllib's own default string. Applied to every request
    # here, not just Riverside's, since other sources' WAFs could plausibly
    # have the same rule even though none observed so far have tripped it.
    "User-Agent": "GeoAI-Agent/1.0 (+stormwater-pipe-lookup)",
}


def _query_page(url: str, fields: str, bounds: tuple[float, float, float, float], offset: int) -> dict:
    lat_min, lat_max, lon_min, lon_max = bounds
    query = urllib.parse.urlencode({
        "geometry": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "outSR": "4326",
        "outFields": fields,
        "returnGeometry": "true",
        "geometryPrecision": "6",
        "resultOffset": str(offset),
        "resultRecordCount": str(_PAGE_SIZE),
        "f": "geojson",
    })
    request = urllib.request.Request(f"{url}?{query}", headers=_REQUEST_HEADERS)
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read())


def _fetch_source(source: dict, bounds: tuple[float, float, float, float]) -> list[dict]:
    """Live bbox query against one registered source's pipe layer(s).
    Returns a list of normalized pipe dicts (possibly empty) -- any
    network/parse failure for this source is swallowed and treated as "no
    pipes from this source," the same graceful-degrade contract every other
    live source in this project follows (e.g. fema_flood's occasional
    courtesy-server 500s)."""
    normalize = source["normalize"]
    pipes: list[dict] = []
    for url in source["urls"]:
        offset = 0
        for _ in range(_MAX_PAGES_PER_SOURCE):
            try:
                page = _query_page(url, source["fields"], bounds, offset)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError):
                break
            features = page.get("features", [])
            if not features:
                break
            for feature in features:
                geom = feature.get("geometry")
                props = feature.get("properties", {})
                if not geom or geom.get("type") != "LineString":
                    continue
                material, diameter_in, slope = normalize(props)
                pipes.append({
                    "source_city": source["name"],
                    "material": material,
                    "diameter_in": diameter_in,
                    "slope": slope,
                    "capacity_m3s": pipe_capacity_m3s(diameter_in, slope),
                    "coords": geom.get("coordinates"),
                })
            if len(features) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
    return pipes


def fetch_nearby_pipes(bounds: tuple[float, float, float, float]) -> list[dict] | None:
    """Live per-request pipe lookup across every registered source (see
    _SOURCES) whose approximate coverage area overlaps `bounds` (lat_min,
    lat_max, lon_min, lon_max) -- no database, no prior load step. Returns
    None if no registered source's bbox overlaps this area at all (the
    common case outside the handful of US regions covered -- not an error)
    or if every overlapping source's live query failed/returned nothing;
    otherwise a list of dicts with: coords ([[lon,lat], ...]), material,
    diameter_in, slope, capacity_m3s, source_city."""
    candidates = [s for s in _SOURCES if _bbox_overlaps(s["bbox"], bounds)]
    if not candidates:
        return None

    pipes: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = [pool.submit(_fetch_source, s, bounds) for s in candidates]
        for future in as_completed(futures):
            try:
                pipes.extend(future.result())
            except Exception:
                continue
    return pipes or None


def pipes_to_geojson(pipes: list[dict]) -> dict:
    """`fetch_nearby_pipes`'s output as a GeoJSON FeatureCollection of
    LineStrings, for drawing on a map (web_app's persistent map and the
    downloadable report's embedded one both use this the same way they
    already do fema_flood.fetch_fema_flood_zones's GeoJSON)."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "material": p["material"],
                    "diameter_in": p["diameter_in"],
                    "capacity_m3s": p["capacity_m3s"],
                },
                "geometry": {"type": "LineString", "coordinates": p["coords"]},
            }
            for p in pipes
        ],
    }


def tag_hotspots_with_drainage_capacity(hotspots: list[dict], pipes: list[dict], search_radius_m: float = 150.0) -> None:
    """Mutates each hotspot dict in place, adding "nearby_pipe_capacity_m3s"
    (the largest Manning's-equation capacity among pipes within
    `search_radius_m` of the hotspot, or None if no pipe with a computable
    capacity is that close) and "nearby_pipe_count" (how many pipe segments
    were within range at all, computable capacity or not -- lets a caller
    distinguish "no pipes nearby" from "pipes nearby but not enough data to
    size them"). Straight-line distance in a local equirectangular
    approximation, same simplification report.py's other meter-conversion
    code already uses -- fine at the ~150m scale this is meant to work at."""
    from shapely.geometry import LineString, Point

    meters_per_deg_lat = 111_320.0

    def _to_line(coords):
        return LineString(coords) if len(coords) >= 2 else None

    lines = [(_to_line(p["coords"]), p) for p in pipes if p.get("coords")]
    lines = [(line, p) for line, p in lines if line is not None]

    for h in hotspots:
        lon, lat = h["coords"]
        meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(lat))
        point = Point(lon, lat)
        nearby = []
        for line, p in lines:
            nearest = line.interpolate(line.project(point))
            dist_m = math.hypot((point.x - nearest.x) * meters_per_deg_lon, (point.y - nearest.y) * meters_per_deg_lat)
            if dist_m <= search_radius_m:
                nearby.append(p)
        h["nearby_pipe_count"] = len(nearby)
        capacities = [p["capacity_m3s"] for p in nearby if p.get("capacity_m3s")]
        h["nearby_pipe_capacity_m3s"] = round(max(capacities), 3) if capacities else None
