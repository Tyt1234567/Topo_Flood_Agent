"""
Self-contained HTML flood-risk report generator.

Turns the output of the real-data -> TTK -> flood_tools pipeline into a
single, shareable HTML file: an interactive Leaflet web map (real
OpenStreetMap basemap + a toggleable elevation/risk heatmap layer, loaded
from CDN -- this file is opened directly by the user, not sandboxed the way
a Claude Artifact is, so a live basemap is fine here), the ranked-hotspot
table, a data-driven narrative (every claim is computed from the actual
result, not fixed prose naming one specific place), the LLM-generated
recommendation, and a disclosure of the exact prompt sent to the model.
Used by web_app's one-box "just name a place" flow (app.py's
/api/agent/analyze_stream) so the deliverable at the end of a run is a real
file, not just numbers in a chat.

No web framework dependency -- pure functions over the same objects the
rest of geoai_agent already passes around (ScalarField, the dict shapes
returned by ttk_engine.analyze_scalar_field_ttk and
flood_tools.rank_flood_basins, and boundary.geocode_place_boundary's
shapely polygon / GeoJSON geometry).
"""
from __future__ import annotations

import html
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np

from geoai_agent.flood_tools import WEIGHT_PRESETS
from geoai_agent.morse_topology import ScalarField
from geoai_agent.visualize import _cell_edges

_LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
_LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"

# Hillshade lighting -- a fixed, conventional NW sun angle (matches the
# default most GIS tools use, so this reads as "normal" terrain shading
# rather than an unusual custom look). Exaggeration is high because these
# DEMs are typically near-flat (tens of meters of relief over kilometers of
# extent) -- at vert_exag=1 real subtle terrain barely shows up at all.
_HILLSHADE_AZDEG = 315
_HILLSHADE_ALTDEG = 45
_HILLSHADE_VERT_EXAG = 4.0


def _hillshade_spacing_meters(field: ScalarField) -> tuple[float, float]:
    """Approximate (dx, dy) grid spacing in meters for `field`'s lon/lat
    grid, for LightSource's hillshade math -- doesn't need to be exact,
    only roughly proportional, since it only affects shading strength."""
    ny, nx = field.shape
    lat0 = float(np.mean(field.ys))
    dx_deg = float(np.mean(np.diff(field.xs))) if nx > 1 else 1.0
    dy_deg = float(np.mean(np.diff(field.ys))) if ny > 1 else 1.0
    dx_m = max(abs(dx_deg) * 111_320 * np.cos(np.radians(lat0)), 1.0)
    dy_m = max(abs(dy_deg) * 111_320, 1.0)
    return dx_m, dy_m


def elevation_hillshade_geojson(
    field: ScalarField, catchment_grid: list[list[int]] | None = None, boundary_mask=None,
) -> dict:
    """Elevation as a terrain-colored hillshade GeoJSON layer (one polygon
    per grid cell, as in visualize.field_to_geojson), instead of a
    value-ramp heatmap. A bright-yellow-to-purple viridis ramp over
    elevation reads as a risk/intensity heatmap at a glance -- exactly the
    wrong association for "this is just the terrain's shape" -- so this
    blends relief shading (a fixed sun angle, via LightSource.hillshade)
    with matplotlib's "terrain" colormap (green lowland/brown highland), the
    way GIS basemaps conventionally render elevation. Each feature also
    reports the raw elevation `value` so a caller can build its own
    min/max legend.

    `catchment_grid`, if given (ttk_engine's per-run "descending_manifold"
    grid, same (ny, nx) shape as `field`), is attached to each cell as
    `catchment_id` -- which basin's descending manifold (the set of cells
    that drain into one specific local minimum) that cell belongs to. A
    caller can then highlight one minimum's contributing area on click by
    filtering features to matching catchment_id, without needing a second
    geometry/segmentation pass of its own.

    `boundary_mask`, if given (boundary.grid_inside_polygon_mask(field.xs,
    field.ys, polygon), same (ny, nx) shape), skips a cell entirely when
    False -- so the rendered hillshade stops at the real place boundary's
    own shape instead of filling the whole padded rectangular analysis box
    around it, matching every other boundary-aware layer this project now
    draws (FEMA zones, pipes, gauges are geometrically clipped via
    boundary.clip_geojson_to_polygon; this is the equivalent for a raster-
    shaped layer, which has no vector geometry to clip -- only whole cells
    to omit). None (the default) keeps every cell, unchanged from before
    this parameter existed."""
    ny, nx = field.shape
    dx_m, dy_m = _hillshade_spacing_meters(field)
    ls = LightSource(azdeg=_HILLSHADE_AZDEG, altdeg=_HILLSHADE_ALTDEG)
    elev_min, elev_max = float(field.values.min()), float(field.values.max())
    rgba = ls.shade(field.values, cmap=plt.cm.terrain, blend_mode="soft",
                     vert_exag=_HILLSHADE_VERT_EXAG, dx=dx_m, dy=dy_m,
                     vmin=elev_min, vmax=elev_max)

    x_edges = _cell_edges(list(field.xs))
    y_edges = _cell_edges(list(field.ys))
    features = []
    for i in range(ny):
        for j in range(nx):
            if boundary_mask is not None and not boundary_mask[i][j]:
                continue
            r, g, b = (max(0, min(255, int(round(float(c) * 255)))) for c in rgba[i, j, :3])
            color = f"#{r:02x}{g:02x}{b:02x}"
            ring = [
                [x_edges[j], y_edges[i]], [x_edges[j + 1], y_edges[i]],
                [x_edges[j + 1], y_edges[i + 1]], [x_edges[j], y_edges[i + 1]],
                [x_edges[j], y_edges[i]],
            ]
            properties = {"value": float(field.values[i, j]), "color": color}
            if catchment_grid is not None:
                properties["catchment_id"] = int(catchment_grid[i][j])
            features.append({
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })
    return {"type": "FeatureCollection", "features": features}


def terrain_legend_stops(field: ScalarField, n: int = 5) -> list[dict]:
    """n evenly-spaced (elevation value, hex color) stops spanning `field`'s
    real min/max, colored by the same terrain colormap elevation_hillshade_
    geojson uses (flat -- no hillshade component, since a legend swatch has
    no slope/aspect to shade) -- for building a min-to-max elevation color
    legend without duplicating the colormap choice at the call site."""
    elev_min, elev_max = float(field.values.min()), float(field.values.max())
    stops = []
    for k in range(n):
        t = k / (n - 1) if n > 1 else 0.0
        r, g, b, _ = plt.cm.terrain(t)
        color = f"#{int(round(r*255)):02x}{int(round(g*255)):02x}{int(round(b*255)):02x}"
        stops.append({"value": elev_min + t * (elev_max - elev_min), "color": color})
    return stops


def all_minima_geojson(hotspots: list[dict]) -> dict:
    """Point features for EVERY local minimum (not just the
    persistence-significant ones), colored by drainage demand
    (estimated_peak_runoff_m3s) rather than risk_score -- a separate ramp
    from field_to_geojson/flood_hotspots_to_geojson's so the two layers
    aren't visually confused for the same thing when both are on. Includes
    persistence_m so callers can size markers by persistence, the same
    convention the persistent map (web_app's main.js) uses.

    A hotspot explicitly tagged "within_boundary": False (boundary.
    tag_hotspots_within_boundary -- meaning a real place boundary WAS
    resolved this run, and this specific raw minimum sits outside it) is
    dropped from this MAP layer entirely, so scattered dots don't spill
    past the drawn boundary outline the same way FEMA zones/pipes/gauges
    no longer do (see boundary.clip_geojson_to_polygon). "within_boundary"
    missing/None (no boundary was resolved this run at all) keeps every
    point, unchanged -- this only filters when there's an actual boundary
    to be consistent WITH. The full, unfiltered set (including outside
    points, each still tagged with its own boundary status) stays available
    in the report's own "All local minima, unfiltered" TABLE -- this
    function's output is a map layer, not the report's source of truth for
    that table."""
    hotspots = [h for h in hotspots if h.get("within_boundary") is not False]
    max_runoff = max((h["estimated_peak_runoff_m3s"] for h in hotspots), default=1.0) or 1.0
    features = []
    for h in hotspots:
        x, y = h["coords"]
        t = h["estimated_peak_runoff_m3s"] / max_runoff
        features.append({
            "type": "Feature",
            "properties": {
                "rank": h["rank"],
                "estimated_peak_runoff_m3s": h["estimated_peak_runoff_m3s"],
                "contributing_area_km2": h["contributing_area_km2"],
                "persistence_m": h["persistence_m"],
                "catchment_id": h.get("catchment_id"),
                "within_boundary": h.get("within_boundary"),
                "t": t,
            },
            "geometry": {"type": "Point", "coordinates": [x, y]},
        })
    return {"type": "FeatureCollection", "features": features}


def _webmap_fragment(
    elevation: ScalarField, bounds: tuple[float, float, float, float],
    boundary_geojson: dict | None, all_minima_hotspots: list[dict] | None = None,
    catchment_grid: list[list[int]] | None = None, fema_flood_geojson: dict | None = None,
    pipes_geojson: dict | None = None, gauges_geojson: dict | None = None,
    buildings_geojson: dict | None = None, boundary_mask=None,
    flow_field_geojson_data: dict | None = None,
    exposure_geojson_data: dict | None = None, land_cover_geojson_data: dict | None = None,
    roads_geojson_data: dict | None = None, water_geojson_data: dict | None = None,
    rainfall_soil_geojson_data: dict | None = None, land_cover_source: str | None = None,
) -> str:
    """An embeddable (not standalone) Leaflet map: real OSM basemap, a
    toggleable terrain-colored hillshade elevation layer with a color
    legend, the boundary polygon outline (if resolved), (if given) a
    toggleable layer of every raw local minimum, sized by persistence and
    colored by drainage demand, filterable by a persistence-threshold
    slider and click-to-highlight-catchment (via `catchment_grid`, ttk_
    engine's descending-manifold segmentation for the same run
    `all_minima_hotspots` came from), and (if given) FEMA's own official
    flood-zone polygons (see geoai_agent.fema_flood.fetch_fema_flood_zones)
    colored by severity, a real regulatory layer independent of this run's
    own terrain analysis. `buildings_geojson` (see spatial_analysis.
    buildings_geojson), if given, is drawn ONLY when a catchment is
    highlighted (not as its own always-on/toggleable layer -- a run can
    have tens of thousands of building polygons, most nowhere near any
    hotspot, so drawing all of them unconditionally would be visual noise
    and a real render-performance cost for no benefit) -- its
    "catchment_id" properties must already be in THIS map's own catchment_
    grid numbering (see the caller-side map_catchment_id reconciliation in
    web_app/app.py: a significant hotspot's own catchment_id is scoped to
    ITS OWN, higher-persistence-threshold TTK run, not the lower-threshold
    run `catchment_grid` here comes from -- passing a hotspot's raw
    catchment_id straight through would highlight an unrelated, essentially
    random catchment on this map, since the two runs assign that integer
    independently). There is no separate "significant hotspots"
    marker layer on this map -- that ranking still drives the report's own
    hotspot table elsewhere, but every raw minimum, filtered live by the
    slider, supersedes it as a map layer. CDN-loaded -- fine here since
    this report is a real file the user opens/downloads, unlike a
    sandboxed Claude Artifact.

    `flow_field_geojson_data` (see geoai_agent.flow_field.
    flow_field_geojson), if given, draws a toggleable layer of D8
    downhill-flow arrows -- direction is each sampled cell's own
    steepest-descent neighbor, length is that cell's own accumulated
    drainage area (short where little water passes through, long where a
    lot does), a plain raster-hydrology signal independent of TTK/
    persistence entirely, complementing rather than duplicating the
    terrain-topology basin analysis the rest of this map shows.

    `exposure_geojson_data` (population density, same per-cell heatmap
    shape as elevation), `land_cover_geojson_data` (NLCD classes, see
    geoai_agent.land_cover.land_cover_geojson), `roads_geojson_data`/
    `water_geojson_data` (OSM roads/water, see geoai_agent.spatial_analysis.
    roads_geojson/water_geojson), and `rainfall_soil_geojson_data` (a single
    info marker for the run's design-storm intensity and dominant soil
    group, both of which are one value for the whole run, not spatially
    varying) each add one more toggleable layer so every data source this
    run actually fetched has something to look at on the map -- not just
    the ones a hotspot's own score already surfaces. Every layer here
    starts OFF except the elevation hillshade, which starts on -- a reader
    opens the map and immediately sees terrain, then opts into whichever
    other source they want to inspect, rather than being shown a
    forest-of-tabs of every source at once."""
    lat_min, lat_max, lon_min, lon_max = bounds
    field_geojson = elevation_hillshade_geojson(elevation, catchment_grid=catchment_grid, boundary_mask=boundary_mask)
    legend_stops = terrain_legend_stops(elevation)
    boundary_js = f"var boundaryData = {json.dumps(boundary_geojson)};" if boundary_geojson else "var boundaryData = null;"
    all_minima_js = (
        f"var allMinimaData = {json.dumps(all_minima_geojson(all_minima_hotspots))};"
        if all_minima_hotspots else "var allMinimaData = null;"
    )
    fema_js = (
        f"var femaFloodData = {json.dumps(fema_flood_geojson)};" if fema_flood_geojson else "var femaFloodData = null;"
    )
    pipes_js = (
        f"var pipesData = {json.dumps(pipes_geojson)};" if pipes_geojson else "var pipesData = null;"
    )
    gauges_js = (
        f"var gaugesData = {json.dumps(gauges_geojson)};" if gauges_geojson else "var gaugesData = null;"
    )
    buildings_js = (
        f"var buildingsData = {json.dumps(buildings_geojson)};" if buildings_geojson else "var buildingsData = null;"
    )
    flow_field_js = (
        f"var flowFieldData = {json.dumps(flow_field_geojson_data)};" if flow_field_geojson_data
        else "var flowFieldData = null;"
    )
    exposure_js = (
        f"var exposureData = {json.dumps(exposure_geojson_data)};" if exposure_geojson_data else "var exposureData = null;"
    )
    land_cover_js = (
        f"var landCoverData = {json.dumps(land_cover_geojson_data)};" if land_cover_geojson_data else "var landCoverData = null;"
    )
    roads_js = (
        f"var roadsData = {json.dumps(roads_geojson_data)};" if roads_geojson_data else "var roadsData = null;"
    )
    water_js = (
        f"var waterData = {json.dumps(water_geojson_data)};" if water_geojson_data else "var waterData = null;"
    )
    rainfall_soil_js = (
        f"var rainfallSoilData = {json.dumps(rainfall_soil_geojson_data)};" if rainfall_soil_geojson_data
        else "var rainfallSoilData = null;"
    )
    land_cover_layer_label = _land_cover_source_label(land_cover_source)

    return f"""
    <link rel="stylesheet" href="{_LEAFLET_CSS}" />
    <div id="webmap" style="height:560px;border:1px solid var(--line);border-radius:3px;"></div>
    <script src="{_LEAFLET_JS}"></script>
    <script>
    (function() {{
      var fieldData = {json.dumps(field_geojson)};
      var elevationLegend = {json.dumps(legend_stops)};
      {boundary_js}
      {all_minima_js}
      {fema_js}
      {pipes_js}
      {gauges_js}
      {buildings_js}
      {flow_field_js}
      {exposure_js}
      {land_cover_js}
      {roads_js}
      {water_js}
      {rainfall_soil_js}
      var bounds = [[{lat_min}, {lon_min}], [{lat_max}, {lon_max}]];
      var persistenceThreshold = 0;

      // viridis-ish ramp for the all-minima drainage-demand layer, kept
      // local to this map instead of reusing the Python-side _ramp_color
      // so the layer can be toggled/recolored without a server round trip.
      function runoffColor(t) {{
        var stops = [[0,[68,1,84]],[0.25,[59,82,139]],[0.5,[33,145,140]],[0.75,[94,201,98]],[1,[253,231,37]]];
        for (var i = 0; i < stops.length - 1; i++) {{
          var a = stops[i], b = stops[i+1];
          if (t >= a[0] && t <= b[0]) {{
            var f = (b[0] === a[0]) ? 0 : (t - a[0]) / (b[0] - a[0]);
            var r = Math.round(a[1][0] + f * (b[1][0] - a[1][0]));
            var g = Math.round(a[1][1] + f * (b[1][1] - a[1][1]));
            var bl = Math.round(a[1][2] + f * (b[1][2] - a[1][2]));
            return 'rgb(' + r + ',' + g + ',' + bl + ')';
          }}
        }}
        return 'rgb(253,231,37)';
      }}

      // Marker size = persistence (deeper/more-persistent depressions draw
      // bigger dots), same convention as the persistent map in main.js.
      function maxPersistence(fc) {{
        var m = 0;
        (fc.features || []).forEach(function(f) {{
          var p = f.properties.persistence_m;
          if (p !== null && p !== undefined) m = Math.max(m, p);
        }});
        return m;
      }}
      function persistenceRadius(persistence_m, maxP, minR, maxR) {{
        if (persistence_m === null || persistence_m === undefined) return maxR;
        if (maxP <= 0) return minR;
        var frac = Math.max(0, Math.min(1, persistence_m / maxP));
        return minR + frac * (maxR - minR);
      }}
      var allMinimaMaxP = allMinimaData ? maxPersistence(allMinimaData) : 0;

      // Persistence is heavily right-skewed (a handful of deep basins, a
      // long tail of shallow noise) -- a slider spanning [0, max] linearly
      // spends nearly all of its travel where almost nothing changes.
      // Mapping slider position to a PERCENTILE of the actual values
      // (nearest-rank) instead means every step of the slider always
      // changes the shown set by roughly the same number of points.
      function sortedPersistenceValues(fc) {{
        return (fc.features || [])
          .map(function(f) {{ return f.properties.persistence_m; }})
          .filter(function(p) {{ return p !== null && p !== undefined; }})
          .sort(function(a, b) {{ return a - b; }});
      }}
      function percentileToThreshold(sorted, pct) {{
        if (!sorted.length) return 0;
        var idx = Math.min(sorted.length - 1, Math.floor((pct / 100) * sorted.length));
        return sorted[idx];
      }}

      var map = L.map('webmap').fitBounds(bounds);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '&copy; OpenStreetMap contributors', maxZoom: 19,
      }}).addTo(map);

      var heat = L.geoJSON(fieldData, {{
        style: function(f) {{ return {{ fillColor: f.properties.color, color: f.properties.color, weight: 0, fillOpacity: 0.8 }}; }}
      }}).addTo(map);

      var overlays = {{ 'Elevation (terrain)': heat }};
      if (boundaryData) {{
        overlays['Boundary'] = L.geoJSON(boundaryData, {{ style: {{ fill: false, color: '#a8492c', weight: 2.5 }} }});
      }}
      if (femaFloodData) {{
        // "color" is precomputed server-side (fema_flood.py) from FEMA's own
        // FLD_ZONE/ZONE_SUBTY codes -- "" for minimal/undetermined zones,
        // which cover most of a typical bounding box and aren't worth
        // filling in (just terrain hillshade shows through instead).
        overlays['FEMA flood zones (NFHL)'] = L.geoJSON(femaFloodData, {{
          style: function(f) {{
            var color = f.properties.color;
            return color
              ? {{ fillColor: color, color: color, weight: 0.5, fillOpacity: 0.35 }}
              : {{ fillOpacity: 0, weight: 0 }};
          }},
          onEachFeature: function(f, layer) {{
            var p = f.properties;
            layer.bindPopup('<b>FEMA Zone ' + p.FLD_ZONE + '</b><br>' + p.severity);
          }},
        }});
      }}
      if (pipesData) {{
        // Real stormwater pipes, queried live per request (geoai_agent/
        // drainage_db.py) -- covers a handful of US cities/counties.
        // Colored by whether a Manning's-equation capacity could be
        // computed for that segment (most DC records are missing enough
        // of diameter/material/slope to size it -- see the report's own
        // Spatial analysis section for what that means for interpretation).
        overlays['Stormwater pipes'] = L.geoJSON(pipesData, {{
          style: function(f) {{
            return {{ color: f.properties.capacity_m3s ? '#2b6cb0' : '#8894a0', weight: 2, opacity: 0.7 }};
          }},
          onEachFeature: function(f, layer) {{
            var p = f.properties;
            var capText = p.capacity_m3s
              ? p.capacity_m3s.toFixed(2) + ' m&sup3;/s (Manning\\'s equation estimate)'
              : 'not enough data to estimate (missing diameter/material/slope)';
            layer.bindPopup('<b>Stormwater pipe</b><br>Material: ' + (p.material || 'unknown') +
              '<br>Diameter: ' + (p.diameter_in ? p.diameter_in + ' in' : 'unknown') +
              '<br>Capacity: ' + capText);
          }},
        }});
      }}
      // Pictorial marker for a single-point info layer (as opposed to the
      // all-minima layer's plain circleMarkers, which encode real scalar
      // data via size/color and shouldn't be replaced by an icon). An emoji
      // glyph on a bordered badge reads as its own category at a glance
      // instead of just another colored dot. Kept in sync with the live
      // map's static/main.js emojiIcon().
      function emojiIcon(emoji, borderColor) {{
        return L.divIcon({{
          className: 'emoji-marker-icon',
          html: '<div style="width:24px;height:24px;border-radius:50%;background:#fff;border:2px solid ' + borderColor + ';' +
            'display:flex;align-items:center;justify-content:center;font-size:13px;line-height:1;' +
            'box-shadow:0 1px 3px rgba(0,0,0,0.45);">' + emoji + '</div>',
          iconSize: [24, 24],
          iconAnchor: [12, 12],
          popupAnchor: [0, -12],
        }});
      }}

      if (gaugesData) {{
        // Real, current USGS NWIS river/stream gauge readings (geoai_agent/
        // streamflow.py) -- a fluvial-risk signal independent of this run's
        // own topological/pluvial basin analysis. Most of the country has
        // none nearby; that's the expected common case, not missing data.
        overlays['River/stream gauges (USGS NWIS)'] = L.geoJSON(gaugesData, {{
          pointToLayer: function(f, latlng) {{
            return L.marker(latlng, {{ icon: emojiIcon('🌊', '#12335c') }});
          }},
          onEachFeature: function(f, layer) {{
            var p = f.properties;
            layer.bindPopup('<b>' + (p.site_name || 'USGS gauge') + '</b>' +
              (p.gage_height_ft !== null && p.gage_height_ft !== undefined ? '<br>Gage height: ' + p.gage_height_ft + ' ft' : '') +
              (p.discharge_cfs !== null && p.discharge_cfs !== undefined ? '<br>Discharge: ' + p.discharge_cfs + ' ft&sup3;/s' : ''));
          }},
        }});
      }}

      if (flowFieldData) {{
        // D8 downhill flow-direction arrows (geoai_agent.flow_field) --
        // plain raster hydrology, computed straight from the elevation
        // grid, independent of TTK/persistence. Arrow length already
        // encodes flow_accumulation_km2 in the geometry itself (see
        // flow_field.flow_field_geojson) -- styling here only needs
        // color/opacity, not another size computation.
        overlays['Flow direction (drainage volume)'] = L.geoJSON(flowFieldData, {{
          style: function(f) {{
            var t = f.properties.t;
            return {{ color: '#c0392b', weight: 1 + t * 2.5, opacity: 0.45 + t * 0.5 }};
          }},
          onEachFeature: function(f, layer) {{
            var p = f.properties;
            layer.bindPopup('<b>Flow direction</b><br>Accumulated drainage area: ' +
              p.flow_accumulation_km2.toFixed(3) + ' km&sup2;');
          }},
        }});
      }}

      if (exposureData) {{
        // Population density (US Census block groups, or a user upload --
        // see geoai_agent.real_data.fetch_population_exposure_field), same
        // per-cell heatmap shape as the elevation layer above.
        overlays['Population density (exposure)'] = L.geoJSON(exposureData, {{
          style: function(f) {{ return {{ fillColor: f.properties.color, color: f.properties.color, weight: 0, fillOpacity: 0.75 }}; }},
        }});
      }}

      if (landCoverData) {{
        // Classified land cover (geoai_agent.land_cover) -- categorical,
        // each cell keeps its own class color rather than a continuous ramp.
        overlays['Land cover ({land_cover_layer_label})'] = L.geoJSON(landCoverData, {{
          style: function(f) {{ return {{ fillColor: f.properties.color, color: f.properties.color, weight: 0, fillOpacity: 0.65 }}; }},
          onEachFeature: function(f, layer) {{ layer.bindPopup('<b>' + f.properties.label + '</b>'); }},
        }});
      }}

      if (roadsData) {{
        overlays['Roads (OpenStreetMap)'] = L.geoJSON(roadsData, {{
          style: {{ color: '#555', weight: 1.5, opacity: 0.8 }},
        }});
      }}

      if (waterData) {{
        overlays['Water (OpenStreetMap)'] = L.geoJSON(waterData, {{
          style: {{ color: '#2b6cb0', weight: 1.5, fillColor: '#4a90d9', fillOpacity: 0.35 }},
        }});
      }}

      if (rainfallSoilData) {{
        // Design-storm intensity and dominant hydrologic soil group are
        // each a SINGLE value for the whole run (one storm, one dominant
        // group -- see rainfall.py/soil_data.py), not spatially varying, so
        // this is one info marker rather than a coverage layer. Drawn as a
        // rain-cloud emoji badge rather than a circle -- the all-minima
        // layer below fills its circles with a viridis ramp that runs up to
        // yellow, so a plain colored circleMarker here used to be
        // indistinguishable from a high-drainage-demand minimum (or, with a
        // blue badge, from a gauge) at a glance. Kept in sync with the live
        // map's static/main.js populateMap().
        overlays['Rainfall & soil (NOAA / USDA)'] = L.geoJSON(rainfallSoilData, {{
          pointToLayer: function(f, latlng) {{
            return L.marker(latlng, {{ icon: emojiIcon('🌧️', '#c2185b') }});
          }},
          onEachFeature: function(f, layer) {{
            var p = f.properties;
            var html = '<b>Design storm</b><br>' + p.design_storm_label + ', 1-hour intensity: ' +
              p.design_storm_intensity_mm_per_hr.toFixed(0) + ' mm/hr<br><small>' + p.rainfall_source + '</small>';
            if (p.hydrologic_soil_group) {{
              html += '<hr style="margin:4px 0;"><b>Dominant soil group</b><br>' + p.hydrologic_soil_group +
                '<br><small>' + p.soil_source + '</small>';
            }}
            layer.bindPopup(html);
          }},
        }});
      }}

      var catchmentLayer = null;
      var catchmentBuildingsLayer = null;
      var highlightedCatchmentId = null;
      function highlightCatchment(catchmentId) {{
        if (catchmentLayer) {{ map.removeLayer(catchmentLayer); catchmentLayer = null; }}
        if (catchmentBuildingsLayer) {{ map.removeLayer(catchmentBuildingsLayer); catchmentBuildingsLayer = null; }}
        highlightedCatchmentId = (catchmentId === null || catchmentId === undefined) ? null : catchmentId;
        var clearRow = document.getElementById('webmap-catchment-clear');
        if (clearRow) clearRow.style.display = highlightedCatchmentId === null ? 'none' : 'block';
        if (legendUpdateFn) legendUpdateFn();
        if (catchmentId === null || catchmentId === undefined) return;
        var cells = fieldData.features.filter(function(f) {{ return f.properties.catchment_id === catchmentId; }});
        catchmentLayer = L.geoJSON({{ type: 'FeatureCollection', features: cells }}, {{
          style: {{ fillColor: '#2b6cb0', color: '#12335c', weight: 1, fillOpacity: 0.55 }}, interactive: false,
        }}).addTo(map);
        catchmentLayer.bringToFront();
        // The REAL building footprints counted toward this basin's
        // "buildings_in_catchment"/"building_density_per_km2" figures (see
        // spatial_analysis.compute_building_exposure) -- drawn on top of
        // the catchment fill so the aggregate number has something visible
        // to check it against, not just a number to trust on its own.
        if (buildingsData) {{
          var buildings = buildingsData.features.filter(function(f) {{ return f.properties.catchment_id === catchmentId; }});
          if (buildings.length) {{
            catchmentBuildingsLayer = L.geoJSON({{ type: 'FeatureCollection', features: buildings }}, {{
              style: {{ fillColor: '#a8492c', color: '#5c2414', weight: 1, fillOpacity: 0.75 }}, interactive: false,
            }}).addTo(map);
            catchmentBuildingsLayer.bringToFront();
          }}
        }}
      }}

      var allMinimaLayer = null;
      if (allMinimaData) {{
        allMinimaLayer = L.geoJSON(null, {{
          pointToLayer: function(f, latlng) {{
            var p = f.properties;
            var r = persistenceRadius(p.persistence_m, allMinimaMaxP, 4, 15);
            return L.circleMarker(latlng, {{
              radius: r, weight: p.within_boundary ? 1.5 : 0.5,
              color: p.within_boundary ? '#222' : '#8894',
              fillColor: runoffColor(p.t), fillOpacity: 0.85,
            }});
          }},
          onEachFeature: function(f, layer) {{
            var p = f.properties;
            var boundaryNote = (p.within_boundary === null || p.within_boundary === undefined)
              ? '' : ('<br>' + (p.within_boundary ? 'inside' : 'outside') + ' the resolved boundary');
            layer.bindPopup('<b>Local minimum #' + p.rank + '</b><br>drainage demand: ' +
              p.estimated_peak_runoff_m3s.toFixed(2) + ' m&sup3;/s' +
              '<br>contributing area: ' + p.contributing_area_km2.toFixed(3) + ' km&sup2;' +
              '<br>persistence: ' + (p.persistence_m === null ? 'global sink' : p.persistence_m.toFixed(2) + ' m') + boundaryNote +
              '<br><em>click the dot to highlight its catchment</em>');
            layer.on('click', function() {{ highlightCatchment(p.catchment_id); }});
          }}
        }}).addTo(map);
        overlays['All local minima (drainage demand)'] = allMinimaLayer;
      }}

      function rebuildAllMinimaLayer() {{
        if (!allMinimaLayer) return;
        allMinimaLayer.clearLayers();
        var shown = allMinimaData.features.filter(function(f) {{
          return (f.properties.persistence_m === null || f.properties.persistence_m === undefined) || f.properties.persistence_m >= persistenceThreshold;
        }});
        allMinimaLayer.addData({{ type: 'FeatureCollection', features: shown }});
        var el = document.getElementById('webmap-minima-count');
        if (el) el.textContent = 'Showing ' + shown.length + ' of ' + allMinimaData.features.length;
      }}

      L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);

      // Rebuilt from scratch every time a layer is checked/unchecked (see
      // the overlayadd/overlayremove handlers below), so it only ever shows
      // entries for what's actually visible right now -- with this many
      // optional layers, a legend listing every one of them regardless of
      // what's on screen would be longer than the map itself.
      //
      // Leaflet 1.9.4's Map.hasLayer is `stamp(layer) in this._layers` with
      // NO null/undefined guard -- stamp() does `'_leaflet_id' in obj`,
      // which throws "Cannot use 'in' operator to search for '_leaflet_id'
      // in undefined" if called on a key `overlays` never got (most data
      // sources are absent on any given run) -- always check truthiness
      // first.
      function layerIsVisible(layer) {{ return !!layer && map.hasLayer(layer); }}
      function legendContentHtml() {{
        var gradientCss = elevationLegend.map(function(s, i) {{ return s.color + ' ' + (i / (elevationLegend.length - 1) * 100) + '%'; }}).join(', ');
        var gradientHtml = (elevationLegend.length && layerIsVisible(overlays['Elevation (terrain)'])) ?
          '<div style="margin-top:6px;"><strong>Elevation</strong></div>' +
          '<div style="width:160px;height:10px;border-radius:2px;margin:3px 0;background:linear-gradient(90deg, ' + gradientCss + ');"></div>' +
          '<div style="display:flex;justify-content:space-between;font-size:10.5px;"><span>' +
          elevationLegend[0].value.toFixed(0) + ' m</span><span>' + elevationLegend[elevationLegend.length - 1].value.toFixed(0) + ' m</span></div>' : '';
        var allMinimaOn = layerIsVisible(overlays['All local minima (drainage demand)']);
        var runoffGradientCss = [[0,[68,1,84]],[0.25,[59,82,139]],[0.5,[33,145,140]],[0.75,[94,201,98]],[1,[253,231,37]]]
          .map(function(s) {{ return 'rgb(' + s[1][0] + ',' + s[1][1] + ',' + s[1][2] + ') ' + (s[0] * 100) + '%'; }}).join(', ');
        return '<div style="margin-bottom:6px;"><strong>Toggle layers above</strong> -- the legend below only shows what\\'s currently checked.</div>' +
          gradientHtml +
          (allMinimaOn ? '<div style="margin-top:6px;"><strong>Dot size</strong> = persistence (bigger = deeper basin)</div>' : '') +
          (allMinimaOn ? '<div><strong>Dot fill</strong> = drainage demand</div>' : '') +
          (allMinimaOn ? '<div style="width:160px;height:8px;border-radius:2px;margin:3px 0;background:linear-gradient(90deg, ' + runoffGradientCss + ');"></div>' +
            '<div style="display:flex;justify-content:space-between;font-size:10.5px;"><span>low</span><span>high</span></div>' : '') +
          (allMinimaOn ? '<div>Click a dot to highlight its catchment area</div>' : '') +
          (layerIsVisible(overlays['FEMA flood zones (NFHL)']) ?
            '<div style="margin-top:6px;"><strong>FEMA flood zones</strong></div>' +
            '<div><span style="display:inline-block;width:10px;height:10px;background:#c0392b;border-radius:2px;"></span> Special Flood Hazard Area</div>' +
            '<div><span style="display:inline-block;width:10px;height:10px;background:#e8ab5c;border-radius:2px;"></span> 0.2% annual chance ("500-year")</div>'
            : '') +
          (layerIsVisible(overlays['Stormwater pipes']) ?
            '<div style="margin-top:6px;"><strong>Stormwater pipes</strong></div>' +
            '<div><span style="display:inline-block;width:14px;height:2px;background:#2b6cb0;"></span> capacity computed</div>' +
            '<div><span style="display:inline-block;width:14px;height:2px;background:#8894a0;"></span> not enough data</div>'
            : '') +
          (layerIsVisible(overlays['River/stream gauges (USGS NWIS)']) ?
            '<div style="margin-top:6px;"><strong>River/stream gauges</strong> (USGS NWIS, live)</div>' +
            '<div><span style="display:inline-block;width:16px;height:16px;border-radius:50%;background:#fff;border:1.5px solid #12335c;text-align:center;line-height:14px;font-size:9px;margin-right:3px;">🌊</span>active gauge -- click for details</div>'
            : '') +
          (buildingsData && highlightedCatchmentId !== null ?
            '<div style="margin-top:6px;"><span style="display:inline-block;width:10px;height:10px;background:#a8492c;border:1px solid #5c2414;"></span> buildings counted in the highlighted catchment (OpenStreetMap)</div>'
            : '') +
          (layerIsVisible(overlays['Flow direction (drainage volume)']) ?
            '<div style="margin-top:6px;"><strong>Flow direction</strong> (D8, computed from elevation)</div>' +
            '<div>Arrow length = accumulated drainage area -- short where little water passes through, long where a lot does</div>'
            : '') +
          (layerIsVisible(overlays['Population density (exposure)']) ?
            '<div style="margin-top:6px;"><strong>Population density</strong> (US Census block groups, live)</div>'
            : '') +
          (layerIsVisible(overlays['Land cover ({land_cover_layer_label})']) ?
            '<div style="margin-top:6px;"><strong>Land cover</strong> ({land_cover_layer_label}, categorical -- click a cell for its class)</div>'
            : '') +
          (layerIsVisible(overlays['Roads (OpenStreetMap)']) || layerIsVisible(overlays['Water (OpenStreetMap)']) ?
            '<div style="margin-top:6px;"><strong>Roads / water</strong> (OpenStreetMap)</div>'
            : '') +
          (layerIsVisible(overlays['Rainfall & soil (NOAA / USDA)']) ?
            '<div style="margin-top:6px;"><strong>Rainfall &amp; soil</strong> (NOAA Atlas 14 / USDA SSURGO)</div>' +
            '<div><span style="display:inline-block;width:16px;height:16px;border-radius:50%;background:#fff;border:1.5px solid #c2185b;text-align:center;line-height:14px;font-size:9px;margin-right:3px;">🌧️</span>click for design-storm / soil details</div>'
            : '') +
          '<div id="webmap-catchment-clear" style="display:' + (highlightedCatchmentId === null ? 'none' : 'block') + ';margin-top:6px;"><a href="#" id="webmap-catchment-clear-link">Clear catchment highlight</a></div>';
      }}

      function attachLegendClearLink() {{
        var link = document.getElementById('webmap-catchment-clear-link');
        if (link) link.addEventListener('click', function(e) {{ e.preventDefault(); highlightCatchment(null); }});
      }}

      var legend = L.control({{position: 'bottomleft'}});
      legend.onAdd = function() {{
        var div = L.DomUtil.create('div');
        div.style.cssText = 'background:var(--paper-raised, #fff);color:var(--ink,#222);' +
          'padding:8px 10px;font-size:12px;line-height:1.5;border-radius:4px;' +
          'box-shadow:0 1px 4px rgba(0,0,0,0.35);max-width:220px;';
        div.innerHTML = legendContentHtml();
        return div;
      }};
      legend.addTo(map);
      L.DomEvent.disableClickPropagation(legend.getContainer());
      attachLegendClearLink();

      var legendUpdateFn = function() {{
        var container = legend.getContainer();
        if (!container) return;
        container.innerHTML = legendContentHtml();
        attachLegendClearLink();
      }};
      map.on('overlayadd', legendUpdateFn);
      map.on('overlayremove', legendUpdateFn);

      if (allMinimaData && allMinimaData.features.length) {{
        var sortedP = sortedPersistenceValues(allMinimaData);
        var slider = L.control({{position: 'topleft'}});
        slider.onAdd = function() {{
          var div = L.DomUtil.create('div');
          div.style.cssText = 'background:var(--paper-raised, #fff);color:var(--ink,#222);' +
            'padding:8px 10px;font-size:12px;line-height:1.5;border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,0.35);';
          div.innerHTML = '<div><strong>Persistence threshold</strong>: <span id="webmap-threshold-value">0.0</span> m ' +
            '(<span id="webmap-threshold-percentile">0</span>th pct.)</div>' +
            '<input type="range" id="webmap-persistence-slider" min="0" max="100" step="1" value="0" style="width:180px;display:block;margin:4px 0;">' +
            '<div id="webmap-minima-count" style="font-size:10.5px;"></div>';
          return div;
        }};
        slider.addTo(map);
        L.DomEvent.disableClickPropagation(slider.getContainer());
        var sliderEl = document.getElementById('webmap-persistence-slider');
        sliderEl.addEventListener('input', function() {{
          var pct = parseFloat(sliderEl.value);
          persistenceThreshold = percentileToThreshold(sortedP, pct);
          document.getElementById('webmap-threshold-value').textContent = persistenceThreshold.toFixed(1);
          document.getElementById('webmap-threshold-percentile').textContent = pct.toFixed(0);
          rebuildAllMinimaLayer();
        }});
        rebuildAllMinimaLayer();
      }}

      // Called from this same document's report text (see report.py's
      // _point_link) when the user clicks a point mentioned in a table or
      // callout, instead of only being able to click the dot on the map.
      window.focusMapPoint = function(lat, lon, catchmentId) {{
        map.setView([lat, lon], Math.max(map.getZoom(), 15));
        highlightCatchment(catchmentId === undefined ? null : catchmentId);
      }};
    }})();
    </script>
    """


def _md_to_html(text: str) -> str:
    """Converts the small markdown subset FLOOD_RECOMMENDATION_SYSTEM's
    output actually uses (###/#### headers, * bullets, **bold**, --- rules)
    into HTML -- not a general markdown parser."""
    lines = text.split("\n")
    out: list[str] = []
    in_list = False
    for raw_line in lines:
        line = raw_line.rstrip()
        if line.strip() == "---":
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append("<hr>")
            continue
        if line.startswith("#### "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h4>{html.escape(line[5:].strip())}</h4>")
            continue
        if line.startswith("### "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h3>{html.escape(line[4:].strip())}</h3>")
            continue
        if line.startswith("*   ") or line.startswith("* "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            item = line.split(" ", 1)[1].strip() if line.startswith("* ") else line[4:].strip()
            out.append(f"<li>{html.escape(item)}</li>")
            continue
        if line.strip() == "":
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        out.append("</ul>")
    joined = "\n".join(out)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", joined)


def _boundary_hotspots(hotspots: list[dict], grid_shape: tuple[int, int]) -> list[dict]:
    """Hotspots sitting on the analysis grid's outer edge -- a real signal
    (usually a channel/watershed exiting the box) but one worth flagging:
    the basin's true shape is truncated by an arbitrary bounding box, not
    fully captured. Computed from grid_index, not guessed."""
    ny, nx = grid_shape
    edge = []
    for h in hotspots:
        row, col = h["grid_index"]
        if row in (0, ny - 1) or col in (0, nx - 1):
            edge.append(h)
    return edge


def _point_link(
    lat: float, lon: float, catchment_id: int | None, label: str, embed_interactive_map: bool,
    hotspot_rank: int | None = None,
) -> str:
    """A clickable mention of a specific point in the report's prose/tables
    -- clicking it pans the map to that lat/lon and, if the point has a
    catchment_id, highlights its contributing area, same as clicking the
    dot on the map itself. `embed_interactive_map` picks which map to talk
    to: True means this document has its own embedded Leaflet map (the
    standalone downloadable report -- see _webmap_fragment's
    window.focusMapPoint), False means this fragment is loaded inside
    web_app's report iframe and the real map lives in the parent page
    (main.js's focusMapPoint); the iframe is same-origin (srcdoc) so
    calling straight into `parent` works without postMessage.

    `hotspot_rank`, if given, ALSO jumps the "Ranked hotspots" carousel
    (see the hotspot_cards_html/window.showHotspotRank in
    render_flood_report_html) to that basin's card -- unlike the map, the
    carousel lives in THIS document regardless of embed_interactive_map (a
    parent-page point-link mention still lives inside a report fragment
    that has its own copy of the carousel), so this call is never
    `parent.`-prefixed. Guarded by a typeof check since showHotspotRank
    only exists when this document actually rendered a hotspot carousel
    (skipped when there are zero significant hotspots)."""
    target = "" if embed_interactive_map else "parent."
    cid = "null" if catchment_id is None else str(int(catchment_id))
    jump_call = (
        f" if (typeof showHotspotRank === 'function') showHotspotRank({int(hotspot_rank)});"
        if hotspot_rank is not None else ""
    )
    return (
        f'<a href="javascript:void(0)" class="point-link" '
        f'onclick="{target}focusMapPoint({lat:.6f}, {lon:.6f}, {cid});{jump_call} return false;">{label}</a>'
    )


def _fema_zone_cell(hotspot: dict) -> str:
    """A `<td>` for the FEMA-zone table column -- a colored badge for the
    raw FLD_ZONE code (see fema_flood.tag_hotspots_with_flood_zone), tiered
    by the same high/moderate/minimal severity language the narrative and
    LLM prompts use, or a plain dash if this point fell outside every zone
    polygon FEMA returned for the area (not the same as "no FEMA data at
    all" -- that's handled by the caller only adding this column when
    fema_flood_geojson was available in the first place)."""
    zone = hotspot.get("fema_flood_zone")
    severity = hotspot.get("fema_flood_severity") or ""
    if zone is None:
        return '<td class="mono">&ndash;</td>'
    if severity.startswith("high"):
        tier = "fema-high"
    elif severity.startswith("moderate"):
        tier = "fema-moderate"
    else:
        tier = "fema-minimal"
    return f'<td><span class="fema-zone {tier}" title="{html.escape(severity)}">{html.escape(zone)}</span></td>'


def _drainage_capacity_flag(hotspot: dict, drainage_pipe_data_available: bool) -> tuple[str, str] | None:
    """The concrete "connect the dots" fix this report was missing: this
    basin's own ESTIMATED PEAK RUNOFF (estimated_peak_runoff_m3s, a Rational-
    Method screening-level indicator, NOT a required pipe size -- see
    "estimated peak runoff" instead of "needs X capacity" below) explicitly
    compared against whatever real stormwater pipe CAPACITY was actually
    found nearby (drainage_db.py's 150m search radius), instead of leaving a
    reader to notice a high runoff number in one table and a pipe-capacity
    number in a different table and do the comparison themselves. Returns
    (severity, message_html) -- severity is "warn" (a real, stated gap or
    shortfall) or "ok" (nearby capacity covers the estimate) -- or None when
    `drainage_pipe_data_available` is false for the whole run (nothing
    area-specific to say; the caller states that absence ONCE for the whole
    report, not per-basin, see render_flood_report_html).

    Deliberately says "estimated peak runoff" and "a single pipe's recorded
    capacity" rather than "needs X drainage capacity" -- the Rational Method
    number is a screening-level runoff ESTIMATE (see _pipeline_narrative_html's
    Rational Method caveat), and in reality a channel, detention pond,
    overland flow path, and infiltration can all share the job of carrying
    it, not just the one nearest pipe this check happens to compare against."""
    if not drainage_pipe_data_available:
        return None
    demand = hotspot["estimated_peak_runoff_m3s"]
    capacity = hotspot.get("nearby_pipe_capacity_m3s")
    count = hotspot.get("nearby_pipe_count") or 0
    if capacity is None:
        if not count:
            return (
                "warn",
                f"Estimated peak runoff <strong>{demand:.1f} m&sup3;/s</strong>, but no stormwater pipes "
                "were found within 150m of this basin to compare it against.",
            )
        return (
            "warn",
            f"Estimated peak runoff <strong>{demand:.1f} m&sup3;/s</strong> -- {count} nearby pipe"
            f"{'s' if count != 1 else ''} found, but none had enough recorded data (diameter/material/slope) "
            "to confirm whether they can convey it.",
        )
    if capacity < demand:
        shortfall = demand - capacity
        return (
            "warn",
            f"Estimated peak runoff <strong>{demand:.1f} m&sup3;/s</strong> exceeds the nearest pipe's known "
            f"capacity (<strong>{capacity:.1f} m&sup3;/s</strong>) by {shortfall:.1f} m&sup3;/s -- a real "
            "gap against THAT pipe specifically, not necessarily against the site's total drainage capacity "
            "(channels, detention, overland flow, and infiltration aren't counted here).",
        )
    return (
        "ok",
        f"The nearest stormwater pipe's confirmed capacity (<strong>{capacity:.1f} m&sup3;/s</strong>) meets or "
        f"exceeds this basin's {demand:.1f} m&sup3;/s estimated peak runoff.",
    )


def _fema_corroboration_flag(hotspot: dict, fema_available: bool) -> tuple[str, str] | None:
    """Whether FEMA's own official National Flood Hazard Layer -- a real
    regulatory data source, not derived from this run's own terrain
    analysis -- agrees with, or is silent on, this specific basin. Returns
    (severity, message_html) or None when there's nothing basin-specific to
    say: `fema_available` is false for the whole run (stated once, not
    per-basin, same as the drainage-capacity flag above), or this point
    fell outside every zone polygon FEMA returned (a real, common outcome,
    not worth a flag line of its own -- the FEMA Zone context is still
    visible in this basin's stat grid as a plain dash)."""
    if not fema_available:
        return None
    zone = hotspot.get("fema_flood_zone")
    severity = hotspot.get("fema_flood_severity") or ""
    if zone is None:
        return None
    if severity.startswith("high"):
        return (
            "ok",
            f"Falls inside FEMA's Special Flood Hazard Area (zone <strong>{html.escape(zone)}</strong>) -- "
            "independent federal regulatory confirmation, not just this run's own terrain-derived inference.",
        )
    if severity.startswith("moderate"):
        return (
            "info",
            f"FEMA maps this as a moderate risk zone (<strong>{html.escape(zone)}</strong>, the 0.2%-annual-"
            "chance \"500-year\" tier).",
        )
    return (
        "info",
        "Outside FEMA's mapped floodplain -- a localized, terrain-driven risk FEMA's regulatory maps don't "
        "capture, not a sign the topological finding above is wrong.",
    )


def _elevation_source_description(elevation_source: str) -> str:
    """Plain-language description of where 'elevation' data actually came
    from, for the pipeline narrative below -- same wording web_app/app.py's
    own _describe_elevation_source uses for the LLM recommendation context,
    duplicated here rather than imported (web_app is this package's
    CALLER, importing back from it would be a layering inversion)."""
    if elevation_source == "usgs_dem_cog":
        return ("the USGS 3DEP 1/3 arc-second (~10m) DEM, streamed directly from its source GeoTIFF and "
                 "clipped to this exact area via GDAL (not pre-downloaded, not point-sampled)")
    if elevation_source == "usgs":
        return "USGS's 3DEP Elevation Point Query Service (genuinely LiDAR-derived where covered)"
    if elevation_source == "user_upload":
        return "a user-uploaded file (not live-fetched, not cross-checked against USGS/Open-Meteo)"
    return "the live Open-Meteo API (global coverage, ~90m resolution)"


def _land_cover_source_label(land_cover_source: str | None) -> str:
    """Plain-language name for whichever land-cover source actually backed
    this run's runoff-coefficient calculation -- "NLCD" (US+territories
    only) or, outside NLCD's coverage, land_cover.fetch_global_landcover_
    grid's global Sentinel-2 10m classifier. Every report string that used
    to hardcode "NLCD" now goes through this instead, so a non-US run
    doesn't claim a US-only federal data source it never actually used."""
    if land_cover_source == "global_sentinel2":
        return "global Sentinel-2 10m land cover"
    return "NLCD"


def _capitalize_first(s: str) -> str:
    """Uppercases only the first character, leaving everything else exactly
    as-is -- NOT the same as str.capitalize(), which also force-lowercases
    every OTHER character in the string. Needed because the pipeline-
    narrative sentences below join several already-correctly-cased
    fragments (real proper nouns/acronyms like "FEMA", "USGS", "NWIS",
    "OpenStreetMap", "US Census" appear mid-sentence, not just at the
    start) -- str.capitalize() would mangle "FEMA's..." into "Fema's..."
    and "...OpenStreetMap building..." into "...openstreetmap building..."
    wherever they land after the first character (this was observed
    directly in this report's own output before this helper existed)."""
    return s[:1].upper() + s[1:] if s else s


def _mini_leaflet_map(map_id: str, height_px: int, bounds: tuple[float, float, float, float], layers_js: str) -> str:
    """A minimal, self-contained embedded Leaflet map fragment: an OSM
    basemap plus whatever `layers_js` builds and .addTo(map)'s directly --
    no layer control, no slider, not interactive beyond marker popups.
    Deliberately much simpler than _webmap_fragment (this project's main,
    full-featured interactive map): these small maps illustrate ONE step
    of the pipeline narrative each, and must stay cheap to embed -- see
    _pipeline_narrative_html's docstring for which real data sources were
    measured too large to duplicate here, and what was used instead.
    scrollWheelZoom is off so scrolling the page past a small embedded map
    doesn't accidentally zoom it."""
    lat_min, lat_max, lon_min, lon_max = bounds
    return f"""
    <div id="{map_id}" style="height:{height_px}px;border:1px solid var(--line);border-radius:3px;"></div>
    <script>
    (function() {{
      var map = L.map('{map_id}', {{scrollWheelZoom: false}}).fitBounds([[{lat_min}, {lon_min}], [{lat_max}, {lon_max}]]);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '&copy; OpenStreetMap contributors', maxZoom: 19,
      }}).addTo(map);
      {layers_js}
    }})();
    </script>"""


def _pipeline_narrative_html(
    *,
    place_label: str,
    bounds: tuple[float, float, float, float],
    elevation: ScalarField,
    elevation_source: str,
    morse_result: dict,
    flood_result: dict,
    boundary_available: bool,
    boundary_display_name: str | None,
    boundary_geojson: dict | None,
    fema_flood_geojson: dict | None,
    svi_geojson: dict | None,
    gauges_geojson: dict | None,
    drainage_pipe_data_available: bool,
    river_gauge_data_available: bool,
    osm_infrastructure_data_available: bool,
    land_cover_data_available: bool,
    land_cover_source: str | None,
    soil_data_available: bool,
    hydrologic_group_source: str | None,
    exposure_source: str | None,
    expected_annual_runoff: dict | None,
    model_name: str,
) -> str:
    """A numbered, plain-language walkthrough of every real pipeline stage
    THIS run actually went through -- what data backed it, and the real
    numbers that came out -- illustrated with 2-3 small embedded Leaflet
    maps. Replaces "How this was produced" being just a raw LLM-prompt
    dump with an actual narrated account a non-technical reviewer can
    follow; the raw prompt is kept too (see render_flood_report_html),
    just demoted to a supporting appendix rather than being the only thing
    shown.

    The mini-maps deliberately do NOT re-embed every data source's full
    payload a second time (it's already in the main "Interactive map"
    section below when embed_interactive_map=True, and re-embedding it
    here would bloat every report regardless of which report view is
    open): measured directly on a real College Park, MD run in this
    project's own development, `fema_flood_geojson` (65 polygons) was
    ~870KB, `buildings_geojson` (8,923 polygons -- every building counted
    toward ANY significant basin's exposure figures, not an unbounded raw
    OSM dump) was ~2.8MB, and `pipes_geojson` (5,026 segments) was
    ~1.0MB -- all too large to duplicate cheaply. `boundary_geojson`
    (~7KB) and the ranked hotspots themselves (a handful of points) were
    genuinely small, so the maps below are built from THOSE instead: basin
    markers carry their own FEMA zone / building density / nearest-gauge
    properties directly (already on each hotspot dict from earlier in the
    pipeline), giving the same cross-check story at a fraction of the
    size, at the cost of showing per-basin summaries rather than full
    zone/building polygon shapes -- the main interactive map below is
    where a reader goes to see those shapes directly."""
    hotspots = flood_result["hotspots"]
    scenario = flood_result["rainfall_scenario"]
    rainfall_source = flood_result.get("rainfall_source")
    weights = flood_result["weights"]

    steps: list[tuple[str, str]] = []

    if boundary_available:
        loc_text = (
            f"The study area is <strong>{html.escape(boundary_display_name or place_label)}</strong>'s real "
            "administrative boundary, resolved from OpenStreetMap (Nominatim) -- not just a padded box around "
            "a single geocoded point."
        )
    else:
        loc_text = (
            f"No administrative boundary polygon was available for <strong>{html.escape(place_label)}</strong>, "
            "so the study area is a padded box around its geocoded center point instead."
        )
    steps.append(("Location & study area", loc_text))

    ny, nx = elevation.shape
    elev_min, elev_max = float(elevation.values.min()), float(elevation.values.max())
    steps.append((
        "Terrain elevation",
        f"Elevation came from {_elevation_source_description(elevation_source)}, delivered as a "
        f"{nx}&times;{ny} grid spanning {elev_min:.1f}&ndash;{elev_max:.1f}m.",
    ))

    if rainfall_source == "noaa_atlas14_live":
        rain_text = (
            f"The design storm ({scenario['label']}, <strong>1-hour</strong> intensity "
            f"{scenario['intensity_mm_per_hr']:.0f} mm/hr) is a genuine, site-specific NOAA Atlas 14 "
            "precipitation-frequency estimate (specifically its 60-minute-duration depth, chosen to "
            "approximate typical urban catchment response time -- NOAA Atlas 14 also publishes 5-min through "
            "24-hour durations for the same return period, which would give different intensities), fetched "
            "live for this exact location -- real federal engineering-standard data, not a regional "
            "approximation."
        )
    else:
        rain_text = (
            f"The design storm ({scenario['label']}, <strong>1-hour</strong> intensity "
            f"{scenario['intensity_mm_per_hr']:.0f} mm/hr) used the illustrative Mid-Atlantic fallback table, "
            "since a live, site-specific NOAA Atlas 14 estimate wasn't available for this location."
        )
    steps.append(("Design storm (precipitation frequency)", rain_text))

    lc_label = _land_cover_source_label(land_cover_source)
    if land_cover_data_available and soil_data_available:
        soil_clause = (
            "USDA SSURGO soil data" if hydrologic_group_source == "measured"
            else "an assumed moderate-infiltration soil group (\"B\") -- no site-specific soil survey exists "
                 "outside the US, so this is a documented engineering default, not a measurement"
        )
        lc_text = (
            f"Each basin's own runoff coefficient was derived from real {lc_label} land cover and {soil_clause} "
            "across its own catchment, via the actual SCS Curve Number runoff equation at this storm's rainfall "
            "depth -- not one constant assumed for the whole area (see each hotspot's \"Runoff coeff.\" column "
            "below)."
        )
    else:
        lc_text = (
            f"{lc_label} land cover / USDA SSURGO soil data wasn't available for this run, so every basin fell "
            f"back to the same flat, assumed runoff coefficient ({flood_result.get('runoff_coefficient', 0.5):.2f})."
        )
    steps.append(("Land cover & soil", lc_text))

    steps.append((
        "Topological analysis (discrete Morse theory)",
        f"TTK's Morse-Smale complex found <strong>{morse_result['num_significant_basins']}</strong> topologically "
        f"significant basins out of <strong>{morse_result['num_raw_minima']}</strong> raw local minima, at a "
        f"persistence-simplification threshold of {morse_result['persistence_threshold']:.2f}m -- no heuristic "
        "depression-filling or breaching preprocessing, just the elevation field's own topology.",
    ))

    weight_text = ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in weights.items())
    steps.append((
        "Flood risk ranking",
        f"Each basin's risk score weighs {weight_text}, and its drainage demand comes from the Rational Method "
        "using that basin's own runoff coefficient above.",
    ))

    checks = []
    if fema_flood_geojson:
        checks.append(f"FEMA's National Flood Hazard Layer returned <strong>{len(fema_flood_geojson['features'])}</strong> zone polygons for this area")
    else:
        checks.append("FEMA's National Flood Hazard Layer had no data for this area")
    if drainage_pipe_data_available:
        checks.append("real stormwater pipe data was found in the local database")
    else:
        checks.append("no real stormwater pipe data is loaded for this area yet (currently DC-only)")
    if river_gauge_data_available:
        n_gauges = len(gauges_geojson["features"]) if gauges_geojson else 0
        checks.append(f"USGS NWIS returned {n_gauges} active river/stream gauge{'s' if n_gauges != 1 else ''} nearby")
    else:
        checks.append("no active USGS river/stream gauges were found nearby (the common case -- gauge coverage is sparse nationally, not a sign this area went unchecked)")
    steps.append(("Real-world cross-checks", _capitalize_first("; ".join(checks)) + "."))

    exp_parts = []
    if exposure_source == "census_live":
        exp_parts.append("population exposure came from a live US Census block-group density fetch")
    elif exposure_source == "worldpop_global":
        exp_parts.append("population exposure came from WorldPop's global 100m density layer (US Census unavailable here)")
    elif exposure_source == "user_upload":
        exp_parts.append("population exposure came from a user-uploaded file")
    else:
        exp_parts.append("no population/exposure data was available, so that risk factor used a neutral default")
    if osm_infrastructure_data_available:
        exp_parts.append("OpenStreetMap building footprints were aggregated per basin into a building-count/density signal")
    if svi_geojson:
        exp_parts.append(
            f"each basin's census tract was matched against CDC/ATSDR's Social Vulnerability Index "
            f"({len(svi_geojson['features'])} tracts returned for this area) -- a published federal composite of "
            "poverty, unemployment, income, and education, among other factors, supplementary context alongside "
            "population density, not itself one of the weighted risk-score factors"
        )
    else:
        exp_parts.append("CDC/ATSDR Social Vulnerability Index data was not available for this area")
    steps.append(("Exposure: population, buildings & social vulnerability", _capitalize_first("; ".join(exp_parts)) + "."))

    if expected_annual_runoff and expected_annual_runoff.get("basins"):
        steps.append((
            "Multi-return-period aggregation",
            "Beyond the single design storm above, every basin was also re-ranked six times (once per NOAA "
            "return period, 2yr&ndash;500yr) and probability-weighted into an aggregate \"return-period-weighted "
            "runoff index\" -- see the table further down this report.",
        ))

    steps.append((
        "Recommendation generation",
        f"The narrative recommendation below was generated by <strong>{html.escape(model_name)}</strong>, given "
        "the full ranked result above as structured context -- the exact prompt sent is in the appendix at the "
        "end of this section.",
    ))

    steps_html = "".join(
        f'<div class="pipeline-narrative-step"><h4>{i}. {html.escape(title)}</h4><p>{text}</p></div>'
        for i, (title, text) in enumerate(steps, start=1)
    )

    # -- Map 1: study area + ranked hotspots (boundary + a handful of ------
    # points -- genuinely cheap, see the docstring above for real payload
    # sizes that ruled out re-embedding FEMA/buildings/pipes wholesale).
    hotspot_features = []
    vmax = max((h["risk_score"] for h in hotspots), default=1.0) or 1.0
    for h in hotspots:
        t = h["risk_score"] / vmax
        hotspot_features.append({
            "type": "Feature",
            "properties": {"rank": h["rank"], "risk_score": h["risk_score"], "t": t},
            "geometry": {"type": "Point", "coordinates": h["coords"]},
        })
    hotspots_geojson_small = {"type": "FeatureCollection", "features": hotspot_features}
    map1_layers_js = f"""
      var hotspotData = {json.dumps(hotspots_geojson_small)};
      {"var boundaryData = " + json.dumps(boundary_geojson) + ";" if boundary_geojson else ""}
      {"L.geoJSON(boundaryData, {style: {fill: false, color: '#a8492c', weight: 2.5}}).addTo(map);" if boundary_geojson else ""}
      L.geoJSON(hotspotData, {{
        pointToLayer: function(f, latlng) {{
          return L.circleMarker(latlng, {{radius: 8, weight: 1.5, color: '#222', fillColor: '#c0392b', fillOpacity: 0.3 + 0.6 * f.properties.t}});
        }},
        onEachFeature: function(f, layer) {{
          layer.bindPopup('<b>Rank ' + f.properties.rank + '</b><br>priority score: ' + f.properties.risk_score.toFixed(3));
        }},
      }}).addTo(map);
    """
    map1_html = f"""
    <div class="mini-map-block">
      <h4>Study area &amp; ranked hotspots</h4>
      {_mini_leaflet_map("pipeline-map-1", 260, bounds, map1_layers_js)}
    </div>"""

    # -- Map 2: real-world cross-check summary, per basin (FEMA zone / ------
    # building density / nearest gauge, all already on each hotspot dict --
    # no FEMA/building polygon geometry re-embedded, just the per-basin
    # SUMMARY of what those checks found; the shapes themselves are in the
    # main interactive map below).
    cross_features = []
    for h in hotspots:
        severity = h.get("fema_flood_severity") or ""
        cross_features.append({
            "type": "Feature",
            "properties": {
                "rank": h["rank"],
                "fema_zone": h.get("fema_flood_zone"),
                "fema_high": severity.startswith("high"),
                "building_density": h.get("building_density_per_km2"),
                "gauge_name": (h.get("nearest_gauge") or {}).get("site_name"),
            },
            "geometry": {"type": "Point", "coordinates": h["coords"]},
        })
    cross_geojson = {"type": "FeatureCollection", "features": cross_features}
    map2_layers_js = f"""
      var crossData = {json.dumps(cross_geojson)};
      L.geoJSON(crossData, {{
        pointToLayer: function(f, latlng) {{
          return L.circleMarker(latlng, {{radius: 8, weight: 1.5, color: '#222', fillColor: f.properties.fema_high ? '#c0392b' : '#5c8a70', fillOpacity: 0.75}});
        }},
        onEachFeature: function(f, layer) {{
          var p = f.properties;
          var lines = ['<b>Rank ' + p.rank + '</b>'];
          lines.push(p.fema_zone ? 'FEMA zone: ' + p.fema_zone : 'FEMA: not mapped');
          if (p.building_density !== null && p.building_density !== undefined) lines.push('Buildings: ' + p.building_density.toFixed(0) + '/km&sup2;');
          if (p.gauge_name) lines.push('Near gauge: ' + p.gauge_name);
          layer.bindPopup(lines.join('<br>'));
        }},
      }}).addTo(map);
    """
    map2_html = f"""
    <div class="mini-map-block">
      <h4>Regulatory &amp; exposure cross-check, per basin</h4>
      {_mini_leaflet_map("pipeline-map-2", 260, bounds, map2_layers_js)}
    </div>"""

    maps = [map1_html, map2_html]

    # -- Map 3 (conditional): river/stream gauges -- only when this run ----
    # actually found any; gauges_geojson is always small (a sparse national
    # network, never more than a handful of points per bounding box) so
    # there's no size concern re-embedding it here, unlike FEMA/buildings.
    if gauges_geojson and gauges_geojson.get("features"):
        map3_layers_js = f"""
      var gaugesData = {json.dumps(gauges_geojson)};
      L.geoJSON(gaugesData, {{
        pointToLayer: function(f, latlng) {{ return L.circleMarker(latlng, {{radius: 7, weight: 1.5, color: '#12335c', fillColor: '#4a90d9', fillOpacity: 0.9}}); }},
        onEachFeature: function(f, layer) {{
          var p = f.properties;
          layer.bindPopup('<b>' + (p.site_name || 'USGS gauge') + '</b>' +
            (p.gage_height_ft !== null && p.gage_height_ft !== undefined ? '<br>Gage height: ' + p.gage_height_ft + ' ft' : '') +
            (p.discharge_cfs !== null && p.discharge_cfs !== undefined ? '<br>Discharge: ' + p.discharge_cfs + ' ft&sup3;/s' : ''));
        }},
      }}).addTo(map);
        """
        maps.append(f"""
    <div class="mini-map-block">
      <h4>Active river/stream gauges (USGS NWIS)</h4>
      {_mini_leaflet_map("pipeline-map-3", 260, bounds, map3_layers_js)}
    </div>""")

    return f"""
    <div class="pipeline-narrative">{steps_html}</div>
    <link rel="stylesheet" href="{_LEAFLET_CSS}" />
    <script src="{_LEAFLET_JS}"></script>
    <div class="mini-maps">{"".join(maps)}</div>"""


def _data_sources_table_html(
    *,
    elevation_source: str,
    boundary_available: bool,
    rainfall_source_label: str,
    soil_data_available: bool,
    hydrologic_group_source: str | None,
    land_cover_data_available: bool,
    land_cover_source: str | None,
    exposure_source: str | None,
    osm_infrastructure_data_available: bool,
    fema_available: bool,
    river_gauge_data_available: bool,
    drainage_pipe_data_available: bool,
) -> str:
    """A scannable "what did this run actually use, and what is it FOR"
    table -- one row per real data source this project can draw on, every
    one of which also has a matching toggleable layer on the interactive
    map above (see _webmap_fragment) so a reader can go look at the raw
    data behind any row, not just take the summary on faith. Complements
    _pipeline_narrative_html's prose walkthrough (which explains the same
    facts woven into a stage-by-stage story) with a flat, at-a-glance
    reference -- the two aren't redundant, they're read differently."""
    exposure_status = {
        "census_live": "used (US Census, live)", "worldpop_global": "used (WorldPop global, live)",
        "user_upload": "used (user-uploaded)",
    }.get(exposure_source, "not available this run")
    rows = [
        ("Elevation (terrain)", _elevation_source_description(elevation_source),
         "Basin topology (TTK Morse-Smale complex), slope, D8 flow direction/accumulation -- everything else in this analysis is built on top of this one layer.", "used"),
        ("Boundary", "OpenStreetMap (Nominatim) administrative polygon lookup",
         "Clips the analysis to the real place shape (not just a padded box), so a basin can't be found outside the actual area.",
         "used" if boundary_available else "not available (padded box used instead)"),
        ("Rainfall (design storm)", "NOAA Atlas 14 precipitation-frequency estimates, or an illustrative fallback table",
         "Sets the storm intensity fed into the Rational Method drainage-demand calculation and the risk score's rainfall factor.", rainfall_source_label),
        ("Soil (hydrologic group)", "USDA SSURGO, via Soil Data Access",
         "Infiltration classification (A-D) -- one input to the SCS Curve Number runoff coefficient, alongside land cover.",
         "used" if hydrologic_group_source == "measured" else
         "assumed default (\"B\") -- outside US soil survey coverage" if hydrologic_group_source == "assumed_default" else
         "not available this run"),
        ("Land cover", "USGS/MRLC NLCD classified land cover" if land_cover_source != "global_sentinel2"
         else "Esri/Impact Observatory global Sentinel-2 10m land cover (NLCD unavailable here)",
         "The other input to the SCS Curve Number runoff coefficient -- how much of a basin's catchment is impervious vs. vegetated.",
         "used" if land_cover_data_available else "not available this run"),
        ("Population density (exposure)", "US Census block-group density (2020), or a user-uploaded file",
         "Who's near a basin's own pour point -- the risk score's exposure factor.", exposure_status),
        ("Buildings", "OpenStreetMap building footprints",
         "Counted per basin's catchment for a building-count/density figure, and used for nearest-building distance.",
         "used" if osm_infrastructure_data_available else "not available this run"),
        ("Roads &amp; water", "OpenStreetMap ways",
         "Nearest-road/nearest-water distance for each hotspot (spatial_analysis.compute_terrain_metrics).",
         "used" if osm_infrastructure_data_available else "not available this run"),
        ("FEMA flood zones", "FEMA National Flood Hazard Layer (NFHL), official federal data",
         "An independent regulatory cross-check against this run's own terrain-derived ranking -- not an input to the risk score.",
         "used" if fema_available else "not available for this area"),
        ("River/stream gauges", "USGS NWIS, real-time readings",
         "A fluvial-risk signal (current gauge height/discharge) near a basin -- context, not a score input; most areas have none nearby.",
         "used" if river_gauge_data_available else "no active gauges nearby"),
        ("Stormwater pipes", "Local database, currently DC-only (see load_dc_stormwater.py)",
         "Manning's-equation pipe capacity near a basin -- whether real infrastructure exists to drain it.",
         "used" if drainage_pipe_data_available else "not available for this area"),
        ("Flow direction (D8)", "Computed directly from the elevation grid above -- not a separate fetch",
         "Shows which way water actually flows and how much accumulates at each point, on the map's arrow layer.", "computed"),
    ]
    rows_html = "".join(
        f"<tr><td><strong>{name}</strong></td><td>{source}</td><td>{purpose}</td><td>{status}</td></tr>"
        for name, source, purpose, status in rows
    )
    return f"""
    <div class="table-wrap">
    <table class="data-sources-table">
      <thead><tr><th>Source</th><th>What it is</th><th>Used for</th><th>This run</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>"""


def _limitations_html(
    *,
    hotspots: list[dict],
    elevation_source: str,
    boundary_available: bool,
    exposure_source: str | None,
    osm_infrastructure_data_available: bool,
    fema_available: bool,
) -> str:
    """A concrete, non-boilerplate limitations section -- each item names
    the SPECIFIC mechanism by which a figure elsewhere in this report can
    be wrong, using this run's own numbers where possible, rather than a
    generic "results may vary" disclaimer. Every point here traces to a
    real methodological caveat raised during this project's own design
    review, not a hypothetical checklist item."""
    top = hotspots[0] if hotspots else None
    items: list[tuple[str, str]] = []

    items.append((
        "The global sink is a property of this analysis's boundary, not necessarily of the real landscape",
        "A basin with no deeper point to compare against (persistence_m is null, topological score "
        "automatically 1.0) is the lowest point <em>inside this run's own analysis grid and boundary</em> -- "
        "not provably the lowest point of any real, larger drainage system. Cells outside a resolved boundary "
        "are raised to an artificial \"wall\" before topology is computed (see \"The terrain, and where it "
        "drains\" above), which can only ever push the apparent global sink further inward, never reveal a "
        "true one that happens to sit just past the boundary. If the real drainage path continues beyond the "
        "analyzed area, this run cannot see that.",
    ))

    if top is not None:
        area = top["contributing_area_km2"]
        rational_note = (
            f"This run's Rank 1 basin has a contributing area of <strong>{area:.2f} km&sup2;</strong>. "
            if area >= 2.0 else
            f"This run's Rank 1 basin has a contributing area of {area:.2f} km&sup2;, small enough that the "
            "Rational Method's assumptions are closer to holding. "
        )
        items.append((
            "The Rational Method (Q = CIA) is a screening-level indicator here, not a design discharge",
            rational_note + "The Rational Method was developed for, and is normally applied to, relatively "
            "small catchments with a well-defined time of concentration close to the storm's own duration -- "
            "it does not model travel time, temporal rainfall distribution, or basin response time, all of "
            "which matter more as contributing area grows. Every \"estimated peak runoff\" / \"drainage "
            "demand\" figure in this report should be read as a screening-level indicator for prioritizing "
            "which basin to look at first, not a design discharge -- a basin-scale hydrologic model and a "
            "proper time-of-concentration assessment would be needed before sizing any actual infrastructure "
            "against these numbers.",
        ))

    dem_note = (
        "Elevation came from the USGS 3DEP ~10m DEM" if elevation_source in ("usgs_dem_cog", "usgs")
        else ("Elevation came from a user-uploaded file, not independently verified against USGS/Open-Meteo"
              if elevation_source == "user_upload"
              else "Elevation came from Open-Meteo's global ~90m-resolution source")
    )
    items.append((
        "DEM resolution and accuracy set a hard floor on what this analysis can resolve",
        f"{dem_note}. A depression narrower than the grid spacing, a culvert or storm drain that moves water "
        "underground (invisible to any surface DEM), or urban microtopography (curbs, driveways, parking-lot "
        "grading) below the DEM's resolution can all be missed entirely or represented incorrectly -- Morse-"
        "theoretic persistence simplification finds real structure IN the DEM it's given, but cannot recover "
        "detail the DEM never captured in the first place.",
    ))

    if boundary_available:
        items.append((
            "Boundary clipping can change which basins exist, not just which ones are shown",
            "Raising every cell outside the resolved boundary to an artificial \"wall\" (a standard watershed-"
            "delineation technique) prevents a basin from being detected outside the real place -- but a basin "
            "whose true low point or true contributing area extends across that boundary line is still "
            "affected: its catchment can be truncated, or a channel that would otherwise continue and merge "
            "with a larger system instead appears to terminate at the wall. Basins on the outer edge of the "
            "grid are flagged individually above where this applies.",
        ))

    if exposure_source in ("census_live", "worldpop_global", "user_upload"):
        density_aggregation_note = (
            "Census block-group density is itself a spatial average over the whole block group, not a "
            "fine-grained population surface -- treating it as a point value at one grid cell inherits that "
            "aggregation's own uncertainty." if exposure_source == "census_live" else
            "WorldPop's density estimate is itself a modeled 100m-cell surface (not a direct census count), "
            "built from national census/survey data disaggregated by satellite-derived settlement patterns -- "
            "a real, worldwide estimate, but a model output, not a direct measurement the way a US Census "
            "block-group tally is." if exposure_source == "worldpop_global" else ""
        )
        items.append((
            "Exposure is measured at one grid cell" + (
                ", and Census density is a block-group average" if exposure_source == "census_live" else ""
            ),
            "The exposure factor samples population/density data at a basin's own pour point ONLY -- a single "
            "grid cell, not an area total (see the exposure formula in the details above) -- so it can "
            "understate a basin whose deepest point happens to sit on a park, parking lot, or stream corridor "
            "even while the surrounding catchment is densely populated (the \"Buildings\" figure is a separate, "
            f"catchment-wide count precisely because of this gap). {density_aggregation_note} Neither measures "
            "actual flood exposure (population or structures physically within a modeled inundation extent) -- "
            "this analysis has no inundation model to intersect with.",
        ))

    if osm_infrastructure_data_available:
        items.append((
            "OpenStreetMap coverage is real but uneven",
            "Building, road, and water-feature counts/distances come from volunteer-mapped OpenStreetMap data. "
            "Coverage completeness varies by area -- a dense, well-mapped city and a sparsely-mapped rural "
            "area can show very different building counts for reasons that have nothing to do with actual "
            "flood exposure, purely how thoroughly that area has been mapped.",
        ))

    if fema_available:
        items.append((
            "FEMA's flood zones are a regulatory floodplain map, not a complete flood-risk assessment",
            "FEMA's National Flood Hazard Layer maps the 1%- and 0.2%-annual-chance floodplains for regulatory "
            "(mostly mortgage-insurance) purposes, based on its own separate hydrologic/hydraulic modeling -- "
            "agreement or disagreement with this run's own terrain-derived ranking is a useful cross-check, "
            "but neither one validates the other, and FEMA maps are not updated frequently enough to reflect "
            "every recent land-use or drainage-infrastructure change.",
        ))

    return "".join(
        f'<div class="pipeline-narrative-step"><h4>{title}</h4><p>{text}</p></div>' for title, text in items
    )


def _agent_decision_log_html(
    *,
    strategy_decision: dict | None,
    threshold_decision: dict | None,
    weight_sensitivity: dict | None,
) -> str:
    """The three real, consequential decision points in geoai_agent.
    agent_decisions -- rendered so a reader can see not just the final
    numbers but WHY this specific run ended up with them: which weighting
    strategy the agent picked and why, which persistence threshold it
    chose out of the real candidates this terrain actually produced and
    why, and whether the #1-ranked basin survives a change of weighting
    scheme at all. Each of these had a real, checkable effect on this run's
    own output -- this section exists so that effect is visible, not
    buried in a JSON blob only the final LLM call ever saw."""
    if strategy_decision is None and threshold_decision is None and weight_sensitivity is None:
        return (
            '<p class="section-note">This run used this project\'s fixed defaults (entropy weighting, a '
            "5%-of-elevation-range persistence threshold) -- the agent's threshold-decision step was "
            "unavailable this run (see the pipeline log below for why). Give an objective in the intro form "
            "(e.g. \"protecting people's safety\" or \"critical infrastructure\") so the agent's threshold "
            "selection has something to reason about beyond the raw candidate numbers.</p>"
        )

    cards = []

    if strategy_decision is not None:
        preset = strategy_decision["weight_preset"]
        preset_weights = WEIGHT_PRESETS.get(preset, {})
        weights_text = ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in preset_weights.items())
        cards.append(f"""
        <div class="agent-decision-card">
          <div class="agent-decision-eyebrow">Agent decision -- analysis strategy</div>
          <h4>Chose the "{html.escape(preset)}" weighting preset</h4>
          <p><strong>Stated objective:</strong> {html.escape(strategy_decision.get("objective_text") or "(none)")}</p>
          <p><strong>Reasoning:</strong> {html.escape(strategy_decision["reasoning"])}</p>
          <p><strong>Resulting weights:</strong> {weights_text}</p>
        </div>""")

    if threshold_decision is not None:
        chosen = threshold_decision["chosen_threshold"]
        rows = "".join(
            f'<tr class="{"agent-chosen-row" if abs(c["threshold"] - chosen) < 1e-6 else ""}">'
            f'<td>{c["threshold"]:.3f} m</td><td>{html.escape(c["label"])}</td>'
            f'<td class="num">{c["num_significant_basins"]}</td>'
            f'<td>{"&larr; chosen" if abs(c["threshold"] - chosen) < 1e-6 else ""}</td></tr>'
            for c in threshold_decision["candidates"]
        )
        actual = threshold_decision.get("actual_num_significant_basins")
        chosen_estimate = next(
            (c["num_significant_basins"] for c in threshold_decision["candidates"] if abs(c["threshold"] - chosen) < 1e-6),
            None,
        )
        reconciliation_note = ""
        if actual is not None and chosen_estimate is not None and actual != chosen_estimate:
            reconciliation_note = (
                f'<p class="section-note">Note: the table below\'s basin count for the chosen threshold is an '
                f"ESTIMATE from a separate scan (see geoai_agent.agent_decisions.build_persistence_candidates's "
                f"docstring for why) -- the actual run at {chosen:.3f}m produced <strong>{actual}</strong> "
                f"significant basins, not the estimated {chosen_estimate}. TTK's own persistence-simplification "
                f"is threshold-dependent, so a small discrepancy like this is a real, expected numerical "
                f"artifact of running TTK twice at different thresholds, not a data error.</p>"
            )
        cards.append(f"""
        <div class="agent-decision-card">
          <div class="agent-decision-eyebrow">Agent decision -- persistence threshold</div>
          <h4>Chose {chosen:.3f} m out of {len(threshold_decision["candidates"])} real candidates computed from this terrain{f" (actually produced {actual} significant basins)" if actual is not None else ""}</h4>
          <p><strong>Reasoning:</strong> {html.escape(threshold_decision["reasoning"])}</p>
          {reconciliation_note}
          <div class="table-wrap"><table>
            <thead><tr><th>Threshold</th><th>Basis</th><th>Basins at this threshold (estimate)</th><th></th></tr></thead>
            <tbody>{rows}</tbody>
          </table></div>
        </div>""")

    if weight_sensitivity is not None:
        stable = weight_sensitivity["rank1_stable_across_schemes"]
        rows = "".join(
            f'<tr><td class="mono">{html.escape(scheme)}</td>'
            f'<td class="mono">{v["top_1_coords"][1]:.4f}&deg;N, {v["top_1_coords"][0]:.4f}&deg;E</td></tr>'
            for scheme, v in weight_sensitivity["per_scheme"].items()
        )
        verdict = (
            "The same basin ranks #1 under every scheme tested -- this finding does not depend on which "
            "prioritization philosophy was used."
            if stable else
            "Different weighting schemes disagree on which basin ranks #1 -- treat the top pick as sensitive "
            "to which prioritization philosophy is used, not a settled finding independent of it."
        )
        cards.append(f"""
        <div class="agent-decision-card">
          <div class="agent-decision-eyebrow">Agent decision -- uncertainty check</div>
          <h4>{"Top-ranked basin is stable across weighting schemes" if stable else "Top-ranked basin CHANGES depending on weighting scheme"}</h4>
          <p>{verdict}</p>
          <div class="table-wrap"><table>
            <thead><tr><th>Weighting scheme</th><th>#1-ranked basin location</th></tr></thead>
            <tbody>{rows}</tbody>
          </table></div>
        </div>""")

    return "".join(cards)


def _morse_vs_d8_html(morse_vs_d8: dict | None) -> str:
    """The central evidence-chain question this report's own methodology
    review kept landing on: is a Morse-persistence-significant basin
    actually a hydrologically meaningful drainage feature, or just a
    topologically robust artifact with no independent correspondence to how
    water actually flows? Presents geoai_agent.morse_validation.
    compare_morse_to_d8's real, numeric cross-check against an independent
    D8 flow model instead of asserting an answer -- see that module's
    docstring for exactly what each number does and doesn't establish."""
    if morse_vs_d8 is None:
        return (
            '<p class="section-note">This cross-check wasn\'t computed for this run (see the pipeline log for '
            "why). Without it, treat \"significant basin\" as a claim about this run's Morse-Smale topology "
            "only, not independently corroborated against a second hydrology model.</p>"
        )
    n_sig = morse_vs_d8["num_significant_morse_basins"]
    n_raw = morse_vs_d8["num_raw_morse_minima"]
    n_d8 = morse_vs_d8["num_d8_sinks"]
    agreement = morse_vs_d8["d8_sink_agreement_fraction"]
    correlation = morse_vs_d8["contributing_area_correlation"]

    noise_reduction_note = (
        f"D8 flow direction alone -- a completely independent, per-pixel method with no notion of "
        f"'significance' -- finds <strong>{n_d8}</strong> raw local sinks in this same area (any cell with no "
        f"downhill neighbor). Morse theory's own raw critical-point detection finds a comparable "
        f"<strong>{n_raw}</strong> raw minima before any filtering -- the two independent methods agree on "
        f"roughly how many local depressions this terrain actually has. The difference is what happens next: "
        f"D8 has no principled way to separate a real depression from a single noisy pixel (a practical D8 "
        f"pipeline bolts on an arbitrary fixed-area or pit-filling threshold to do that); Morse persistence "
        f"gives each one a continuous significance value derived from the scalar field's own topology, letting "
        f"a defensible threshold (see \"How the agent worked\" above) reduce {n_raw} raw candidates to "
        f"<strong>{n_sig}</strong> actually-analyzed basins -- this is the concrete methodological gain over "
        f"plain D8, and it's a real number, not a claim taken on faith."
    )

    if agreement is None:
        sink_check_note = "No significant basins this run to check."
    else:
        sink_check_note = (
            f"<strong>{agreement:.0%}</strong> of this run's Morse-significant basins have a pour point that is "
            f"ALSO a local sink under the independently-computed D8 model. This is close to a tautology by "
            f"construction (a true local minimum has no downhill neighbor under either method's own comparison "
            f"rule) -- so it functions as a correctness check, not a discovery: a value noticeably below 100% "
            f"would mean the two methods disagree about basic local-minimum detection somewhere (e.g. a flat "
            f"region handled differently), which would be worth investigating rather than assuming away."
        )

    if correlation is None:
        correlation_note = "Too few significant basins this run to compute a meaningful correlation."
    else:
        strength = (
            "a strong" if abs(correlation) >= 0.7 else "a moderate" if abs(correlation) >= 0.4 else "a weak"
        )
        correlation_note = (
            f"Across this run's significant basins, Morse's own contributing_area_km2 (from the descending-"
            f"manifold catchment segmentation) correlates with D8's flow accumulation at the same pour point "
            f"(an entirely separately-computed measure of upstream drainage area) at <strong>r = "
            f"{correlation:.2f}</strong> -- {strength} correlation. This is real, checkable evidence for "
            f"whether Morse's notion of \"how much area drains into this basin\" tracks the same physical "
            f"quantity D8 accumulation measures, reported honestly whichever way it comes out -- a low "
            f"correlation would be an important limitation to know about, not something to hide."
        )

    return f"""
    <div class="agent-decision-card">
      <div class="agent-decision-eyebrow">Evidence-chain check -- Morse vs. D8</div>
      <h4>Sink agreement &amp; noise reduction</h4>
      <p>{sink_check_note}</p>
      <p>{noise_reduction_note}</p>
    </div>
    <div class="agent-decision-card">
      <div class="agent-decision-eyebrow">Evidence-chain check -- Morse vs. D8</div>
      <h4>Contributing-area correlation</h4>
      <p>{correlation_note}</p>
    </div>"""


def _reflection_html(reflection: dict | None) -> str:
    """Renders the Reflection-pattern step (geoai_agent.agent_decisions.
    reflect_on_recommendation): execute (write the recommendation above) ->
    critique (a second LLM call re-checks it against the same evidence) ->
    refine (only shown here as "revised" when the reviewer both found a
    real issue and returned a usable correction). Shown directly under the
    recommendation it reviewed so a reader can see the self-check happened
    and what it did or didn't find, not just the final polished text with
    no visible scrutiny behind it."""
    if reflection is None:
        return (
            '<p class="section-note">Self-review of this recommendation wasn\'t run for this run (see the '
            "pipeline log below for why) -- treat the recommendation above as unreviewed.</p>"
        )
    # Three distinct outcomes, not two: issues_found (the model's own
    # verdict) and revision_applied (whether a usable rewrite actually came
    # back) can disagree -- a model can flag a real issue but return an
    # empty/malformed revision. Collapsing these into one field used to
    # print "found no unsupported claims" directly above a critique
    # describing a real problem (a real bug caught by review).
    if reflection["revision_applied"]:
        heading, verb = "Revised after self-review", "What it found and fixed"
    elif reflection["issues_found"]:
        heading, verb = "Self-review flagged an issue, but no usable revision was returned", "What it found (kept the original text)"
    else:
        heading, verb = "Self-review found no unsupported claims", "What it verified"
    return f"""
    <div class="agent-decision-card">
      <div class="agent-decision-eyebrow">Agent decision -- self-review (reflection)</div>
      <h4>{heading}</h4>
      <p>A second LLM call re-read the same evidence bundle the recommendation above was written from, checking
        specifically for claims the evidence doesn't support -- not a general rewrite.</p>
      <p><strong>{verb}:</strong> {html.escape(reflection["critique"])}</p>
    </div>"""


def _narrative_html(
    hotspots: list[dict], grid_shape: tuple[int, int], exposure_provided: bool,
    boundary_available: bool, fema_available: bool, embed_interactive_map: bool,
) -> str:
    """Data-driven analysis paragraphs -- every claim traces back to a
    computed value (risk_score, contributing_area_km2, grid_index,
    within_boundary, ...), so this reads correctly for any location
    instead of describing one specific watershed by name."""
    if not hotspots:
        return (
            "<p>No basin in this area survived topological-persistence noise filtering -- either the "
            "terrain here is essentially flat/featureless at this resolution, or genuine depressions are "
            "smaller than the analysis grid can resolve. No flood hotspots to rank.</p>"
        )

    top = hotspots[0]
    lat, lon = top["coords"][1], top["coords"][0]
    # NOT top.get("catchment_id") -- that's scoped to this basin's OWN,
    # higher-persistence-threshold TTK run, a different, unrelated numbering
    # from the lower-threshold run the map's catchment_grid comes from (see
    # web_app/app.py's map_catchment_id reconciliation and _webmap_fragment's
    # docstring). Falls back to no highlight (None) rather than a wrong one
    # if that reconciliation wasn't available this run.
    coord_link = _point_link(
        lat, lon, top.get("map_catchment_id"), f"{lat:.4f}°N, {lon:.4f}°E", embed_interactive_map,
        hotspot_rank=top.get("rank"),
    )
    parts = [
        f"<p><strong>Rank 1</strong> ({coord_link}) is the highest-priority basin: "
        f"priority score <strong>{top['risk_score']:.3f}</strong>, contributing area "
        f"<strong>{top['contributing_area_km2']:.2f} km²</strong>, estimated peak runoff "
        f"<strong>{top['estimated_peak_runoff_m3s']:.1f} m³/s</strong> under the design storm used below."
    ]
    if top["persistence_m"] is None:
        parts.append(" It is the single deepest depression <em>within this analysis's own boundary and grid</em> "
                      "(the global sink -- no deeper point exists inside the analyzed area to compare it against, "
                      "so its topological factor score is automatically the maximum, 1.0). That is a statement "
                      "about this run's terrain data, not a guarantee it is the worst real flood risk in the "
                      "wider area -- see this basin's own contributing-area factor score below to check whether "
                      "its high overall score is driven mainly by depth, by catchment size, or genuinely both "
                      "(see Limitations &amp; uncertainty on how boundary clipping can affect which point ends up "
                      "as the global sink).")
    elif top["persistence_m"] >= 0.05 * (max(h["elevation_m"] for h in hotspots) - min(h["elevation_m"] for h in hotspots) or 1):
        parts.append(f" It holds {top['persistence_m']:.1f}m of relief before spilling over -- a real, well-defined depression, not a shallow dip.")
    if len(hotspots) > 1:
        second = hotspots[1]
        ratio = top["estimated_peak_runoff_m3s"] / max(second["estimated_peak_runoff_m3s"], 1e-6)
        parts.append(f" Its drainage demand is {ratio:.1f}× the next-ranked basin.")
    if boundary_available and top.get("within_boundary") is False:
        parts.append(" Note this basin's coordinate falls just outside the resolved place boundary -- it is "
                      "still real terrain draining toward this area, but is technically adjacent to, not inside, "
                      "the named place.")
    if fema_available and top.get("fema_flood_zone") and (top.get("fema_flood_severity") or "").startswith("high"):
        parts.append(f" This point also falls inside FEMA's own mapped Special Flood Hazard Area "
                      f"(zone {html.escape(top['fema_flood_zone'])}) -- independent regulatory confirmation, not "
                      "just this run's own terrain-derived inference.")
    lead = "".join(parts) + "</p>"

    out = [f'<div class="callout"><h3>Top priority</h3>{lead}</div>']

    def _rank_links(hs: list[dict]) -> str:
        """Every basin reference elsewhere in this report is a clickable
        coordinate link (see _point_link) that pans the map AND jumps the
        "Ranked hotspots" carousel to that basin's own card -- these rank
        mentions ("#3, #5") shouldn't be the one place left as plain text
        a reader can't act on, especially now that the carousel shows only
        one card at a time (see hotspot_cards_html in
        render_flood_report_html) instead of every card stacked and
        scrollable."""
        return ", ".join(
            _point_link(
                h["coords"][1], h["coords"][0], h.get("map_catchment_id"), f"#{h['rank']}",
                embed_interactive_map, hotspot_rank=h["rank"],
            )
            for h in hs
        )

    edge = _boundary_hotspots(hotspots, grid_shape)
    if edge:
        ranks = _rank_links(edge)
        out.append(
            f'<div class="caveat"><strong>Read this before acting on it:</strong> basin{"es" if len(edge) > 1 else ""} '
            f"{ranks} sit{'s' if len(edge) == 1 else ''} on the outer edge of the analyzed bounding box, not an interior "
            "low point. That's consistent with a real channel or watershed draining out of this box, not necessarily "
            "noise -- but it also means its true shape and contributing area are truncated by an arbitrary boundary, "
            "not fully captured. If this basin is a priority, re-run with the bounding box extended past that edge "
            "to see whether it stays a distinct feature or merges into a longer drainage path.</div>"
        )

    if boundary_available:
        outside = [h for h in hotspots if h.get("within_boundary") is False]
        if outside:
            ranks = _rank_links(outside)
            out.append(
                f'<p class="section-note">Basin{"es" if len(outside) > 1 else ""} {ranks} '
                f"{'fall' if len(outside) > 1 else 'falls'} outside the resolved place boundary shown on the map "
                "below (shaded/outlined area) -- included because the analysis grid is a padded rectangle around "
                "the boundary, not the polygon itself, but worth knowing before prioritizing them as \"this place's\" problem.</p>"
            )

    if fema_available:
        sfha = [h for h in hotspots if (h.get("fema_flood_severity") or "").startswith("high")]
        if sfha:
            ranks = _rank_links(sfha)
            out.append(
                f'<p class="section-note">{len(sfha)} of {len(hotspots)} significant basin'
                f"{'s' if len(hotspots) != 1 else ''} ({ranks}) {'fall' if len(sfha) > 1 else 'falls'} inside FEMA's "
                "own mapped Special Flood Hazard Area (the regulatory 1%-annual-chance floodplain or worse) -- "
                "see the FEMA Zone column in the table below.</p>"
            )
        else:
            out.append(
                '<p class="section-note">None of the significant basins fall inside FEMA\'s mapped Special Flood '
                "Hazard Area, per its National Flood Hazard Layer for this area -- these are ponding/drainage "
                "risks this run's own terrain analysis identified, not locations FEMA has separately flagged.</p>"
            )
    else:
        out.append(
            '<p class="section-note">FEMA flood hazard data (National Flood Hazard Layer) was not available for '
            "this area -- every ranking above is based solely on this run's own terrain analysis, not "
            "cross-checked against FEMA's official flood mapping.</p>"
        )

    if not exposure_provided:
        out.append(
            '<p class="section-note">No population/exposure data was supplied for this run, so every ranking above '
            "is purely hydrological and terrain-based -- it does not account for how many people or structures are "
            "actually near each point.</p>"
        )

    return "\n".join(out)


_CSS = """
:root {
  --paper: #eef1ea; --paper-raised: #ffffff; --ink: #1c2620; --ink-muted: #5c6b5f;
  --line: #d6dccb; --water: #2f6f8a; --water-deep: #1e4b60; --water-tint: #e4eef1;
  --hazard: #a8492c; --hazard-bg: #f6e4d9; --hazard-2: #c07a2e; --hazard-2-bg: #f7ecd9;
  --mono: ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  --serif: Georgia, "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
  --sans: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
:root[data-theme="dark"] {
  --paper: #171d19; --paper-raised: #202822; --ink: #e7ece5; --ink-muted: #97a498;
  --line: #34402f; --water: #7fb8cf; --water-deep: #a9d4e3; --water-tint: #1c2c30;
  --hazard: #e08a68; --hazard-bg: #3a2419; --hazard-2: #dcae5f; --hazard-2-bg: #362c17;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #171d19; --paper-raised: #202822; --ink: #e7ece5; --ink-muted: #97a498;
    --line: #34402f; --water: #7fb8cf; --water-deep: #a9d4e3; --water-tint: #1c2c30;
    --hazard: #e08a68; --hazard-bg: #3a2419; --hazard-2: #dcae5f; --hazard-2-bg: #362c17;
  }
}
:root[data-theme="light"] {
  --paper: #eef1ea; --paper-raised: #ffffff; --ink: #1c2620; --ink-muted: #5c6b5f;
  --line: #d6dccb; --water: #2f6f8a; --water-deep: #1e4b60; --water-tint: #e4eef1;
  --hazard: #a8492c; --hazard-bg: #f6e4d9; --hazard-2: #c07a2e; --hazard-2-bg: #f7ecd9;
}
* { box-sizing: border-box; }
body { background: var(--paper); color: var(--ink); font-family: var(--sans); font-size: 16px; line-height: 1.6; margin: 0; padding: 0 20px 80px; }
.page { max-width: 860px; margin: 0 auto; }
.masthead { padding: 48px 0 24px; border-bottom: 3px solid var(--ink); }
.eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--water-deep); margin: 0 0 10px; }
h1 { font-family: var(--serif); font-size: clamp(30px, 5vw, 42px); font-weight: 700; margin: 0 0 6px; text-wrap: balance; letter-spacing: -0.01em; }
.subtitle { font-family: var(--serif); font-style: italic; font-size: 18px; color: var(--ink-muted); margin: 0 0 16px; }
.meta-line { font-family: var(--mono); font-size: 12.5px; color: var(--ink-muted); display: flex; flex-wrap: wrap; gap: 4px 16px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); margin: 24px 0 8px; }
.stat { background: var(--paper-raised); padding: 14px 16px; display: flex; flex-direction: column; gap: 4px; }
.stat-label { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-muted); }
.stat-value { font-size: 15px; font-weight: 600; }
.report-toc {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px;
  padding: 12px 16px; margin: 20px 0; border: 1px solid var(--line); border-radius: 6px;
  background: var(--paper-raised); position: sticky; top: 0; z-index: 10;
}
.report-toc-label { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-muted); margin-right: 4px; }
.report-toc a {
  font-size: 12.5px; color: var(--ink); text-decoration: none; padding: 3px 9px;
  border-radius: 12px; background: var(--paper); border: 1px solid var(--line); white-space: nowrap;
}
.report-toc a:hover { border-color: var(--water-deep); color: var(--water-deep); }
section { margin: 44px 0; scroll-margin-top: 12px; }
h2 { font-family: var(--serif); font-size: 23px; font-weight: 700; margin: 0 0 6px; text-wrap: balance; }
.section-note { color: var(--ink-muted); font-size: 14.5px; margin: 0 0 16px; max-width: 68ch; }
p { margin: 0 0 14px; max-width: 68ch; }
figure { margin: 0; }
figure img { width: 100%; height: auto; display: block; border: 1px solid var(--line); border-radius: 3px; background: var(--paper-raised); }
figcaption { font-size: 13px; color: var(--ink-muted); margin-top: 10px; max-width: 68ch; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 3px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; min-width: 520px; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
th { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-muted); font-weight: 600; background: var(--paper); }
td.num { text-align: right; }
td.strong { font-weight: 700; color: var(--hazard); }
td.mono { font-family: var(--mono); font-size: 12.5px; }
.point-link { color: var(--water-deep); text-decoration: underline dotted; text-underline-offset: 2px; cursor: pointer; }
.point-link:hover { color: var(--hazard); }
.fema-zone { display: inline-block; padding: 1px 7px; border-radius: 3px; font-family: var(--mono); font-size: 11.5px; font-weight: 600; }
.fema-zone.fema-high { background: var(--hazard-bg); color: var(--hazard); }
.fema-zone.fema-moderate { background: var(--hazard-2-bg); color: var(--hazard-2); }
.fema-zone.fema-minimal { color: var(--ink-muted); }
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--water-tint); }
.data-sources-table { min-width: 720px; }
.data-sources-table td:nth-child(2), .data-sources-table td:nth-child(3) { font-size: 13px; color: var(--ink-muted); }
.data-sources-table td:last-child { font-family: var(--mono); font-size: 12px; white-space: nowrap; }
.callout { background: var(--hazard-bg); border-left: 4px solid var(--hazard); padding: 16px 20px; border-radius: 0 3px 3px 0; margin: 16px 0; }
.callout h3 { font-family: var(--serif); font-size: 16px; margin: 0 0 8px; color: var(--hazard); }
.callout p { margin: 0; max-width: none; }
.caveat { background: var(--hazard-2-bg); border-left: 4px solid var(--hazard-2); padding: 14px 20px; border-radius: 0 3px 3px 0; margin: 16px 0; font-size: 14.5px; }
.caveat strong { color: var(--hazard-2); }
.weights { margin: 4px 0 0; display: flex; flex-direction: column; gap: 8px; max-width: 460px; }
.wbar-row { display: grid; grid-template-columns: 130px 1fr 40px; align-items: center; gap: 10px; }
.wbar-label { font-size: 13px; text-transform: capitalize; color: var(--ink-muted); }
.wbar-track { height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; }
.wbar-fill { height: 100%; background: var(--water); border-radius: 4px; }
.wbar-val { font-family: var(--mono); font-size: 12px; text-align: right; font-variant-numeric: tabular-nums; }
.hotspot-card { border: 1px solid var(--line); border-radius: 6px; background: var(--paper-raised); padding: 22px 24px; margin-bottom: 18px; }
.hotspot-carousel-nav { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
.carousel-btn {
  font-family: inherit; font-size: 13px; font-weight: 600; padding: 7px 14px; border-radius: 5px;
  border: 1px solid var(--line); background: var(--paper-raised); color: var(--ink); cursor: pointer;
}
.carousel-btn:hover:not(:disabled) { border-color: var(--water-deep); color: var(--water-deep); }
.carousel-btn:disabled { opacity: 0.35; cursor: default; }
.hotspot-carousel-counter { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--ink-muted); font-family: var(--mono); }
.hotspot-carousel-counter select {
  font-family: inherit; font-size: 13px; padding: 5px 8px; border-radius: 5px; border: 1px solid var(--line);
  background: var(--paper-raised); color: var(--ink); max-width: 220px;
}
.hotspot-card-header { display: flex; align-items: baseline; justify-content: space-between; gap: 14px; flex-wrap: wrap; margin-bottom: 4px; }
.hotspot-card-header h3 { font-family: var(--serif); font-size: 20px; font-weight: 700; margin: 0; }
.hotspot-risk-badge { font-family: var(--mono); font-size: 12.5px; color: var(--ink-muted); white-space: nowrap; }
.hotspot-risk-badge strong { color: var(--ink); font-size: 14px; }
.basin-flag { margin: 10px 0; padding: 9px 13px; border-radius: 0 3px 3px 0; font-size: 13.5px; border-left: 3px solid; max-width: 68ch; }
.basin-flag.warn { background: var(--hazard-bg); border-color: var(--hazard); }
.basin-flag.warn strong { color: var(--hazard); }
.basin-flag.ok { background: var(--water-tint); border-color: var(--water-deep); }
.basin-flag.ok strong { color: var(--water-deep); }
.basin-flag.info { background: transparent; border-color: var(--line); color: var(--ink-muted); }
.hotspot-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(115px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); margin: 14px 0; }
.hotspot-stats .stat { padding: 10px 12px; }
.hotspot-scores-heading { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-muted); margin: 16px 0 6px; }
.hotspot-scores { margin: 0; display: flex; flex-direction: column; gap: 6px; max-width: 460px; }
.hotspot-scores .wbar-row { grid-template-columns: 115px 1fr 36px; }
.hotspot-scores .wbar-fill { background: var(--hazard); }
.hotspot-detail { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 4px; }
.hotspot-detail summary { cursor: pointer; font-size: 13px; font-weight: 600; color: var(--water-deep); padding: 8px 0; }
.hotspot-detail-body { padding: 6px 0 4px; }
.hotspot-detail-body dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; font-size: 13.5px; margin: 0 0 12px; }
.hotspot-detail-body dt { color: var(--ink-muted); }
.hotspot-detail-body dd { margin: 0; }
.rec-block { background: var(--paper-raised); border: 1px solid var(--line); border-radius: 4px; padding: 24px 26px; }
.rec-block h3 { font-family: var(--serif); font-size: 18px; margin: 18px 0 8px; }
.rec-block h3:first-child { margin-top: 0; }
.rec-block h4 { font-family: var(--serif); font-size: 15px; margin: 16px 0 6px; }
.rec-block ul { margin: 0 0 14px; padding-left: 20px; }
.rec-block li { margin-bottom: 6px; max-width: 64ch; }
.rec-block hr { border: none; border-top: 1px solid var(--line); margin: 18px 0; }
.rec-block p { max-width: none; }
.rec-block strong { color: var(--water-deep); }
details.appendix { border: 1px solid var(--line); border-radius: 4px; background: var(--paper-raised); }
details.appendix summary { padding: 14px 18px; cursor: pointer; font-weight: 600; font-size: 14.5px; list-style: none; }
details.appendix summary::-webkit-details-marker { display: none; }
.appendix-body { padding: 0 18px 18px; }
.appendix-block { margin-top: 14px; }
.appendix-block h4 { font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-muted); margin: 0 0 8px; }
pre.prompt { background: var(--paper); border: 1px solid var(--line); border-radius: 3px; padding: 14px 16px; font-family: var(--mono); font-size: 12px; line-height: 1.55; white-space: pre-wrap; word-break: break-word; overflow-x: auto; margin: 0; max-height: 400px; overflow-y: auto; }
footer { margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--line); font-size: 12.5px; color: var(--ink-muted); font-family: var(--mono); }
.pipeline-narrative { display: flex; flex-direction: column; gap: 14px; margin-bottom: 18px; }
.pipeline-narrative-step { border-left: 2px solid var(--line); padding-left: 16px; }
.pipeline-narrative-step h4 { font-family: var(--serif); font-size: 15px; margin: 0 0 4px; }
.pipeline-narrative-step p { margin: 0; max-width: 68ch; color: var(--ink-muted); font-size: 14px; }
.agent-decision-card { border-left: 3px solid var(--water-deep); padding: 12px 16px; margin-bottom: 14px; background: var(--paper-raised); border-radius: 0 3px 3px 0; }
.agent-decision-card h4 { font-family: var(--serif); font-size: 15.5px; margin: 0 0 6px; }
.agent-decision-card .agent-decision-eyebrow { font-family: var(--mono); font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--water-deep); margin-bottom: 4px; }
.agent-decision-card p { margin: 0 0 8px; max-width: 68ch; font-size: 14px; }
.agent-decision-card table { font-size: 13px; }
tr.agent-chosen-row td { background: var(--water-tint); font-weight: 600; }
.mini-maps { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 18px 0; }
.mini-map-block h4 { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-muted); margin: 0 0 6px; }
"""


def render_flood_report_html(
    *,
    place_label: str,
    bounds: tuple[float, float, float, float],
    elevation: ScalarField,
    elevation_source: str,
    morse_result: dict,
    flood_result: dict,
    recommendation: str | None,
    system_prompt: str,
    user_context: str,
    model_name: str,
    exposure_provided: bool,
    generated_at: str,
    boundary_polygon=None,
    boundary_geojson: dict | None = None,
    boundary_display_name: str | None = None,
    boundary_mask=None,
    all_minima_hotspots: list[dict] | None = None,
    catchment_grid: list[list[int]] | None = None,
    fema_flood_geojson: dict | None = None,
    svi_geojson: dict | None = None,
    pipes_geojson: dict | None = None,
    gauges_geojson: dict | None = None,
    buildings_geojson: dict | None = None,
    flow_field_geojson: dict | None = None,
    exposure_geojson: dict | None = None,
    land_cover_geojson_data: dict | None = None,
    roads_geojson: dict | None = None,
    water_geojson: dict | None = None,
    rainfall_soil_geojson: dict | None = None,
    drainage_pipe_data_available: bool = False,
    osm_infrastructure_data_available: bool = False,
    river_gauge_data_available: bool = False,
    exposure_source: str | None = None,
    land_cover_data_available: bool = False,
    land_cover_source: str | None = None,
    soil_data_available: bool = False,
    hydrologic_group_source: str | None = None,
    expected_annual_runoff: dict | None = None,
    strategy_decision: dict | None = None,
    threshold_decision: dict | None = None,
    weight_sensitivity: dict | None = None,
    morse_vs_d8: dict | None = None,
    reflection: dict | None = None,
    embed_interactive_map: bool = True,
) -> str:
    """Assembles the full standalone HTML report. Returns the document as
    a string (title/style/body only -- no <html>/<head> wrapper needed by
    callers that embed it directly, but it's also a complete, valid
    standalone page as-is when saved to a .html file). If
    `boundary_polygon` is given (see boundary.geocode_place_boundary --
    the caller clips the elevation field fed to TTK to it BEFORE calling
    this function), the report notes that a real boundary was resolved and
    `boundary_geojson` draws its outline on the web map; hotspots are
    tagged "within_boundary" by the caller beforehand
    (boundary.tag_hotspots_within_boundary). `all_minima_hotspots`, if
    given, is EVERY raw local minimum (not just the persistence-significant
    ones in flood_result) ranked the same way -- see
    ttk_engine.analyze_scalar_field_ttk(..., persistence_threshold=~0) --
    drawn on the web map colored by drainage demand and covered in its own
    section, since most real ponding-relevant terrain doesn't survive
    noise filtering into the "significant" list above.

    `fema_flood_geojson`, if given (see geoai_agent.fema_flood.
    fetch_fema_flood_zones), is FEMA's own official National Flood Hazard
    Layer for this area -- a real regulatory data source, not derived from
    this run's terrain analysis. Hotspots are tagged "fema_flood_zone"/
    "fema_flood_severity" by the caller beforehand
    (fema_flood.tag_hotspots_with_flood_zone); when present, both hotspot
    tables get a FEMA Zone column and the web map gets a toggleable overlay.

    The significant hotspots (not all_minima_hotspots) are also expected to
    already carry geoai_agent.spatial_analysis's per-hotspot enrichment --
    "aspect_compass"/"local_relief_m"/"flow_length_m" (always present, pure
    terrain math, no extra data source), "nearest_road_m"/"nearest_
    building_m"/"nearest_water_m" (from OpenStreetMap, present only when
    `osm_infrastructure_data_available`), "nearby_pipe_capacity_m3s"
    (from geoai_agent.drainage_db, live-queried per request, present only
    when `drainage_pipe_data_available` -- covers a handful of US cities/
    counties, see that module's docstring), "twi" (Topographic Wetness Index, from
    spatial_analysis.compute_topographic_wetness_index -- always present
    whenever a slope layer was fetched this run, pure terrain math like
    aspect/relief, no extra data source), and "nearest_gauge" (from
    geoai_agent.streamflow, present only when `river_gauge_data_available`
    -- a real, current USGS NWIS river-stage/discharge reading, the one
    fluvial signal in an otherwise pluvial/topological pipeline). All of
    this renders as a "Spatial analysis" section -- entirely computed, not
    just carried through, so the report itself explains what each column
    means.

    `gauges_geojson` (see geoai_agent.streamflow.gauges_to_geojson) draws a
    toggleable USGS gauge-location layer on the web map, same role as
    `pipes_geojson`. `exposure_source` distinguishes a user-uploaded
    exposure file ("user_upload") from a live US Census block-group density
    fetch ("census_live") for the "Exposure data" summary chip -- None means
    neither was available this run (the neutral-default case).
    flood_result's own "rainfall_source" (added by flood_tools.
    rank_flood_basins) says whether the design storm intensity came from a
    live, site-specific NOAA Atlas 14 fetch or the static illustrative
    fallback table -- shown next to the "Design storm" chip and in the
    Ranked Hotspots section, since which one was used materially changes
    how much to trust the reported intensity.

    Each hotspot is also expected to carry "runoff_coefficient"/
    "runoff_coefficient_source" ("land_cover_derived" -- a real per-basin
    value from NLCD land cover + USDA SSURGO soil via the SCS Curve Number
    method, present whenever `land_cover_data_available` and `soil_data_
    available` are both true and that basin's catchment had usable data --
    or "constant", the flat fallback) and "buildings_in_catchment"/
    "building_density_per_km2" (OpenStreetMap building counts aggregated
    per basin, present whenever `osm_infrastructure_data_available` --
    reuses the same building fetch `nearest_building_m` already uses, at no
    extra data cost). `expected_annual_runoff`, if given (see flood_tools.
    compute_expected_annual_runoff), is a separate multi-return-period
    aggregate ranking -- {"basins": [...], "note": <caveat text>} -- shown
    as its own section alongside the primary single-scenario hotspot table,
    with the caveat text surfaced plainly (this figure is meant to look
    smaller than any one storm's own runoff number; that's a genuine
    property of a probability-weighted aggregate across all six return
    periods, not an error).

    `embed_interactive_map=False` skips the "Interactive map" section
    entirely -- used by web_app's two-column live UI, which renders its own
    persistent Leaflet map in a separate panel (built from the same
    elevation_hillshade_geojson/flood_hotspots_to_geojson/all_minima_geojson
    data via app.py's map_data payload) rather than embedding one inside
    this report fragment. The standalone downloadable .html file always
    keeps the map embedded (embed_interactive_map=True, the default) so it
    stays a complete, self-contained artifact once saved."""
    hotspots = flood_result["hotspots"]
    scenario = flood_result["rainfall_scenario"]
    weights = flood_result["weights"]
    lat_min, lat_max, lon_min, lon_max = bounds
    boundary_available = boundary_polygon is not None
    fema_available = fema_flood_geojson is not None

    webmap_html = (
        _webmap_fragment(
            elevation, bounds, boundary_geojson, all_minima_hotspots, catchment_grid,
            fema_flood_geojson, pipes_geojson, gauges_geojson, buildings_geojson, boundary_mask,
            flow_field_geojson,
            exposure_geojson, land_cover_geojson_data, roads_geojson, water_geojson, rainfall_soil_geojson,
            land_cover_source=land_cover_source,
        )
        if embed_interactive_map else ""
    )

    coefficient_varies = len({h.get("runoff_coefficient_source") for h in hotspots}) > 1 or any(
        h.get("runoff_coefficient_source") == "land_cover_derived" for h in hotspots
    )

    # Multi-return-period aggregate (flood_tools.compute_expected_annual_runoff),
    # if this run computed it -- joined onto each significant basin by
    # catchment_id, the SAME id space this basin's own hotspot dict already
    # uses (both come from the SAME morse_result/TTK run) -- NOT the same id
    # space as map_catchment_id (a separate, lower-threshold run used only
    # for map-click consistency, see _point_link's callers throughout this
    # file). Computed once here so both the per-card stat grid and each
    # card's "More detail" per-storm breakdown can look a basin's own figure
    # up without re-deriving this join.
    ead_by_catchment = {}
    if expected_annual_runoff and expected_annual_runoff.get("basins"):
        ead_by_catchment = {b["catchment_id"]: b for b in expected_annual_runoff["basins"]}

    has_osm = any(h.get("nearest_road_m") is not None for h in hotspots)
    has_pipes = any(h.get("nearby_pipe_capacity_m3s") is not None for h in hotspots)
    has_twi = any(h.get("twi") is not None for h in hotspots)
    has_gauge = any(h.get("nearest_gauge") is not None for h in hotspots)
    has_buildings = any(h.get("buildings_in_catchment") is not None for h in hotspots)
    has_svi = any(h.get("svi_percentile") is not None for h in hotspots)

    def _fmt_m(value: float | None) -> str:
        return f"{value:,.0f} m" if value is not None else "&ndash;"

    # These explain the DATA SOURCE itself, not any one basin's own number --
    # shown once above the whole set of cards below rather than repeated
    # inside every card (the same "state a run-wide fact once, not per-
    # basin" discipline _drainage_capacity_flag/_fema_corroboration_flag
    # already apply above for their own "not available this run" cases).
    osm_note = (
        '<p class="section-note">Nearest road/building/water distances (see each basin\'s "More detail" below) '
        "are straight-line, from OpenStreetMap (live, community-mapped -- coverage varies by area, and a feature "
        "missing from OSM doesn't mean it doesn't exist on the ground).</p>" if has_osm else
        '<p class="section-note">OpenStreetMap infrastructure data was not available for this area, so nearest-'
        "road/building/water distances aren't shown below.</p>" if osm_infrastructure_data_available is False else ""
    )
    pipe_note = (
        '<p class="section-note">Drainage-capacity comparisons below use the largest Manning\'s-equation full-'
        "flow estimate (see geoai_agent/drainage_db.py) among real stormwater pipes within 150m of each basin, "
        "queried live from that area's own public GIS pipe network (diameter/material/slope, where available) "
        "-- covers a handful of US cities/counties, not a nationwide dataset. It's an order-of-magnitude "
        "comparison against each basin's own drainage demand, not an engineering-grade capacity study, and the "
        "nearest pipe by distance may be a small lateral or a large trunk line -- this doesn't know which one "
        "actually serves that specific basin.</p>"
        if has_pipes else
        '<p class="section-note">No real stormwater pipe data is available for this area (live coverage is '
        "limited to a handful of US cities/counties) -- the drainage-capacity notes below are each basin's own "
        "demand estimate only, not compared against actual infrastructure.</p>" if drainage_pipe_data_available is False else ""
    )
    twi_note = (
        '<p class="section-note">TWI (Topographic Wetness Index) = ln(contributing area / tan(slope)) -- higher '
        "values mean greater tendency to accumulate/retain water given a basin's own contributing area and local "
        "slope. Supplementary terrain context alongside each basin's persistence score, <strong>not</strong> one "
        "of the weighted factors in its risk score below.</p>" if has_twi else ""
    )
    svi_note = (
        '<p class="section-note">SVI percentile is CDC/ATSDR\'s published Social Vulnerability Index for that '
        "basin's census tract -- an overall percentile rank (0-100%) against every US tract, built from poverty, "
        "unemployment, income, education, disability, housing, and other factors. Supplementary socioeconomic "
        "context alongside population-density exposure, <strong>not</strong> one of the weighted factors in the "
        "risk score below.</p>" if has_svi else ""
    )
    gauge_note = (
        '<p class="section-note">Nearest active USGS river/stream gauge (NWIS, live) within 2km, if any (see each '
        "basin's \"More detail\" below) -- a real fluvial (river-stage) signal independent of this run's own "
        "topological/pluvial analysis, not a comparison this tool makes automatically. Gauge coverage is sparse "
        "nationally, so \"none within 2km\" is the common case, not a sign the area went unchecked.</p>" if has_gauge else
        '<p class="section-note">No active USGS river/stream gauges were found near this area.</p>'
        if river_gauge_data_available else ""
    )
    buildings_note = (
        '<p class="section-note">OpenStreetMap-mapped buildings whose centroid falls within a basin\'s own '
        "catchment (the area draining into it), and their density per km&sup2; below -- reuses the same building "
        "fetch used for \"nearest building\" detail, aggregated per basin instead of just the closest one. "
        "Community-mapped coverage, not an official building count, and supplementary exposure context, "
        "<strong>not</strong> one of the weighted factors in risk score below.</p>" if has_buildings else ""
    )
    shared_notes_html = "".join(n for n in (osm_note, pipe_note, twi_note, gauge_note, buildings_note, svi_note) if n)

    def _hotspot_card_html(h: dict) -> str:
        """One significant basin's complete profile -- rank/coords/risk
        score, the computed drainage-capacity and FEMA cross-check flags
        (the actual "connect the dots" fix for what used to be two separate
        tables a reader had to cross-reference by hand), every key number
        in one compact stat grid (replacing that same cross-referencing for
        every OTHER fact about this basin -- runoff coefficient, TWI,
        building density, the multi-return-period expected-annual-runoff
        headline figure), this basin's OWN per-factor risk-score breakdown
        (not the run-level WEIGHTS shown further down -- this is how THIS
        basin actually scored against those weights, new content this
        report never showed before), and a collapsed "More detail" for the
        secondary terrain/proximity facts that don't need to be in the
        primary read."""
        persistence = "global sink (infinite)" if h["persistence_m"] is None else f"{h['persistence_m']:.1f} m"
        # NOT h.get("catchment_id") -- that's scoped to this basin's OWN,
        # higher-persistence-threshold TTK run, a different, unrelated
        # numbering from the lower-threshold run the map's catchment_grid
        # comes from (see web_app/app.py's map_catchment_id reconciliation).
        coord_link = _point_link(
            h["coords"][1], h["coords"][0], h.get("map_catchment_id"),
            f"{h['coords'][1]:.4f}, {h['coords'][0]:.4f}", embed_interactive_map,
            hotspot_rank=h["rank"],
        )

        flags_html = ""
        drainage_flag = _drainage_capacity_flag(h, drainage_pipe_data_available)
        if drainage_flag:
            sev, msg = drainage_flag
            flags_html += f'<div class="basin-flag {sev}">{msg}</div>'
        fema_flag = _fema_corroboration_flag(h, fema_available)
        if fema_flag:
            sev, msg = fema_flag
            flags_html += f'<div class="basin-flag {sev}">{msg}</div>'

        ead = ead_by_catchment.get(h.get("catchment_id"))

        coefficient_stat = ""
        if coefficient_varies:
            source = h.get("runoff_coefficient_source")
            tag = "land cover + soil derived" if source == "land_cover_derived" else "assumed constant"
            coefficient_stat = (
                f'<div class="stat"><span class="stat-label">Runoff coeff.</span>'
                f'<span class="stat-value">{h.get("runoff_coefficient", 0.0):.2f} '
                f'<span style="font-weight:400;color:var(--ink-muted);font-size:11.5px;">({tag})</span></span></div>'
            )
        boundary_stat = (
            f'<div class="stat"><span class="stat-label">Boundary</span>'
            f'<span class="stat-value">{"inside" if h.get("within_boundary") else "outside"}</span></div>'
        ) if boundary_available else ""
        fema_stat = ""
        if fema_available:
            zone = h.get("fema_flood_zone")
            fema_stat = (
                f'<div class="stat"><span class="stat-label">FEMA zone</span>'
                f'<span class="stat-value">{html.escape(zone) if zone else "&ndash;"}</span></div>'
            )
        twi_stat = (
            f'<div class="stat"><span class="stat-label">TWI</span><span class="stat-value">{h["twi"]:.2f}</span></div>'
            if has_twi and h.get("twi") is not None else ""
        )
        svi_stat = (
            f'<div class="stat"><span class="stat-label">SVI percentile</span>'
            f'<span class="stat-value">{h["svi_percentile"]:.0%}</span></div>'
            if has_svi and h.get("svi_percentile") is not None else ""
        )
        buildings_stat = ""
        if has_buildings and h.get("buildings_in_catchment") is not None:
            density = h.get("building_density_per_km2")
            density_note = f" ({density:,.0f}/km&sup2;)" if density is not None else ""
            buildings_stat = (
                f'<div class="stat"><span class="stat-label">Buildings</span>'
                f'<span class="stat-value">{h["buildings_in_catchment"]:,}{density_note}</span></div>'
            )
        ead_stat = ""
        if ead is not None:
            ead_stat = (
                f'<div class="stat"><span class="stat-label">Return-period-weighted runoff index</span>'
                f'<span class="stat-value">{ead["expected_annual_runoff_m3s"]:.2f} m&sup3;/s</span></div>'
            )

        stats_html = f"""
        <div class="hotspot-stats">
          <div class="stat"><span class="stat-label">Elevation</span><span class="stat-value">{h['elevation_m']:.1f} m</span></div>
          <div class="stat"><span class="stat-label">Contrib. area</span><span class="stat-value">{h['contributing_area_km2']:.2f} km&sup2;</span></div>
          <div class="stat"><span class="stat-label">Persistence</span><span class="stat-value">{persistence}</span></div>
          <div class="stat"><span class="stat-label">Peak runoff</span><span class="stat-value">{h['estimated_peak_runoff_m3s']:.1f} m&sup3;/s</span></div>
          {coefficient_stat}
          {boundary_stat}
          {fema_stat}
          {twi_stat}
          {buildings_stat}
          {svi_stat}
          {ead_stat}
        </div>"""

        factor_bars = "".join(
            f'<div class="wbar-row"><span class="wbar-label">{label}</span>'
            f'<div class="wbar-track"><div class="wbar-fill" style="width:{h.get(key, 0.0) * 100:.0f}%"></div></div>'
            f'<span class="wbar-val">{h.get(key, 0.0):.2f}</span></div>'
            for label, key in (
                ("topological", "topological_score"), ("contributing area", "contributing_area_score"),
                ("rainfall", "rainfall_score"), ("slope", "slope_score"), ("exposure", "exposure_score"),
            )
        )

        detail_rows = []
        if h.get("aspect_compass"):
            detail_rows.append(f"<dt>Aspect</dt><dd>{html.escape(h['aspect_compass'])}</dd>")
        if h.get("local_relief_m") is not None:
            detail_rows.append(f"<dt>Local relief</dt><dd>{h['local_relief_m']:.1f} m</dd>")
        if h.get("flow_length_m") is not None:
            detail_rows.append(f"<dt>Flow length</dt><dd>{_fmt_m(h.get('flow_length_m'))}</dd>")
        if has_osm:
            detail_rows.append(f"<dt>Nearest road</dt><dd>{_fmt_m(h.get('nearest_road_m'))}</dd>")
            detail_rows.append(f"<dt>Nearest building</dt><dd>{_fmt_m(h.get('nearest_building_m'))}</dd>")
            detail_rows.append(f"<dt>Nearest water</dt><dd>{_fmt_m(h.get('nearest_water_m'))}</dd>")
        if has_gauge:
            gauge = h.get("nearest_gauge")
            if gauge is not None:
                reading_bits = []
                if gauge.get("gage_height_ft") is not None:
                    reading_bits.append(f"{gauge['gage_height_ft']:.1f} ft stage")
                if gauge.get("discharge_cfs") is not None:
                    reading_bits.append(f"{gauge['discharge_cfs']:.0f} ft&sup3;/s")
                reading = ", ".join(reading_bits) or "no current reading"
                detail_rows.append(
                    f"<dt>Nearest gauge</dt><dd>{html.escape(gauge['site_name'] or 'USGS gauge')} "
                    f"({gauge['distance_m']:,.0f}m away, {reading})</dd>"
                )
            else:
                detail_rows.append("<dt>Nearest gauge</dt><dd>none within 2km</dd>")
        if has_svi and h.get("svi_percentile") is not None:
            detail_rows.append(
                f"<dt>Social vulnerability</dt><dd>{html.escape(h.get('svi_severity') or '')} "
                f"(socioeconomic-status theme: {h['svi_socioeconomic_percentile']:.0%} percentile)</dd>"
                if h.get("svi_socioeconomic_percentile") is not None else
                f"<dt>Social vulnerability</dt><dd>{html.escape(h.get('svi_severity') or '')}</dd>"
            )
        if ead is not None:
            per = ead["per_scenario"]
            detail_rows.append(
                "<dt>Per-storm runoff</dt><dd>"
                f"2yr {per.get('2yr', 0.0):.2f} &middot; 10yr {per.get('10yr', 0.0):.2f} &middot; "
                f"25yr {per.get('25yr', 0.0):.2f} &middot; 50yr {per.get('50yr', 0.0):.2f} &middot; "
                f"100yr {per.get('100yr', 0.0):.2f} &middot; 500yr {per.get('500yr', 0.0):.2f} m&sup3;/s</dd>"
            )
        detail_html = ""
        if detail_rows:
            detail_html = f"""
        <details class="hotspot-detail">
          <summary>More detail</summary>
          <div class="hotspot-detail-body"><dl>{''.join(detail_rows)}</dl></div>
        </details>"""

        return f"""
    <div class="hotspot-card">
      <div class="hotspot-card-header">
        <h3>#{h['rank']} &middot; {coord_link}</h3>
        <span class="hotspot-risk-badge">priority score <strong>{h['risk_score']:.3f}</strong></span>
      </div>
      {flags_html}
      {stats_html}
      <div class="hotspot-scores-heading">This basin's own factor scores</div>
      <div class="weights hotspot-scores">{factor_bars}</div>
      {detail_html}
    </div>"""

    # One basin visible at a time (see .hotspot-carousel-slide's inline
    # display:none default below) with prev/next + a rank jump-list --
    # `hotspots` can run into the dozens, and stacking every card
    # vertically made the "Ranked hotspots" section the longest scroll in
    # the whole report for no reason, when a reader almost always wants ONE
    # basin's full profile at a time anyway. Pure inline <script>, no
    # framework -- runs fine both as a downloaded standalone file and via
    # the live UI's iframe(srcdoc=...) embedding (unlike innerHTML
    # insertion, srcdoc is a real document load, so inline scripts execute
    # normally in both contexts).
    hotspot_slides_html = "".join(
        f'<div class="hotspot-carousel-slide" data-rank="{h["rank"]}" style="display:{"" if i == 0 else "none"};">'
        f'{_hotspot_card_html(h)}</div>'
        for i, h in enumerate(hotspots)
    )
    hotspot_jump_options = "".join(
        f'<option value="{i}">#{h["rank"]} &middot; priority {h["risk_score"]:.3f}</option>'
        for i, h in enumerate(hotspots)
    )
    hotspot_cards_html = f"""
    <div class="hotspot-carousel" id="hotspot-carousel">
      <div class="hotspot-carousel-nav">
        <button type="button" class="carousel-btn" id="hotspot-carousel-prev">&larr; Previous</button>
        <span class="hotspot-carousel-counter">
          <select id="hotspot-carousel-jump">{hotspot_jump_options}</select>
          <span>of {len(hotspots)}</span>
        </span>
        <button type="button" class="carousel-btn" id="hotspot-carousel-next">Next &rarr;</button>
      </div>
      {hotspot_slides_html}
    </div>
    <script>
    (function() {{
      var slides = document.querySelectorAll('#hotspot-carousel .hotspot-carousel-slide');
      var jumpEl = document.getElementById('hotspot-carousel-jump');
      var prevBtn = document.getElementById('hotspot-carousel-prev');
      var nextBtn = document.getElementById('hotspot-carousel-next');
      var current = 0;
      function show(i) {{
        current = Math.max(0, Math.min(slides.length - 1, i));
        slides.forEach(function(s, idx) {{ s.style.display = idx === current ? '' : 'none'; }});
        jumpEl.value = String(current);
        prevBtn.disabled = current === 0;
        nextBtn.disabled = current === slides.length - 1;
      }}
      prevBtn.addEventListener('click', function() {{ show(current - 1); }});
      nextBtn.addEventListener('click', function() {{ show(current + 1); }});
      jumpEl.addEventListener('change', function() {{ show(parseInt(jumpEl.value, 10)); }});
      // Reused by _point_link's onclick (a report mention like "see #3
      // above") when the click target is one of these carousel slides --
      // jumps the carousel to that rank instead of just scrolling to a
      // hidden slide the reader can't see. See _point_link below.
      window.showHotspotRank = function(rank) {{
        for (var i = 0; i < slides.length; i++) {{
          if (parseInt(slides[i].dataset.rank, 10) === rank) {{ show(i); slides[i].scrollIntoView({{behavior: 'smooth', block: 'center'}}); return; }}
        }}
      }};
      show(0);
    }})();
    </script>"""

    weight_bars = "".join(
        f'<div class="wbar-row"><span class="wbar-label">{k.replace("_", " ")}</span>'
        f'<div class="wbar-track"><div class="wbar-fill" style="width:{v * 100:.0f}%"></div></div>'
        f'<span class="wbar-val">{v:.0%}</span></div>'
        for k, v in weights.items()
    )

    weighting_method = flood_result.get("weighting_method", "preset")
    _rainfall_invariance_note = (
        "One thing worth saying plainly about \"rainfall\" specifically: this run applies a single selected "
        "storm uniformly across the whole area, so rainfall_score is the exact same number for every basin "
        "here. Whatever weight it's given only shifts every basin's risk_score by the same constant amount, "
        "which cannot change WHICH basin ranks highest -- verified directly: re-ranking these same basins "
        "with rainfall's raw weight set to 0%, 20%, and 55% (holding the other four weights' relative "
        "proportions fixed) produces an identical order every time. It only matters if risk_score is ever "
        "compared ACROSS different storm scenarios (see the expected-annual-runoff aggregation below)."
    )
    if weighting_method == "entropy":
        weighting_method_note = (
            "<strong>Important distinction:</strong> the weights above are <strong>data-driven relative "
            "weights</strong>, not \"objective importance\" for flooding. The Entropy Weight Method (Shannon "
            "entropy applied to each factor's own variation across these specific basins; see Zhu et al., "
            "\"Effectiveness of the Entropy Weight Method in Decision-Making,\" <em>Mathematical Problems in "
            "Engineering</em>, 2020, and its application to urban flooding specifically in Wang et al.'s "
            "Beijing FAHP-EWM flood-risk study) determines the <strong>discriminative</strong> weight of each "
            "factor -- how much it varies across THESE basins, i.e. how useful it is for telling them apart -- "
            "it does not, and cannot, estimate a factor's <strong>causal or predictive importance</strong> for "
            "flooding. Those are different questions with different answers: a factor that varies more here "
            "gets more weight regardless of whether it actually drives flood risk, and a factor that's "
            "genuinely important but happens to be constant in this specific run gets weight 0 for that reason "
            "alone -- see \"rainfall\" below, which is IDENTICAL for every basin in a single run since one "
            "storm applies uniformly across the whole area, and therefore contributes nothing to distinguishing "
            "them no matter how much rainfall actually matters physically. That is a real, correct property of "
            "the method, not a bug -- but it means this run's entropy weights answer \"which factor best "
            "distinguishes THESE five basins from each other,\" not \"which factor best predicts an actual "
            "flood.\" The same basins analyzed alongside a different set could get different weights, and this "
            "is still not a model validated against real historical flood outcomes."
        )
    elif weighting_method == "custom":
        weighting_method_note = (
            "The weights above were set by hand, using the Modify-weights control -- a person's own numbers, "
            "not an expert-named preset and not computed by the Entropy Weight Method. Like every other "
            "weighting mode here, this is NOT a statistically calibrated flood-risk model fit or validated "
            "against historical flood outcome data -- risk_score stays a transparent, reproducible way to "
            f"combine five heuristic signals into one ordering. {_rainfall_invariance_note}"
        )
    else:
        weighting_method_note = (
            "The weights above are an expert-defined prioritization scheme (see <code>flood_tools."
            "WEIGHT_PRESETS</code> -- named presets like \"balanced\"/\"life_safety\"/\"infrastructure_"
            "protection\"), NOT a statistically calibrated flood-risk model fit or validated against "
            "historical flood outcome data. They aren't arbitrary, though: grouping this project's five "
            "factors against the nine physical factors in a real published AHP flood-hazard-index study "
            "(Mohammed, Hussein &amp; Abood, \"Potential Flood Hazard Mapping Based on GIS and Analytical "
            "Hierarchy Process,\" <em>Journal of Water Management Modeling</em> 32, C528, 2024, "
            "<a href=\"https://doi.org/10.14796/JWMM.C528\" target=\"_blank\" rel=\"noopener\">"
            "doi.org/10.14796/JWMM.C528</a>) lands in a similar ballpark to that paper's own AHP-derived "
            "weights -- a real, checkable anchor for \"is this a reasonable prioritization,\" not proof these "
            "exact numbers are correct for this study area (one river-basin study doesn't generalize, and "
            "the factor definitions only approximately correspond -- see flood_tools.py's module docstring "
            "for the full breakdown). A data-driven relative-weighting alternative (no expert judgment needed, "
            "computed from how much each factor varies across this run's own basins -- NOT a measure of each "
            "factor's causal importance for flooding, see that mode's own note) is also available -- see "
            "<code>flood_tools.entropy_weights</code> -- with its own real, citable methodology and its own "
            f"honest caveats. {_rainfall_invariance_note}"
        )

    rainfall_source = flood_result.get("rainfall_source", "static_illustrative_table")
    rainfall_source_label = (
        "NOAA Atlas 14 (live, site-specific)" if rainfall_source == "noaa_atlas14_live"
        else "illustrative Mid-Atlantic fallback"
    )
    if exposure_source == "census_live":
        exposure_label = "US Census block groups (live)"
        exposure_score_formula_note = (
            "US Census block-group population density (people per km&sup2;, 2020 Census, via Esri Living "
            "Atlas' <code>USA_Census_BlockGroups</code> layer -- see geoai_agent.real_data."
            "fetch_population_exposure_field) <strong>at this basin's own pour point only</strong> (a single "
            "grid cell, not a catchment total or average), min-max normalized against that field's range "
            "across the whole analysis grid. This is a DIFFERENT measurement from \"Buildings\" in the stat "
            "grid above (which counts OpenStreetMap buildings across the basin's entire catchment, a different "
            "data source) -- a basin can have many buildings in its catchment while still scoring low on "
            "exposure here, if the population-density value specifically at its exact low point happens to be "
            "low (e.g. a park, stream corridor, or parking lot sitting at the literal bottom of the depression) "
            "even while the broader catchment around it is dense."
        )
    elif exposure_source == "worldpop_global":
        exposure_label = "WorldPop global 100m (live)"
        exposure_score_formula_note = (
            "WorldPop's global 100m gridded population density (people per km&sup2;, 2020 estimate, via its "
            "public ArcGIS Living Atlas ImageServer -- see geoai_agent.real_data."
            "fetch_global_population_density_field, used because no US Census coverage exists for this area) "
            "<strong>at this basin's own pour point only</strong> (a single grid cell, not a catchment total or "
            "average), min-max normalized against that field's range across the whole analysis grid. A modeled "
            "estimate (built from census/survey data disaggregated by satellite-derived settlement patterns), "
            "not a direct government count the way US Census block-group data is. This is a DIFFERENT "
            "measurement from \"Buildings\" in the stat grid above (OpenStreetMap buildings across the basin's "
            "entire catchment, a different data source)."
        )
    elif exposure_source == "user_upload":
        exposure_label = "user-supplied"
        exposure_score_formula_note = (
            "the user-uploaded exposure/population value <strong>at this basin's own pour point only</strong> "
            "(a single grid cell, not a catchment total or average), min-max normalized against that field's "
            "range across the whole analysis grid. This is a DIFFERENT measurement from \"Buildings\" in the "
            "stat grid above (OpenStreetMap buildings across the basin's entire catchment, a different data "
            "source)."
        )
    else:
        exposure_label = "not available (neutral default)"
        exposure_score_formula_note = (
            "no exposure/population data was available for this run (no US Census coverage or fetch failure, "
            "and no user upload), so every basin was given the same neutral 0.5 -- exposure contributed nothing "
            "to distinguishing these basins from each other this run."
        )

    stat_chips = f"""
  <div class="stat"><span class="stat-label">Data source</span><span class="stat-value">{html.escape(elevation_source)}</span></div>
  <div class="stat"><span class="stat-label">Analysis grid</span><span class="stat-value">{elevation.shape[1]}&times;{elevation.shape[0]}</span></div>
  <div class="stat"><span class="stat-label">Elevation range</span><span class="stat-value">{float(elevation.values.min()):.1f}&ndash;{float(elevation.values.max()):.1f} m</span></div>
  <div class="stat"><span class="stat-label">Significant basins</span><span class="stat-value">{morse_result['num_significant_basins']} <span style="font-weight:400;color:var(--ink-muted);font-size:12.5px;">of {morse_result['num_raw_minima']} raw minima</span></span></div>
  <div class="stat"><span class="stat-label">Design storm</span><span class="stat-value">{scenario['label']} &middot; {scenario['intensity_mm_per_hr']:.0f} mm/hr, 1-hour intensity <span style="font-weight:400;color:var(--ink-muted);font-size:12.5px;">({rainfall_source_label})</span></span></div>
  <div class="stat"><span class="stat-label">Boundary source</span><span class="stat-value">{'OpenStreetMap (Nominatim)' if boundary_available else 'padded box around geocoded point'}</span></div>
  <div class="stat"><span class="stat-label">Exposure data</span><span class="stat-value">{exposure_label}</span></div>
  <div class="stat"><span class="stat-label">FEMA flood data</span><span class="stat-value">{'NFHL (live)' if fema_available else 'not available for this area'}</span></div>
  <div class="stat"><span class="stat-label">River gauge data</span><span class="stat-value">{'USGS NWIS (live)' if river_gauge_data_available else 'no active gauges nearby'}</span></div>
  <div class="stat"><span class="stat-label">Runoff coefficient</span><span class="stat-value">{f"{_land_cover_source_label(land_cover_source)} + {'SSURGO soil' if hydrologic_group_source == 'measured' else 'assumed soil group'} (per basin)" if land_cover_data_available and soil_data_available else 'assumed constant'}</span></div>
"""

    rec_html = _md_to_html(recommendation) if recommendation else (
        "<p><em>A narrative summary couldn't be generated for this run (a transient issue reaching the "
        "language model, not a problem with the analysis itself). The ranked hotspot table, drainage-demand "
        "figures, and every data source noted above are complete and independent of this step -- they come "
        "directly from flood_tools.rank_flood_basins and the live data fetches, not from the model. Re-running "
        "the analysis will normally produce this summary; in the meantime, the \"Ranked hotspots\" and "
        "\"What this shows\" sections below cover the same top-priority findings this paragraph would have "
        "narrated.</em></p>"
    )
    narrative_html = _narrative_html(
        hotspots, elevation.shape, exposure_provided, boundary_available, fema_available, embed_interactive_map,
    )

    boundary_section = ""
    if boundary_available:
        boundary_section = f"""
    <p class="section-note">Boundary resolved to <strong>{html.escape(boundary_display_name or place_label)}</strong>
      via OpenStreetMap (Nominatim). The elevation data fed to TTK was clipped to this shape BEFORE analysis --
      every grid cell outside the polygon was raised to high "wall" ground (a standard watershed-delineation
      technique; the analysis grid itself stays a complete rectangle, since TTK's topology needs one, but a basin
      can no longer form outside the real boundary once everything outside it is artificially higher than
      everything inside). A basin whose true lowest point sits very close to the boundary edge can still be
      pulled toward that wall -- the same edge effect real GIS watershed tools accept when clipping to a
      boundary that isn't itself a topographic ridge.</p>"""

    interactive_map_section = ""
    if embed_interactive_map:
        interactive_map_section = f"""
  <section id="sec-map">
    <h2>Interactive map</h2>
    <p class="section-note">Real OpenStreetMap basemap with the elevation field as a toggleable hillshade layer{', the resolved boundary outlined in red,' if boundary_available else ''}{", FEMA's official flood zones as a toggleable overlay," if fema_available else ""} and each hotspot as a clickable marker.</p>
    {webmap_html}
  </section>"""

    pipeline_narrative_html = _pipeline_narrative_html(
        place_label=place_label, bounds=bounds, elevation=elevation, elevation_source=elevation_source,
        morse_result=morse_result, flood_result=flood_result, boundary_available=boundary_available,
        boundary_display_name=boundary_display_name, boundary_geojson=boundary_geojson,
        fema_flood_geojson=fema_flood_geojson, svi_geojson=svi_geojson, gauges_geojson=gauges_geojson,
        drainage_pipe_data_available=drainage_pipe_data_available,
        river_gauge_data_available=river_gauge_data_available,
        osm_infrastructure_data_available=osm_infrastructure_data_available,
        land_cover_data_available=land_cover_data_available, land_cover_source=land_cover_source,
        soil_data_available=soil_data_available, hydrologic_group_source=hydrologic_group_source,
        exposure_source=exposure_source, expected_annual_runoff=expected_annual_runoff, model_name=model_name,
    )
    agent_decision_log_html = _agent_decision_log_html(
        strategy_decision=strategy_decision, threshold_decision=threshold_decision,
        weight_sensitivity=weight_sensitivity,
    )
    morse_vs_d8_html = _morse_vs_d8_html(morse_vs_d8)
    reflection_html = _reflection_html(reflection)

    # Table of contents -- entries conditional on whatever this specific
    # run actually rendered (embed_interactive_map, all_minima_hotspots),
    # so a link is never dead. Anchors, not JS -- works identically in the
    # standalone downloaded file and inside the live UI's floating report
    # window, no cross-frame scripting needed either way.
    toc_entries = [("sec-agent", "How the agent worked"), ("sec-d8", "Morse vs. D8 cross-check")]
    toc_entries.append(("sec-terrain", "The terrain, and where it drains"))
    if embed_interactive_map:
        toc_entries.append(("sec-map", "Interactive map"))
    toc_entries.append(("sec-shows", "What this shows"))
    toc_entries.append(("sec-recommendation", "Generated recommendation"))
    toc_entries.append(("sec-hotspots", "Ranked hotspots"))
    if all_minima_hotspots:
        toc_entries.append(("sec-allminima", "All local minima, unfiltered"))
    toc_entries.append(("sec-limitations", "Limitations & uncertainty"))
    toc_entries.append(("sec-pipeline", "Analysis pipeline & data sources"))
    toc_html = (
        '<nav class="report-toc"><span class="report-toc-label">On this page</span>'
        + "".join(f'<a href="#{anchor}">{html.escape(label)}</a>' for anchor, label in toc_entries)
        + "</nav>"
    )
    data_sources_table_html = _data_sources_table_html(
        elevation_source=elevation_source, boundary_available=boundary_available,
        rainfall_source_label=rainfall_source_label, soil_data_available=soil_data_available,
        hydrologic_group_source=hydrologic_group_source,
        land_cover_data_available=land_cover_data_available, land_cover_source=land_cover_source,
        exposure_source=exposure_source,
        osm_infrastructure_data_available=osm_infrastructure_data_available, fema_available=fema_available,
        river_gauge_data_available=river_gauge_data_available,
        drainage_pipe_data_available=drainage_pipe_data_available,
    )
    limitations_html = _limitations_html(
        hotspots=hotspots, elevation_source=elevation_source, boundary_available=boundary_available,
        exposure_source=exposure_source, osm_infrastructure_data_available=osm_infrastructure_data_available,
        fema_available=fema_available,
    )

    all_minima_section = ""
    if all_minima_hotspots:
        by_runoff = sorted(all_minima_hotspots, key=lambda h: h["estimated_peak_runoff_m3s"], reverse=True)
        top_n = by_runoff[:15]
        am_boundary_th = "<th>Boundary</th>" if boundary_available else ""
        am_fema_th = "<th>FEMA Zone</th>" if fema_available else ""
        am_rows = []
        for h in top_n:
            am_boundary_cell = ""
            if boundary_available:
                am_boundary_cell = f"<td>{'inside' if h.get('within_boundary') else 'outside'}</td>"
            am_fema_cell = _fema_zone_cell(h) if fema_available else ""
            am_coord_link = _point_link(
                h["coords"][1], h["coords"][0], h.get("catchment_id"),
                f"{h['coords'][1]:.4f}, {h['coords'][0]:.4f}", embed_interactive_map,
            )
            am_rows.append(f"""
      <tr>
        <td>{h['rank']}</td>
        <td class="mono">{am_coord_link}</td>
        <td class="num strong">{h['estimated_peak_runoff_m3s']:.2f}</td>
        <td class="num">{h['contributing_area_km2']:.3f}</td>
        <td class="num">{h['risk_score']:.3f}</td>
        {am_boundary_cell}
        {am_fema_cell}
      </tr>""")

        inside_note = ""
        if boundary_available:
            inside = [h for h in by_runoff if h.get("within_boundary")]
            if inside:
                top_inside = inside[0]
                top_inside_link = _point_link(
                    top_inside["coords"][1], top_inside["coords"][0], top_inside.get("catchment_id"),
                    f"{top_inside['coords'][1]:.4f}°N, {top_inside['coords'][0]:.4f}°E", embed_interactive_map,
                )
                inside_note = (
                    f'<div class="callout"><h3>Highest drainage demand actually inside the boundary</h3>'
                    f"<p>Of every local minimum, the one with the largest drainage demand that falls "
                    f"<strong>inside</strong> {html.escape(boundary_display_name or place_label)}'s real boundary is "
                    f"<strong>#{top_inside['rank']}</strong> ({top_inside_link}): "
                    f"<strong>{top_inside['estimated_peak_runoff_m3s']:.2f} m³/s</strong>, contributing area "
                    f"{top_inside['contributing_area_km2']:.3f} km². This is a much smaller feature than the "
                    "boundary-crossing basins in the significant-basin table above, but it's the one actually "
                    "within the named place rather than adjacent to it.</p></div>"
                )
            else:
                inside_note = (
                    '<div class="caveat"><strong>None of the local minima fall inside the resolved boundary.</strong> '
                    "Every candidate low point in this analysis -- significant or not -- sits outside the real place "
                    "polygon, just inside the padded analysis box around it. That's a genuine finding about this "
                    "place's terrain (it may sit on a local topographic high, or its boundary may exclude the "
                    "nearby drainage channel/floodplain by design), not a processing error.</div>"
                )

        all_minima_section = f"""
  <section id="sec-allminima">
    <h2>All local minima, unfiltered</h2>
    <p class="section-note">Same DEM, same TTK Morse-Smale complex, but re-run with the persistence-simplification
      threshold dropped to ~0 instead of the auto-computed value above -- so every raw local minimum ({len(all_minima_hotspots)}
      total, not just the {morse_result['num_significant_basins']} that survive noise filtering) gets its own
      contributing area, drainage demand, and risk score. Shown as small dots on the maps above, colored by
      drainage demand. Most of these are shallow/noisy microtopography, not real flood basins -- see the caveat below.</p>
    {inside_note}
    <div class="table-wrap">
      <table>
        <thead><tr><th>Rank</th><th>Location (lat, lon)</th><th>Peak runoff (m&sup3;/s)</th><th>Contrib. area (km&sup2;)</th><th>Priority score</th>{am_boundary_th}{am_fema_th}</tr></thead>
        <tbody>{''.join(am_rows)}</tbody>
      </table>
    </div>
    <p class="section-note" style="margin-top:10px;">Top 15 of {len(all_minima_hotspots)} shown, ranked by drainage demand. Treat this as a map of where
      microtopographic ponding could occur (useful for storm-drain inlet placement), not a second infrastructure
      priority ranking -- most of these have persistence well below the significance threshold used for the
      table above.</p>
  </section>"""

    return f"""<title>{html.escape(place_label)} &mdash; Flood Risk &amp; Drainage Demand</title>
<style>{_CSS}</style>
<div class="page">
  <div class="masthead">
    <p class="eyebrow">Field Report &middot; GeoAI Agent, Morse-Theoretic Terrain Analysis</p>
    <h1>{html.escape(place_label)}</h1>
    <p class="subtitle">Where floodwater actually pools, and where drainage capacity is most exposed</p>
    <div class="meta-line">
      <span>{lat_min:.3f}&ndash;{lat_max:.3f}&deg;N, {abs(lon_max):.3f}&ndash;{abs(lon_min):.3f}&deg;W</span>
      <span>&middot;</span><span>Generated {html.escape(generated_at)}</span>
      <span>&middot;</span><span>Model: {html.escape(model_name)}</span>
    </div>
  </div>

  <div class="stats">{stat_chips}</div>

  {toc_html}

  <section id="sec-agent">
    <h2>How the agent worked</h2>
    <p class="section-note">This run isn't just a fixed pipeline with an LLM writing the final summary --
      real decision points below actually changed this run's own output: which persistence threshold counts
      as "a real basin, not noise" for this specific terrain, and whether the resulting #1 pick holds up
      under a different reasonable weighting choice. A third decision -- the agent reviewing its own
      recommendation against the evidence before it's shown to you -- appears with the recommendation itself,
      further down this page.</p>
    {agent_decision_log_html}
  </section>

  <section id="sec-d8">
    <h2>Does a Morse basin correspond to real drainage? (D8 cross-check)</h2>
    <p class="section-note">The central methodological question underneath everything else in this report:
      is a Morse-persistence-significant basin a hydrologically meaningful feature, or just a topologically
      robust artifact of the Morse-Smale complex? Rather than assert an answer, this cross-checks every
      significant basin against an independent, standard raster-hydrology model (D8 flow direction/
      accumulation) computed separately from the same elevation data.</p>
    {morse_vs_d8_html}
  </section>

  <section id="sec-terrain">
    <h2>The terrain, and where it drains</h2>
    <p class="section-note">Elevation fetched live for this run (see data source above) and analyzed by TTK's Morse-Smale complex on the persistence-simplified field, then ranked by flood_tools.rank_flood_basins -- see the map below for where each basin sits.</p>
    {boundary_section}
  </section>

  {interactive_map_section}

  <section id="sec-shows">
    <h2>What this shows</h2>
    {narrative_html}
  </section>

  <section id="sec-recommendation">
    <h2>Generated recommendation</h2>
    <div class="rec-block">{rec_html}</div>
    {reflection_html}
  </section>

  <section id="sec-hotspots">
    <h2>Ranked hotspots</h2>
    <p class="section-note">{scenario['label']}, 1-hour intensity ({scenario['intensity_mm_per_hr']:.0f} mm/hr, {rainfall_source_label})
      {'&middot; runoff coefficient varies per basin -- see the "Runoff coeff." figure on each card below (land-cover-derived where ' + _land_cover_source_label(land_cover_source) + '/soil data was available, assumed ' + str(flood_result['runoff_coefficient']) + ' elsewhere)' if coefficient_varies else '&middot; runoff coefficient ' + str(flood_result['runoff_coefficient']) + ' (assumed, same for every basin)'}
      -- each basin below shows every fact this run has about it in one place, including how it drains
      (runoff/pipe-capacity comparison), what real-world data corroborates or complicates the finding (FEMA,
      gauges), and its own breakdown of why it scored the way it did.</p>
    {shared_notes_html}
    <h3 style="font-family:var(--serif);font-size:16px;margin:22px 0 4px;">How the Flood Priority Score is computed</h3>
    <p class="section-note">The percentage each factor contributes to every basin's priority score below -- each
      card's own "This basin's own factor scores" then shows how that specific basin scored against these
      weights. Called a <strong>Flood Priority Score</strong>, deliberately not a "flood risk score" or
      "probability of flooding": it is a weighted sum of five normalized heuristic indicators
      (<code>risk_score</code> internally), not a statistically fitted or historically validated risk model --
      see the caveat immediately below.</p>
    <div class="weights">{weight_bars}</div>
    <div class="caveat">
      <strong>This is a prioritization tool, not a validated flood-risk prediction.</strong> {weighting_method_note}
      Apply a different, equally legitimate weighting to the exact same underlying per-factor scores below and
      these same basins would rank differently. Treat the Flood Priority Score as a transparent, reproducible way
      to combine five heuristic signals into one ordering for deciding what to look at first -- not as a
      probability, a percentage chance of flooding, or an objectively "correct" ranking of actual flood risk.
    </div>
    <details class="hotspot-detail">
      <summary>Exact formula for each per-basin factor score</summary>
      <div class="hotspot-detail-body">
        <dl>
          <dt>Topological</dt><dd>persistence_m &divide; (this study area's total elevation range), capped at
            1.0. A basin with no deeper point to compare against (the global sink, persistence_m is null/
            infinite) automatically scores 1.0 -- <strong>this is a scoring-function CONVENTION for handling an
            undefined value, not a Morse-theoretic derivation</strong>: Morse theory identifies the global sink
            as a critical point with no death time, full stop; assigning that case the maximum possible score
            is this project's own choice of how to normalize an undefined quantity, the same way contributing-
            area's own normalization (below) is a choice, not a law of nature. Worded precisely: "Morse theory
            identified this point as the global sink" is a fact about the terrain; "therefore it scores 1.0 on
            topological significance" is a fact about this scoring function.</dd>
          <dt>Contributing area</dt><dd>this basin's own contributing_area_km2 &divide; the LARGEST
            contributing_area_km2 among this run's significant basins. By construction, whichever ONE basin
            drains the most area always scores exactly 1.0 -- every other basin's score is relative to that
            one basin, not a fixed area threshold.</dd>
          <dt>Rainfall</dt><dd>the selected design storm's intensity &divide; the largest intensity available
            across all six return periods this run fetched. This is the SAME number for every basin in a single
            run (one storm applies uniformly across the whole study area) -- it is not basin-specific.</dd>
          <dt>Slope</dt><dd>1 minus the min-max-normalized ground slope <strong>at this basin's own pour point
            only</strong> (a single grid cell at its exact low point, not averaged across its catchment),
            normalized against the slope range across the whole analysis grid. Flatter ground scores higher
            (ponds more readily than a steep channel).</dd>
          <dt>Exposure</dt><dd>{exposure_score_formula_note}</dd>
        </dl>
      </div>
    </details>
    {hotspot_cards_html}
  </section>
  {all_minima_section}

  <section id="sec-limitations">
    <h2>Limitations &amp; uncertainty</h2>
    <p class="section-note">What this analysis does NOT establish, and where its numbers can be wrong for
      reasons that have nothing to do with a coding bug -- read this before treating any figure above as more
      certain than it is.</p>
    {limitations_html}
  </section>

  <section id="sec-pipeline">
    <h2>Analysis pipeline &amp; data sources</h2>
    <p class="section-note">What this specific run actually did, stage by stage, and what real data backed each
      one -- the maps below draw straight from this run's own ranked basins and cross-check results, not a
      generic illustration.</p>
    <h3>Data sources used in this run</h3>
    <p class="section-note">Every row below also has a matching toggleable layer on the interactive map (elevation
      is on by default; everything else starts off -- switch it on to see the raw data behind any row here).</p>
    {data_sources_table_html}
    {pipeline_narrative_html}
    <details class="appendix">
      <summary>Final prompt sent to the model (system instruction + user context)</summary>
      <div class="appendix-body">
        <div class="appendix-block"><h4>System instruction</h4><pre class="prompt">{html.escape(system_prompt)}</pre></div>
        <div class="appendix-block"><h4>User context</h4><pre class="prompt">{html.escape(user_context)}</pre></div>
      </div>
    </details>
  </section>

  <footer>
    Pipeline: geocode{'+boundary lookup' if boundary_available else ''} &rarr; build_real_world_dataset(elevation_source=&quot;auto&quot;) &middot;
    analyze_scalar_field_ttk (persistence threshold {morse_result['persistence_threshold']:.2f}, {morse_result['num_significant_basins']} of {morse_result['num_raw_minima']} raw minima kept) &middot;
    flood_tools.rank_flood_basins &middot; LLMClient(model=&quot;{html.escape(model_name)}&quot;)
  </footer>
</div>
"""
