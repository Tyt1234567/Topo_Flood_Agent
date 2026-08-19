"""
Self-contained HTML flood-risk report generator.

Turns the output of the real-data -> TTK -> flood_tools pipeline into a
single, shareable HTML file: an embedded static matplotlib overview map
(elevation + slope with ranked hotspots, works fully offline), an
interactive Leaflet web map (real OpenStreetMap basemap + a toggleable
elevation/risk heatmap layer, loaded from CDN -- this file is opened
directly by the user, not sandboxed the way a Claude Artifact is, so a
live basemap is fine here), the ranked-hotspot table, a data-driven
narrative (every claim is computed from the actual result, not fixed
prose naming one specific place), the LLM-generated recommendation, and a
disclosure of the exact prompt sent to the model. Used by web_app's
one-box "just name a place" flow (app.py's /api/agent/analyze_stream) so
the deliverable at the end of a run is a real file, not just numbers in a
chat.

No web framework dependency -- pure functions over the same objects the
rest of geoai_agent already passes around (ScalarField, the dict shapes
returned by ttk_engine.analyze_scalar_field_ttk and
flood_tools.rank_flood_basins, and boundary.geocode_place_boundary's
shapely polygon / GeoJSON geometry).
"""
from __future__ import annotations

import base64
import html
import io
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import numpy as np

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


def elevation_hillshade_geojson(field: ScalarField, catchment_grid: list[list[int]] | None = None) -> dict:
    """Elevation as a terrain-colored hillshade GeoJSON layer (one polygon
    per grid cell, as in visualize.field_to_geojson), instead of a
    value-ramp heatmap. A bright-yellow-to-purple viridis ramp over
    elevation reads as a risk/intensity heatmap at a glance -- exactly the
    wrong association for "this is just the terrain's shape" -- so this
    blends relief shading (a fixed sun angle, via LightSource.hillshade)
    with matplotlib's "terrain" colormap (green lowland/brown highland,
    the same palette render_flood_map_png's static image uses), the way
    GIS basemaps conventionally render elevation. Each feature also
    reports the raw elevation `value` so a caller can build its own
    min/max legend.

    `catchment_grid`, if given (ttk_engine's per-run "descending_manifold"
    grid, same (ny, nx) shape as `field`), is attached to each cell as
    `catchment_id` -- which basin's descending manifold (the set of cells
    that drain into one specific local minimum) that cell belongs to. A
    caller can then highlight one minimum's contributing area on click by
    filtering features to matching catchment_id, without needing a second
    geometry/segmentation pass of its own."""
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


def _persistence_to_size(persistence_m: float | None, max_persistence: float, min_size: float, max_size: float) -> float:
    """Maps a basin's topological persistence to a matplotlib/Leaflet
    marker size -- deeper (more persistent) depressions draw bigger dots.
    A global sink (persistence_m is None -- no deeper point exists to
    compare it against) is treated as maximally persistent."""
    if persistence_m is None:
        return max_size
    if max_persistence <= 0:
        return min_size
    frac = max(0.0, min(1.0, persistence_m / max_persistence))
    return min_size + frac * (max_size - min_size)


def _max_finite_persistence(hotspots: list[dict]) -> float:
    finite = [h["persistence_m"] for h in hotspots if h["persistence_m"] is not None]
    return max(finite) if finite else 1.0


def _polygon_clip_patch(polygon, ax) -> mpatches.PathPatch:
    """A matplotlib PathPatch tracing `polygon`'s exterior ring(s) (a
    shapely Polygon or MultiPolygon), pre-transformed to `ax`'s data
    coordinates so it's ready to pass straight to Artist.set_clip_path
    (which needs a single already-transformed Patch, not a bare Path plus
    a separate transform kwarg -- that combination raises "Invalid
    arguments to set_clip_path"). Interior holes are ignored -- real
    administrative boundaries with holes are rare enough, and a
    slightly-too-generous clip at a hole is a much smaller error than not
    clipping at all."""
    geoms = polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for geom in geoms:
        coords = list(geom.exterior.coords)
        if len(coords) < 3:
            continue
        vertices.extend(coords)
        codes.append(MplPath.MOVETO)
        codes.extend([MplPath.LINETO] * (len(coords) - 2))
        codes.append(MplPath.CLOSEPOLY)
    path = MplPath(vertices, codes)
    patch = mpatches.PathPatch(path, facecolor="none", edgecolor="none")
    patch.set_transform(ax.transData)
    return patch


def render_flood_map_png(
    elevation: ScalarField, slope: ScalarField | None, hotspots: list[dict],
    boundary_polygon=None, all_minima_hotspots: list[dict] | None = None,
) -> bytes:
    """Elevation (+ slope, if available) heatmap with ranked flood hotspots
    marked, as PNG bytes. If `boundary_polygon` (a shapely geometry) is
    given, both heatmaps are clipped to it -- the analysis grid is still
    the full padded rectangle underneath (TTK needs a complete rectangular
    domain), this only restricts what's drawn to the real place shape.
    `all_minima_hotspots` (every raw local minimum, not just the
    persistence-significant ones -- see flood_tools.rank_flood_basins run
    on a near-zero threshold) are drawn as small dots on the elevation
    panel, colored by drainage demand (peak runoff), so the full ponding
    picture is visible even where it didn't survive noise filtering."""
    has_slope = slope is not None
    fig, axes = plt.subplots(1, 2 if has_slope else 1, figsize=(14 if has_slope else 8, 6.4))
    ax_elev = axes[0] if has_slope else axes

    # Hillshade relief blended with a muted terrain colormap (LightSource.
    # shade) instead of a flat color-by-elevation fill -- reads as terrain,
    # not a heat/risk map, and the relief shading makes real depressions
    # visually legible even on near-flat DEMs.
    dx_m, dy_m = _hillshade_spacing_meters(elevation)
    ls = LightSource(azdeg=_HILLSHADE_AZDEG, altdeg=_HILLSHADE_ALTDEG)
    elev_min, elev_max = float(elevation.values.min()), float(elevation.values.max())
    rgb = ls.shade(elevation.values, cmap=plt.cm.terrain, blend_mode="soft",
                    vert_exag=_HILLSHADE_VERT_EXAG, dx=dx_m, dy=dy_m,
                    vmin=elev_min, vmax=elev_max)
    extent = [float(elevation.xs.min()), float(elevation.xs.max()), float(elevation.ys.min()), float(elevation.ys.max())]
    im = ax_elev.imshow(rgb, extent=extent, origin="lower", aspect="auto", zorder=1)
    if boundary_polygon is not None:
        im.set_clip_path(_polygon_clip_patch(boundary_polygon, ax_elev))
        bx, by = _polygon_boundary_xy(boundary_polygon)
        ax_elev.plot(bx, by, color="#1c2620", linewidth=1.6, zorder=6)
    # im itself is an RGB image, not a scalar-mappable -- a separate
    # ScalarMappable proxy is what lets a normal elevation colorbar still
    # appear alongside the hillshade-blended render.
    sm = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(elev_min, elev_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_elev, fraction=0.046, pad=0.04)
    cbar.set_label("Elevation (m)")

    if all_minima_hotspots:
        runoff = np.array([h["estimated_peak_runoff_m3s"] for h in all_minima_hotspots])
        max_runoff = max(runoff.max(), 1e-6)
        max_persist_all = _max_finite_persistence(all_minima_hotspots)
        mx = [h["coords"][0] for h in all_minima_hotspots]
        my = [h["coords"][1] for h in all_minima_hotspots]
        sizes_all = [_persistence_to_size(h["persistence_m"], max_persist_all, 10, 140) for h in all_minima_hotspots]
        within = [h.get("within_boundary") for h in all_minima_hotspots]
        edge_colors = ["#1c2620" if w else "none" for w in within] if boundary_polygon is not None else "#1c2620"
        sc = ax_elev.scatter(mx, my, c=runoff, s=sizes_all, cmap="plasma", norm=plt.Normalize(0, max_runoff),
                              edgecolor=edge_colors, linewidth=0.8, alpha=0.85, zorder=4)
        cbar3 = fig.colorbar(sc, ax=ax_elev, fraction=0.046, pad=0.10)
        cbar3.set_label("All local minima -- drainage demand (m³/s)")

    max_persist = _max_finite_persistence(hotspots)
    for h in hotspots:
        x, y = h["coords"]
        size = _persistence_to_size(h["persistence_m"], max_persist, 90, 520)
        ax_elev.scatter([x], [y], s=size, facecolor="none", edgecolor="red", linewidth=2.2, zorder=5)
        ax_elev.annotate(f"#{h['rank']}", (x, y), textcoords="offset points", xytext=(8, 8),
                          fontsize=12, fontweight="bold", color="red")
    title = "Elevation (hillshade) -- flood hotspots ranked by risk, sized by persistence"
    if all_minima_hotspots:
        title += "\nsmall dots = every local minimum, filled = inside boundary" if boundary_polygon is not None else "\nsmall dots = every local minimum"
    ax_elev.set_title(title)
    ax_elev.set_xlabel("Longitude")
    ax_elev.set_ylabel("Latitude")
    ax_elev.set_aspect("equal")

    if hotspots:
        legend_fracs = [0.15, 0.5, 1.0]
        handles = []
        for frac in legend_fracs:
            s = _persistence_to_size(frac * max_persist, max_persist, 90, 520)
            marker_size = 2.0 * (s / np.pi) ** 0.5
            label = f"~{frac * max_persist:.1f} m" if max_persist > 0 else "n/a"
            handles.append(Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none",
                                   markeredgecolor="red", markeredgewidth=2, markersize=marker_size, label=label))
        ax_elev.legend(handles=handles, loc="lower left", fontsize=7.5, title="Dot size = persistence",
                        title_fontsize=7.5, framealpha=0.85, labelspacing=1.1)

    if has_slope:
        ax_slope = axes[1]
        im2 = ax_slope.pcolormesh(slope.xs, slope.ys, slope.values, cmap="inferno", shading="auto")
        if boundary_polygon is not None:
            im2.set_clip_path(_polygon_clip_patch(boundary_polygon, ax_slope))
            bx, by = _polygon_boundary_xy(boundary_polygon)
            ax_slope.plot(bx, by, color="#e7ece5", linewidth=1.6, zorder=6)
        cbar2 = fig.colorbar(im2, ax=ax_slope, fraction=0.046, pad=0.04)
        cbar2.set_label("Terrain steepness (relative units)")
        for h in hotspots:
            x, y = h["coords"]
            ax_slope.scatter([x], [y], s=120, facecolor="none", edgecolor="cyan", linewidth=2, zorder=5)
            ax_slope.annotate(f"#{h['rank']}", (x, y), textcoords="offset points", xytext=(8, 8),
                               fontsize=12, fontweight="bold", color="cyan")
        ax_slope.set_title("Slope (gradient magnitude) -- brighter = faster runoff concentration")
        ax_slope.set_xlabel("Longitude")
        ax_slope.set_ylabel("Latitude")
        ax_slope.set_aspect("equal")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return buf.getvalue()


def _polygon_boundary_xy(polygon) -> tuple[list[float], list[float]]:
    """Flattened x/y coordinate lists (with None gaps between separate
    rings, matplotlib's convention for drawing disjoint line segments in
    one plot() call) tracing every exterior ring of a Polygon/MultiPolygon."""
    geoms = polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]
    xs: list[float] = []
    ys: list[float] = []
    for i, geom in enumerate(geoms):
        if i > 0:
            xs.append(None)
            ys.append(None)
        cx, cy = geom.exterior.xy
        xs.extend(cx)
        ys.extend(cy)
    return xs, ys


def all_minima_geojson(hotspots: list[dict]) -> dict:
    """Point features for EVERY local minimum (not just the
    persistence-significant ones), colored by drainage demand
    (estimated_peak_runoff_m3s) rather than risk_score -- a separate ramp
    from field_to_geojson/flood_hotspots_to_geojson's so the two layers
    aren't visually confused for the same thing when both are on. Includes
    persistence_m so callers can size markers by persistence the same way
    the static map does."""
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
    catchment_grid: list[list[int]] | None = None,
) -> str:
    """An embeddable (not standalone) Leaflet map: real OSM basemap, a
    toggleable terrain-colored hillshade elevation layer with a color
    legend, the boundary polygon outline (if resolved), and (if given) a
    toggleable layer of every raw local minimum, sized by persistence and
    colored by drainage demand, filterable by a persistence-threshold
    slider and click-to-highlight-catchment (via `catchment_grid`, ttk_
    engine's descending-manifold segmentation for the same run
    `all_minima_hotspots` came from). There is no separate "significant
    hotspots" marker layer on this map -- that ranking still drives the
    report's own hotspot table elsewhere, but every raw minimum, filtered
    live by the slider, supersedes it as a map layer. CDN-loaded -- fine
    here since this report is a real file the user opens/downloads,
    unlike a sandboxed Claude Artifact."""
    lat_min, lat_max, lon_min, lon_max = bounds
    field_geojson = elevation_hillshade_geojson(elevation, catchment_grid=catchment_grid)
    legend_stops = terrain_legend_stops(elevation)
    boundary_js = f"var boundaryData = {json.dumps(boundary_geojson)};" if boundary_geojson else "var boundaryData = null;"
    all_minima_js = (
        f"var allMinimaData = {json.dumps(all_minima_geojson(all_minima_hotspots))};"
        if all_minima_hotspots else "var allMinimaData = null;"
    )

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
      // bigger dots), matching the static map's encoding.
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
        overlays['Boundary'] = L.geoJSON(boundaryData, {{ style: {{ fill: false, color: '#a8492c', weight: 2.5 }} }}).addTo(map);
      }}

      var catchmentLayer = null;
      function highlightCatchment(catchmentId) {{
        if (catchmentLayer) {{ map.removeLayer(catchmentLayer); catchmentLayer = null; }}
        document.getElementById('webmap-catchment-clear').style.display = (catchmentId === null || catchmentId === undefined) ? 'none' : 'block';
        if (catchmentId === null || catchmentId === undefined) return;
        var cells = fieldData.features.filter(function(f) {{ return f.properties.catchment_id === catchmentId; }});
        catchmentLayer = L.geoJSON({{ type: 'FeatureCollection', features: cells }}, {{
          style: {{ fillColor: '#2b6cb0', color: '#12335c', weight: 1, fillOpacity: 0.55 }}, interactive: false,
        }}).addTo(map);
        catchmentLayer.bringToFront();
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

      var legend = L.control({{position: 'bottomleft'}});
      legend.onAdd = function() {{
        var div = L.DomUtil.create('div');
        div.style.cssText = 'background:var(--paper-raised, #fff);color:var(--ink,#222);' +
          'padding:8px 10px;font-size:12px;line-height:1.5;border-radius:4px;' +
          'box-shadow:0 1px 4px rgba(0,0,0,0.35);max-width:220px;';
        var gradientCss = elevationLegend.map(function(s, i) {{ return s.color + ' ' + (i / (elevationLegend.length - 1) * 100) + '%'; }}).join(', ');
        var gradientHtml = elevationLegend.length ?
          '<div><strong>Elevation</strong></div>' +
          '<div style="width:160px;height:10px;border-radius:2px;margin:3px 0;background:linear-gradient(90deg, ' + gradientCss + ');"></div>' +
          '<div style="display:flex;justify-content:space-between;font-size:10.5px;"><span>' +
          elevationLegend[0].value.toFixed(0) + ' m</span><span>' + elevationLegend[elevationLegend.length - 1].value.toFixed(0) + ' m</span></div>' : '';
        div.innerHTML = gradientHtml +
          '<div style="margin-top:6px;"><strong>Dot size</strong> = persistence (bigger = deeper basin)</div>' +
          (allMinimaData ? '<div><strong>Dot fill</strong> = drainage demand</div>' : '') +
          (allMinimaData ? '<div>Click a dot to highlight its catchment area</div>' : '') +
          '<div id="webmap-catchment-clear" style="display:none;margin-top:6px;"><a href="#" id="webmap-catchment-clear-link">Clear catchment highlight</a></div>';
        return div;
      }};
      legend.addTo(map);
      L.DomEvent.disableClickPropagation(legend.getContainer());
      var clearLink = document.getElementById('webmap-catchment-clear-link');
      if (clearLink) clearLink.addEventListener('click', function(e) {{ e.preventDefault(); highlightCatchment(null); }});

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


def _point_link(lat: float, lon: float, catchment_id: int | None, label: str, embed_interactive_map: bool) -> str:
    """A clickable mention of a specific point in the report's prose/tables
    -- clicking it pans the map to that lat/lon and, if the point has a
    catchment_id, highlights its contributing area, same as clicking the
    dot on the map itself. `embed_interactive_map` picks which map to talk
    to: True means this document has its own embedded Leaflet map (the
    standalone downloadable report -- see _webmap_fragment's
    window.focusMapPoint), False means this fragment is loaded inside
    web_app's report iframe and the real map lives in the parent page
    (main.js's focusMapPoint); the iframe is same-origin (srcdoc) so
    calling straight into `parent` works without postMessage."""
    target = "" if embed_interactive_map else "parent."
    cid = "null" if catchment_id is None else str(int(catchment_id))
    return (
        f'<a href="javascript:void(0)" class="point-link" '
        f'onclick="{target}focusMapPoint({lat:.6f}, {lon:.6f}, {cid}); return false;">{label}</a>'
    )


def _narrative_html(
    hotspots: list[dict], grid_shape: tuple[int, int], exposure_provided: bool,
    boundary_available: bool, embed_interactive_map: bool,
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
    coord_link = _point_link(lat, lon, top.get("catchment_id"), f"{lat:.4f}°N, {lon:.4f}°E", embed_interactive_map)
    parts = [
        f"<p><strong>Rank 1</strong> ({coord_link}) is the highest-priority basin: "
        f"risk score <strong>{top['risk_score']:.3f}</strong>, contributing area "
        f"<strong>{top['contributing_area_km2']:.2f} km²</strong>, estimated peak runoff "
        f"<strong>{top['estimated_peak_runoff_m3s']:.1f} m³/s</strong> under the design storm used below."
    ]
    if top["persistence_m"] is None:
        parts.append(" It is the single deepest depression in the whole analyzed area (the global sink -- "
                      "no deeper point exists to compare it against).")
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
    lead = "".join(parts) + "</p>"

    out = [f'<div class="callout"><h3>Top priority</h3>{lead}</div>']

    edge = _boundary_hotspots(hotspots, grid_shape)
    if edge:
        ranks = ", ".join(f"#{h['rank']}" for h in edge)
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
            ranks = ", ".join(f"#{h['rank']}" for h in outside)
            out.append(
                f'<p class="section-note">Basin{"es" if len(outside) > 1 else ""} {ranks} '
                f"{'fall' if len(outside) > 1 else 'falls'} outside the resolved place boundary shown on the map "
                "below (shaded/outlined area) -- included because the analysis grid is a padded rectangle around "
                "the boundary, not the polygon itself, but worth knowing before prioritizing them as \"this place's\" problem.</p>"
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
section { margin: 44px 0; }
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
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--water-tint); }
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
"""


def render_flood_report_html(
    *,
    place_label: str,
    bounds: tuple[float, float, float, float],
    elevation: ScalarField,
    slope: ScalarField | None,
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
    all_minima_hotspots: list[dict] | None = None,
    catchment_grid: list[list[int]] | None = None,
    embed_interactive_map: bool = True,
) -> str:
    """Assembles the full standalone HTML report. Returns the document as
    a string (title/style/body only -- no <html>/<head> wrapper needed by
    callers that embed it directly, but it's also a complete, valid
    standalone page as-is when saved to a .html file). If
    `boundary_polygon`/`boundary_geojson` are given (see
    boundary.geocode_place_boundary), the static map is clipped to the
    real place shape and the web map draws its outline; hotspots are
    tagged "within_boundary" by the caller beforehand
    (boundary.tag_hotspots_within_boundary). `all_minima_hotspots`, if
    given, is EVERY raw local minimum (not just the persistence-significant
    ones in flood_result) ranked the same way -- see
    ttk_engine.analyze_scalar_field_ttk(..., persistence_threshold=~0) --
    drawn on both maps colored by drainage demand and covered in its own
    section, since most real ponding-relevant terrain doesn't survive
    noise filtering into the "significant" list above.

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

    png_bytes = render_flood_map_png(
        elevation, slope, hotspots, boundary_polygon=boundary_polygon,
        all_minima_hotspots=all_minima_hotspots,
    )
    img_b64 = base64.b64encode(png_bytes).decode("ascii")
    webmap_html = _webmap_fragment(elevation, bounds, boundary_geojson, all_minima_hotspots, catchment_grid) if embed_interactive_map else ""

    rows = []
    for h in hotspots:
        persistence = "global sink (infinite)" if h["persistence_m"] is None else f"{h['persistence_m']:.1f} m"
        boundary_cell = ""
        if boundary_available:
            boundary_cell = f"<td>{'inside' if h.get('within_boundary') else 'outside'}</td>"
        coord_link = _point_link(
            h["coords"][1], h["coords"][0], h.get("catchment_id"),
            f"{h['coords'][1]:.4f}, {h['coords'][0]:.4f}", embed_interactive_map,
        )
        rows.append(f"""
      <tr>
        <td>{h['rank']}</td>
        <td class="mono">{coord_link}</td>
        <td class="num">{h['elevation_m']:.1f}</td>
        <td class="num">{h['contributing_area_km2']:.2f}</td>
        <td class="num strong">{h['estimated_peak_runoff_m3s']:.1f}</td>
        <td class="num">{h['risk_score']:.3f}</td>
        <td class="mono">{persistence}</td>
        {boundary_cell}
      </tr>""")
    hotspot_rows = "".join(rows)
    boundary_th = "<th>Boundary</th>" if boundary_available else ""

    weight_bars = "".join(
        f'<div class="wbar-row"><span class="wbar-label">{k.replace("_", " ")}</span>'
        f'<div class="wbar-track"><div class="wbar-fill" style="width:{v * 100:.0f}%"></div></div>'
        f'<span class="wbar-val">{v:.0%}</span></div>'
        for k, v in weights.items()
    )

    stat_chips = f"""
  <div class="stat"><span class="stat-label">Data source</span><span class="stat-value">{html.escape(elevation_source)}</span></div>
  <div class="stat"><span class="stat-label">Analysis grid</span><span class="stat-value">{elevation.shape[1]}&times;{elevation.shape[0]}</span></div>
  <div class="stat"><span class="stat-label">Elevation range</span><span class="stat-value">{float(elevation.values.min()):.1f}&ndash;{float(elevation.values.max()):.1f} m</span></div>
  <div class="stat"><span class="stat-label">Significant basins</span><span class="stat-value">{morse_result['num_significant_basins']} <span style="font-weight:400;color:var(--ink-muted);font-size:12.5px;">of {morse_result['num_raw_minima']} raw minima</span></span></div>
  <div class="stat"><span class="stat-label">Design storm</span><span class="stat-value">{scenario['label']} &middot; {scenario['intensity_mm_per_hr']:.0f} mm/hr</span></div>
  <div class="stat"><span class="stat-label">Boundary source</span><span class="stat-value">{'OpenStreetMap (Nominatim)' if boundary_available else 'padded box around geocoded point'}</span></div>
  <div class="stat"><span class="stat-label">Exposure data</span><span class="stat-value">{'user-supplied' if exposure_provided else 'not available (neutral default)'}</span></div>
"""

    rec_html = _md_to_html(recommendation) if recommendation else "<p><em>Not generated for this run.</em></p>"
    narrative_html = _narrative_html(hotspots, elevation.shape, exposure_provided, boundary_available, embed_interactive_map)

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
  <section>
    <h2>Interactive map</h2>
    <p class="section-note">Real OpenStreetMap basemap with the elevation field as a toggleable hillshade layer{', the resolved boundary outlined in red,' if boundary_available else ''} and each hotspot as a clickable marker.</p>
    {webmap_html}
  </section>"""

    all_minima_section = ""
    if all_minima_hotspots:
        by_runoff = sorted(all_minima_hotspots, key=lambda h: h["estimated_peak_runoff_m3s"], reverse=True)
        top_n = by_runoff[:15]
        am_boundary_th = "<th>Boundary</th>" if boundary_available else ""
        am_rows = []
        for h in top_n:
            am_boundary_cell = ""
            if boundary_available:
                am_boundary_cell = f"<td>{'inside' if h.get('within_boundary') else 'outside'}</td>"
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
  <section>
    <h2>All local minima, unfiltered</h2>
    <p class="section-note">Same DEM, same TTK Morse-Smale complex, but re-run with the persistence-simplification
      threshold dropped to ~0 instead of the auto-computed value above -- so every raw local minimum ({len(all_minima_hotspots)}
      total, not just the {morse_result['num_significant_basins']} that survive noise filtering) gets its own
      contributing area, drainage demand, and risk score. Shown as small dots on the maps above, colored by
      drainage demand. Most of these are shallow/noisy microtopography, not real flood basins -- see the caveat below.</p>
    {inside_note}
    <div class="table-wrap">
      <table>
        <thead><tr><th>Rank</th><th>Location (lat, lon)</th><th>Peak runoff (m&sup3;/s)</th><th>Contrib. area (km&sup2;)</th><th>Risk score</th>{am_boundary_th}</tr></thead>
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

  <section>
    <h2>The terrain, and where it drains</h2>
    <p class="section-note">Elevation fetched live for this run (see data source above) and analyzed by TTK's Morse-Smale complex on the persistence-simplified field, then ranked by flood_tools.rank_flood_basins. Circles mark the basins that survived simplification, sized/ranked by risk score.</p>
    {boundary_section}
    <figure>
      <img src="data:image/png;base64,{img_b64}" alt="Elevation and slope maps with ranked flood hotspots marked">
    </figure>
  </section>

  {interactive_map_section}

  <section>
    <h2>Ranked hotspots</h2>
    <p class="section-note">{scenario['label']} ({scenario['intensity_mm_per_hr']:.0f} mm/hr, illustrative design intensity) &middot; runoff coefficient {flood_result['runoff_coefficient']}</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Rank</th><th>Location (lat, lon)</th><th>Elev. (m)</th><th>Contrib. area (km&sup2;)</th><th>Peak runoff (m&sup3;/s)</th><th>Risk score</th><th>Persistence</th>{boundary_th}</tr></thead>
        <tbody>{hotspot_rows}</tbody>
      </table>
    </div>
    <h3 style="font-family:var(--serif);font-size:16px;margin:22px 0 4px;">Risk-scoring weights used</h3>
    <div class="weights">{weight_bars}</div>
  </section>
  {all_minima_section}
  <section>
    <h2>What this shows</h2>
    {narrative_html}
  </section>

  <section>
    <h2>Generated recommendation</h2>
    <div class="rec-block">{rec_html}</div>
  </section>

  <section>
    <h2>How this was produced</h2>
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
