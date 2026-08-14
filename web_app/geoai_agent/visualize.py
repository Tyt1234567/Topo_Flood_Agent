"""
Self-contained Leaflet map for discrete-Morse analysis results, over one or
more scalar-field layers at once.

No new Python dependency: Leaflet is loaded from its CDN inside the
generated HTML file, which embeds the layer data as GeoJSON and needs
nothing but a browser to open. Two coordinate modes:

- Geographic (RealWorldDataset): field x/y are lon/lat, drawn over a real
  OpenStreetMap tile basemap.
- Local (synthetic GeoDataset): field x/y are arbitrary town-local units,
  drawn with L.CRS.Simple (no basemap, just the data's own coordinate
  plane).

Each layer contributes two toggleable overlays: a colored grid-cell heatmap
of the field values, and markers for its significant critical points
(minima, maxima, saddles), via Leaflet's layer control.
"""

from __future__ import annotations

import json

from geoai_agent.morse_topology import ScalarField

_LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
_LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"

# Viridis-ish anchor colors: perceptually ordered, colorblind-safe, and
# reads sensibly in grayscale printouts.
_RAMP_STOPS = [
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
]

_CRITICAL_COLORS = {"minimum": "#2b83ba", "maximum": "#d7191c", "saddle": "#e6a817"}


def _ramp_color(t: float) -> str:
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(_RAMP_STOPS, _RAMP_STOPS[1:]):
        if t0 <= t <= t1:
            frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            r = round(c0[0] + frac * (c1[0] - c0[0]))
            g = round(c0[1] + frac * (c1[1] - c0[1]))
            b = round(c0[2] + frac * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return f"#{_RAMP_STOPS[-1][1][0]:02x}{_RAMP_STOPS[-1][1][1]:02x}{_RAMP_STOPS[-1][1][2]:02x}"


def _cell_edges(coords) -> list[float]:
    """Midpoint boundaries between consecutive grid coordinates, extended
    by a half-step at each end, so every node gets a Voronoi-like cell."""
    n = len(coords)
    edges = [0.0] * (n + 1)
    edges[0] = coords[0] - (coords[1] - coords[0]) / 2
    edges[n] = coords[-1] + (coords[-1] - coords[-2]) / 2
    for k in range(1, n):
        edges[k] = (coords[k - 1] + coords[k]) / 2
    return edges


def field_to_geojson(field: ScalarField) -> dict:
    """One rectangular polygon feature per grid node, colored by value.
    GeoJSON coordinates are always [x, y] (lon, lat for geographic data),
    which matches ScalarField's xs/ys convention directly."""
    ny, nx = field.shape
    x_edges = _cell_edges(list(field.xs))
    y_edges = _cell_edges(list(field.ys))
    vmin, vmax = float(field.values.min()), float(field.values.max())
    span = (vmax - vmin) or 1.0

    features = []
    for i in range(ny):
        for j in range(nx):
            value = float(field.values[i, j])
            t = (value - vmin) / span
            ring = [
                [x_edges[j], y_edges[i]],
                [x_edges[j + 1], y_edges[i]],
                [x_edges[j + 1], y_edges[i + 1]],
                [x_edges[j], y_edges[i + 1]],
                [x_edges[j], y_edges[i]],
            ]
            features.append({
                "type": "Feature",
                "properties": {"value": value, "color": _ramp_color(t)},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })
    return {"type": "FeatureCollection", "features": features}


def critical_points_to_geojson(result: dict, significant_only: bool = True) -> dict:
    """Point features for a field's critical points (minima/maxima/saddles)
    from an analyze_scalar_field() result dict."""
    features = []
    for kind in ("minimum", "maximum", "saddle"):
        for entry in result["critical_points"][kind]:
            if significant_only and entry.get("significant") is False:
                continue
            x, y = entry["coords"]
            features.append({
                "type": "Feature",
                "properties": {
                    "kind": kind,
                    "value": entry["value"],
                    "persistence": entry["persistence"],
                    "significant": entry.get("significant"),
                    "color": _CRITICAL_COLORS[kind],
                },
                "geometry": {"type": "Point", "coordinates": [x, y]},
            })
    return {"type": "FeatureCollection", "features": features}


def flood_hotspots_to_geojson(hotspots: list[dict]) -> dict:
    """Point features for a flood_tools.rank_flood_basins() result's
    "hotspots" list, colored by risk_score (same viridis-ish ramp as
    field_to_geojson's heatmap, so a reader doesn't have to learn a second
    color scale)."""
    features = []
    for h in hotspots:
        x, y = h["coords"]
        features.append({
            "type": "Feature",
            "properties": {
                "rank": h["rank"],
                "risk_score": h["risk_score"],
                "contributing_area_km2": h["contributing_area_km2"],
                "estimated_peak_runoff_m3s": h["estimated_peak_runoff_m3s"],
                "persistence_m": h["persistence_m"],
                "color": _ramp_color(h["risk_score"]),
            },
            "geometry": {"type": "Point", "coordinates": [x, y]},
        })
    return {"type": "FeatureCollection", "features": features}


def render_leaflet_map(
    fields: dict[str, ScalarField],
    results: dict[str, dict],
    is_geographic: bool,
    output_path: str,
    title: str = "Scalar field topology",
) -> str:
    """Writes a self-contained Leaflet HTML map with one heatmap + one
    critical-points overlay per layer in `fields`, toggleable via a layer
    control. `results` maps the same layer names to their
    analyze_scalar_field() output. Returns output_path."""
    layer_names = list(fields)
    if not layer_names:
        raise ValueError("no layers to visualize")

    field_geojson = {name: field_to_geojson(fields[name]) for name in layer_names}
    points_geojson = {name: critical_points_to_geojson(results[name]) for name in layer_names}

    all_xs = [x for f in fields.values() for x in (float(f.xs.min()), float(f.xs.max()))]
    all_ys = [y for f in fields.values() for y in (float(f.ys.min()), float(f.ys.max()))]
    bounds = [[min(all_ys), min(all_xs)], [max(all_ys), max(all_xs)]]  # [[lat,lon],[lat,lon]]

    basemap_js = (
        """
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 19,
        }).addTo(map);
        """
        if is_geographic
        else ""
    )
    map_init_js = (
        f"var map = L.map('map').fitBounds({json.dumps(bounds)});"
        if is_geographic
        else f"var map = L.map('map', {{crs: L.CRS.Simple}}).fitBounds({json.dumps(bounds)});"
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="{_LEAFLET_CSS}" />
<style>
  html, body {{ margin: 0; height: 100%; font-family: sans-serif; }}
  #map {{ height: 100%; }}
  .legend {{
    background: white; padding: 8px 10px; line-height: 1.4; font-size: 13px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.4); border-radius: 4px;
  }}
  .legend .swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; border-radius: 50%; }}
</style>
</head>
<body>
<div id="map"></div>
<script src="{_LEAFLET_JS}"></script>
<script>
var fieldData = {json.dumps(field_geojson)};
var pointData = {json.dumps(points_geojson)};
var layerNames = {json.dumps(layer_names)};

{map_init_js}
{basemap_js}

var overlays = {{}};
layerNames.forEach(function(name, idx) {{
  var heat = L.geoJSON(fieldData[name], {{
    style: function(feature) {{
      return {{ fillColor: feature.properties.color, color: feature.properties.color,
                weight: 0, fillOpacity: 0.65 }};
    }}
  }});
  var points = L.geoJSON(pointData[name], {{
    pointToLayer: function(feature, latlng) {{
      return L.circleMarker(latlng, {{
        radius: 6, color: '#222', weight: 1, fillColor: feature.properties.color, fillOpacity: 0.9
      }});
    }},
    onEachFeature: function(feature, layer) {{
      var p = feature.properties;
      layer.bindPopup(
        '<b>' + name + ' - ' + p.kind + '</b><br>value: ' + p.value.toFixed(3) +
        '<br>persistence: ' + (p.persistence === null ? 'infinite (global extremum)' : p.persistence.toFixed(3))
      );
    }}
  }});
  overlays[name + ' \\u2014 field'] = heat;
  overlays[name + ' \\u2014 critical points'] = points;
  if (idx === 0) {{ heat.addTo(map); points.addTo(map); }}
}});

L.control.layers(null, overlays, {{collapsed: false}}).addTo(map);

var legend = L.control({{position: 'bottomright'}});
legend.onAdd = function() {{
  var div = L.DomUtil.create('div', 'legend');
  div.innerHTML =
    '<span class="swatch" style="background:{_CRITICAL_COLORS['minimum']}"></span>minimum<br>' +
    '<span class="swatch" style="background:{_CRITICAL_COLORS['maximum']}"></span>maximum<br>' +
    '<span class="swatch" style="background:{_CRITICAL_COLORS['saddle']}"></span>saddle<br>' +
    '<small>heatmap: dark = low value, bright yellow = high value</small>';
  return div;
}};
legend.addTo(map);
</script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
