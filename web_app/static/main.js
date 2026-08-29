"use strict";

// -- shared small helpers ----------------------------------------------------

function setStatus(id, message, isError) {
  const el = document.getElementById(id);
  el.textContent = message || "";
  el.classList.toggle("error", !!isError);
}

function openModal(id) {
  document.getElementById(id).hidden = false;
}

function closeModal(id) {
  document.getElementById(id).hidden = true;
}

document.querySelectorAll(".modal-overlay").forEach((overlay) => {
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.hidden = true;
  });
});

// -- persistent right-column map ---------------------------------------------
//
// One Leaflet map instance lives for the whole page session (ArcGIS/QGIS-
// style: the map is always there, not something that appears only once a
// report finishes) -- populateMap() clears and rebuilds its data layers each
// time a new "done" event arrives, rather than creating a new map per run.

let persistentMap = null;
let mapLayers = { heat: null, allMinima: null, boundary: null, catchment: null, catchmentBuildings: null, reportFocus: null, femaFlood: null, pipes: null, gauges: null, buildings: null, flowField: null, exposure: null, landCover: null, roads: null, water: null, rainfallSoil: null };
let mapLayerControl = null;
let mapLegendControl = null;
let mapSliderControl = null;
let currentMapData = null;
let persistenceThreshold = 0;
// Reassigned fresh by populateMap() every run (it closes over that run's own
// mapData) and invoked by the overlayadd/overlayremove handlers registered
// ONCE below, so the legend content always matches whichever layers are
// actually checked right now -- see the legend-building block in populateMap.
let legendUpdateFn = null;
let currentHighlightedCatchmentId = null;

function initPersistentMap() {
  persistentMap = L.map("persistent-map").setView([39.5, -98.35], 4);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(persistentMap);

  // Registered ONCE for the whole page session (persistentMap itself is a
  // single long-lived instance -- see the comment above) rather than
  // re-registered on every populateMap() call, which would stack duplicate
  // handlers across repeated analyses in the same session. Reads
  // mapLayers.allMinima/mapSliderControl at FIRE time (both module-level
  // `let` bindings this closure sees the current value of, even though
  // they're reassigned wholesale by clearMapLayers/populateMap on every
  // run), so this stays correct across runs without needing per-run setup.
  // The persistence-threshold slider only makes sense while the "All local
  // minima" layer is actually visible -- see rebuildAllMinimaLayer/the
  // slider-construction block in populateMap.
  persistentMap.on("overlayadd", (e) => {
    if (e.layer === mapLayers.allMinima && mapSliderControl) {
      mapSliderControl.getContainer().style.display = "block";
    }
    if (legendUpdateFn) legendUpdateFn();
  });
  persistentMap.on("overlayremove", (e) => {
    if (e.layer === mapLayers.allMinima && mapSliderControl) {
      mapSliderControl.getContainer().style.display = "none";
    }
    if (legendUpdateFn) legendUpdateFn();
  });
}

// viridis-ish ramp for the all-minima drainage-demand layer -- matches the
// one embedded in the downloadable report (geoai_agent/report.py's
// _webmap_fragment) so the two views read the same way.
function runoffColor(t) {
  const stops = [[0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]], [0.75, [94, 201, 98]], [1, [253, 231, 37]]];
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i], b = stops[i + 1];
    if (t >= a[0] && t <= b[0]) {
      const f = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
      const r = Math.round(a[1][0] + f * (b[1][0] - a[1][0]));
      const g = Math.round(a[1][1] + f * (b[1][1] - a[1][1]));
      const bl = Math.round(a[1][2] + f * (b[1][2] - a[1][2]));
      return `rgb(${r},${g},${bl})`;
    }
  }
  return "rgb(253,231,37)";
}

function maxPersistence(featureCollection) {
  let m = 0;
  ((featureCollection && featureCollection.features) || []).forEach((f) => {
    const p = f.properties.persistence_m;
    if (p !== null && p !== undefined) m = Math.max(m, p);
  });
  return m;
}

// Real-world persistence values are heavily right-skewed (a handful of deep
// basins, a long tail of shallow noise) -- a slider spanning [0, max]
// linearly spends nearly all of its travel on a range where almost nothing
// is between the smallest values and the few large ones, so most of the
// track does nothing. Mapping slider position to a PERCENTILE of the
// actual persistence values instead (nearest-rank) means every step of the
// slider always changes the shown set by roughly the same number of
// points, regardless of how skewed the distribution is.
function sortedPersistenceValues(featureCollection) {
  return ((featureCollection && featureCollection.features) || [])
    .map((f) => f.properties.persistence_m)
    .filter((p) => p !== null && p !== undefined)
    .sort((a, b) => a - b);
}
function percentileToThreshold(sorted, pct) {
  if (!sorted.length) return 0;
  const idx = Math.min(sorted.length - 1, Math.floor((pct / 100) * sorted.length));
  return sorted[idx];
}

// Marker size = persistence (deeper/more-persistent depressions draw bigger
// dots) -- a global sink (persistence_m === null, no deeper point to compare
// against) is treated as maximally persistent, same convention as the static
// map in report.py.
function persistenceRadius(persistenceM, maxP, minR, maxR) {
  if (persistenceM === null || persistenceM === undefined) return maxR;
  if (maxP <= 0) return minR;
  const frac = Math.max(0, Math.min(1, persistenceM / maxP));
  return minR + frac * (maxR - minR);
}

function clearMapLayers() {
  Object.values(mapLayers).forEach((layer) => { if (layer) persistentMap.removeLayer(layer); });
  mapLayers = { heat: null, allMinima: null, boundary: null, catchment: null, catchmentBuildings: null, reportFocus: null, femaFlood: null, pipes: null, gauges: null, buildings: null, flowField: null, exposure: null, landCover: null, roads: null, water: null, rainfallSoil: null };
  if (mapLayerControl) { persistentMap.removeControl(mapLayerControl); mapLayerControl = null; }
  if (mapLegendControl) { persistentMap.removeControl(mapLegendControl); mapLegendControl = null; }
  if (mapSliderControl) { persistentMap.removeControl(mapSliderControl); mapSliderControl = null; }
  legendUpdateFn = null;
  currentHighlightedCatchmentId = null;
}

// Called from inside the report iframe (report.py renders each hotspot/
// local-minimum mention as a clickable element that invokes this via
// `parent.focusMapPoint(...)`) -- the iframe is same-origin (srcdoc), so it
// can reach this directly rather than needing postMessage. Flies the
// persistent map to the point the report is describing and, when the
// clicked mention has a catchment_id, also highlights that basin's actual
// contributing area (same visual as clicking the dot on the map itself).
function focusMapPoint(lat, lon, catchmentId) {
  if (!persistentMap) return;
  if (mapLayers.reportFocus) { persistentMap.removeLayer(mapLayers.reportFocus); mapLayers.reportFocus = null; }
  mapLayers.reportFocus = L.circleMarker([lat, lon], {
    radius: 12, weight: 3, color: "#e63946", fillColor: "#e63946", fillOpacity: 0.15,
  }).addTo(persistentMap);
  persistentMap.flyTo([lat, lon], Math.max(persistentMap.getZoom(), 15));
  if (catchmentId !== null && catchmentId !== undefined) highlightCatchment(catchmentId);
}
window.focusMapPoint = focusMapPoint;

// Highlights one local minimum's contributing area: every terrain cell
// whose catchment_id (see report.py's elevation_hillshade_geojson) matches
// the clicked feature's own catchment_id -- i.e. TTK's descending-manifold
// segmentation for that basin, not an approximation drawn after the fact.
function highlightCatchment(catchmentId) {
  if (mapLayers.catchment) { persistentMap.removeLayer(mapLayers.catchment); mapLayers.catchment = null; }
  if (mapLayers.catchmentBuildings) { persistentMap.removeLayer(mapLayers.catchmentBuildings); mapLayers.catchmentBuildings = null; }
  currentHighlightedCatchmentId = catchmentId === null || catchmentId === undefined ? null : catchmentId;
  const clearRow = document.getElementById("catchment-clear-row");
  if (clearRow) clearRow.hidden = currentHighlightedCatchmentId === null;
  if (catchmentId === null || catchmentId === undefined || !currentMapData) return;
  const cells = currentMapData.field_geojson.features.filter((f) => f.properties.catchment_id === catchmentId);
  mapLayers.catchment = L.geoJSON({ type: "FeatureCollection", features: cells }, {
    style: { fillColor: "#2b6cb0", color: "#12335c", weight: 1, fillOpacity: 0.55 },
    interactive: false,
  }).addTo(persistentMap);
  mapLayers.catchment.bringToFront();

  // The REAL building footprints counted toward this basin's
  // "buildings_in_catchment"/"building_density_per_km2" figures (see
  // geoai_agent/spatial_analysis.py's compute_building_exposure) -- drawn
  // on top of the catchment fill so that aggregate number has something
  // visible to check it against, not just a figure to take on trust.
  // "catchment_id" here is already reconciled server-side (web_app/app.py)
  // against this SAME all-minima catchment_grid numbering, not each
  // significant hotspot's own separate TTK-run numbering.
  if (currentMapData.buildings_geojson) {
    const buildings = currentMapData.buildings_geojson.features.filter((f) => f.properties.catchment_id === catchmentId);
    if (buildings.length) {
      mapLayers.catchmentBuildings = L.geoJSON({ type: "FeatureCollection", features: buildings }, {
        style: { fillColor: "#a8492c", color: "#5c2414", weight: 1, fillOpacity: 0.75 },
        interactive: false,
      }).addTo(persistentMap);
      mapLayers.catchmentBuildings.bringToFront();
    }
  }
}

function rebuildAllMinimaLayer() {
  if (!currentMapData || !currentMapData.all_minima_geojson) return;
  if (mapLayers.allMinima) mapLayers.allMinima.clearLayers();
  const shown = currentMapData.all_minima_geojson.features.filter(
    (f) => (f.properties.persistence_m === null || f.properties.persistence_m === undefined) || f.properties.persistence_m >= persistenceThreshold
  );
  mapLayers.allMinima.addData({ type: "FeatureCollection", features: shown });
  const countEl = document.getElementById("minima-count-label");
  if (countEl) countEl.textContent = `Showing ${shown.length} of ${currentMapData.all_minima_geojson.features.length}`;
}

function populateMap(mapData) {
  clearMapLayers();
  currentMapData = mapData;
  persistenceThreshold = 0;
  document.getElementById("map-empty-state").hidden = true;

  const [latMin, latMax, lonMin, lonMax] = mapData.bounds;
  const leafletBounds = [[latMin, lonMin], [latMax, lonMax]];

  // Elevation is on by default (a reader should see terrain the moment the
  // map appears) -- every other layer below is built and registered with
  // the layer control but not .addTo(persistentMap)'d, so the user
  // explicitly opts into whichever OTHER data source they want to see on
  // top of it (see the legend hint added below).
  mapLayers.heat = L.geoJSON(mapData.field_geojson, {
    style: (f) => ({ fillColor: f.properties.color, color: f.properties.color, weight: 0, fillOpacity: 0.8 }),
  }).addTo(persistentMap);

  const overlays = { "Elevation (terrain)": mapLayers.heat };

  if (mapData.boundary_geojson) {
    mapLayers.boundary = L.geoJSON(mapData.boundary_geojson, {
      style: { fill: false, color: "#a8492c", weight: 2.5 },
    });
    overlays["Boundary"] = mapLayers.boundary;
  }

  // "color" is precomputed server-side (geoai_agent/fema_flood.py) from
  // FEMA's own FLD_ZONE/ZONE_SUBTY codes -- "" for minimal/undetermined
  // zones, which cover most of a typical bounding box and aren't worth
  // filling in (terrain hillshade shows through instead).
  if (mapData.fema_flood_geojson) {
    mapLayers.femaFlood = L.geoJSON(mapData.fema_flood_geojson, {
      style: (f) => {
        const color = f.properties.color;
        return color
          ? { fillColor: color, color, weight: 0.5, fillOpacity: 0.35 }
          : { fillOpacity: 0, weight: 0 };
      },
      onEachFeature: (f, layer) => {
        const p = f.properties;
        layer.bindPopup(`<b>FEMA Zone ${p.FLD_ZONE}</b><br>${p.severity}`);
      },
    });
    overlays["FEMA flood zones (NFHL)"] = mapLayers.femaFlood;
  }

  // Real stormwater pipes from a local database (geoai_agent/drainage_db.py)
  // -- currently DC-only, see load_dc_stormwater.py.
  if (mapData.pipes_geojson) {
    mapLayers.pipes = L.geoJSON(mapData.pipes_geojson, {
      style: (f) => ({ color: f.properties.capacity_m3s ? "#2b6cb0" : "#8894a0", weight: 2, opacity: 0.7 }),
      onEachFeature: (f, layer) => {
        const p = f.properties;
        const capText = p.capacity_m3s
          ? `${p.capacity_m3s.toFixed(2)} m&sup3;/s (Manning's equation estimate)`
          : "not enough data to estimate (missing diameter/material/slope)";
        layer.bindPopup(
          `<b>Stormwater pipe</b><br>Material: ${p.material || "unknown"}` +
          `<br>Diameter: ${p.diameter_in ? p.diameter_in + " in" : "unknown"}<br>Capacity: ${capText}`
        );
      },
    });
    overlays["Stormwater pipes"] = mapLayers.pipes;
  }

  // Real, current USGS NWIS river/stream gauge readings (geoai_agent/
  // streamflow.py) -- a fluvial-risk signal independent of this run's own
  // topological/pluvial basin analysis. Most of the country has none
  // nearby; that's the expected common case, not missing data. Same
  // marker style the downloadable report's own embedded map already uses
  // (geoai_agent/report.py's _webmap_fragment) so the two views read the
  // same way.
  if (mapData.gauges_geojson) {
    mapLayers.gauges = L.geoJSON(mapData.gauges_geojson, {
      pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 7, weight: 1.5, color: "#12335c", fillColor: "#4a90d9", fillOpacity: 0.9 }),
      onEachFeature: (f, layer) => {
        const p = f.properties;
        layer.bindPopup(
          `<b>${p.site_name || "USGS gauge"}</b>` +
          (p.gage_height_ft !== null && p.gage_height_ft !== undefined ? `<br>Gage height: ${p.gage_height_ft} ft` : "") +
          (p.discharge_cfs !== null && p.discharge_cfs !== undefined ? `<br>Discharge: ${p.discharge_cfs} ft&sup3;/s` : "")
        );
      },
    });
    overlays["River/stream gauges (USGS NWIS)"] = mapLayers.gauges;
  }

  // Every OpenStreetMap building counted toward ANY significant basin's
  // "buildings_in_catchment"/"building_density_per_km2" figures (see
  // geoai_agent/spatial_analysis.py's compute_building_exposure) -- a
  // general, always-available view of that same data, distinct from
  // highlightCatchment's brighter per-basin highlight (mapLayers.
  // catchmentBuildings, fillOpacity 0.75) shown only while one basin is
  // selected. Lower opacity here so the two don't visually compete when
  // both happen to be on at once.
  if (mapData.buildings_geojson) {
    mapLayers.buildings = L.geoJSON(mapData.buildings_geojson, {
      style: { fillColor: "#a8492c", color: "#5c2414", weight: 1, fillOpacity: 0.55 },
    });
    overlays["Buildings (OpenStreetMap, counted toward exposure)"] = mapLayers.buildings;
  }

  // D8 downhill flow-direction arrows (geoai_agent/flow_field.py) -- plain
  // raster hydrology computed straight from the elevation grid, independent
  // of TTK/persistence entirely. Arrow length already encodes each sampled
  // cell's own flow_accumulation_km2 (short = little water passes through
  // that spot, long = a lot does) -- styling here only needs color/opacity,
  // the size itself is baked into the geometry server-side.
  if (mapData.flow_field_geojson) {
    mapLayers.flowField = L.geoJSON(mapData.flow_field_geojson, {
      style: (f) => {
        const t = f.properties.t;
        return { color: "#c0392b", weight: 1 + t * 2.5, opacity: 0.45 + t * 0.5 };
      },
      onEachFeature: (f, layer) => {
        const p = f.properties;
        layer.bindPopup(`<b>Flow direction</b><br>Accumulated drainage area: ${p.flow_accumulation_km2.toFixed(3)} km&sup2;`);
      },
    });
    overlays["Flow direction (drainage volume)"] = mapLayers.flowField;
  }

  // Population density (US Census block groups, or a user upload -- see
  // geoai_agent/real_data.py's fetch_population_exposure_field), the same
  // per-cell heatmap shape as the elevation layer above.
  if (mapData.exposure_geojson) {
    mapLayers.exposure = L.geoJSON(mapData.exposure_geojson, {
      style: (f) => ({ fillColor: f.properties.color, color: f.properties.color, weight: 0, fillOpacity: 0.75 }),
    });
    overlays["Population density (exposure)"] = mapLayers.exposure;
  }

  // NLCD classified land cover (geoai_agent/land_cover.py) -- categorical,
  // each cell keeps its own class color instead of a continuous ramp (see
  // land_cover_geojson's docstring for why interpolating between class
  // codes would be meaningless).
  if (mapData.land_cover_geojson) {
    mapLayers.landCover = L.geoJSON(mapData.land_cover_geojson, {
      style: (f) => ({ fillColor: f.properties.color, color: f.properties.color, weight: 0, fillOpacity: 0.65 }),
      onEachFeature: (f, layer) => layer.bindPopup(`<b>${f.properties.label}</b>`),
    });
    overlays["Land cover (NLCD)"] = mapLayers.landCover;
  }

  // OpenStreetMap roads/water (geoai_agent/spatial_analysis.py) -- fetched
  // alongside buildings for each hotspot's nearest-infrastructure distance,
  // drawn here so the raw data behind that distance is actually visible.
  if (mapData.roads_geojson) {
    mapLayers.roads = L.geoJSON(mapData.roads_geojson, { style: { color: "#555", weight: 1.5, opacity: 0.8 } });
    overlays["Roads (OpenStreetMap)"] = mapLayers.roads;
  }
  if (mapData.water_geojson) {
    mapLayers.water = L.geoJSON(mapData.water_geojson, {
      style: { color: "#2b6cb0", weight: 1.5, fillColor: "#4a90d9", fillOpacity: 0.35 },
    });
    overlays["Water (OpenStreetMap)"] = mapLayers.water;
  }

  // Design-storm intensity and dominant hydrologic soil group are each a
  // SINGLE value for the whole run (one storm, one dominant soil group --
  // see rainfall.py/soil_data.py), not spatially varying, so this is one
  // info marker at the study area's center rather than a coverage layer.
  if (mapData.rainfall_soil_geojson) {
    mapLayers.rainfallSoil = L.geoJSON(mapData.rainfall_soil_geojson, {
      pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 8, weight: 1.5, color: "#5c3a1e", fillColor: "#dcd939", fillOpacity: 0.9 }),
      onEachFeature: (f, layer) => {
        const p = f.properties;
        let html = `<b>Design storm</b><br>${p.design_storm_label}, 1-hour intensity: ${p.design_storm_intensity_mm_per_hr.toFixed(0)} mm/hr<br><small>${p.rainfall_source}</small>`;
        if (p.hydrologic_soil_group) {
          html += `<hr style="margin:4px 0;"><b>Dominant soil group</b><br>${p.hydrologic_soil_group}<br><small>${p.soil_source}</small>`;
        }
        layer.bindPopup(html);
      },
    });
    overlays["Rainfall & soil (NOAA / USDA)"] = mapLayers.rainfallSoil;
  }

  if (mapData.all_minima_geojson) {
    const allMinimaMaxP = maxPersistence(mapData.all_minima_geojson);
    mapLayers.allMinima = L.geoJSON(null, {
      pointToLayer: (f, latlng) => {
        const p = f.properties;
        const r = persistenceRadius(p.persistence_m, allMinimaMaxP, 4, 16);
        return L.circleMarker(latlng, {
          radius: r,
          weight: p.within_boundary ? 1.5 : 0.5,
          color: p.within_boundary ? "#222" : "#8894",
          fillColor: runoffColor(p.t),
          fillOpacity: 0.85,
        });
      },
      onEachFeature: (f, layer) => {
        const p = f.properties;
        const boundaryNote = p.within_boundary === null || p.within_boundary === undefined
          ? ""
          : `<br>${p.within_boundary ? "inside" : "outside"} the resolved boundary`;
        layer.bindPopup(
          `<b>Local minimum #${p.rank}</b><br>drainage demand: ${p.estimated_peak_runoff_m3s.toFixed(2)} m&sup3;/s` +
          `<br>contributing area: ${p.contributing_area_km2.toFixed(3)} km&sup2;` +
          `<br>persistence: ${p.persistence_m === null ? "global sink" : p.persistence_m.toFixed(2) + " m"}${boundaryNote}` +
          `<br><em>click the dot to highlight its catchment</em>`
        );
        layer.on("click", () => highlightCatchment(p.catchment_id));
      },
    });
    overlays["All local minima (drainage demand)"] = mapLayers.allMinima;
  }

  mapLayerControl = L.control.layers(null, overlays, { collapsed: false }).addTo(persistentMap);

  // -- legend: rebuilt from scratch every time a layer is checked/unchecked
  // (see the overlayadd/overlayremove handlers in initPersistentMap), so it
  // only ever shows entries for what's actually visible right now -- with
  // this many optional layers, a legend listing every one of them
  // regardless of what's on screen would be longer than the map itself.
  //
  // Leaflet 1.9.4's Map.hasLayer is `stamp(layer) in this._layers` with NO
  // null guard -- stamp() does `'_leaflet_id' in obj`, which throws
  // "Cannot use 'in' operator to search for '_leaflet_id' in null" if
  // called on null (verified directly against leaflet-src.js). Most of
  // mapLayers.* IS null on any given run (whichever data sources weren't
  // fetched) -- always check truthiness first.
  function layerIsVisible(layer) {
    return !!layer && persistentMap.hasLayer(layer);
  }
  function legendContentHtml() {
    let gradientHtml = "";
    if (layerIsVisible(mapLayers.heat) && mapData.elevation_legend && mapData.elevation_legend.length) {
      const stops = mapData.elevation_legend;
      const gradientCss = stops.map((s, i) => `${s.color} ${(i / (stops.length - 1)) * 100}%`).join(", ");
      gradientHtml =
        `<div style="margin-top:6px;"><strong>Elevation</strong></div>` +
        `<div style="width:160px;height:10px;border-radius:2px;margin:3px 0;background:linear-gradient(90deg, ${gradientCss});"></div>` +
        `<div style="display:flex;justify-content:space-between;font-size:10.5px;">` +
        `<span>${stops[0].value.toFixed(0)} m</span><span>${stops[stops.length - 1].value.toFixed(0)} m</span></div>`;
    }
    return (
      `<div style="margin-bottom:6px;"><strong>Toggle layers above</strong> -- the legend below only shows what's currently checked.</div>` +
      gradientHtml +
      (layerIsVisible(mapLayers.allMinima)
        ? `<div style="margin-top:6px;"><strong>Dot size</strong> = persistence (bigger = deeper basin)</div>` +
          `<div><strong>Dot fill</strong> = drainage demand</div>` +
          `<div>Click a dot -- or a point mentioned in the report -- to highlight its catchment area</div>`
        : "") +
      (layerIsVisible(mapLayers.femaFlood)
        ? `<div style="margin-top:6px;"><strong>FEMA flood zones</strong></div>` +
          `<div><span style="display:inline-block;width:10px;height:10px;background:#c0392b;border-radius:2px;"></span> Special Flood Hazard Area</div>` +
          `<div><span style="display:inline-block;width:10px;height:10px;background:#e8ab5c;border-radius:2px;"></span> 0.2% annual chance ("500-year")</div>`
        : "") +
      (layerIsVisible(mapLayers.pipes)
        ? `<div style="margin-top:6px;"><strong>Stormwater pipes</strong></div>` +
          `<div><span style="display:inline-block;width:14px;height:2px;background:#2b6cb0;"></span> capacity computed</div>` +
          `<div><span style="display:inline-block;width:14px;height:2px;background:#8894a0;"></span> not enough data</div>`
        : "") +
      (layerIsVisible(mapLayers.gauges)
        ? `<div style="margin-top:6px;"><strong>River/stream gauges</strong> (USGS NWIS, live)</div>` +
          `<div><span style="display:inline-block;width:10px;height:10px;background:#4a90d9;border-radius:50%;border:1.5px solid #12335c;"></span> active gauge</div>`
        : "") +
      (layerIsVisible(mapLayers.buildings)
        ? `<div style="margin-top:6px;"><strong>Buildings</strong> (OpenStreetMap)</div>` +
          `<div><span style="display:inline-block;width:10px;height:10px;background:#a8492c;border:1px solid #5c2414;"></span> counted toward exposure, brighter when highlighting one basin's catchment</div>`
        : "") +
      (layerIsVisible(mapLayers.flowField)
        ? `<div style="margin-top:6px;"><strong>Flow direction</strong> (D8, computed from elevation)</div>` +
          `<div>Arrow length = accumulated drainage area at that spot -- short where little water passes through, long where a lot does</div>`
        : "") +
      (layerIsVisible(mapLayers.exposure)
        ? `<div style="margin-top:6px;"><strong>Population density</strong> (US Census block groups, live)</div>`
        : "") +
      (layerIsVisible(mapLayers.landCover)
        ? `<div style="margin-top:6px;"><strong>Land cover</strong> (NLCD, categorical -- click a cell for its class)</div>`
        : "") +
      (layerIsVisible(mapLayers.roads) || layerIsVisible(mapLayers.water)
        ? `<div style="margin-top:6px;"><strong>Roads / water</strong> (OpenStreetMap)</div>`
        : "") +
      (layerIsVisible(mapLayers.rainfallSoil)
        ? `<div style="margin-top:6px;"><strong>Rainfall &amp; soil</strong> (NOAA Atlas 14 / USDA SSURGO -- click the marker)</div>`
        : "") +
      `<div id="catchment-clear-row"${currentHighlightedCatchmentId === null ? " hidden" : ""}><a href="#" id="catchment-clear-link">Clear catchment highlight</a></div>`
    );
  }

  function attachLegendClearLink() {
    const link = document.getElementById("catchment-clear-link");
    if (link) link.addEventListener("click", (e) => { e.preventDefault(); highlightCatchment(null); });
  }

  currentHighlightedCatchmentId = null;
  mapLegendControl = L.control({ position: "bottomleft" });
  mapLegendControl.onAdd = () => {
    const div = L.DomUtil.create("div", "webmap-legend");
    div.innerHTML = legendContentHtml();
    return div;
  };
  mapLegendControl.addTo(persistentMap);
  L.DomEvent.disableClickPropagation(mapLegendControl.getContainer());
  attachLegendClearLink();

  // Reassigned every populateMap() call so it always closes over THIS run's
  // mapData/mapLegendControl -- invoked by initPersistentMap's overlayadd/
  // overlayremove handlers (registered once, outside this function) every
  // time a layer is checked or unchecked.
  legendUpdateFn = () => {
    const container = mapLegendControl.getContainer();
    if (!container) return;
    container.innerHTML = legendContentHtml();
    attachLegendClearLink();
  };

  // -- persistence-threshold slider: filters the all-minima layer client- --
  // side (no re-fetch/re-analyze -- every raw minimum's persistence_m is
  // already in mapData, this just changes which of them are drawn).
  if (mapData.all_minima_geojson && mapData.all_minima_geojson.features.length) {
    const sortedP = sortedPersistenceValues(mapData.all_minima_geojson);
    mapSliderControl = L.control({ position: "topleft" });
    mapSliderControl.onAdd = () => {
      const div = L.DomUtil.create("div", "webmap-legend webmap-slider");
      div.innerHTML =
        `<div><strong>Persistence threshold</strong>: <span id="threshold-value">0.0</span> m ` +
        `(<span id="threshold-percentile">0</span>th pct.)</div>` +
        `<input type="range" id="persistence-slider" min="0" max="100" step="1" value="0" style="width:180px;">` +
        `<div id="minima-count-label" style="font-size:10.5px;color:inherit;"></div>`;
      return div;
    };
    mapSliderControl.addTo(persistentMap);
    // Hidden until the "All local minima" overlay is actually turned on
    // (see the overlayadd/overlayremove handlers in initPersistentMap) --
    // every layer in this map starts off, so a threshold control for a
    // layer that isn't even showing yet would just be confusing chrome.
    mapSliderControl.getContainer().style.display = "none";
    const sliderEl = document.getElementById("persistence-slider");
    L.DomEvent.disableClickPropagation(mapSliderControl.getContainer());
    sliderEl.addEventListener("input", () => {
      const pct = parseFloat(sliderEl.value);
      persistenceThreshold = percentileToThreshold(sortedP, pct);
      document.getElementById("threshold-value").textContent = persistenceThreshold.toFixed(1);
      document.getElementById("threshold-percentile").textContent = pct.toFixed(0);
      rebuildAllMinimaLayer();
    });
    rebuildAllMinimaLayer();
  }

  persistentMap.fitBounds(leafletBounds);
}

// -- draw-a-boundary mode -----------------------------------------------
//
// Alternative to naming a place: click points directly on the persistent
// map to build a polygon, then send its vertices to analyze_stream instead
// of a place name (see boundary_from_drawn_polygon in geoai_agent/
// boundary.py, which builds the same {name, bounds, polygon, geojson}
// shape a real geocoded boundary would) -- only how the location itself
// gets chosen differs from the place-name path below; both go straight
// into runAgentAnalysis with no popup in between (population/exposure now
// comes from a live US Census fetch automatically -- see real_data.
// fetch_population_exposure_field -- so there's nothing to ask upfront).

let drawMode = false;
let drawPoints = []; // [[lat, lon], ...] in click order
let drawPreviewLayer = null;
let mapDrawControl = null;
let lastDrawnPolygon = null; // kept so a chat-triggered resolution re-run can reuse the same polygon

function updateDrawPreview() {
  if (drawPreviewLayer) { persistentMap.removeLayer(drawPreviewLayer); drawPreviewLayer = null; }
  if (drawPoints.length >= 2) {
    drawPreviewLayer = L.polygon(drawPoints, {
      color: "#2b6cb0", weight: 2, dashArray: "5,5", fillOpacity: 0.12,
    }).addTo(persistentMap);
  } else if (drawPoints.length === 1) {
    drawPreviewLayer = L.circleMarker(drawPoints[0], { radius: 4, color: "#2b6cb0" }).addTo(persistentMap);
  }
  const countEl = document.getElementById("draw-point-count");
  if (countEl) countEl.textContent = `${drawPoints.length} point${drawPoints.length === 1 ? "" : "s"}`;
  const finishBtn = document.getElementById("draw-finish-btn");
  if (finishBtn) finishBtn.disabled = drawPoints.length < 3;
}

function onDrawClick(e) {
  drawPoints.push([e.latlng.lat, e.latlng.lng]);
  updateDrawPreview();
}

function showDrawControl() {
  mapDrawControl = L.control({ position: "topright" });
  mapDrawControl.onAdd = () => {
    const div = L.DomUtil.create("div", "webmap-legend");
    div.innerHTML =
      `<div><strong>Draw a boundary</strong></div>` +
      `<div>Click the map to add points (need at least 3).</div>` +
      `<div id="draw-point-count">0 points</div>` +
      `<div class="button-row">` +
      `<button id="draw-finish-btn" type="button" class="primary" disabled>Finish</button>` +
      `<button id="draw-cancel-btn" type="button" class="secondary">Cancel</button>` +
      `</div>`;
    return div;
  };
  mapDrawControl.addTo(persistentMap);
  L.DomEvent.disableClickPropagation(mapDrawControl.getContainer());
  document.getElementById("draw-finish-btn").addEventListener("click", finishDrawMode);
  document.getElementById("draw-cancel-btn").addEventListener("click", cancelDrawMode);
}

function hideDrawControl() {
  if (mapDrawControl) { persistentMap.removeControl(mapDrawControl); mapDrawControl = null; }
}

function startDrawMode() {
  if (!persistentMap || drawMode) return;
  drawMode = true;
  drawPoints = [];
  document.getElementById("map-empty-state").hidden = true;
  persistentMap.doubleClickZoom.disable(); // a double-click would otherwise both add a point and zoom the map
  persistentMap.getContainer().style.cursor = "crosshair";
  persistentMap.on("click", onDrawClick);
  showDrawControl();
  updateDrawPreview();
}

function endDrawMode() {
  drawMode = false;
  persistentMap.off("click", onDrawClick);
  persistentMap.doubleClickZoom.enable();
  persistentMap.getContainer().style.cursor = "";
  hideDrawControl();
  if (drawPreviewLayer) { persistentMap.removeLayer(drawPreviewLayer); drawPreviewLayer = null; }
  if (!currentMapData) document.getElementById("map-empty-state").hidden = false;
}

function cancelDrawMode() {
  drawPoints = [];
  endDrawMode();
}

function finishDrawMode() {
  if (drawPoints.length < 3) return;
  const polygon = drawPoints.slice();
  drawPoints = [];
  endDrawMode();
  pendingBoundaryShapefile = null; // a freshly drawn polygon replaces any earlier uploaded boundary
  pendingExposureFile = null; // a fresh drawn-boundary run shouldn't silently reuse a file from an unrelated earlier run
  pendingElevationFile = null;
  setStatus("intro-status", "");
  runAgentAnalysis(document.getElementById("place-input").value.trim(), null, null, { drawnPolygon: polygon });
}

document.getElementById("draw-boundary-btn").addEventListener("click", () => {
  setStatus("intro-status", "");
  startDrawMode();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && drawMode) cancelDrawMode();
});

// -- terminal panel: a collapsible slide-in log, not a full-page view ------

function setTerminalOpen(open) {
  document.getElementById("terminal-panel").classList.toggle("open", open);
}

document.getElementById("terminal-toggle").addEventListener("click", () => {
  const panel = document.getElementById("terminal-panel");
  setTerminalOpen(!panel.classList.contains("open"));
});

// Which of the pipeline's data-fetching steps a "Modify" button on that row
// should let the user replace with their own upload -- matched by the exact
// step.label text agent_analyze_stream (web_app/app.py) sends for each one.
// Every other step (boundary lookup, TTK analysis, flood ranking, the LLM
// recommendation...) has no user-suppliable data source, so gets no button.
function pipelineStepModifyKind(label) {
  if (label === "Fetching elevation data" || label === "Fetched elevation data") return "elevation";
  if (
    label === "Loading exposure/population data" || label === "Exposure data loaded" ||
    label === "No exposure/population data supplied -- that factor will use a neutral default" ||
    label === "Exposure data failed to load -- ignoring that factor"
  ) return "exposure";
  if (
    label === "Boundary ready" || label === "Resolved real boundary (OpenStreetMap)" ||
    label === "Resolved location (point + padded box)" ||
    label === "No boundary polygon available -- geocoding to a point instead"
  ) return "boundary";
  if (label === "Flood ranking complete") return "weights";
  return null;
}

function addProgressStep(step) {
  const log = document.getElementById("progress-log");
  const prevLive = log.querySelector(".step-dot.live");
  if (prevLive) prevLive.classList.remove("live");

  const row = document.createElement("div");
  row.className = "pipeline-step";
  const detailText = step.detail ? JSON.stringify(step.detail, null, 2) : "";
  const modifyKind = pipelineStepModifyKind(step.label);
  // Starts disabled: this row belongs to a run that's still in progress (the
  // pipeline steps after this one haven't happened yet), and re-running with
  // new data before this run has even finished would race it -- see the
  // "done" handler in runAgentAnalysis, which enables every button still in
  // the log once the run it belongs to actually completes.
  const modifyBtnHtml = modifyKind
    ? `<button type="button" class="secondary step-modify-btn" data-modify-kind="${modifyKind}" disabled>Modify</button>`
    : "";
  row.innerHTML = `
    <div class="pipeline-step-summary"><span class="step-dot live"></span><span class="pipeline-step-label">${step.label}</span>${modifyBtnHtml}</div>
    ${detailText ? `<pre>${detailText}</pre>` : ""}`;
  log.appendChild(row);
  if (modifyKind) {
    row.querySelector(".step-modify-btn").addEventListener("click", () => {
      if (modifyKind === "weights") openModifyWeightsModal(); else openModifyDataModal(modifyKind);
    });
  }
  row.scrollIntoView({ behavior: "smooth", block: "end" });
}

// A real agent decision (see geoai_agent.agent_decisions -- which weighting
// strategy to use, which persistence threshold counts as "a real basin" for
// this terrain, whether the top pick is robust to a different weighting
// choice), rendered visually distinct from a plain pipeline step so it
// doesn't get lost among "fetching X" log lines -- this is the one place in
// the log where the agent is shown actually CHOOSING something with real
// consequences, not just reporting that a fetch succeeded.
function addAgentDecisionStep(decision) {
  const log = document.getElementById("progress-log");
  const prevLive = log.querySelector(".step-dot.live");
  if (prevLive) prevLive.classList.remove("live");

  const row = document.createElement("div");
  row.className = "pipeline-step agent-decision-step";
  row.innerHTML = `
    <div class="pipeline-step-summary">
      <span class="step-dot live"></span>
      <span class="agent-decision-badge">Agent decision</span>
      <span class="pipeline-step-label">${decision.label}</span>
    </div>
    <p class="agent-decision-reasoning">${decision.reasoning}</p>`;
  log.appendChild(row);
  row.scrollIntoView({ behavior: "smooth", block: "end" });
}

function finalizeProgressDots() {
  const lastDot = document.querySelector("#progress-log .step-dot.live");
  if (lastDot) lastDot.classList.remove("live");
}

function enableModifyButtons() {
  document.querySelectorAll("#progress-log .step-modify-btn").forEach((btn) => { btn.disabled = false; });
}

// -- modify-data modal: re-run with a user-uploaded elevation or exposure --
// file, triggered by a "Modify" button on the matching pipeline-step row
// (see pipelineStepModifyKind above). Feeds the same runAgentAnalysis path
// as everything else, just with an extra file field and preserveChat so the
// conversation carries across the re-run like the resolution-change path.

let pendingElevationFile = null; // sticky across re-runs of the same place, like pendingExposureFile
let pendingBoundaryShapefile = null; // sticky the same way; supersedes lastDrawnPolygon once set
let modifyDataKind = null; // "elevation" | "exposure" | "boundary"

function openModifyDataModal(kind) {
  modifyDataKind = kind;
  const titleEl = document.getElementById("modify-data-title");
  const hintEl = document.getElementById("modify-data-hint");
  const fileInput = document.getElementById("modify-data-file");
  if (kind === "elevation") {
    titleEl.textContent = "Upload your own elevation data";
    hintEl.textContent = "A GeoTIFF (.tif/.tiff) with a coordinate system -- e.g. your own DEM. This replaces the live-fetched USGS/Open-Meteo elevation for a full re-run of the current place/boundary.";
    fileInput.accept = ".tif,.tiff";
  } else if (kind === "exposure") {
    titleEl.textContent = "Upload your own exposure/population data";
    hintEl.textContent = "CSV (header x,y,value), a shapefile (.zip with POINT geometry + a value field), or a GeoTIFF (.tif). This replaces the exposure/population weighting for a full re-run of the current place/boundary.";
    fileInput.accept = ".csv,.zip,.shp,.tif,.tiff";
  } else {
    titleEl.textContent = "Upload your own boundary";
    hintEl.textContent = "A shapefile (.zip bundling .shp/.shx/.dbf, POLYGON geometry) defining the exact analysis area. This replaces the geocoded/drawn boundary for a full re-run over that shape.";
    fileInput.accept = ".zip,.shp";
  }
  fileInput.value = "";
  setStatus("modify-data-status", "");
  openModal("modify-data-modal");
}

document.getElementById("modify-data-cancel").addEventListener("click", () => closeModal("modify-data-modal"));
document.getElementById("modify-data-submit").addEventListener("click", () => {
  const file = document.getElementById("modify-data-file").files[0];
  if (!file) {
    setStatus("modify-data-status", "Choose a file first.", true);
    return;
  }
  let kindLabel;
  if (modifyDataKind === "elevation") {
    pendingElevationFile = file;
    kindLabel = "elevation";
  } else if (modifyDataKind === "exposure") {
    pendingExposureFile = file;
    kindLabel = "exposure/population";
  } else {
    pendingBoundaryShapefile = file;
    lastDrawnPolygon = null; // the uploaded shapefile is now the boundary's source of truth, not whatever was drawn before
    kindLabel = "boundary";
  }
  closeModal("modify-data-modal");
  addChatMessage("assistant", `Got it -- re-running ${lastAnalyzedPlace || "the analysis"} with your uploaded ${kindLabel} data…`);
  runAgentAnalysis(lastAnalyzedPlace, pendingExposureFile, null, {
    preserveChat: true, drawnPolygon: lastDrawnPolygon,
    elevationFile: pendingElevationFile, boundaryShapefile: pendingBoundaryShapefile,
    customWeights: lastCustomWeights,
  });
});

// -- modify-weights modal: re-run with hand-set risk-factor weights ---------
// Triggered by the "Modify" button on the "Flood ranking complete" step row.
// Five linked sliders that always sum to 100% (dragging one proportionally
// rescales the other four, same UX as e.g. a budget-allocation slider set) --
// applying them sends weighting_method=custom + custom_weights=<json> instead
// of the default entropy weighting, for this and every subsequent re-run of
// the same analysis (see runAgentAnalysis).

const WEIGHT_FACTORS = [
  { key: "topological", label: "Topological (basin persistence)" },
  { key: "contributing_area", label: "Contributing area" },
  { key: "rainfall", label: "Rainfall intensity" },
  { key: "slope", label: "Slope" },
  { key: "exposure", label: "Exposure (population/buildings)" },
];
let modifyWeightsState = null; // {key: percent 0-100, ...}, always summing to 100

function openModifyWeightsModal() {
  const current = (qaContext && qaContext.weights) || {
    topological: 0.30, contributing_area: 0.25, rainfall: 0.20, slope: 0.15, exposure: 0.10,
  };
  modifyWeightsState = {};
  WEIGHT_FACTORS.forEach((f) => { modifyWeightsState[f.key] = (current[f.key] || 0) * 100; });
  renderWeightSliders();
  setStatus("modify-weights-status", "");
  openModal("modify-weights-modal");
}

// Redistributes `delta` (the change just made to `changedKey`) across the
// other four factors in proportion to their CURRENT weights, so relative
// priorities among the untouched sliders are preserved; a final renormalize
// pass corrects any rounding/clamping drift back to exactly 100.
function redistributeWeights(changedKey, newValue, state) {
  const keys = WEIGHT_FACTORS.map((f) => f.key);
  newValue = Math.max(0, Math.min(100, newValue));
  const delta = newValue - state[changedKey];
  const others = keys.filter((k) => k !== changedKey);
  const othersTotal = others.reduce((sum, k) => sum + state[k], 0);
  const updated = { ...state, [changedKey]: newValue };
  others.forEach((k) => {
    const share = othersTotal > 0 ? state[k] / othersTotal : 1 / others.length;
    updated[k] = Math.max(0, state[k] - delta * share);
  });
  const total = keys.reduce((sum, k) => sum + updated[k], 0) || 1;
  keys.forEach((k) => { updated[k] = (updated[k] / total) * 100; });
  return updated;
}

function renderWeightSliders() {
  const container = document.getElementById("weight-sliders");
  container.innerHTML = WEIGHT_FACTORS.map((f) => `
    <div class="weight-slider-row" data-key="${f.key}">
      <label>${f.label}</label>
      <input type="range" min="0" max="100" step="1" value="${Math.round(modifyWeightsState[f.key])}" data-key="${f.key}">
      <span class="weight-slider-val">${Math.round(modifyWeightsState[f.key])}%</span>
    </div>`).join("");
  container.querySelectorAll("input[type=range]").forEach((input) => {
    input.addEventListener("input", () => {
      modifyWeightsState = redistributeWeights(input.dataset.key, Number(input.value), modifyWeightsState);
      container.querySelectorAll(".weight-slider-row").forEach((row) => {
        const val = Math.round(modifyWeightsState[row.dataset.key]);
        row.querySelector("input[type=range]").value = val;
        row.querySelector(".weight-slider-val").textContent = `${val}%`;
      });
    });
  });
}

document.getElementById("modify-weights-cancel").addEventListener("click", () => closeModal("modify-weights-modal"));
document.getElementById("modify-weights-submit").addEventListener("click", () => {
  const customWeights = {};
  WEIGHT_FACTORS.forEach((f) => { customWeights[f.key] = modifyWeightsState[f.key] / 100; });
  closeModal("modify-weights-modal");
  addChatMessage("assistant", `Got it -- re-running ${lastAnalyzedPlace || "the analysis"} with your own risk weights…`);
  runAgentAnalysis(lastAnalyzedPlace, pendingExposureFile, null, {
    preserveChat: true, drawnPolygon: lastDrawnPolygon,
    elevationFile: pendingElevationFile, boundaryShapefile: pendingBoundaryShapefile,
    customWeights,
  });
});

// -- one-box agent flow -------------------------------------------------------
//
// place-input -> POST /api/agent/analyze_stream directly (no popup first --
// population/exposure is fetched live from US Census automatically now),
// whose response body is newline-delimited JSON, one {"type": "step"|
// "error"|"done", ...} object per line, flushed as each pipeline stage
// finishes server-side. Read with fetch()'s streaming body reader (not
// EventSource -- EventSource can't POST a file upload, needed for the
// elevation/exposure/boundary "Modify" re-run flow below) so the terminal
// panel fills in live instead of only appearing once the whole multi-stage
// analysis is done.

let pendingExposureFile = null; // only ever set via the post-run "Modify" flow now (see openModifyDataModal)

// -- report Q&A chat -----------------------------------------------------
//
// The LLM as the front-facing agent, not just the backend step that writes
// the recommendation paragraph once and disappears: qaContext is the
// compact per-run summary analyze_stream's "done" event includes
// specifically for this (ranked hotspots, elevation stats, rainfall/
// weights, the recommendation text -- not the full map_data/report HTML).
// Stateless like the rest of this app -- chatHistory lives only in this
// tab and is resent with every question rather than kept server-side.

let qaContext = null;
let chatHistory = [];
let lastAnalyzedPlace = null;
let lastCustomWeights = null; // {topological, contributing_area, rainfall, slope, exposure} set via the Modify-weights sliders, or null
let currentRunId = null; // set from the stream's "run_id" event, cleared on done/error -- see runAgentAnalysis and the #interject-send handler below
let checkpointCountdownTimer = null; // setInterval id for the currently-open checkpoint's countdown, or null -- see openCheckpointUI/closeCheckpointUI

// A narrow, deterministic (no LLM call -- every chat turn already costs one
// against the free-tier daily quota, doubling that per message just to
// classify intent isn't worth it) detector for "change the resolution and
// re-run": requires both a resolution-ish keyword and a number+unit near
// it, so it doesn't misfire on unrelated numbers (rainfall mm/hr, ranks,
// coordinates) elsewhere in a normal question.
function parseResolutionRequest(text) {
  if (!/resolution|分辨率|精度/i.test(text)) return null;
  const match = text.match(/(\d+(?:\.\d+)?)\s*(米|m|meters?)(?!m)\b/i);
  if (!match) return null;
  const value = parseFloat(match[1]);
  return isFinite(value) && value > 0 ? value : null;
}

function addChatMessage(role, content, extraClass) {
  const log = document.getElementById("chat-log");
  const row = document.createElement("div");
  row.className = `chat-msg ${role}${extraClass ? " " + extraClass : ""}`;
  row.textContent = content;
  log.appendChild(row);
  row.scrollIntoView({ behavior: "smooth", block: "end" });
  return row;
}

async function sendChatMessage() {
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question) return;
  if (!qaContext) {
    setStatus("chat-status", "Run an analysis first.", true);
    return;
  }

  setStatus("chat-status", "");
  input.value = "";
  addChatMessage("user", question);

  const requestedResolution = parseResolutionRequest(question);
  if (requestedResolution !== null) {
    addChatMessage("assistant", `Got it -- re-running the analysis for ${lastAnalyzedPlace || "the drawn area"} at ${requestedResolution}m resolution…`);
    runAgentAnalysis(lastAnalyzedPlace, pendingExposureFile, requestedResolution, {
      preserveChat: true, drawnPolygon: lastDrawnPolygon,
      elevationFile: pendingElevationFile, boundaryShapefile: pendingBoundaryShapefile,
      customWeights: lastCustomWeights,
    });
    return;
  }

  document.getElementById("chat-send").disabled = true;
  const pendingRow = addChatMessage("assistant", "Thinking…", "pending");

  const formData = new FormData();
  formData.append("question", question);
  formData.append("context", JSON.stringify(qaContext));
  formData.append("history", JSON.stringify(chatHistory));

  try {
    const res = await fetch("/api/agent/ask", { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `request failed (${res.status})`);
    pendingRow.textContent = data.answer;
    pendingRow.classList.remove("pending");
    chatHistory.push({ role: "user", content: question });
    chatHistory.push({ role: "assistant", content: data.answer });
  } catch (err) {
    pendingRow.textContent = "Couldn't get an answer: " + err.message;
    pendingRow.classList.remove("pending");
    pendingRow.classList.add("error");
  } finally {
    document.getElementById("chat-send").disabled = false;
    input.focus();
  }
}

document.getElementById("chat-send").addEventListener("click", sendChatMessage);
document.getElementById("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); sendChatMessage(); }
});

// One placeholder per checkpoint (keyed by the exact checkpoint_label
// strings web_app/app.py's checkpoint_wait calls are made with), so the
// hint text in the feedback box actually matches what that checkpoint can
// act on instead of one generic prompt for all three -- see
// openCheckpointUI below, which sets #interject-input's placeholder from
// this map each time a checkpoint opens.
const CHECKPOINT_PROMPTS = {
  "persistence threshold": "Too many small basins, or too few? e.g. raise the threshold, or focus on fewer, bigger basins",
  "ranking & prioritization": "Prioritize differently? e.g. weight life safety more, or focus on critical infrastructure",
  "the recommendation": "Anything off in the write-up? e.g. tone down a claim, or focus on the top basin",
};
const DEFAULT_CHECKPOINT_PROMPT = "Have a comment on this step?";

// Mid-run feedback (see web_app/app.py's agent_analyze_stream: three
// checkpoints -- persistence threshold, ranking & prioritization, the
// recommendation -- each poll for whatever's here the moment they open and
// loop back to re-decide/re-rank/re-write using it). Fire-and-forget: the
// checkpoint that picks this up (if any -- a checkpoint may have already
// closed, or the run may not have reached one yet) surfaces what happened
// as its own agent_decision step in the log above, so this handler doesn't
// need to poll for or display a result of its own.
async function sendInterjection() {
  const input = document.getElementById("interject-input");
  const message = input.value.trim();
  const statusEl = document.getElementById("interject-status");
  if (!message || !currentRunId) return;
  const formData = new FormData();
  formData.append("run_id", currentRunId);
  formData.append("message", message);
  input.value = "";
  statusEl.hidden = false;
  statusEl.textContent = "Sent -- the agent will pick it up at its next checkpoint.";
  try {
    await fetch("/api/agent/interject", { method: "POST", body: formData });
  } catch (err) {
    statusEl.textContent = "Couldn't send that -- " + err.message;
  }
}
document.getElementById("interject-send").addEventListener("click", sendInterjection);
document.getElementById("interject-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); sendInterjection(); }
});

// The feedback box (Send) and the countdown + Continue button together --
// both live only while a checkpoint is actually open (see web_app/app.py's
// checkpoint_wait, which emits "checkpoint_open" -- carrying wait_seconds
// and which checkpoint it is -- right before it starts polling, and
// "checkpoint_close" however the wait ends: real feedback, Continue, or
// timeout). Outside that window there's nothing listening server-side for
// either box to send to, so both stay hidden the rest of the time.
//
// The countdown auto-clicks Continue when it reaches 0 -- unless the user
// has clicked into the feedback input first, which freezes the countdown
// (no more ticking, no auto-continue) until they click Continue by hand.
// That's a deliberate override of the input's own focus, not a generic
// "any interaction pauses it": clicking into the box is a clear signal the
// user is about to type something, so auto-advancing out from under them
// would cut them off mid-thought.
function openCheckpointUI(waitSeconds, checkpointLabel) {
  closeCheckpointUI(); // guard against a stray still-running timer from an earlier checkpoint
  const row = document.getElementById("checkpoint-row");
  const countdownEl = document.getElementById("checkpoint-countdown");
  let remaining = waitSeconds;
  row.hidden = false;
  countdownEl.textContent = `auto-continuing in ${remaining}s…`;
  checkpointCountdownTimer = setInterval(() => {
    remaining -= 1;
    if (remaining > 0) {
      countdownEl.textContent = `auto-continuing in ${remaining}s…`;
      return;
    }
    clearInterval(checkpointCountdownTimer);
    checkpointCountdownTimer = null;
    countdownEl.textContent = "Continuing…";
    sendContinue();
  }, 1000);

  const interjectRow = document.getElementById("interject-row");
  const input = document.getElementById("interject-input");
  interjectRow.hidden = false;
  input.placeholder = CHECKPOINT_PROMPTS[checkpointLabel] || DEFAULT_CHECKPOINT_PROMPT;
  input.value = "";
  document.getElementById("interject-status").hidden = true;
  // No auto-focus here -- focusing this box is exactly the signal that
  // freezes the countdown below, so doing it programmatically on open
  // would freeze every checkpoint's countdown instantly.
}
function closeCheckpointUI() {
  if (checkpointCountdownTimer !== null) {
    clearInterval(checkpointCountdownTimer);
    checkpointCountdownTimer = null;
  }
  document.getElementById("checkpoint-row").hidden = true;
  document.getElementById("interject-row").hidden = true;
  const input = document.getElementById("interject-input");
  input.value = "";
  input.blur(); // explicit, not left to the browser -- #interject-input is
  // the SAME element reused across all 3 checkpoints (just shown/hidden),
  // so if it's still focused when this checkpoint closes, leaving that
  // focus in place risks the NEXT checkpoint's freeze-on-focus handler
  // firing again with no real click behind it (observed while recording a
  // demo: checkpoint 2 showed "Timer stopped" despite never being clicked,
  // right after checkpoint 1's hold-then-Continue interaction).
  document.getElementById("interject-status").hidden = true;
}
document.getElementById("interject-input").addEventListener("focus", () => {
  if (checkpointCountdownTimer === null) return; // no checkpoint open (or already frozen/expired) -- nothing to freeze
  clearInterval(checkpointCountdownTimer);
  checkpointCountdownTimer = null;
  document.getElementById("checkpoint-countdown").textContent = "Timer stopped -- click Continue when you're ready.";
  // The countdown above is purely a client-side display -- freezing it
  // doesn't touch web_app/app.py's own independent wait loop, which would
  // otherwise still time the checkpoint out on its own and effectively
  // "auto-continue" regardless of what the UI shows. This is what actually
  // tells the backend to stop counting down (see checkpoint_wait's
  // handling of HOLD_SIGNAL).
  if (currentRunId) {
    const formData = new FormData();
    formData.append("run_id", currentRunId);
    fetch("/api/agent/checkpoint_hold", { method: "POST", body: formData }).catch(() => {});
  }
});
async function sendContinue() {
  if (!currentRunId) return;
  document.getElementById("checkpoint-continue").disabled = true;
  const formData = new FormData();
  formData.append("run_id", currentRunId);
  try {
    await fetch("/api/agent/continue", { method: "POST", body: formData });
  } finally {
    document.getElementById("checkpoint-continue").disabled = false;
  }
  // Doesn't hide the row itself -- the stream's own "checkpoint_close"
  // event (arriving within ~1s, once checkpoint_wait's poll loop notices
  // CONTINUE_SIGNAL) is the one reliable signal that the checkpoint has
  // actually closed server-side.
}
document.getElementById("checkpoint-continue").addEventListener("click", sendContinue);

// The "Report" toggle lives in the LEFT panel, styled and behaving exactly
// like the terminal panel's own toggle above it (#report-toggle-panel
// reuses .terminal-panel/.terminal-toggle as-is -- same collapsible-widget
// look, just with no body of its own), but instead of expanding a body in
// place it shows/hides #report-card, a floating window over the map (see
// initFloatingReportWindow below for its drag/resize behavior). Resets to
// shown every time a fresh report lands (showReport), rather than staying
// hidden across runs. The resize grip is a child of #report-card now, so
// hiding the card hides it too -- nothing else to toggle here.
function setReportPanelHidden(hide) {
  document.getElementById("report-card").hidden = hide;
  document.getElementById("report-toggle-panel").classList.toggle("open", !hide);
}
document.getElementById("report-toggle").addEventListener("click", () => {
  setReportPanelHidden(!document.getElementById("report-card").hidden);
});

function showReport(event, placeLabel, options) {
  options = options || {};
  document.getElementById("report-toggle-panel").hidden = false;
  document.getElementById("report-toggle-label").textContent = `Report -- ${placeLabel}`;
  document.getElementById("chat-panel").hidden = false;
  setReportPanelHidden(false);
  document.getElementById("report-title").textContent = `Report -- ${placeLabel}`;
  document.getElementById("report-frame").srcdoc = event.report_body_html;

  const blob = new Blob([event.report_html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const link = document.getElementById("report-download");
  link.href = url;
  const safeName = placeLabel.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
  link.download = `flood-report-${safeName || "location"}.html`;

  qaContext = event.qa_context || null;
  // A chat-triggered re-run (e.g. "redo this at 30m resolution") keeps the
  // conversation visible -- the request that caused the re-run is itself
  // part of that conversation, wiping it the moment the new report lands
  // would erase the question right as it gets answered.
  if (!options.preserveChat) {
    chatHistory = [];
    document.getElementById("chat-log").innerHTML = "";
    setStatus("chat-status", "");
  }

  if (event.map_data) populateMap(event.map_data);
}

async function runAgentAnalysis(place, exposureFile, resolutionMeters, options) {
  options = options || {};
  const drawnPolygon = options.drawnPolygon || null;
  const elevationFile = options.elevationFile || null;
  const boundaryShapefile = options.boundaryShapefile || null;
  const customWeights = options.customWeights || null;
  lastAnalyzedPlace = place;
  lastDrawnPolygon = drawnPolygon;
  lastCustomWeights = customWeights; // sticky across re-runs (Modify-weights sliders), so a chat-triggered re-run (e.g. resolution change) keeps whichever weighting the user picked, instead of silently reverting to the entropy default
  const label = place || "custom drawn area";
  document.getElementById("terminal-panel").hidden = false;
  setTerminalOpen(true);
  document.getElementById("terminal-toggle-label").textContent = `Analyzing ${label}…`;
  document.getElementById("terminal-dot").className = "step-dot live";
  document.getElementById("progress-place").textContent = label;
  document.getElementById("progress-log").innerHTML = "";
  document.getElementById("progress-error").hidden = true;
  document.getElementById("progress-retry").hidden = true;
  document.getElementById("report-card").hidden = true;
  document.getElementById("report-toggle-panel").hidden = true;
  document.getElementById("chat-panel").hidden = true;
  currentRunId = null;
  document.getElementById("interject-row").hidden = true;
  document.getElementById("interject-input").value = "";
  document.getElementById("interject-status").hidden = true;
  closeCheckpointUI();
  setStatus("intro-status", "");

  const formData = new FormData();
  if (place) formData.append("place", place);
  if (boundaryShapefile) formData.append("boundary_shapefile", boundaryShapefile);
  if (drawnPolygon) formData.append("boundary_polygon", JSON.stringify(drawnPolygon));
  if (elevationFile) formData.append("elevation_file", elevationFile);
  if (exposureFile) formData.append("exposure_file", exposureFile);
  if (resolutionMeters) formData.append("resolution_meters", String(resolutionMeters));
  // Read live rather than threaded through `options` -- the box's current
  // value (whatever the user has typed, possibly since the last run) is
  // always what a fresh or re-run analysis should use; see geoai_agent.
  // agent_decisions.decide_persistence_threshold for what this actually drives.
  const objectiveInput = document.getElementById("objective-input");
  const objectiveText = objectiveInput ? objectiveInput.value.trim() : "";
  if (objectiveText) formData.append("objective_text", objectiveText);
  // Every run uses entropy weighting (flood_tools.entropy_weights) unless
  // the user has hand-set weights via the Modify-weights sliders -- there
  // is no other choice to send here anymore (see web_app/app.py's
  // weighting_method handling).
  if (customWeights) {
    formData.append("weighting_method", "custom");
    formData.append("custom_weights", JSON.stringify(customWeights));
  }

  // Flown to as soon as the FIRST step carrying bounds arrives (boundary/
  // geocoding resolves almost immediately, long before the rest of the
  // pipeline -- elevation fetch, TTK, ranking, agent decisions, the LLM
  // recommendation -- which can take minutes). Without this the map just
  // sits on whatever it showed before until the whole run finishes, giving
  // no visual feedback that anything is happening yet. No-ops on every
  // subsequent bounds-bearing step this same run (there's normally only
  // one anyway) so this doesn't re-fly and interrupt a user who's already
  // panned/zoomed elsewhere while waiting.
  let flewToBoundsThisRun = false;
  function flyToBoundsOnce(bounds) {
    if (flewToBoundsThisRun || !bounds || bounds.length !== 4) return;
    flewToBoundsThisRun = true;
    const [latMin, latMax, lonMin, lonMax] = bounds;
    document.getElementById("map-empty-state").hidden = true;
    persistentMap.flyToBounds([[latMin, lonMin], [latMax, lonMax]]);
  }

  try {
    const res = await fetch("/api/agent/analyze_stream", { method: "POST", body: formData });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `request failed (${res.status})`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // last element may be a partial line -- keep it for the next chunk

      for (const rawLine of lines) {
        const trimmed = rawLine.trim();
        if (!trimmed) continue;
        const event = JSON.parse(trimmed);

        if (event.type === "run_id") {
          // Identifies this run to POST /api/agent/interject and /api/
          // agent/continue -- see web_app/app.py's agent_analyze_stream.
          // The feedback box itself only appears once a checkpoint_open
          // event actually arrives (see openCheckpointUI) -- nothing is
          // listening for it outside that window, so showing it earlier
          // would invite a comment that just sits unseen.
          currentRunId = event.run_id;
        } else if (event.type === "step") {
          addProgressStep(event);
          if (event.detail && event.detail.bounds) flyToBoundsOnce(event.detail.bounds);
        } else if (event.type === "checkpoint_open") {
          openCheckpointUI(event.wait_seconds, event.checkpoint);
        } else if (event.type === "checkpoint_close") {
          closeCheckpointUI();
        } else if (event.type === "agent_decision") {
          addAgentDecisionStep(event);
        } else if (event.type === "error") {
          finalizeProgressDots();
          currentRunId = null;
          document.getElementById("interject-row").hidden = true;
          closeCheckpointUI();
          const errEl = document.getElementById("progress-error");
          errEl.textContent = event.message;
          errEl.hidden = false;
          document.getElementById("progress-retry").hidden = false;
          document.getElementById("terminal-toggle-label").textContent = "Analysis failed";
          document.getElementById("terminal-dot").className = "step-dot error";
          if (options.preserveChat) addChatMessage("assistant", "The re-run failed: " + event.message, "error");
          return;
        } else if (event.type === "done") {
          finalizeProgressDots();
          enableModifyButtons();
          currentRunId = null;
          document.getElementById("interject-row").hidden = true;
          closeCheckpointUI();
          lastAnalyzedPlace = event.place_label || place;
          document.getElementById("terminal-toggle-label").textContent = `Done in ${event.total_elapsed_seconds}s -- ${lastAnalyzedPlace}`;
          document.getElementById("terminal-dot").className = "step-dot";
          setTerminalOpen(false);
          showReport(event, lastAnalyzedPlace, options);
          if (options.preserveChat) addChatMessage("assistant", `Done (${event.total_elapsed_seconds}s) -- map and report above are now from the new run. Ask away.`);
          return;
        }
      }
    }
  } catch (err) {
    finalizeProgressDots();
    currentRunId = null;
    document.getElementById("interject-row").hidden = true;
    closeCheckpointUI();
    const errEl = document.getElementById("progress-error");
    errEl.textContent = "Error: " + err.message;
    errEl.hidden = false;
    document.getElementById("progress-retry").hidden = false;
    document.getElementById("terminal-toggle-label").textContent = "Analysis failed";
    document.getElementById("terminal-dot").className = "step-dot error";
    if (options.preserveChat) addChatMessage("assistant", "The re-run failed: " + err.message, "error");
  }
}

document.getElementById("place-submit").addEventListener("click", () => {
  const place = document.getElementById("place-input").value.trim();
  if (!place) {
    setStatus("intro-status", "Type a place first.", true);
    return;
  }
  setStatus("intro-status", "");
  pendingElevationFile = null;
  pendingBoundaryShapefile = null;
  pendingExposureFile = null; // a fresh named-place run shouldn't silently reuse a file from an unrelated earlier run
  runAgentAnalysis(place, null, null, {});
});
document.getElementById("place-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); document.getElementById("place-submit").click(); }
});

function resetToIntro() {
  if (drawMode) cancelDrawMode();
  pendingElevationFile = null;
  pendingBoundaryShapefile = null;
  document.getElementById("report-card").hidden = true;
  document.getElementById("report-toggle-panel").hidden = true;
  document.getElementById("chat-panel").hidden = true;
  document.getElementById("place-input").value = "";
  document.getElementById("place-input").focus();
  setStatus("intro-status", "");
}
document.getElementById("progress-retry").addEventListener("click", resetToIntro);

// -- floating report window: drag by the toolbar, resize by the corner ----
//
// #report-card.floating-window's CSS default anchors it to a viewport
// corner (top/right). Dragging switches it to explicit inline left/top the
// first time (wins over the class rule for the same property, same
// "inline beats class" pattern the old bottom-strip resize used) and
// clears `right` so the two don't fight over the horizontal position.
// Resizing sets explicit inline width/height directly -- position:fixed
// takes the window out of any flex flow, so there's no flex-basis
// quirk to work around here. Pointer Events (not separate mouse/touch
// listeners) so one code path handles both input types.

function initFloatingReportWindow() {
  const reportCard = document.getElementById("report-card");
  const dragHandle = document.getElementById("report-drag-handle");
  const resizeHandle = document.getElementById("report-resize-handle");
  const MARGIN_PX = 8; // keeps the toolbar reachable even if dragged toward an edge

  // -- drag --
  let dragging = false;
  let dragOffsetX = 0;
  let dragOffsetY = 0;

  dragHandle.addEventListener("pointerdown", (e) => {
    if (e.target.closest("a, button")) return; // the download link/buttons in the toolbar shouldn't start a drag
    const rect = reportCard.getBoundingClientRect();
    // Take over positioning via left/top from here on -- clearing `right`
    // stops the CSS class's right-anchored default from also applying,
    // which would otherwise fight with the inline left we're about to set.
    reportCard.style.right = "auto";
    reportCard.style.left = `${rect.left}px`;
    reportCard.style.top = `${rect.top}px`;
    dragOffsetX = e.clientX - rect.left;
    dragOffsetY = e.clientY - rect.top;
    dragging = true;
    dragHandle.setPointerCapture(e.pointerId);
    e.preventDefault(); // avoid selecting page text while dragging
  });
  dragHandle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const rect = reportCard.getBoundingClientRect();
    const maxLeft = window.innerWidth - MARGIN_PX;
    const maxTop = window.innerHeight - MARGIN_PX;
    const left = Math.min(maxLeft - rect.width, Math.max(MARGIN_PX - rect.width + 40, e.clientX - dragOffsetX));
    const top = Math.min(maxTop - 32, Math.max(0, e.clientY - dragOffsetY));
    reportCard.style.left = `${left}px`;
    reportCard.style.top = `${top}px`;
  });
  function stopDragging(e) {
    if (!dragging) return;
    dragging = false;
    if (dragHandle.hasPointerCapture(e.pointerId)) dragHandle.releasePointerCapture(e.pointerId);
  }
  dragHandle.addEventListener("pointerup", stopDragging);
  dragHandle.addEventListener("pointercancel", stopDragging);

  // -- resize (bottom-right corner grip) --
  let resizing = false;
  let resizeStartX = 0;
  let resizeStartY = 0;
  let resizeStartWidth = 0;
  let resizeStartHeight = 0;

  resizeHandle.addEventListener("pointerdown", (e) => {
    const rect = reportCard.getBoundingClientRect();
    resizeStartX = e.clientX;
    resizeStartY = e.clientY;
    resizeStartWidth = rect.width;
    resizeStartHeight = rect.height;
    resizing = true;
    resizeHandle.classList.add("dragging");
    resizeHandle.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  resizeHandle.addEventListener("pointermove", (e) => {
    if (!resizing) return;
    const rect = reportCard.getBoundingClientRect();
    const maxWidth = window.innerWidth - rect.left - MARGIN_PX;
    const maxHeight = window.innerHeight - rect.top - MARGIN_PX;
    const width = Math.min(maxWidth, Math.max(380, resizeStartWidth + (e.clientX - resizeStartX)));
    const height = Math.min(maxHeight, Math.max(240, resizeStartHeight + (e.clientY - resizeStartY)));
    reportCard.style.width = `${width}px`;
    reportCard.style.height = `${height}px`;
  });
  function stopResizing(e) {
    if (!resizing) return;
    resizing = false;
    resizeHandle.classList.remove("dragging");
    if (resizeHandle.hasPointerCapture(e.pointerId)) resizeHandle.releasePointerCapture(e.pointerId);
  }
  resizeHandle.addEventListener("pointerup", stopResizing);
  resizeHandle.addEventListener("pointercancel", stopResizing);
}

initPersistentMap();
initFloatingReportWindow();
