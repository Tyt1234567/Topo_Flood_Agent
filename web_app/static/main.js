"use strict";

function setStatus(id, message, isError) {
  const el = document.getElementById(id);
  el.textContent = message || "";
  el.classList.toggle("error", !!isError);
}

// -- modals -----------------------------------------------------------------

function openModal(id) {
  document.getElementById(id).hidden = false;
}

function closeModal(id) {
  document.getElementById(id).hidden = true;
}

document.querySelectorAll(".modal-overlay").forEach((overlay) => {
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.hidden = true; // click outside the modal card closes it
  });
});
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  document.querySelectorAll(".modal-overlay").forEach((overlay) => { overlay.hidden = true; });
});

// -- shared GeoJSON -> Leaflet layer helpers -------------------------------

function fieldLayer(geojson) {
  return L.geoJSON(geojson, {
    style: (feature) => ({
      fillColor: feature.properties.color,
      color: feature.properties.color,
      weight: 0,
      fillOpacity: 0.65,
    }),
  });
}

function pointsLayer(geojson, labelPrefix) {
  return L.geoJSON(geojson, {
    pointToLayer: (feature, latlng) =>
      L.circleMarker(latlng, {
        radius: 6, color: "#222", weight: 1,
        fillColor: feature.properties.color, fillOpacity: 0.9,
      }),
    onEachFeature: (feature, layer) => {
      const p = feature.properties;
      const persistence = p.persistence === null ? "infinite (global extremum)" : p.persistence.toFixed(3);
      layer.bindPopup(
        `<b>${labelPrefix} - ${p.kind}</b><br>value: ${p.value.toFixed(3)}<br>persistence: ${persistence}`
      );
    },
  });
}

// -- Real-World Data ---------------------------------------------------------

let realworldMap;
let realworldTileLayer;
let realworldLayerControl;
const realworldOverlays = {};
let sqPreviewRect = null; // outline of the bounding box shown right after parsing, before raw data loads
let rwDatasetId = null; // server-cached raw dataset from the last load; reused by Analyze so data isn't fetched twice
let rwElevationSource = "auto"; // "auto" tries USGS 3DEP first for elevation, falling back to Open-Meteo automatically

function initRealworldMap() {
  realworldMap = L.map("realworld-map");
  realworldMap.setView([40.0, -105.3], 11);
  realworldTileLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(realworldMap);
}

function clearRealworldOverlays() {
  Object.values(realworldOverlays).forEach((layer) => realworldMap.removeLayer(layer));
  for (const key in realworldOverlays) delete realworldOverlays[key];
  if (realworldLayerControl) realworldMap.removeControl(realworldLayerControl);
  realworldLayerControl = null;
}

// Switches the map to the resolved location as soon as it's known (right
// after Parse, before the user even clicks "Fetch live & analyze"), so
// they can see where the analysis will run and back out if it's wrong.
function previewBounds(bounds, popupText) {
  const [latMin, latMax, lonMin, lonMax] = bounds;
  const leafletBounds = [[latMin, lonMin], [latMax, lonMax]];
  if (sqPreviewRect) realworldMap.removeLayer(sqPreviewRect);
  sqPreviewRect = L.rectangle(leafletBounds, {
    color: "#2b6cb0", weight: 2, fillOpacity: 0.08, dashArray: "6",
  }).addTo(realworldMap);
  if (popupText) sqPreviewRect.bindPopup(popupText).openPopup();
  realworldMap.fitBounds(leafletBounds, { maxZoom: 13 });
}

document.getElementById("rw-advanced-open").addEventListener("click", () => openModal("rw-advanced-modal"));
document.getElementById("rw-advanced-close").addEventListener("click", () => closeModal("rw-advanced-modal"));

let customLayerCount = 0;
document.getElementById("rw-add-layer").addEventListener("click", () => {
  customLayerCount += 1;
  const row = document.createElement("div");
  row.className = "custom-layer-row";
  row.innerHTML = `
    <input type="text" placeholder="layer name" class="rw-layer-name">
    <input type="file" accept=".csv,.zip,.shp" class="rw-layer-file">
    <input type="text" placeholder="what does this measure? (optional, helps the recommendation)" class="rw-layer-description">
    <button type="button" class="secondary rw-remove-layer">x</button>
  `;
  row.querySelector(".rw-remove-layer").addEventListener("click", () => row.remove());
  document.getElementById("rw-custom-layers").appendChild(row);
});

function buildRealworldFormData() {
  const formData = new FormData();
  formData.append("bounds", document.getElementById("rw-bounds").value.trim());
  formData.append("resolution_meters", document.getElementById("rw-resolution-meters").value);
  formData.append("primary_variable", document.getElementById("rw-variable").value);
  formData.append("use_sample_layer", document.getElementById("rw-sample").checked ? "true" : "false");
  formData.append("elevation_source", rwElevationSource);

  document.querySelectorAll(".custom-layer-row").forEach((row) => {
    const name = row.querySelector(".rw-layer-name").value.trim();
    const file = row.querySelector(".rw-layer-file").files[0];
    const description = row.querySelector(".rw-layer-description").value.trim();
    if (name && file) {
      formData.append("layer_name", name);
      formData.append(`layer_file::${name}`, file);
      if (description) formData.append(`layer_description::${name}`, description);
    }
  });
  return formData;
}

// Fetches just the raw scalar field(s) -- no critical-point analysis, no
// recommendation -- as soon as a location is confirmed, rather than only
// as part of the final Analyze step. Renders the real heatmap on the map
// immediately and caches the dataset server-side (rwDatasetId) so the
// later Analyze call can reuse it instead of fetching all over again.
async function loadRawDataPreview() {
  const formData = buildRealworldFormData();
  const variable = document.getElementById("rw-variable").value;
  setStatus("rw-status", (variable === "elevation" && rwElevationSource !== "open_meteo")
    ? "Trying USGS 3DEP first (one request per point -- this can take up to a minute or two)..."
    : "Loading raw data for the confirmed location...");
  try {
    const res = await fetch("/api/realworld/load_data", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "request failed");
    rwDatasetId = data.dataset_id;
    if (data.elevation_source) rwElevationSource = data.elevation_source;

    realworldMap.fitBounds(data.bounds);
    clearRealworldOverlays();
    if (sqPreviewRect) {
      realworldMap.removeLayer(sqPreviewRect);
      sqPreviewRect = null;
    }
    let first = true;
    for (const [name, layer] of Object.entries(data.layers)) {
      const field = fieldLayer(layer.field);
      realworldOverlays[name + " (raw field)"] = field;
      if (first) { field.addTo(realworldMap); first = false; }
    }
    realworldLayerControl = L.control.layers(null, realworldOverlays, { collapsed: false }).addTo(realworldMap);
    setStatus("rw-status", data.elevation_source_fallback_note ||
      "Raw data loaded -- click \"Fetch & Analyze\" to find critical points.");
  } catch (err) {
    setStatus("rw-status", "Error loading raw data: " + err.message, true);
  }
}

// Any manual edit invalidates the cached raw dataset -- it no longer
// matches what's in the fields, so Analyze must fetch fresh instead of
// reusing stale data. It also resets the elevation source back to the
// default: source selection came from analyzing the query's own wording,
// which no longer applies once the user is editing fields by hand. Only
// fires on genuine user edits: Smart Query sets these fields' .value
// programmatically, which does not dispatch input events, so the
// auto-load in renderSmartQueryClarify is unaffected.
function invalidateRwDataset() {
  rwDatasetId = null;
  rwElevationSource = "auto";
}
["rw-bounds", "rw-variable", "rw-resolution-meters"].forEach((id) => {
  document.getElementById(id).addEventListener("input", invalidateRwDataset);
  document.getElementById(id).addEventListener("change", invalidateRwDataset);
});
document.getElementById("rw-add-layer").addEventListener("click", invalidateRwDataset);
document.getElementById("rw-sample").addEventListener("change", invalidateRwDataset);

async function submitRealworldAnalyze(feedback) {
  const formData = buildRealworldFormData();
  const recommend = document.getElementById("rw-recommend").checked;
  formData.append("recommend", recommend ? "true" : "false");
  if (feedback) formData.append("feedback", feedback);
  if (rwDatasetId) formData.append("dataset_id", rwDatasetId);

  setStatus("rw-status", feedback
    ? "Re-running with your feedback..."
    : rwDatasetId
      ? "Running Morse analysis on the already-loaded data..."
      : "Fetching real-world data and running Morse analysis...");
  document.getElementById("rw-result").hidden = true;
  try {
    const res = await fetch("/api/realworld/analyze", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "request failed");
    renderRealworldResult(data);
    setStatus("rw-status", "");
  } catch (err) {
    setStatus("rw-status", "Error: " + err.message, true);
  }
}

// Mirrors resolution_for_target_spacing / ground_spacing_meters in
// geoai_agent/real_data.py -- shows the grid point count and actually
// achievable per-axis spacing a requested meters-based pixel size resolves
// to, including when it gets capped, before the request is even sent. Each
// axis (longitude/x, latitude/y) is sized independently from the box's own
// physical length on that axis, so a non-square box gets a non-square
// point count instead of both axes being forced to share one number.
const RW_MIN_GRID_RESOLUTION = 6;
const RW_MAX_GRID_RESOLUTION = 80;
const RW_MAX_USGS_GRID_RESOLUTION = 10;

function updateGroundResolutionHint() {
  const el = document.getElementById("rw-ground-resolution");
  const parts = document.getElementById("rw-bounds").value.split(",").map(Number);
  const meters = Number(document.getElementById("rw-resolution-meters").value);
  if (parts.length !== 4 || parts.some(Number.isNaN) || !meters || meters <= 0) {
    el.textContent = "~ grid points";
    return;
  }
  const [latMin, latMax, lonMin, lonMax] = parts;
  const midLatRad = ((latMin + latMax) / 2) * (Math.PI / 180);
  const metersPerDegLat = 111320;
  const metersPerDegLon = 111320 * Math.cos(midLatRad);
  const lonSpanM = Math.abs(lonMax - lonMin) * metersPerDegLon;
  const latSpanM = Math.abs(latMax - latMin) * metersPerDegLat;

  const axisPoints = (spanM) => Math.max(
    RW_MIN_GRID_RESOLUTION, Math.min(Math.ceil(spanM / meters) + 1, RW_MAX_GRID_RESOLUTION)
  );
  const nx = axisPoints(lonSpanM);
  const ny = axisPoints(latSpanM);
  const cappedByMax = nx === RW_MAX_GRID_RESOLUTION || ny === RW_MAX_GRID_RESOLUTION;
  const achievedLon = lonSpanM / (nx - 1);
  const achievedLat = latSpanM / (ny - 1);

  let note = "";
  if (cappedByMax) note = " (capped for practicality)";
  else if (nx > RW_MAX_USGS_GRID_RESOLUTION || ny > RW_MAX_USGS_GRID_RESOLUTION) {
    note = ` -- if USGS is used for elevation, it caps lower (${RW_MAX_USGS_GRID_RESOLUTION}x${RW_MAX_USGS_GRID_RESOLUTION})`;
  }
  el.textContent = `~${nx}x${ny} grid (${nx * ny} points, ` +
    `~${Math.round(achievedLon)}m x ~${Math.round(achievedLat)}m between samples)${note}`;
}
document.getElementById("rw-bounds").addEventListener("input", updateGroundResolutionHint);
document.getElementById("rw-resolution-meters").addEventListener("input", updateGroundResolutionHint);

function renderRealworldResult(data) {
  realworldMap.fitBounds(data.bounds);
  clearRealworldOverlays();
  if (sqPreviewRect) {
    realworldMap.removeLayer(sqPreviewRect);
    sqPreviewRect = null;
  }

  const summaryContainer = document.getElementById("rw-layer-summaries");
  summaryContainer.innerHTML = "";

  if (data.elevation_source) {
    const sourceNote = document.createElement("p");
    sourceNote.className = "hint";
    sourceNote.textContent = `Elevation data source actually used: ${data.elevation_source}`;
    summaryContainer.appendChild(sourceNote);
  }

  let first = true;
  for (const [name, layer] of Object.entries(data.layers)) {
    const field = fieldLayer(layer.field);
    const points = pointsLayer(layer.points, name);
    realworldOverlays[name + " (field)"] = field;
    realworldOverlays[name + " (critical points)"] = points;
    if (first) {
      field.addTo(realworldMap);
      points.addTo(realworldMap);
      first = false;
    }

    const s = layer.summary;
    const euler = s.euler_characteristic;
    const [ny, nx] = s.shape;
    const div = document.createElement("div");
    div.className = "layer-summary";
    div.innerHTML = `<b>${name}</b> (${nx}x${ny} grid, ${nx * ny} points)<br>
      range: [${s.value_range[0].toFixed(2)}, ${s.value_range[1].toFixed(2)}]
      &nbsp;|&nbsp; significant pit basins: ${s.num_significant_basins}
      &nbsp;|&nbsp; significant peak basins: ${s.num_significant_peak_basins}
      &nbsp;|&nbsp; threshold: ${s.persistence_threshold.toFixed(3)}
      <br>domain Euler characteristic: ${euler.domain_euler_characteristic} (disk, expected 1)`;
    summaryContainer.appendChild(div);
  }

  realworldLayerControl = L.control.layers(null, realworldOverlays, { collapsed: false }).addTo(realworldMap);

  document.getElementById("rw-result").hidden = false;
  document.getElementById("rw-recommendation").textContent =
    data.recommendation || "(not requested)";
  document.getElementById("rw-feedback-box").style.display = data.recommendation ? "" : "none";
  document.getElementById("rw-feedback").value = "";
}

document.getElementById("rw-submit").addEventListener("click", () => submitRealworldAnalyze());
document.getElementById("rw-rerun").addEventListener("click", () => {
  const feedback = document.getElementById("rw-feedback").value.trim();
  if (!feedback) {
    setStatus("rw-status", "Type feedback first, or click Accept.", true);
    return;
  }
  submitRealworldAnalyze(feedback);
});
document.getElementById("rw-accept").addEventListener("click", () => {
  document.getElementById("rw-feedback-box").style.display = "none";
  setStatus("rw-status", "Recommendation accepted.");
});

// -- Smart Query parsing (free-text -> fills in the fields above) -----------
//
// Parsing only ever fills in rw-bounds/rw-variable below and previews the
// area on the map -- it never fetches data itself. There is exactly one
// "Fetch & Analyze" button (submitRealworldAnalyze, above) and exactly one
// bounds/topic state (the rw-* fields), so there is no way to click a
// "run" action against a stale or different location than what was just
// parsed.

async function submitSmartQueryParse() {
  const query = document.getElementById("sq-query").value.trim();
  if (!query) {
    setStatus("sq-status", "Describe what to analyze first.", true);
    return;
  }
  setStatus("sq-status", "Parsing bounding box and topic...");

  const formData = new FormData();
  formData.append("query", query);

  try {
    const res = await fetch("/api/realworld/smart_query", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "request failed");
    renderSmartQueryClarify(data);
    setStatus("sq-status", "");
  } catch (err) {
    setStatus("sq-status", "Error: " + err.message, true);
  }
}

function renderSmartQueryClarify(data) {
  const [latMin, latMax, lonMin, lonMax] = data.bounds;
  const place = data.resolved_place;
  const placeLine = place
    ? `<br><b>Resolved place:</b> ${place.name}, ${place.admin1 || ""} ${place.country || ""} ` +
      `(${place.latitude.toFixed(4)}, ${place.longitude.toFixed(4)})`
    : "";

  previewBounds(
    data.bounds,
    place ? `${place.name}${place.admin1 ? ", " + place.admin1 : ""}` : "Bounding box to analyze"
  );

  document.getElementById("rw-bounds").value =
    [latMin, latMax, lonMin, lonMax].map((v) => v.toFixed(5)).join(",");
  document.getElementById("rw-variable").value = data.variable;
  rwElevationSource = data.elevation_source || "auto";

  let resolutionLine = "";
  if (data.requested_resolution_meters) {
    const requested = data.requested_resolution_meters;
    const [achievedLon, achievedLat] = data.achieved_resolution_meters;
    const achievedWorst = Math.max(achievedLon, achievedLat); // the coarser of the two axes
    document.getElementById("rw-resolution-meters").value = requested;
    const closeEnough = achievedWorst <= requested * 1.1; // within ~10% counts as "hit the target"
    const achievedText = Math.round(achievedLon) === Math.round(achievedLat)
      ? `~${Math.round(achievedWorst)}m`
      : `~${Math.round(achievedLon)}m x ~${Math.round(achievedLat)}m`;
    resolutionLine = closeEnough
      ? `<br><b>Resolution:</b> ${achievedText} between samples, close to the requested ${requested}m.`
      : `<br><b>Resolution:</b> this bounding box is too large to actually reach ${requested}m at a ` +
        `practical grid size; the closest achievable is ${achievedText}. Shrink the bounding ` +
        "box for finer resolution.";
  }

  const sourceLine = data.source_note ? `<br><b>Data source:</b> ${data.source_note}` : "";

  document.getElementById("sq-clarify-summary").innerHTML =
    `<b>Bounding box:</b> ${latMin.toFixed(3)}, ${latMax.toFixed(3)}, ${lonMin.toFixed(3)}, ${lonMax.toFixed(3)}` +
    placeLine +
    `<br><b>Topic:</b> ${data.variable}` +
    resolutionLine +
    sourceLine;
  updateGroundResolutionHint();
  openModal("sq-modal"); // confirm before loading anything -- especially important for the slow USGS path
}

document.getElementById("sq-parse").addEventListener("click", submitSmartQueryParse);
document.getElementById("sq-modal-confirm").addEventListener("click", () => {
  closeModal("sq-modal");
  loadRawDataPreview(); // fetch the raw field as soon as the location is confirmed, not at the end of the pipeline
});
document.getElementById("sq-modal-edit").addEventListener("click", () => {
  closeModal("sq-modal");
  setStatus("sq-status", "Edit the fields below, then click \"Fetch & Analyze\" when ready.");
});

// -- init ---------------------------------------------------------------

initRealworldMap();
updateGroundResolutionHint();
