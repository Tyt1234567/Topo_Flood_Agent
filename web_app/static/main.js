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
let mapLayers = { heat: null, allMinima: null, boundary: null, catchment: null, reportFocus: null };
let mapLayerControl = null;
let mapLegendControl = null;
let mapSliderControl = null;
let currentMapData = null;
let persistenceThreshold = 0;

function initPersistentMap() {
  persistentMap = L.map("persistent-map").setView([39.5, -98.35], 4);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(persistentMap);
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
  mapLayers = { heat: null, allMinima: null, boundary: null, catchment: null, reportFocus: null };
  if (mapLayerControl) { persistentMap.removeControl(mapLayerControl); mapLayerControl = null; }
  if (mapLegendControl) { persistentMap.removeControl(mapLegendControl); mapLegendControl = null; }
  if (mapSliderControl) { persistentMap.removeControl(mapSliderControl); mapSliderControl = null; }
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
  document.getElementById("catchment-clear-row").hidden = catchmentId === null || catchmentId === undefined;
  if (catchmentId === null || catchmentId === undefined || !currentMapData) return;
  const cells = currentMapData.field_geojson.features.filter((f) => f.properties.catchment_id === catchmentId);
  mapLayers.catchment = L.geoJSON({ type: "FeatureCollection", features: cells }, {
    style: { fillColor: "#2b6cb0", color: "#12335c", weight: 1, fillOpacity: 0.55 },
    interactive: false,
  }).addTo(persistentMap);
  mapLayers.catchment.bringToFront();
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

  mapLayers.heat = L.geoJSON(mapData.field_geojson, {
    style: (f) => ({ fillColor: f.properties.color, color: f.properties.color, weight: 0, fillOpacity: 0.8 }),
  }).addTo(persistentMap);

  const overlays = { "Elevation (terrain)": mapLayers.heat };

  if (mapData.boundary_geojson) {
    mapLayers.boundary = L.geoJSON(mapData.boundary_geojson, {
      style: { fill: false, color: "#a8492c", weight: 2.5 },
    }).addTo(persistentMap);
    overlays["Boundary"] = mapLayers.boundary;
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
    }).addTo(persistentMap);
    overlays["All local minima (drainage demand)"] = mapLayers.allMinima;
  }

  mapLayerControl = L.control.layers(null, overlays, { collapsed: false }).addTo(persistentMap);

  // -- legend: elevation gradient + marker encoding + catchment hint -----
  mapLegendControl = L.control({ position: "bottomleft" });
  mapLegendControl.onAdd = () => {
    const div = L.DomUtil.create("div", "webmap-legend");
    let gradientHtml = "";
    if (mapData.elevation_legend && mapData.elevation_legend.length) {
      const stops = mapData.elevation_legend;
      const gradientCss = stops.map((s, i) => `${s.color} ${(i / (stops.length - 1)) * 100}%`).join(", ");
      gradientHtml =
        `<div><strong>Elevation</strong></div>` +
        `<div style="width:160px;height:10px;border-radius:2px;margin:3px 0;background:linear-gradient(90deg, ${gradientCss});"></div>` +
        `<div style="display:flex;justify-content:space-between;font-size:10.5px;">` +
        `<span>${stops[0].value.toFixed(0)} m</span><span>${stops[stops.length - 1].value.toFixed(0)} m</span></div>`;
    }
    div.innerHTML =
      gradientHtml +
      `<div style="margin-top:6px;"><strong>Dot size</strong> = persistence (bigger = deeper basin)</div>` +
      (mapData.all_minima_geojson ? `<div><strong>Dot fill</strong> = drainage demand</div>` : "") +
      `<div>Click a dot -- or a point mentioned in the report -- to highlight its catchment area</div>` +
      `<div id="catchment-clear-row" hidden><a href="#" id="catchment-clear-link">Clear catchment highlight</a></div>`;
    return div;
  };
  mapLegendControl.addTo(persistentMap);
  L.DomEvent.disableClickPropagation(mapLegendControl.getContainer());
  document.getElementById("catchment-clear-link").addEventListener("click", (e) => {
    e.preventDefault();
    highlightCatchment(null);
  });

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

// -- terminal panel: a collapsible slide-in log, not a full-page view ------

function setTerminalOpen(open) {
  document.getElementById("terminal-panel").classList.toggle("open", open);
}

document.getElementById("terminal-toggle").addEventListener("click", () => {
  const panel = document.getElementById("terminal-panel");
  setTerminalOpen(!panel.classList.contains("open"));
});

function addProgressStep(step) {
  const log = document.getElementById("progress-log");
  const prevLive = log.querySelector(".step-dot.live");
  if (prevLive) prevLive.classList.remove("live");

  const row = document.createElement("div");
  row.className = "pipeline-step";
  const detailText = step.detail ? JSON.stringify(step.detail, null, 2) : "";
  row.innerHTML = `
    <div class="pipeline-step-summary"><span class="step-dot live"></span>${step.label}</div>
    ${detailText ? `<pre>${detailText}</pre>` : ""}`;
  log.appendChild(row);
  row.scrollIntoView({ behavior: "smooth", block: "end" });
}

function finalizeProgressDots() {
  const lastDot = document.querySelector("#progress-log .step-dot.live");
  if (lastDot) lastDot.classList.remove("live");
}

// -- one-box agent flow -------------------------------------------------------
//
// place-input -> (exposure-modal: optional file, or skip) -> POST
// /api/agent/analyze_stream, whose response body is newline-delimited JSON,
// one {"type": "step"|"error"|"done", ...} object per line, flushed as each
// pipeline stage finishes server-side. Read with fetch()'s streaming body
// reader (not EventSource -- EventSource can't POST a file upload for the
// exposure layer) so the terminal panel fills in live instead of only
// appearing once the whole multi-stage analysis is done.

let pendingExposureFile = null;

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
    addChatMessage("assistant", `Got it -- re-running the analysis for ${lastAnalyzedPlace} at ${requestedResolution}m resolution…`);
    runAgentAnalysis(lastAnalyzedPlace, pendingExposureFile, requestedResolution, { preserveChat: true });
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

function showReport(event, placeLabel, options) {
  options = options || {};
  const card = document.getElementById("report-card");
  card.hidden = false;
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
  lastAnalyzedPlace = place;
  document.getElementById("terminal-panel").hidden = false;
  setTerminalOpen(true);
  document.getElementById("terminal-toggle-label").textContent = `Analyzing ${place}…`;
  document.getElementById("terminal-dot").className = "step-dot live";
  document.getElementById("progress-place").textContent = place;
  document.getElementById("progress-log").innerHTML = "";
  document.getElementById("progress-error").hidden = true;
  document.getElementById("progress-retry").hidden = true;
  document.getElementById("report-card").hidden = true;
  setStatus("intro-status", "");

  const formData = new FormData();
  formData.append("place", place);
  if (exposureFile) formData.append("exposure_file", exposureFile);
  if (resolutionMeters) formData.append("resolution_meters", String(resolutionMeters));

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

        if (event.type === "step") {
          addProgressStep(event);
        } else if (event.type === "error") {
          finalizeProgressDots();
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
          document.getElementById("terminal-toggle-label").textContent = `Done in ${event.total_elapsed_seconds}s -- ${place}`;
          document.getElementById("terminal-dot").className = "step-dot";
          setTerminalOpen(false);
          showReport(event, event.place_label || place, options);
          if (options.preserveChat) addChatMessage("assistant", `Done (${event.total_elapsed_seconds}s) -- map and report above are now from the new run. Ask away.`);
          return;
        }
      }
    }
  } catch (err) {
    finalizeProgressDots();
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
  document.getElementById("exposure-file").value = "";
  pendingExposureFile = null;
  openModal("exposure-modal");
});
document.getElementById("place-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); document.getElementById("place-submit").click(); }
});

document.getElementById("exposure-continue").addEventListener("click", () => {
  pendingExposureFile = document.getElementById("exposure-file").files[0] || null;
  closeModal("exposure-modal");
  runAgentAnalysis(document.getElementById("place-input").value.trim(), pendingExposureFile);
});
document.getElementById("exposure-skip").addEventListener("click", () => {
  pendingExposureFile = null;
  closeModal("exposure-modal");
  runAgentAnalysis(document.getElementById("place-input").value.trim(), null);
});

function resetToIntro() {
  document.getElementById("report-card").hidden = true;
  document.getElementById("place-input").value = "";
  document.getElementById("place-input").focus();
  setStatus("intro-status", "");
}
document.getElementById("progress-retry").addEventListener("click", resetToIntro);
document.getElementById("report-new").addEventListener("click", resetToIntro);

initPersistentMap();
