# GeoAI Agent -- Web UI

A Flask + vanilla JS/Leaflet front end over the GeoAI agent's real-world
data and discrete-Morse-theory analysis engine. This folder is
self-contained: `geoai_agent/` below is a standalone copy of the
discrete-Morse-theory/real-world-data/GeoJSON package (not an import of
anything outside `web_app/`), and `examples/` holds its own sample CSV
layer. Nothing in this folder reaches outside it -- you can copy just
`web_app/` elsewhere and it still runs.

## Run

From inside `web_app/` (or from the project root, both work):

```
pip install -r requirements.txt
python app.py
```

Requires a free-tier Gemini API key (https://aistudio.google.com/apikey)
set as the `GEMINI_API_KEY` environment variable, for the LLM-driven
routing and recommendation steps. Open http://localhost:5000 once it's
running.

## Configuration

Copy the repo root's `.env.example` to `.env` (either at the repo root or
inside `web_app/` -- both work) and fill in what you need. Full details
in the root `README.md`'s Configuration section; short version:

| Variable | Required? | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | LLM routing/recommendation calls |
| `DATABASE_URL` | No | Real stormwater pipe/catch-basin data (`geoai_agent/drainage_db.py`) -- see `../db_dump/README.md` to restore the bundled sample dataset |
| `TTK_PYTHON`, `TTK_BIN_DIR` | No | Discrete-Morse (TTK) terrain analysis, if not at the defaults in `geoai_agent/ttk_engine.py` |
| `QGIS_PYTHON` | No | Primary US elevation source (streamed USGS 3DEP DEM via GDAL), if not at the default in `geoai_agent/dem_cog_source.py`. Without it, elevation automatically falls back to USGS's Elevation Point Query Service, then Open-Meteo. |

## What's in the panel

**Real-World Data** -- one bounding box, one topic (elevation, temperature,
precipitation, wind, humidity, or pressure -- live-fetched from Open-Meteo,
never pre-downloaded), and any number of extra user-uploaded layers, all
behind a single "Fetch & Analyze" button. Fill in the bounds/topic fields
either by hand, or describe the query in plain English ("find critical
points in Boulder, Colorado") and click "Parse query" -- an LLM extraction
step (natural-language phrasing varies too much for regex to keep up with
reliably) works out the place/coordinates and topic and fills the fields
in for you; it never fetches anything itself. Add your own CSV
(`x,y,value`) or shapefile (`.zip` bundling `.shp`/`.shx`/`.dbf`, POINT
geometry + a `value` field) layers by name + file upload -- naming a
custom layer the same as the selected topic (e.g. "elevation") overrides
the live fetch with your own data. Results render on a real OpenStreetMap
basemap with one heatmap + critical-points overlay per layer.

**Flood & Drainage Analysis** -- a Track-A-style domain module on top of
the same elevation layer: every significant topological depression
(Morse-Smale basin) becomes a candidate flood hotspot, scored by a
weighted risk formula (topological persistence, contributing area, slope,
selected rainfall scenario, and an optional exposure layer) and given a
Rational-Method drainage demand (m3/s) under the selected design storm.
Two HITL controls -- a rainfall-scenario dropdown (2/10/25/50/100/500-year
design storms) and a priority-weighting preset (balanced / life-safety /
infrastructure-protection) -- change which basins rank highest. See
`geoai_agent/flood_tools.py`.

**Analysis process** -- both the main analysis and the flood panel show a
collapsible step-by-step trace of what the backend actually did for that
request (fetch data, run TTK, [rank flood basins,] generate the LLM
recommendation), each with how long it took and a JSON detail payload --
the same visibility the CLI agent prints to a terminal, surfaced as
structured data since a web request has no terminal to watch it run in.

## API endpoints

- `POST /api/realworld/smart_query` -- `{"query": "..."}` -> an LLM
  extraction call works out `{bounds, variable, resolved_place}` from free
  text; returns what was understood without fetching anything. The
  frontend fills the manual fields in from this response.
- `POST /api/realworld/load_data` -- multipart form: `bounds`, `resolution_meters`,
  `primary_variable`, `use_sample_layer`, plus any custom `layer_name` +
  `layer_file::<name>` pairs -> fetches just the raw scalar field(s) and
  caches them server-side under a `dataset_id`, so the map can show the
  real heatmap immediately after a place is confirmed.
- `POST /api/realworld/analyze` -- same form fields, plus `recommend` and
  optionally a cached `dataset_id` -> runs `build_real_world_dataset` (if
  not reusing a cached dataset) + `analyze_scalar_field_ttk` per layer,
  returns GeoJSON + summaries (+ an LLM recommendation for
  `primary_variable` if requested) + a `pipeline_steps` trace. The final
  fetch+analyze action for the panel.
- `POST /api/realworld/flood_analyze` -- same form fields plus
  `rainfall_scenario`, `weight_preset`, optionally `runoff_coefficient` and
  a cached `dataset_id` -> ranks flood hotspots on the elevation layer
  (`flood_tools.rank_flood_basins`), returns the ranked list + GeoJSON
  markers (+ an LLM recommendation) + a `pipeline_steps` trace.

## Notes

- `app.run(debug=True)` is a development server -- fine for local use, not
  meant to be exposed to the internet as-is.
- Uploaded layer files are written to a temp file for the duration of one
  request and deleted immediately after.
