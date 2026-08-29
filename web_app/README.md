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

## What's in the app

One box: type a real place name (e.g. "College Park, MD") and, optionally,
what matters most there (e.g. "protecting critical infrastructure"), or
skip the name and draw a custom boundary directly on the map. Click
Analyze -- everything else happens automatically:

- **Parallel real-data retrieval** -- FEMA flood zones (NFHL), NOAA Atlas 14
  rainfall, USGS 3DEP elevation, USDA SSURGO soil, Census population
  density, local stormwater pipes (`geoai_agent/drainage_db.py`), USGS
  river gauges, and OpenStreetMap roads/buildings, fetched concurrently.
- **Real discrete-Morse (TTK) terrain analysis** -- watershed basins,
  persistence-based noise filtering, cross-checked against an independent
  D8 flow-direction computation.
- **Flood-hotspot ranking** -- entropy-weighted scoring across five risk
  factors (topology, contributing area, rainfall, slope, exposure), an
  Expected-Annual-Damage-style aggregate, and a weight-sensitivity check.
- **Three human-in-the-loop checkpoints** (persistence threshold, ranking &
  prioritization, the final recommendation) -- each pauses with a 15s
  countdown and a Continue button; type feedback instead and the agent
  re-runs that step with it folded in. The agent can also autonomously
  redo the threshold or ranking on its own if its self-review flags an
  inconsistency, independent of any human feedback.
- **Map** -- terrain heatmap, boundary, FEMA zones, pipes, river gauges,
  buildings, D8 flow field, population density, land cover, roads/water,
  rainfall & soil, and a persistence-threshold slider over all local
  minima (client-side, no re-fetch).
- **Report** -- a floating, draggable, resizable window with the full
  write-up, ranked hotspots, data-source citations, and a downloadable
  standalone HTML copy. Report mentions of a hotspot jump to and highlight
  that basin's catchment on the map.
- **Chat** -- ask follow-up questions grounded in the same run's data.
- **Modify & re-run** -- upload your own elevation/exposure data, or
  hand-set the five risk-factor weights, and re-run the whole analysis
  with it substituted in.

## API endpoints

- `POST /api/agent/analyze_stream` -- multipart form (`place` or
  `boundary_polygon`, optional `objective_text`, `elevation_file`,
  `exposure_file`, `resolution_meters`, `custom_weights`) -> the whole
  pipeline above, streamed as newline-delimited JSON (`step` /
  `agent_decision` / `checkpoint_open` / `checkpoint_close` / `done` /
  `error` events) so the UI can render each stage as it finishes.
- `POST /api/agent/interject` -- `{run_id, feedback}` -> real feedback text
  for whichever checkpoint is currently open.
- `POST /api/agent/continue` -- `{run_id}` -> skip the rest of a
  checkpoint's wait immediately (same effect as the countdown reaching 0).
- `POST /api/agent/checkpoint_hold` -- `{run_id}` -> freeze a checkpoint's
  countdown (the frontend calls this when the feedback box is focused).
- `POST /api/agent/ask` -- `{question, context, history}` -> an LLM answer
  grounded in the run's compact data summary (`context`, from
  `analyze_stream`'s `done` event) and prior chat turns.

(`/api/realworld/*` routes also exist for an older bounding-box/topic
panel that predates the place-name-driven flow above and isn't wired into
the current frontend.)

## Notes

- `app.run(debug=True)` is a development server -- fine for local use, not
  meant to be exposed to the internet as-is.
- Uploaded layer files are written to a temp file for the duration of one
  request and deleted immediately after.
