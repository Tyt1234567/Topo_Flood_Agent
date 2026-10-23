# Topo Flood Agent

A flood-risk / drainage-demand agent that takes a real place name and
produces a human-reviewed recommendation, grounded in real terrain,
rainfall, soil, flood-zone, and infrastructure data -- no coordinates or
settings to fill in. See `web_app/README.md` for the full feature list and
API surface.

```
web_app/
  app.py               Flask backend: data retrieval, TTK terrain
                        analysis, flood-hotspot ranking, LLM
                        recommendation, checkpoints, chat
  geoai_agent/          Self-contained analysis package (see below)
  static/, templates/   Frontend (vanilla JS + Leaflet)
  load_*_stormwater.py  One-time ETL scripts for stormwater pipe data
db_dump/
  drainage_dump.sql.gz  Bundled stormwater pipe/catch-basin data dump
```

## Quick start

```
cd web_app
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. Requires `GEMINI_API_KEY` at minimum (see
Configuration below) -- everything else degrades gracefully without it.

## How it works

Given a place name, the agent:

1. Geocodes it and concurrently fetches real data: FEMA flood zones,
   NOAA Atlas 14 rainfall, USGS 3DEP elevation, USDA SSURGO soil, Census
   population density, local stormwater pipes, USGS river gauges, and
   OpenStreetMap roads/buildings.
2. Runs a real discrete Morse-Smale complex over the elevation field via
   TTK (the Topology ToolKit, https://topology-tool-kit.github.io/) to
   find watershed basins and rank them by topological persistence, so a
   real flood-accumulation basin can be told apart from noise -- cross-
   checked against an independent D8 flow-direction computation.
3. Scores every significant basin (persistence, contributing area,
   rainfall, slope, exposure) with entropy-weighted risk factors and a
   Rational-Method drainage demand, and ranks them.
4. Writes an LLM recommendation, reviews its own draft against the
   evidence, and can autonomously redo the threshold or ranking if that
   self-review finds a real inconsistency.
5. Pauses at three human-in-the-loop checkpoints (persistence threshold,
   ranking, the recommendation) -- each auto-continues after 15s, or
   folds in typed feedback and re-runs that step.

`geoai_agent/ttk_engine.py` is the active Morse-Smale engine (shells out
to a TTK-equipped Python environment as a subprocess); `geoai_agent/
morse_topology.py` is a from-scratch pure-Python reimplementation of the
same PL-Morse/persistence/watershed methodology, kept as the shared
`ScalarField` data structure and a reference for the algorithm, not the
active compute path. `geoai_agent/flood_tools.py` turns a Morse-Smale
result into the flood-risk/drainage-demand scoring above.

## Configuration

Copy `.env.example` to `.env` and fill in whichever of these you need --
`.env` is gitignored, loaded automatically via `python-dotenv`. Every
setting below has a name-matched environment variable, so setting them
directly in your shell instead of a `.env` file works the same way.

| Variable | Required? | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | LLM backend (`geoai_agent/llm_client.py`, via the `google-genai` SDK). Free tier: https://aistudio.google.com/apikey |
| `DATABASE_URL` | No | Real stormwater pipe/catch-basin data (`geoai_agent/drainage_db.py`), a `postgresql://user:pass@host:port/dbname` PostGIS connection. Without it, that layer is silently skipped, same as any other missing data source. See `db_dump/README.md` to restore the bundled sample dataset (DC, Austin, Charlotte, Anne Arundel, Fairfax, Loudoun, PG County), or `web_app/load_*_stormwater.py` to load another city yourself. |
| `TTK_PYTHON`, `TTK_BIN_DIR` | No (needed for Morse-Smale analysis) | Point at a Python environment with TTK's VTK-based Python bindings installed, if not at the defaults baked into `geoai_agent/ttk_engine.py` (`D:\anaconda3\envs\tda_env\python.exe` / `D:\TTK\bin`). |
| `QGIS_PYTHON` | No (needed for the primary US elevation source) | Point at QGIS's bundled Python (`python-qgis.bat`), if not at the default baked into `geoai_agent/dem_cog_source.py` (`D:\QGIS\bin\python-qgis.bat`). Without it, elevation automatically falls back to USGS's Elevation Point Query Service, then Open-Meteo. |

`GEMINI_API_KEY` defaults `LLMClient` to a pinned stable model
(`LLMClient.DEFAULT_MODEL`) rather than a `-latest` alias -- Google's
`-latest` aliases can point at a brand-new preview model with a much
tighter free-tier daily quota than an established stable snapshot.
