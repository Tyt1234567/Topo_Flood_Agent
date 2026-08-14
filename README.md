# GeoAI Agent

An agentic pipeline that answers natural-language spatial questions by routing
between simple geometry tools, network-topology analysis, and continuous
scalar-field topology (discrete Morse theory), then produces a
human-reviewed recommendation. Implements the pipeline diagrammed in
`geoai-workflow.png`, extended with a third routing branch.

```
User -> GeoAI Agent (LLM)
     -> [1] Understand Task
     -> [2] Spatial Reasoning (routing)
          geometry_sufficient -> Standard GIS Tools -> done
          topology_needed     -> [3] Select Representation
                               -> [4] Topology Analysis (connectivity, adjacency, critical structures)
                               -> [5] Integrate Results (hazard, population, accessibility)
                               -> [6] Spatial Reasoning (synthesis)
                               -> [7] Recommendation
                               -> Human (feedback) -> Agent Re-runs -> back to GeoAI Agent
          morse_needed         -> [4] Morse Topology Analysis (critical points, persistence, watershed basins
                                    on a continuous scalar field)
                               -> [7] Recommendation
                               -> Human (feedback) -> Agent Re-runs -> back to GeoAI Agent
```

## Scenario

The agent reasons over a synthetic town split by a river (`geoai_agent/data.py`):
two residential clusters, each reachable from the hospital by exactly one
bridge. A flood hazard zone covers one of the two bridges. This setup lets
the agent weigh hazard exposure, population served, and accessibility loss
against each other when recommending which bridge to protect first.

The same dataset also carries three continuous scalar fields sampled on a
grid over the town (synthetic terrain elevation, flood hazard intensity, and
population density). These feed the `morse_needed` branch, which runs a
real discrete Morse-Smale complex via TTK (the Topology ToolKit,
`geoai_agent/ttk_engine.py`) to find critical points (basins, peaks,
saddles) and rank them by topological persistence, so the agent can tell a
real feature (e.g. an actual flood-accumulation basin) from noise (a
boundary artifact or a minor bump). A from-scratch pure-Python
reimplementation of the same PL-Morse/persistence/watershed methodology
(following the UMIACS GeoVis Lab's approach, https://geovis.umiacs.io/,
e.g. `FormanGradient2D`, `Terrain_Trees`) is kept at
`geoai_agent/morse_topology.py` as a reference/fallback, but is no longer
what the pipeline actually calls.

On top of that, `geoai_agent/flood_tools.py` turns the Morse-Smale basins
of a real elevation field into a flood-risk / drainage-demand analysis:
every significant depression is scored (topological persistence,
contributing area, slope, a selectable design-storm rainfall scenario, and
an optional exposure layer) and given a Rational-Method drainage demand.
See `web_app/`'s "Flood & drainage analysis" panel for the HITL controls
(rainfall scenario, life-safety-vs-infrastructure weighting).

## Requirements

- Python 3.11+
- A free-tier Gemini API key (https://aistudio.google.com/apikey), set as
  the `GEMINI_API_KEY` environment variable
- Python packages:
  ```
  pip install -r requirements.txt
  ```
- For the discrete-Morse (`morse_needed`) branch and flood/drainage
  analysis: TTK (the Topology ToolKit, https://topology-tool-kit.github.io/),
  installed in a separate Python environment with its VTK-based Python
  bindings (`geoai_agent/ttk_engine.py` shells out to it as a subprocess --
  see `TTK_PYTHON` / `TTK_BIN_DIR` in that file to point it at your install)

## Usage

```
python main.py "Which bridge should we reinforce first, considering flood risk and the population that depends on it to reach the hospital?"
python main.py "Where would floodwater actually pool in this town, and how significant are those low points compared to noise?"
python main.py "Where are the population density peaks, and what's the natural boundary between the two residential clusters?"
```

Or run without arguments to pick from example queries interactively:

```
python main.py
```

The pipeline prints each step's output as it runs. When the topology branch
reaches step 7, it prints a `Recommendation` and prompts for human review:

- Press Enter to accept and finish.
- Type feedback (e.g. `"ignore flood risk, focus on accessibility only"`) to
  trigger a new pass — the agent re-runs the full pipeline with your
  feedback folded into its reasoning.

A geometry-sufficient query (e.g. `"What is the area of zone_cluster_a?"`)
routes directly to `Standard GIS Tools` and exits without a human review
step, matching the diagram. A `morse_needed` query goes through the same
human review loop as the topology branch.

## Project layout

| File | Role |
|---|---|
| `geoai_agent/data.py` | Synthetic GIS dataset: road graph, hazard polygon, zones, population points, and scalar fields (elevation, hazard intensity, population density) |
| `geoai_agent/geometry_tools.py` | Standard GIS tools for the geometry-sufficient branch (area, distance, containment, buffer) |
| `geoai_agent/topology_tools.py` | Connectivity, adjacency, and critical-structure (graph bridge) analysis over the discrete network |
| `geoai_agent/morse_topology.py` | Pure-Python discrete Morse engine (reference/fallback, not the active path): critical point classification, persistence pairing, watershed segmentation |
| `geoai_agent/ttk_engine.py` | Active Morse-Smale engine: shells out to a TTK-equipped Python environment as a subprocess, returns the same result shape as `morse_topology.analyze_scalar_field` plus separatrices |
| `geoai_agent/_ttk_worker.py` | Runs inside the TTK environment only -- builds a vtkImageData from a `ScalarField`, runs TTK's persistence-simplification + Morse-Smale complex |
| `geoai_agent/flood_tools.py` | Flood-risk / drainage-demand domain engine on top of a Morse-Smale result: weighted risk scoring, Rational-Method runoff, rainfall scenarios, HITL weight presets |
| `geoai_agent/real_data.py` | Real-world scalar field layers: online elevation fetch (USGS 3DEP / Open-Meteo), user-CSV layers via IDW interpolation, derived slope layer |
| `geoai_agent/visualize.py` | Self-contained multi-layer Leaflet map generation for Morse analysis and flood-hotspot results |
| `geoai_agent/integration.py` | Combines topology output with hazard/population/accessibility per structure |
| `geoai_agent/llm_client.py` | Gemini API client (google-genai SDK) |
| `geoai_agent/prompts.py` | System prompts and JSON schemas for each LLM-driven step |
| `geoai_agent/agent.py` | Orchestrator implementing the full pipeline and the human feedback re-run loop |
| `main.py` | CLI entry point (synthetic town) |
| `real_world_example.py` | CLI entry point for real-world data, multi-layer input, and Leaflet visualization |
| `examples/sample_hazard_layer.csv` | Example user-supplied CSV layer for `real_world_example.py` |

## Scalar field topology (discrete Morse theory)

`geoai_agent/morse_topology.py` is a generic engine: it takes any
`ScalarField` (a named grid of values) and returns:

- **Critical points** -- minima, maxima, and saddles, classified via
  connected components of each vertex's lower/upper link (the standard PL
  Morse definition).
- **Persistence** -- each critical point is paired with the saddle that
  would merge/split its feature, via a union-find sweep over sub/superlevel
  sets (a merge tree and its dual split tree). The persistence value is how
  much the field would have to change to erase that feature -- low
  persistence means noise, high persistence means a real, load-bearing
  feature of the surface.
- **Watershed basins** -- every grid cell is assigned to a basin by
  steepest descent to a local minimum, then basins below the persistence
  threshold are merged into their spill partner, giving a simplified,
  noise-free segmentation.

It is not tied to terrain: the same code runs on `elevation`,
`hazard_intensity`, `population_density`, or any other field later added to
`GeoDataset.scalar_fields`. `geoai_agent/topology_tools.analyze_scalar_topology`
is the thin dataset-aware entry point the agent calls.

## Real-world data and multi-layer visualization

By default the `morse_needed` branch analyzes the synthetic town's fields.
`real_world_example.py` swaps that out for genuine data over an actual
place, with support for multiple simultaneous layers and a Leaflet map:

```
python real_world_example.py
```

It:

1. Fetches real terrain elevation for a lat/lon bounding box (default:
   the Boulder, CO Flatirons foothills) from the free, keyless
   [Open-Meteo elevation API](https://open-meteo.com/en/docs/elevation-api).
2. Derives a second real layer, `slope` (gradient magnitude of the
   elevation), at no extra data cost.
3. Optionally loads any number of additional user-supplied layers from CSV
   files (header `x,y,value`, x=longitude/y=latitude), interpolated onto
   the analysis grid with inverse-distance weighting
   (`geoai_agent/real_data.py`) -- a bundled example,
   `examples/sample_hazard_layer.csv`, is offered interactively. Add your
   own by entering a layer name and CSV path when prompted, or edit
   `real_world_example.py`'s `layer_sources` dict directly.
4. Runs the same discrete-Morse engine (`morse_topology.py`) on every
   layer and prints a plain-language recommendation for the elevation
   layer.
5. Renders all layers together as a self-contained Leaflet map
   (`geoai_agent/visualize.py`, Leaflet loaded from its CDN, no new Python
   dependency) over a real OpenStreetMap basemap, with a layer control to
   toggle each field's heatmap and its significant critical points on and
   off independently. Open `real_world_map.html` in a browser.

Run non-interactively with defaults only (elevation + derived slope, no
prompts, no LLM recommendation call skipped) via:

```
python real_world_example.py --no-input
```

The synthetic-town path (`main.py`, `geoai_agent/data.py`) is unchanged and
still used for the network-topology (`topology_needed`) branch, which needs
the town's road graph and zones and has no real-world equivalent here.

## Configuration

The LLM backend is Gemini (`geoai_agent/llm_client.py`, via the
`google-genai` SDK), reading the API key from the `GEMINI_API_KEY`
environment variable and defaulting to a pinned stable model
(`LLMClient.DEFAULT_MODEL`) rather than a `-latest` alias -- Google's
`-latest` aliases can point at a brand-new preview model with a much
tighter free-tier daily quota than an established stable snapshot. To use
a different model or a separate API key, construct `GeoAIAgent` with a
custom `LLMClient`:

```python
from geoai_agent.agent import GeoAIAgent
from geoai_agent.llm_client import LLMClient

llm = LLMClient(model="gemini-3.5-flash", api_key="...")
agent = GeoAIAgent(llm_client=llm)
agent.run("your query")
```

The Morse-Smale engine (`geoai_agent/ttk_engine.py`) looks for TTK's
Python environment at `D:\anaconda3\envs\tda_env\python.exe` /
`D:\TTK\bin` by default -- override with the `TTK_PYTHON` / `TTK_BIN_DIR`
environment variables if your install lives elsewhere.
