"""
Real-world data example for the GeoAI Agent's Morse topology branch.

Unlike main.py (synthetic town), this pulls real terrain elevation for an
actual place from the free Open-Meteo elevation API, derives a second real
layer (slope) from it, optionally loads any number of additional
user-supplied CSV layers (x,y,value samples, interpolated onto the grid),
runs the discrete-Morse analysis on every layer, and renders all of them
together as a toggleable Leaflet map over a real OpenStreetMap basemap.

Usage:
    python real_world_example.py
    python real_world_example.py --no-input      # skip interactive prompts, use defaults only

The default location is the Boulder, CO Flatirons foothills (real
lat/lon box with genuine terrain relief). You can point it anywhere by
answering the bounding-box prompt.
"""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

from geoai_agent import prompts
from geoai_agent.llm_client import LLMClient
from geoai_agent.ttk_engine import analyze_scalar_field_ttk
from geoai_agent.real_data import build_real_world_dataset
from geoai_agent.visualize import render_leaflet_map

DEFAULT_BOUNDS = (39.95, 40.05, -105.35, -105.25)  # lat_min, lat_max, lon_min, lon_max
DEFAULT_RESOLUTION = 20
SAMPLE_CSV = "examples/sample_hazard_layer.csv"
OUTPUT_HTML = "real_world_map.html"


def _prompt_bounds(no_input: bool) -> tuple[float, float, float, float]:
    if no_input:
        return DEFAULT_BOUNDS
    print(
        f"Bounding box to analyze (default: Boulder, CO Flatirons "
        f"{DEFAULT_BOUNDS}):"
    )
    raw = input(
        "Enter 'lat_min,lat_max,lon_min,lon_max' or press Enter for the default: "
    ).strip()
    if not raw:
        return DEFAULT_BOUNDS
    parts = [float(p) for p in raw.split(",")]
    if len(parts) != 4:
        print("Could not parse 4 numbers, using default.")
        return DEFAULT_BOUNDS
    return tuple(parts)


def _prompt_layer_sources(no_input: bool) -> dict[str, str | None]:
    """elevation always comes from the online fetch here; additional layers
    can be added interactively from user CSV files."""
    layer_sources: dict[str, str | None] = {"elevation": None}
    if no_input:
        return layer_sources

    use_sample = input(
        f"\nAdd the bundled example layer ({SAMPLE_CSV}, a small sample "
        "'hazard_survey' CSV) to demonstrate a user-supplied layer? [y/N]: "
    ).strip().lower()
    if use_sample == "y":
        layer_sources["hazard_survey"] = SAMPLE_CSV

    print(
        "\nYou can add your own layers too: a CSV with header 'x,y,value' "
        "(x=longitude, y=latitude). Leave the name blank to stop."
    )
    while True:
        name = input("Layer name (blank to finish): ").strip()
        if not name:
            break
        path = input(f"CSV path for layer '{name}': ").strip()
        if not path:
            print("No path given, skipping this layer.")
            continue
        layer_sources[name] = path
    return layer_sources


def main() -> None:
    no_input = "--no-input" in sys.argv

    bounds = _prompt_bounds(no_input)
    layer_sources = _prompt_layer_sources(no_input)

    print(f"\nFetching/loading layers for bounds {bounds} ...")
    dataset = build_real_world_dataset(
        name="real_world_example",
        bounds=bounds,
        layer_sources=layer_sources,
        resolution=DEFAULT_RESOLUTION,
    )
    print(f"Layers built: {list(dataset.scalar_fields)}")

    results = {}
    for layer_name, field in dataset.scalar_fields.items():
        print(f"\n--- Morse analysis: {layer_name} ---")
        result = analyze_scalar_field_ttk(field)
        results[layer_name] = result
        print(json.dumps({
            "value_range": result["value_range"],
            "persistence_threshold": result["persistence_threshold"],
            "num_significant_basins": result["num_significant_basins"],
            "significant_maxima": sum(1 for e in result["critical_points"]["maximum"] if e["significant"]),
            "significant_minima": sum(1 for e in result["critical_points"]["minimum"] if e["significant"]),
        }, indent=2))

    print("\n--- Recommendation (elevation) ---")
    llm = LLMClient()
    recommendation = llm.chat(
        prompts.MORSE_RECOMMENDATION_SYSTEM,
        f"Morse analysis result: {json.dumps(results['elevation'], ensure_ascii=False)}",
    )
    print(recommendation)

    output_path = render_leaflet_map(
        fields=dataset.scalar_fields,
        results=results,
        is_geographic=dataset.is_geographic,
        output_path=OUTPUT_HTML,
        title=f"Real-world scalar field topology ({dataset.name})",
    )
    print(f"\nMap written to {output_path} -- open it in a browser to explore all layers.")


if __name__ == "__main__":
    main()
