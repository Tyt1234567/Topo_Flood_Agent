"""
Topology-based analysis: step 4 in the pipeline.

Three independent sub-analyses, run when the routing decision at step 2
determines geometry alone is not sufficient:

- Connectivity: graph-theoretic reachability and shortest paths over the
  road network (networkx).
- Adjacency: which polygon zones touch/intersect each other or the hazard
  zone (shapely).
- Critical structures: road segments whose removal disconnects part of the
  network (networkx graph bridges) -- in this scenario these correspond to
  the physical river-crossing bridges.

This module handles topology of the *discrete network* (graph, polygons).
For topology of *continuous scalar fields* (elevation, hazard intensity,
population density -- critical points, persistence, watershed basins via
discrete Morse theory) see geoai_agent/ttk_engine.py (TTK-backed Morse-Smale
complex), invoked below through analyze_scalar_topology.
"""

import networkx as nx

from geoai_agent.data import GeoDataset
from geoai_agent.ttk_engine import analyze_scalar_field_ttk


def analyze_connectivity(dataset: GeoDataset) -> dict:
    graph = dataset.road_graph
    components = list(nx.connected_components(graph))
    shortest_paths = {}
    for pop in dataset.population_points:
        try:
            length = nx.shortest_path_length(
                graph, source=pop.nearest_node, target=dataset.hospital_node, weight="length"
            )
        except nx.NetworkXNoPath:
            length = None
        shortest_paths[pop.nearest_node] = length
    return {
        "is_fully_connected": nx.is_connected(graph),
        "num_components": len(components),
        "shortest_path_to_hospital": shortest_paths,
    }


def analyze_adjacency(dataset: GeoDataset) -> dict:
    zone_items = list(dataset.zones.items())
    zone_adjacency = []
    for i in range(len(zone_items)):
        for j in range(i + 1, len(zone_items)):
            name_a, poly_a = zone_items[i]
            name_b, poly_b = zone_items[j]
            if poly_a.touches(poly_b) or poly_a.intersects(poly_b):
                zone_adjacency.append((name_a, name_b))

    hazard_adjacent_zones = [
        name for name, poly in dataset.zones.items()
        if poly.intersects(dataset.hazard_zone)
    ]
    return {
        "zone_adjacency_pairs": zone_adjacency,
        "zones_touching_hazard": hazard_adjacent_zones,
    }


def analyze_critical_structures(dataset: GeoDataset) -> dict:
    graph = dataset.road_graph
    graph_bridges = set(nx.bridges(graph))

    critical_structures = []
    for u, v in graph_bridges:
        edge = graph.edges[u, v]
        critical_structures.append({
            "edge": (u, v),
            "is_physical_bridge": edge.get("is_bridge_structure", False),
            "geom": edge["geom"],
            "length": edge["length"],
        })
    return {"critical_structures": critical_structures}


def analyze_scalar_topology(dataset: GeoDataset, field_name: str, persistence_threshold: float | None = None) -> dict:
    """Discrete-Morse topological analysis of one scalar field in the
    dataset (elevation, hazard_intensity, population_density, or any other
    field later added to GeoDataset.scalar_fields). See ttk_engine.py."""
    field = dataset.scalar_fields.get(field_name)
    if field is None:
        return {"error": f"unknown scalar field '{field_name}'", "available_fields": list(dataset.scalar_fields)}
    return analyze_scalar_field_ttk(field, persistence_threshold=persistence_threshold)


def run_topology_analysis(dataset: GeoDataset) -> dict:
    return {
        "connectivity": analyze_connectivity(dataset),
        "adjacency": analyze_adjacency(dataset),
        "critical_structures": analyze_critical_structures(dataset),
    }
