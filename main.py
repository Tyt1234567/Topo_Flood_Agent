"""
CLI entry point for the GeoAI Agent Spatial Reasoning Pipeline.

Prerequisites:
1. pip install -r requirements.txt
2. GEMINI_API_KEY environment variable set to a free-tier Gemini API key
   (https://aistudio.google.com/apikey).

Usage:
    python main.py
    python main.py "Which bridge should we reinforce first, considering flood risk?"
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from geoai_agent.agent import GeoAIAgent

EXAMPLE_QUERIES = [
    "Which bridge should we reinforce first, considering flood risk and the population that depends on it to reach the hospital?",
    "What is the area of zone_cluster_a?",
    "Where would floodwater actually pool in this town, and how significant are those low points compared to noise?",
    "Where are the population density peaks, and what's the natural boundary between the two residential clusters?",
]


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("No query given on the command line. Example queries:")
        for example in EXAMPLE_QUERIES:
            print(f"  - {example}")
        query = input("\nEnter your spatial query: ").strip()
        if not query:
            query = EXAMPLE_QUERIES[0]
            print(f"Using default example query:\n  {query}")

    agent = GeoAIAgent()
    agent.run(query)


if __name__ == "__main__":
    main()
