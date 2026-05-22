"""
Defines available world generation presets.
Used by SimManager and MultiAgentSim.
"""

TOPOLOGIES = [
    {
        "name": "edge_farmland",
        "width": 120,
        "height": 120,
        "water_ratio": 0.08,
        "crop_ratio": 0.25,
        "forest_ratio": 0.2,
    },
    {
        "name": "forest_core",
        "width": 120,
        "height": 120,
        "water_ratio": 0.12,
        "crop_ratio": 0.05,
        "forest_ratio": 0.5,
    },
    {
        "name": "river_delta",
        "width": 120,
        "height": 120,
        "water_ratio": 0.18,
        "crop_ratio": 0.15,
        "forest_ratio": 0.3,
    },
    {
        "name": "bannerghatta_osm",
        "width": 240,
        "height": 720,
        "water_ratio": 0.0,
        "crop_ratio": 0.0,
        "forest_ratio": 0.0,
    },
]