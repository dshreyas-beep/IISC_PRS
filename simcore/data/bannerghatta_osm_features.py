"""
OSM feature constants for the Bannerghatta map mode.

This module is intentionally separate from `simcore.data.bannerghatta_ecology`,
which must stay free of any hardcoded population baselines.
"""

from __future__ import annotations

# OSM relation id for "Bannerghatta National Park" (used to reconstruct boundary rings
# from cached Overpass payloads, if present).
BANNERGHATTA_RELATION_ID = 8124064

SUVARNA_MUKHI_OSM_WAY_IDS = [
    254665790,
    325489259,
    325489260,
    325489262,
    699281412,
    699281416,
    781691089,
    1190779821,
    1190779822,
    1266419333,
    1317850440,
    1317850441,
    1337328423,
    1366035457,
    1411464128,
]

# Deterministic fallback polyline anchors for Suvarnamukhi drainage.
SUVARNA_MUKHI_ANCHOR_COORDS = [
    {"lat": 12.8867824, "lon": 77.5655150},
    {"lat": 12.8789632, "lon": 77.5589114},
    {"lat": 12.8527609, "lon": 77.5456104},
    {"lat": 12.8367585, "lon": 77.5325851},
    {"lat": 12.8070592, "lon": 77.5271087},
    {"lat": 12.8035052, "lon": 77.5105192},
    {"lat": 12.7692336, "lon": 77.5101059},
    {"lat": 12.7612086, "lon": 77.4882696},
    {"lat": 12.7269409, "lon": 77.4756688},
    {"lat": 12.7092246, "lon": 77.4715229},
    {"lat": 12.6639559, "lon": 77.4142891},
]

SAFARI_ZONE_COORDS = {
    "tiger_safari": {
        "label": "Tiger Safari / Rescue Holding",
        "center": {"lat": 12.8008, "lon": 77.5770},
        "radius_m": 950,
    },
    "lion_safari": {
        "label": "Lion Safari / Rescue Holding",
        "center": {"lat": 12.8035, "lon": 77.5814},
        "radius_m": 700,
    },
}

# Publicly described BBP-style recharge pit clusters (approximate anchors).
RECHARGE_PIT_COORDS = [
    {"lat": 12.8009, "lon": 77.5775, "label": "BBP recharge pit cluster 1"},
    {"lat": 12.8025, "lon": 77.5812, "label": "BBP recharge pit cluster 2"},
    {"lat": 12.7986, "lon": 77.5744, "label": "BBP recharge pit cluster 3"},
    {"lat": 12.8052, "lon": 77.5787, "label": "BBP recharge pit cluster 4"},
    {"lat": 12.7969, "lon": 77.5802, "label": "BBP recharge pit cluster 5"},
    {"lat": 12.8018, "lon": 77.5727, "label": "BBP recharge pit cluster 6"},
    {"lat": 12.8041, "lon": 77.5750, "label": "BBP recharge pit cluster 7"},
    {"lat": 12.7996, "lon": 77.5828, "label": "BBP recharge pit cluster 8"},
]

