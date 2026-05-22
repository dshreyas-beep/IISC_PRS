"""
Bannerghatta National Park geography constants.

CRITICAL: This module must remain *lightweight* and contain **no** hardcoded
species counts, baseline arrays, or population defaults. Population sizing is
owned by the UI/websocket `species_counts` payload and enforced in the engine.
"""

BANNERGHATTA_BBOX = {
    "north": 12.8229972,
    "south": 12.3443525,
    "east": 77.6368603,
    "west": 77.4828815,
}

# Declared park centroid used for UI camera framing / metadata only.
BANNERGHATTA_CENTER_LATLON = (12.8008, 77.5756)

# Declared area for BNP (sq km) used in metadata only.
BANNERGHATTA_DECLARED_AREA_SQ_KM = 260.51
