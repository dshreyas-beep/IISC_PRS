"""
Species-level traits used by the simulation runtime (not a population baseline).

Keep this separate from any geography modules so that map selection never forces
animal counts or species lists.
"""

from __future__ import annotations

SPECIES_WEIGHTS_KG: dict[str, float] = {
    "elephant": 3000.0,
    "leopard": 60.0,
    "sloth_bear": 130.0,
    "tiger": 180.0,
    "lion": 170.0,
    "gaur": 650.0,
    "sambar_deer": 180.0,
    "spotted_deer": 65.0,
    "barking_deer": 22.0,
    "wild_boar": 75.0,
    "human": 65.0,
}

