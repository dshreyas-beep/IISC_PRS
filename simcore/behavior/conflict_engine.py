from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

from simcore.world.zones import Zone


@dataclass
class ConflictHotspot:
    id: str
    type: str  # "tension" | "fight" | "raid"
    cx: float
    cy: float
    radius: float
    score: float
    label: str


def compute_zone_crowd(zones: List[Zone], agents: List[Any], radius_factor: float = 1.1) -> Dict[str, int]:
    crowd: Dict[str, int] = {z.id: 0 for z in zones}
    for a in agents:
        if not getattr(a, "alive", True):
            continue
        ax, ay = float(a.x), float(a.y)
        for z in zones:
            r = float(z.radius) * radius_factor
            if (ax - z.centroid_x) ** 2 + (ay - z.centroid_y) ** 2 <= r * r:
                crowd[z.id] += 1
    return crowd


def compute_conflicts(
    zones: List[Zone],
    agents: List[Any],
    herd_map: Dict[int, int],
) -> List[ConflictHotspot]:
    """
    Creates conflict hotspots that you can draw in UI.
    - Elephant vs elephant (herd overlap)
    - Human vs wildlife (near crops/settlements)
    - Overcrowded waterholes (resource pressure)
    """

    alive = [a for a in agents if getattr(a, "alive", True)]
    if not zones or not alive:
        return []

    crowd = compute_zone_crowd(zones, alive)

    hotspots: List[ConflictHotspot] = []

    # Build per-zone species + herd distributions
    for z in zones:
        in_zone: List[Any] = []
        r2 = (z.radius * 1.15) ** 2
        for a in alive:
            dx = float(a.x) - float(z.centroid_x)
            dy = float(a.y) - float(z.centroid_y)
            if dx * dx + dy * dy <= r2:
                in_zone.append(a)

        if not in_zone:
            continue

        kinds = [str(getattr(a, "kind", getattr(a, "species", ""))).lower() for a in in_zone]
        n = len(kinds)

        elephants = [a for a in in_zone if str(getattr(a, "kind", getattr(a, "species", ""))).lower() == "elephant"]
        humans = [a for a in in_zone if str(getattr(a, "kind", getattr(a, "species", ""))).lower() == "human"]
        predators = [a for a in in_zone if str(getattr(a, "kind", getattr(a, "species", ""))).lower() == "leopard"]

        # Herd overlap score (elephants)
        herd_ids = [herd_map.get(int(getattr(a, "id", getattr(a, "agent_id", -1))), -1) for a in elephants]
        unique_herds = len(set(h for h in herd_ids if h != -1))

        herd_overlap = 0.0
        if len(elephants) >= 4 and unique_herds >= 2:
            herd_overlap = 1.0 + 0.25 * (len(elephants) - 4)

        # Human-wildlife conflict score
        human_wildlife = 0.0
        if humans and (elephants or predators):
            human_wildlife = 0.8 + 0.2 * (len(humans) + len(elephants) + len(predators))

        # Resource pressure score (crowding)
        resource_pressure = 0.0
        if z.type == "water" and crowd.get(z.id, 0) >= 5:
            resource_pressure = 0.6 + 0.1 * (crowd[z.id] - 5)

        score = herd_overlap + human_wildlife + resource_pressure

        if score < 0.9:
            continue

        # classify
        if herd_overlap >= 1.2:
            ctype = "fight"
            label = f"🐘 Herd dominance fight risk ({unique_herds} herds)"
        elif human_wildlife >= 1.0 and z.type in ("crop", "settlement"):
            ctype = "raid"
            label = f"⚠️ Human–wildlife conflict risk"
        else:
            ctype = "tension"
            label = f"⚠️ Resource pressure / tension"

        hotspots.append(
            ConflictHotspot(
                id=z.id,
                type=ctype,
                cx=float(z.centroid_x),
                cy=float(z.centroid_y),
                radius=float(z.radius) * 1.25,
                score=float(score),
                label=label,
            )
        )

    # keep top hotspots
    hotspots.sort(key=lambda h: h.score, reverse=True)
    return hotspots[:12]
