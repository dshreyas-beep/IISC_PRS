# simcore/world/zones.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np


@dataclass(frozen=True)
class Zone:
    id: str
    type: str  # "water" | "crop" | "settlement"
    centroid_x: float
    centroid_y: float
    radius: float
    area: int
    mean_strength: float
    bbox: Tuple[int, int, int, int]  # x0,y0,x1,y1


def extract_zones(
    layer: np.ndarray,
    threshold: float,
    ztype: str,
    *,
    min_cells: int = 25,
    max_zones: int = 12,
) -> List[Zone]:
    """
    Connected-component clustering (8-connected) for high-value regions.
    Returns zones sorted by (area, mean_strength) desc.
    """
    H, W = layer.shape
    visited = np.zeros((H, W), dtype=bool)
    zones: List[Zone] = []
    zid = 0

    for y in range(H):
        for x in range(W):
            if visited[y, x] or layer[y, x] < threshold:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            coords = []
            strength_sum = 0.0

            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                strength_sum += float(layer[cy, cx])

                for ny in range(max(0, cy - 1), min(H, cy + 2)):
                    for nx in range(max(0, cx - 1), min(W, cx + 2)):
                        if not visited[ny, nx] and layer[ny, nx] >= threshold:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            if len(coords) < min_cells:
                continue

            ys = np.array([c[0] for c in coords], dtype=np.float32)
            xs = np.array([c[1] for c in coords], dtype=np.float32)

            cy = float(ys.mean())
            cx = float(xs.mean())

            area = int(len(coords))
            mean_strength = float(strength_sum / max(1, area))

            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())

            radius = float(np.sqrt(area / np.pi))  # equivalent disk

            zones.append(
                Zone(
                    id=f"{ztype}_{zid}",
                    type=ztype,
                    centroid_x=cx,
                    centroid_y=cy,
                    radius=radius,
                    area=area,
                    mean_strength=mean_strength,
                    bbox=(x0, y0, x1, y1),
                )
            )
            zid += 1

    zones.sort(key=lambda z: (z.area, z.mean_strength), reverse=True)
    return zones[:max_zones]


def zones_to_dicts(zones: List[Zone]) -> List[Dict[str, Any]]:
    return [
        {
            "id": z.id,
            "type": z.type,
            "centroid_x": z.centroid_x,
            "centroid_y": z.centroid_y,
            "radius": z.radius,
            "area": z.area,
            "mean_strength": z.mean_strength,
            "bbox": list(z.bbox),
        }
        for z in zones
    ]
