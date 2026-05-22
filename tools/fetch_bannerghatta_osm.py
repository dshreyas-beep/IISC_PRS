"""
ETL utility to fetch and cache the real-world Bannerghatta OSM movement graph.

IMPORTANT:
- This script is *offline preprocessing* only. Do not call it inside the sim loop.
- The simulation runtime reads the cached outputs from `data/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from simcore.data.bannerghatta_ecology import (
    BANNERGHATTA_BBOX,
    BANNERGHATTA_CENTER_LATLON,
    BANNERGHATTA_DECLARED_AREA_SQ_KM,
)

N_LAYERS = 7


def _bbox_linspace(bbox: dict[str, float], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    lons = np.linspace(float(bbox["west"]), float(bbox["east"]), int(width), dtype=np.float64)
    lats = np.linspace(float(bbox["north"]), float(bbox["south"]), int(height), dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lon_grid, lat_grid


def _fetch_features(bbox: dict[str, float]):
    import osmnx as ox

    def features(tags: dict[str, Any]):
        try:
            return ox.features_from_bbox((bbox["west"], bbox["south"], bbox["east"], bbox["north"]), tags)
        except TypeError:
            return ox.features_from_bbox(bbox["north"], bbox["south"], bbox["east"], bbox["west"], tags)

    water_tags = {
        "natural": ["water", "wetland"],
        "water": True,
        "waterway": ["river", "stream", "canal", "ditch", "drain", "riverbank"],
        "landuse": ["reservoir", "basin"],
    }
    crop_tags = {
        "landuse": ["farmland", "farm", "farmyard", "orchard", "meadow", "plant_nursery"],
    }
    cover_tags = {
        "landuse": ["forest"],
        "natural": ["wood", "scrub", "grassland"],
        "leaf_type": True,
    }

    return {
        "water": features(water_tags),
        "crop": features(crop_tags),
        "cover": features(cover_tags),
    }


def rasterize_layers(bbox: dict[str, float], width: int, height: int) -> np.ndarray:
    from shapely import contains, points, unary_union
    from shapely.geometry.base import BaseGeometry

    layers = np.zeros((int(height), int(width), N_LAYERS), dtype=np.float32)

    feats = _fetch_features(bbox)
    lon_grid, lat_grid = _bbox_linspace(bbox, width, height)
    pts = points(lon_grid.reshape(-1), lat_grid.reshape(-1))

    def union_geom(gdf) -> BaseGeometry | None:
        if gdf is None or len(gdf) == 0 or "geometry" not in gdf:
            return None
        geom = gdf.geometry
        geom = geom[geom.notna()]
        if len(geom) == 0:
            return None
        try:
            merged = unary_union(list(geom.values))
        except Exception:
            merged = unary_union([g for g in geom.values if g is not None])
        if merged is None or getattr(merged, "is_empty", True):
            return None
        return merged

    water_geom = union_geom(feats.get("water"))
    crop_geom = union_geom(feats.get("crop"))
    cover_geom = union_geom(feats.get("cover"))

    def mask_for(geom: BaseGeometry | None) -> np.ndarray:
        if geom is None:
            return np.zeros((int(height) * int(width),), dtype=bool)
        try:
            return contains(geom, pts)
        except Exception:
            out = np.zeros((pts.shape[0],), dtype=bool)
            for i, p in enumerate(pts):
                try:
                    out[i] = geom.contains(p)
                except Exception:
                    out[i] = False
            return out

    water_mask = mask_for(water_geom).reshape(int(height), int(width))
    crop_mask = mask_for(crop_geom).reshape(int(height), int(width))
    cover_mask = mask_for(cover_geom).reshape(int(height), int(width))

    layers[:, :, 0] = water_mask.astype(np.float32)
    layers[:, :, 1] = crop_mask.astype(np.float32)
    layers[:, :, 5] = cover_mask.astype(np.float32)

    layers[water_mask, 1] = 0.0
    layers[water_mask, 5] = 0.0

    return layers


def fetch_osmnx_graph(bbox: dict[str, float]) -> dict[str, Any]:
    import osmnx as ox

    custom_filter = (
        '["highway"~"primary|primary_link|secondary|secondary_link|tertiary|'
        'tertiary_link|unclassified|residential|service|track|path|footway|'
        'bridleway|steps|living_street"]["area"!~"yes"]'
    )

    try:
        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.timeout = 240
    except Exception:
        pass

    # OSMnx 2.x: bbox=(left, bottom, right, top); OSMnx 1.x: north/south/east/west args.
    try:
        graph = ox.graph_from_bbox(
            (bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            network_type="all",
            custom_filter=custom_filter,
            retain_all=True,
            simplify=True,
            truncate_by_edge=True,
        )
    except TypeError:
        graph = ox.graph_from_bbox(
            bbox["north"],
            bbox["south"],
            bbox["east"],
            bbox["west"],
            network_type="all",
            custom_filter=custom_filter,
            retain_all=True,
            simplify=True,
            truncate_by_edge=True,
        )

    nodes: dict[str, dict[str, float]] = {}
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for u, v, data in graph.edges(data=True):
        su, sv = str(u), str(v)
        a = graph.nodes[u]
        b = graph.nodes[v]
        nodes.setdefault(su, {"lat": float(a["y"]), "lon": float(a["x"])})
        nodes.setdefault(sv, {"lat": float(b["y"]), "lon": float(b["x"])})
        key = tuple(sorted((su, sv)))
        if key in seen:
            continue
        seen.add(key)
        highway = data.get("highway", "path")
        if isinstance(highway, list):
            highway = highway[0] if highway else "path"
        edges.append(
            {
                "u": su,
                "v": sv,
                "osmid": data.get("osmid", ""),
                "kind": str(highway),
                "length_m": float(data.get("length", 1.0)),
            }
        )

    return {
        "bbox": dict(bbox),
        "nodes": nodes,
        "edges": edges,
        "source": "OpenStreetMap via OSMnx graph_from_bbox",
        "metadata": {
            "center_latlon": list(BANNERGHATTA_CENTER_LATLON),
            "declared_area_sq_km": float(BANNERGHATTA_DECLARED_AREA_SQ_KM),
            "note": "Use with attribution: OpenStreetMap contributors.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache Bannerghatta OSM graph + raster layers.")
    parser.add_argument("--graph-out", type=Path, default=Path("data/bannerghatta_graph.json"))
    parser.add_argument("--layers-out", type=Path, default=Path("data/bannerghatta_osm_layers.npz"))
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    args.graph_out.parent.mkdir(parents=True, exist_ok=True)
    args.layers_out.parent.mkdir(parents=True, exist_ok=True)

    bbox = dict(BANNERGHATTA_BBOX)
    try:
        payload = fetch_osmnx_graph(bbox)
    except Exception as exc:
        raise SystemExit(
            "Failed to fetch OSMnx graph. Install deps then retry:\n"
            "  pip install osmnx\n"
            f"Error: {exc}"
        ) from exc

    args.graph_out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[bannerghatta_osm] Wrote graph: {args.graph_out} ({len(payload['nodes'])} nodes, {len(payload['edges'])} edges)")

    layers = rasterize_layers(bbox, int(args.width), int(args.height)).astype(np.float32)
    np.savez_compressed(
        args.layers_out,
        layers=layers,
        bbox=np.array([bbox["west"], bbox["south"], bbox["east"], bbox["north"]], dtype=np.float32),
    )
    print(f"[bannerghatta_osm] Wrote raster layers: {args.layers_out} shape={layers.shape} dtype={layers.dtype}")


if __name__ == "__main__":
    main()
