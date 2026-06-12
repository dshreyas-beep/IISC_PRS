import random
import numpy as np
import math
import traceback
import csv
import os
import dataclasses
from pathlib import Path

from simcore.world.grid_world import (
    GridWorld, make_topology_layers,
    LAYER_WATER, LAYER_CROP, LAYER_OBSTACLE, LAYER_SETTLEMENT
)
from simcore.world.world_runtime import WorldRuntime
from simcore.world.geo_graph import load_bannerghatta_graph
from simcore.agents.agent import make_agent
from simcore.world.zones import extract_zones
from simcore.behavior.conflict_engine import compute_conflicts
from simcore.behavior.behavior_profiles import get_ecological_modifiers
from simcore.data.bannerghatta_ecology import (
    BANNERGHATTA_BBOX,
    BANNERGHATTA_DECLARED_AREA_SQ_KM,
)
from simcore.data.species_traits import SPECIES_WEIGHTS_KG

LAYER_COVER = 5 

MAX_GROUP_SIZE   = 8   
MAX_HUMANS       = 50  
MAX_ELEPHANTS    = 25

WATER_BASIN_MAX_RADIUS  = 15.0      
CROP_MAX_INTENSITY      = 1.0
SPATIAL_BUCKET_SIZE     = 72.0
INTERACTION_SPECIES     = {"elephant", "human", "leopard", "tiger", "lion", "sloth_bear"}
_TORCH = None


def _torch_module():
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH

class MultiAgentSim:
    """
    Multi-agent ecological simulation. 
    """
    def __init__(self, topology_config: dict, seed: int = 1, fps: int = 20, species_counts: dict = None):
        self.seed   = seed
        self.fps    = fps
        self.rng    = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.tick   = 0
        self.is_night      = False 
        self.season        = "wet"
        self.topology_name = topology_config["name"]
        self.region_metadata = self._build_region_metadata()

        # Population sizing is owned by the UI/websocket payload. Never inject
        # a hardcoded baseline when switching to a real-world map.
        safe_fallback = {"elephant": 20, "leopard": 5, "sloth_bear": 0}
        self.species_counts = dict(species_counts or {})
        if not self.species_counts:
            self.species_counts = safe_fallback

        self.W = topology_config["width"]
        self.H = topology_config["height"]

        if self._is_bannerghatta_osm():
            root = Path(__file__).resolve().parents[2]
            cache_path = root / "data" / "bannerghatta_osm_layers.npz"
            if not cache_path.exists():
                layers = make_topology_layers(self.topology_name, self.W, self.H, self.np_rng)
            else:
                data = np.load(cache_path, allow_pickle=False)
                layers = data["layers"].astype(np.float32)
                if layers.shape != (self.H, self.W, 7):
                    layers = make_topology_layers(self.topology_name, self.W, self.H, self.np_rng)
        else:
            layers = make_topology_layers(self.topology_name, self.W, self.H, self.np_rng)

        self.layers = layers
        self.grid = GridWorld(W=self.W, H=self.H, layers=layers)

        # bannerghatta_osm uses a cached OSM movement graph. The raster layers are
        # precomputed (loaded inside `make_topology_layers`) and are not fetched
        # during runtime.
        self.geo_graph = load_bannerghatta_graph(self.W, self.H) if self._is_bannerghatta_osm() else None
        if self.geo_graph and self.geo_graph.nodes:
            self.geo_graph.paint_layers(self.grid.layers, LAYER_WATER, LAYER_OBSTACLE, LAYER_COVER)
        self.safari_zones = self.geo_graph.to_frontend(max_edges=0).get("safari_zones", []) if self.geo_graph else []

        self.base_water_map      = np.copy(self.grid.layers[:, :, LAYER_WATER])
        self.base_crop_map       = np.copy(self.grid.layers[:, :, LAYER_CROP])
        self.base_settlement_map = np.copy(self.grid.layers[:, :, LAYER_SETTLEMENT])
        self.static_water_map      = np.copy(self.base_water_map)
        self.static_crop_map       = np.copy(self.base_crop_map)
        self.static_settlement_map = np.copy(self.base_settlement_map)
        self.static_cover_map      = np.copy(self.grid.layers[:, :, LAYER_COVER])

        if not self._is_bannerghatta():
            self.grid.layers[:, :, LAYER_COVER] = 1.0
        else:
            np.clip(self.grid.layers[:, :, LAYER_COVER], 0.0, 1.0, out=self.grid.layers[:, :, LAYER_COVER])
        self.grid.layers[:, :, LAYER_COVER][self.base_water_map        > 0.1] = 0.0
        self.grid.layers[:, :, LAYER_COVER][self.base_crop_map         > 0.1] = 0.0
        self.grid.layers[:, :, LAYER_COVER][self.base_settlement_map   > 0.1] = 0.0

        # --- FENCE & ANIMAL BRIDGE SETUP ---
        self.fence_y = self.H / 2.0
        self.bridge_center = (self.W / 2.0, self.H / 2.0)
        self.bridge_width = 30.0
        
        # Draw the physical obstacle fence across the middle
        self._draw_fence()

        self.world = WorldRuntime(self.grid, self.np_rng)
        self.zones = []
        
        self.water_nodes = []
        self.crop_nodes = []
        self._init_resource_nodes()
        self._recalculate_zones(initial=True)

        self.agents          = []
        self.herd_map        = {}
        self.agent_id_counter = 0

        self.human_group_counter  = 0
        self.human_group_centers  : dict[int, tuple] = {}

        self.mortality_events = []
        self.hotspots = []
        self._spatial_index = {}
        self._sample_coord_cache = {}

        self._init_agents()

        self.analytics_data = {
            "carrying_capacity": 100.0,
            "eco_scores":        {},
            "survival_report":   "Analyzing initial adaptation patterns…"
        }

        # --- UPGRADED STATISTICAL LOGGER ---
        self.log_file = "ecosystem_data.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                csv.writer(f).writerow([
                    "Tick", "Agent_ID", "Species", "Weight_kg", "Energy", "Hydration", "Action_Mode", "Behavioral_Condition"
                ])

    def _is_bannerghatta(self) -> bool:
        return self.topology_name in ("bannerghatta", "bannerghatta_bnp", "bannerghatta_osm")

    def _is_bannerghatta_osm(self) -> bool:
        return self.topology_name == "bannerghatta_osm"

    def _build_region_metadata(self):
        if self.topology_name not in ("bannerghatta", "bannerghatta_bnp", "bannerghatta_osm"):
            return None
        is_osm = self.topology_name == "bannerghatta_osm"
        return {
            "id": "bannerghatta_osm" if is_osm else "bannerghatta",
            "name": "Bannerghatta National Park OSM graph" if is_osm else "Bannerghatta National Park region",
            "center_lat": 12.8008,
            "center_lon": 77.5756,
            "declared_area_sq_km": BANNERGHATTA_DECLARED_AREA_SQ_KM,
            "bbox": dict(BANNERGHATTA_BBOX),
            "cell_note": "Agents move on OSM roads, tracks, footways, and paths; raster layers are visualization/perception surfaces." if is_osm else "120x120 ecological proxy grid, not a surveyed legal boundary",
            "source": "OpenStreetMap / OSMnx graph + Overpass ecological features" if is_osm else "Calibrated procedural proxy",
            "assumptions": [
                "BNP declared area is injected as 260.51 sq km / 65,127.5 acres",
                "OSM relation 8124064 supplies the park geometry when present",
                "Suvarnamukhi OSM waterway segments are injected as primary water resources",
                "Natural water polygons are read from OSM and recharge pits are represented as BBP clusters unless exact public OSM points exist",
                "Only the OSM graph constrains movement in bannerghatta_osm mode"
            ]
        }

    def _draw_fence(self):
        """ Draws the fence into the Obstacle layer so the UI renders it as a dark barrier """
        if self._is_bannerghatta_osm():
            return
        fence_y_int = int(self.fence_y)
        for x in range(self.W):
            # Leave a gap for the bridge
            if abs(x - self.bridge_center[0]) > (self.bridge_width / 2.0):
                if 0 <= fence_y_int < self.H:
                    self.grid.layers[fence_y_int-1:fence_y_int+2, x, LAYER_OBSTACLE] = 1.0
                    self.grid.layers[fence_y_int-1:fence_y_int+2, x, LAYER_COVER] = 0.0

    def _init_resource_nodes(self):
        if self._is_bannerghatta_osm():
            if self.geo_graph:
                self.water_nodes = []
                for line in self.geo_graph.water_polylines:
                    pts = line.get("points", [])
                    stride = max(1, len(pts) // 10)
                    for idx, p in enumerate(pts[::stride]):
                        node_id = self.geo_graph.nearest_node_xy(float(p["x"]), float(p["y"])) if self.geo_graph.nodes else None
                        self.water_nodes.append({
                            "cx": float(p["x"]),
                            "cy": float(p["y"]),
                            "lat": float(p["lat"]),
                            "lon": float(p["lon"]),
                            "label": f"{line.get('name', 'Suvarnamukhi')} {idx + 1}",
                            "kind": "stream",
                            "graph_node": node_id,
                            "intensity": 0.95,
                            "active": True,
                        })
                for p in self.geo_graph.water_points:
                    node_id = self.geo_graph.nearest_node_xy(float(p["x"]), float(p["y"])) if self.geo_graph.nodes else None
                    self.water_nodes.append({
                        "cx": float(p["x"]),
                        "cy": float(p["y"]),
                        "lat": float(p["lat"]),
                        "lon": float(p["lon"]),
                        "label": p.get("label", "OSM water body"),
                        "kind": p.get("type", "lake"),
                        "graph_node": node_id,
                        "intensity": 0.85,
                        "active": True,
                    })
                for p in self.geo_graph.recharge_pits:
                    node_id = self.geo_graph.nearest_node_xy(float(p["x"]), float(p["y"])) if self.geo_graph.nodes else None
                    self.water_nodes.append({
                        "cx": float(p["x"]),
                        "cy": float(p["y"]),
                        "lat": float(p["lat"]),
                        "lon": float(p["lon"]),
                        "label": p.get("label", "Recharge pit"),
                        "kind": "recharge_pit",
                        "graph_node": node_id,
                        "intensity": 0.55,
                        "active": True,
                    })

            water_zones = extract_zones(self.static_water_map, threshold=0.08, ztype="water", min_cells=4, max_zones=24)
            crop_zones = extract_zones(self.static_crop_map, threshold=0.08, ztype="crop", min_cells=8, max_zones=32)
            if not self.water_nodes:
                self.water_nodes = [
                    {
                        "cx": z.centroid_x,
                        "cy": z.centroid_y,
                        "intensity": max(0.25, min(1.0, z.mean_strength)),
                        "active": True,
                    }
                    for z in water_zones
                ]
            self.crop_nodes = [
                {
                    "cx": z.centroid_x,
                    "cy": z.centroid_y,
                    "intensity": max(0.10, min(1.0, z.mean_strength)),
                }
                for z in crop_zones
            ]
            if self.geo_graph and self.geo_graph.nodes:
                for node in self.water_nodes:
                    if "graph_node" not in node or node["graph_node"] is None:
                        node["graph_node"] = self.geo_graph.nearest_node_xy(node["cx"], node["cy"])
                        if node["graph_node"] in self.geo_graph.nodes:
                            g = self.geo_graph.nodes[node["graph_node"]]
                            node["cx"], node["cy"] = g["x"], g["y"]
                for node in self.crop_nodes:
                    node["graph_node"] = self.geo_graph.nearest_node_xy(node["cx"], node["cy"])
                    if node["graph_node"] in self.geo_graph.nodes:
                        g = self.geo_graph.nodes[node["graph_node"]]
                        node["cx"], node["cy"] = g["x"], g["y"]
            if not self.water_nodes:
                self.water_nodes = [{"cx": self.W * 0.50, "cy": self.H * 0.55, "intensity": 0.65, "active": True}]
            if not self.crop_nodes:
                self.crop_nodes = [{"cx": self.W * 0.25, "cy": self.H * 0.35, "intensity": 0.35}]
            return

        if self._is_bannerghatta():
            self.water_nodes = [
                {"cx": self.W * 0.43, "cy": self.H * 0.18, "intensity": 0.85, "active": True},
                {"cx": self.W * 0.57, "cy": self.H * 0.31, "intensity": 0.70, "active": True},
                {"cx": self.W * 0.45, "cy": self.H * 0.51, "intensity": 0.78, "active": True},
                {"cx": self.W * 0.62, "cy": self.H * 0.70, "intensity": 0.86, "active": True},
                {"cx": self.W * 0.38, "cy": self.H * 0.86, "intensity": 0.70, "active": True},
            ]
            self.crop_nodes = [
                {"cx": self.W * 0.22, "cy": self.H * 0.18, "intensity": 0.45},
                {"cx": self.W * 0.78, "cy": self.H * 0.22, "intensity": 0.50},
                {"cx": self.W * 0.23, "cy": self.H * 0.38, "intensity": 0.38},
                {"cx": self.W * 0.80, "cy": self.H * 0.45, "intensity": 0.42},
                {"cx": self.W * 0.24, "cy": self.H * 0.62, "intensity": 0.35},
                {"cx": self.W * 0.77, "cy": self.H * 0.70, "intensity": 0.34},
            ]
            for w in self.water_nodes:
                self._paint_circle(w["cx"], w["cy"], WATER_BASIN_MAX_RADIUS * w["intensity"], LAYER_WATER, self.base_water_map)
            for c in self.crop_nodes:
                self._paint_circle(c["cx"], c["cy"], WATER_BASIN_MAX_RADIUS * c["intensity"], LAYER_CROP, self.base_crop_map)
            return

        self.water_nodes = [
            {"cx": self.W * 0.3, "cy": self.H * 0.3, "intensity": 1.0, "active": True},
            {"cx": self.W * 0.7, "cy": self.H * 0.7, "intensity": 1.0, "active": True}
        ]
        self.crop_nodes = []
        for w in self.water_nodes:
            self._paint_circle(w["cx"], w["cy"], WATER_BASIN_MAX_RADIUS, LAYER_WATER, self.base_water_map)

    def _paint_circle(self, cx, cy, r, layer_idx, base_map):
        r_int = int(math.ceil(r))
        for dy in range(-r_int, r_int + 1):
            for dx in range(-r_int, r_int + 1):
                if dx*dx + dy*dy <= r*r:
                    ny, nx = int(cy) + dy, int(cx) + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W:
                        self.grid.layers[ny, nx, layer_idx] = 1.0
                        base_map[ny, nx] = 1.0
                        self.grid.layers[ny, nx, LAYER_COVER] = 0.0

    def _clear_layer(self, layer_idx, base_map):
        self.grid.layers[:, :, layer_idx] = 0.0
        base_map[:, :] = 0.0

    def _update_dynamic_resources(self):
        self._clear_layer(LAYER_WATER, self.base_water_map)
        self._clear_layer(LAYER_CROP, self.base_crop_map)

        if self._is_bannerghatta_osm():
            self.grid.layers[:, :, LAYER_WATER] = np.maximum(
                self.grid.layers[:, :, LAYER_WATER],
                self.static_water_map * 0.45
            )
            self.grid.layers[:, :, LAYER_CROP] = np.maximum(
                self.grid.layers[:, :, LAYER_CROP],
                self.static_crop_map * 0.30
            )
            self.base_water_map[:, :] = self.static_water_map
            self.base_crop_map[:, :] = self.static_crop_map

        alive_elephants = [a for a in self.agents if a.species == "elephant" and getattr(a, "alive", True)]
        alive_humans = [a for a in self.agents if a.species == "human" and getattr(a, "alive", True)]

        active_waters = [w for w in self.water_nodes if w["active"]]
        
        if not active_waters:
            if self._is_bannerghatta_osm():
                for w in self.water_nodes:
                    w["intensity"] = max(float(w.get("intensity", 0.0)), 0.35)
                    w["active"] = True
            else:
                self.water_nodes = [
                    {"cx": float(self.rng.randint(20, self.W-20)), "cy": float(self.rng.randint(20, self.H-20)), "intensity": 1.0, "active": True},
                    {"cx": float(self.rng.randint(20, self.W-20)), "cy": float(self.rng.randint(20, self.H-20)), "intensity": 1.0, "active": True}
                ]
            active_waters = self.water_nodes
            for a in self.agents:
                if getattr(a, "alive", True):
                    a.migrating = True
                    tgt = self.rng.choice(active_waters)
                    a.migration_target = (tgt["cx"], tgt["cy"])
                    a.mode = "MIGRATE"

        for w in active_waters:
            r = WATER_BASIN_MAX_RADIUS * w["intensity"]
            drain_multiplier = 3.0 if getattr(self, "season", "wet") == "dry" else 1.0
            drain = sum(0.005 for a in alive_elephants if math.hypot(a.x - w["cx"], a.y - w["cy"]) <= r + 8) * drain_multiplier
            drain += sum(0.001 for a in alive_humans if math.hypot(a.x - w["cx"], a.y - w["cy"]) <= r + 8) * drain_multiplier
            
            w["intensity"] = max(0.0, w["intensity"] - drain)
            
            if w["intensity"] < 0.1:
                if self._is_bannerghatta_osm():
                    w["intensity"] = 0.25 if w.get("kind") == "stream" else 0.15
                    w["active"] = True
                else:
                    w["cx"] = float(self.rng.randint(20, self.W-20))
                    w["cy"] = float(self.rng.randint(20, self.H-20))
                    w["intensity"] = 1.0
                    w["active"] = True
                
                for a in self.agents:
                    if getattr(a, "alive", True) and a.mode != "HUNT":
                        a.migrating = True
                        a.migration_target = (w["cx"], w["cy"])
                        a.mode = "MIGRATE"
            else:
                self._paint_circle(w["cx"], w["cy"], WATER_BASIN_MAX_RADIUS * w["intensity"], LAYER_WATER, self.base_water_map)

        dead_crops = []
        for c in self.crop_nodes:
            r = WATER_BASIN_MAX_RADIUS * c["intensity"]
            e_near = sum(1 for a in alive_elephants if math.hypot(a.x - c["cx"], a.y - c["cy"]) <= r + 8)
            h_near = sum(1 for a in alive_humans if math.hypot(a.x - c["cx"], a.y - c["cy"]) <= r + 15)

            c["intensity"] -= (e_near * 0.05) 
            c["intensity"] += (h_near * 0.02) 
            c["intensity"] -= 0.001          

            c["intensity"] = min(1.0, max(0.0, c["intensity"]))
            if c["intensity"] < 0.05:
                if self._is_bannerghatta_osm():
                    c["intensity"] = 0.05
                else:
                    dead_crops.append(c)
            else: self._paint_circle(c["cx"], c["cy"], WATER_BASIN_MAX_RADIUS * c["intensity"], LAYER_CROP, self.base_crop_map)

        for c in dead_crops: self.crop_nodes.remove(c)
        if self.tick % 10 == 0:
            self._recalculate_zones()

    def _new_human_group(self, cx: float, cy: float) -> int:
        gid = self.human_group_counter
        self.human_group_counter += 1
        self.human_group_centers[gid] = (cx, cy)
        return gid

    def _sample_from_layer(self, layer_idx: int, threshold: float, *, avoid_settlement: bool = False) -> tuple[float, float]:
        layer = self.grid.layers[:, :, layer_idx]
        cache_key = (int(layer_idx), float(threshold), bool(avoid_settlement))
        coords = self._sample_coord_cache.get(cache_key)
        if coords is None:
            mask = layer >= threshold
            if avoid_settlement:
                mask = mask & (self.grid.layers[:, :, LAYER_SETTLEMENT] < 0.25)
            coords = np.argwhere(mask)
            self._sample_coord_cache[cache_key] = coords
        if coords.size == 0:
            x, y = self.rng.uniform(20, self.grid.W - 20), self.rng.uniform(20, self.grid.H - 20)
            return self._snap_xy_to_graph(x, y)
        y, x = coords[int(self.np_rng.integers(0, len(coords)))]
        return self._snap_xy_to_graph(
            float(np.clip(x + self.rng.uniform(-2.0, 2.0), 2.0, self.grid.W - 2.0)),
            float(np.clip(y + self.rng.uniform(-2.0, 2.0), 2.0, self.grid.H - 2.0)),
        )

    def _graph_enabled(self) -> bool:
        return bool(self._is_bannerghatta_osm() and self.geo_graph and self.geo_graph.nodes)

    def _snap_xy_to_graph(self, x: float, y: float) -> tuple[float, float]:
        if not self._graph_enabled():
            return float(x), float(y)
        sx, sy, _ = self.geo_graph.snap_xy(float(x), float(y), self.W, self.H)
        return sx, sy

    def _snap_agent_to_graph(self, agent) -> None:
        if not self._graph_enabled():
            return
        agent.x, agent.y, node_id = self.geo_graph.snap_xy(float(agent.x), float(agent.y), self.W, self.H)
        agent.graph_node = node_id
        agent.graph_next_node = None

    def _sample_safari_zone(self, zone_id: str) -> tuple[float, float]:
        if not self._graph_enabled():
            return self.rng.uniform(20, self.grid.W - 20), self.rng.uniform(20, self.grid.H - 20)
        zones = {z["id"]: z for z in getattr(self, "safari_zones", [])}
        zone = zones.get(zone_id)
        if not zone:
            return self.geo_graph.sample_node(self.rng)[:2]
        angle = self.rng.uniform(0.0, math.tau)
        radius = self.rng.uniform(0.0, float(zone["radius"]) * 0.75)
        return self._snap_xy_to_graph(float(zone["cx"]) + math.cos(angle) * radius, float(zone["cy"]) + math.sin(angle) * radius)

    def _nearest_water_target(self, x: float, y: float) -> dict | None:
        active = [w for w in self.water_nodes if w.get("active", True)]
        if not active:
            return None
        return min(active, key=lambda w: math.hypot(float(x) - w["cx"], float(y) - w["cy"]))

    def _rebuild_spatial_index(self) -> None:
        buckets = {}
        for agent in self.agents:
            if not getattr(agent, "alive", True):
                continue
            key = (int(agent.x // SPATIAL_BUCKET_SIZE), int(agent.y // SPATIAL_BUCKET_SIZE))
            buckets.setdefault(key, []).append(agent)
        self._spatial_index = buckets

    def _nearby_agents(self, agent, radius: float):
        if agent.species not in INTERACTION_SPECIES:
            return
        if not self._spatial_index:
            yield from (a for a in self.agents if a.id != agent.id)
            return
        bx = int(agent.x // SPATIAL_BUCKET_SIZE)
        by = int(agent.y // SPATIAL_BUCKET_SIZE)
        reach = int(math.ceil(radius / SPATIAL_BUCKET_SIZE))
        r2 = radius * radius
        for gy in range(by - reach, by + reach + 1):
            for gx in range(bx - reach, bx + reach + 1):
                for other in self._spatial_index.get((gx, gy), []):
                    if other.id == agent.id:
                        continue
                    if (agent.x - other.x) ** 2 + (agent.y - other.y) ** 2 <= r2:
                        yield other

    def _init_agents(self):
        if self._is_bannerghatta_osm():
            herd_centers = {
                1: self._sample_from_layer(LAYER_COVER, 0.35, avoid_settlement=True),
                2: self._sample_from_layer(LAYER_COVER, 0.35, avoid_settlement=True)
            }
        else:
            herd_centers = {
                1: (self.rng.uniform(30, self.grid.W - 30), self.rng.uniform(30, self.grid.H - 30)),
                2: (self.rng.uniform(30, self.grid.W - 30), self.rng.uniform(30, self.grid.H - 30))
            }
        herd_leaders_assigned = {1: False, 2: False}

        human_groups = []
        active_waters = [w for w in self.water_nodes if w["active"]]
        for _ in range(4):
            if self._is_bannerghatta_osm():
                cx, cy = self._sample_from_layer(LAYER_SETTLEMENT, 0.25)
            elif active_waters:
                spawn = self.rng.choice(active_waters)
                cx = spawn["cx"] + self.rng.uniform(-10, 10)
                cy = spawn["cy"] + self.rng.uniform(-10, 10)
            else:
                cx = self.rng.uniform(30, self.grid.W - 30)
                cy = self.rng.uniform(30, self.grid.H - 30)
            human_groups.append(self._new_human_group(cx, cy))

        for species, count in self.species_counts.items():
            for _ in range(count):
                agent = make_agent(self.agent_id_counter, self.world, species)
                agent.eco_mods = get_ecological_modifiers(species, self.topology_name)
                agent.heading  = self.rng.uniform(0, 2 * math.pi)

                agent.return_home_timer = 0
                agent.attack_cooldown   = 0  
                agent.migrating         = False
                mt = self._nearest_water_target(agent.x, agent.y)
                agent.migration_target  = (mt["cx"], mt["cy"]) if mt else self._snap_xy_to_graph(self.rng.uniform(20, self.W - 20), self.rng.uniform(20, self.H - 20))
                
                # --- BASE WEIGHTS FOR STATISTICS ---
                agent.weight = float(SPECIES_WEIGHTS_KG.get(species, 25.0))

                if hasattr(agent, 'adaptation_period'): agent.adaptation_period = 200 + (self.agent_id_counter * 20)

                if species == "elephant":
                    agent.herd_id  = 1 if self.agent_id_counter % 2 == 0 else 2
                    self.herd_map[agent.id] = agent.herd_id
                    agent.x, agent.y = herd_centers[agent.herd_id][0] + self.rng.uniform(-5, 5), herd_centers[agent.herd_id][1] + self.rng.uniform(-5, 5)
                    agent.home_x, agent.home_y = agent.x, agent.y

                    if not herd_leaders_assigned[agent.herd_id]:
                        agent.is_leader = True
                        herd_leaders_assigned[agent.herd_id] = True

                elif species == "human":
                    grp = self.rng.choice(human_groups)
                    agent.group_id = grp
                    cx, cy = self.human_group_centers[grp]
                    # Restrict human spawning to settlement or crop cells only, not deep forest
                    if self._is_bannerghatta_osm():
                        # Try to sample from settlement first, then crop as fallback
                        try:
                            agent.x, agent.y = self._sample_from_layer(LAYER_SETTLEMENT, 0.15)
                        except:
                            agent.x, agent.y = self._sample_from_layer(LAYER_CROP, 0.15)
                    else:
                        agent.x, agent.y = cx + self.rng.uniform(-5, 5), cy + self.rng.uniform(-5, 5)
                    agent.home_x, agent.home_y = cx, cy

                elif species == "tiger" and self._is_bannerghatta_osm():
                    agent.x, agent.y = self._sample_safari_zone("tiger_safari")
                    agent.home_x, agent.home_y = agent.x, agent.y

                elif species == "lion" and self._is_bannerghatta_osm():
                    agent.x, agent.y = self._sample_safari_zone("lion_safari")
                    agent.home_x, agent.home_y = agent.x, agent.y

                else:
                    if self._is_bannerghatta_osm():
                        agent.x, agent.y = self._sample_from_layer(LAYER_COVER, 0.25, avoid_settlement=True)
                    else:
                        agent.x, agent.y = self.rng.uniform(20, self.grid.W - 20), self.rng.uniform(20, self.grid.H - 20)
                    agent.home_x, agent.home_y = agent.x, agent.y

                self._snap_agent_to_graph(agent)
                self.agents.append(agent)
                self.agent_id_counter += 1

    def _recalculate_zones(self, initial=False):
        self.zones.clear()
        raw_water = extract_zones(self.grid.layers[:, :, LAYER_WATER], threshold=0.1, ztype="water")
        for z in raw_water: self.zones.append(dataclasses.replace(z, radius=min(z.radius, 25.0)))

        raw_crops = extract_zones(self.grid.layers[:, :, LAYER_CROP], threshold=0.1, ztype="crop")
        for z in raw_crops: self.zones.append(dataclasses.replace(z, radius=min(z.radius * 0.5, 18.0)))

        raw_settlements = extract_zones(self.grid.layers[:, :, LAYER_SETTLEMENT], threshold=0.2, ztype="settlement")
        for z in raw_settlements: self.zones.append(dataclasses.replace(z, radius=min(z.radius * 0.5, 15.0)))

    def _dist_to_nearest_water(self, x: float, y: float) -> float:
        dists = [math.hypot(x - w["cx"], y - w["cy"]) for w in self.water_nodes if w["active"]]
        return min(dists) if dists else 999.0

    def _update_analytics(self):
        crop_pixels      = np.sum(self.grid.layers[:, :, LAYER_CROP] > 0.1)
        herbivore_demand = sum((100 - a.energy) for a in self.agents if a.species in ["elephant", "sloth_bear", "gaur", "sambar_deer", "spotted_deer", "barking_deer", "wild_boar"] and getattr(a, "alive", True))
        cap_ratio = min(100.0, (crop_pixels * 8.0) / max(1.0, herbivore_demand) * 100.0)

        eco_scores = {}
        for sp in list(self.species_counts.keys())[:24]:
            sp_agents = [a for a in self.agents if a.species == sp and getattr(a, "alive", True)]
            if not sp_agents: continue
            avg_energy    = sum(a.energy for a in sp_agents) / len(sp_agents)
            avg_thirst    = sum(a.thirst  for a in sp_agents) / len(sp_agents)
            panic_penalty = sum(15 for a in sp_agents if a.mode in ["FLEE", "DEFEND", "MIGRATE"]) / len(sp_agents)
            eco_scores[sp] = round(max(10.0, min(100.0, avg_energy - (avg_thirst * 0.25) - panic_penalty)), 1)

        human_agents     = [a for a in self.agents if a.species == "human" and getattr(a, "alive", True)]
        if not human_agents: report = "No active human settlement agents in this baseline."
        else: report = f"Expanding Populations: {len({getattr(a, 'group_id', 0) for a in human_agents})} settlement group(s)."
        alive_counts = {}
        for a in self.agents:
            if getattr(a, "alive", True):
                alive_counts[a.species] = alive_counts.get(a.species, 0) + 1
        self.analytics_data = {"carrying_capacity": round(cap_ratio, 1), "eco_scores": eco_scores, "survival_report": report, "alive_counts": alive_counts}

    def _build_settlement_and_crops(self, agent, y_int: int, x_int: int, dist_to_water: float, local_humans: int):
        if dist_to_water > 80.0 or agent.energy <= 20.0 or self.rng.random() > 0.20: return
        
        merge = False
        for c in self.crop_nodes:
            if math.hypot(agent.x - c["cx"], agent.y - c["cy"]) < 20.0:
                merge = True
                break
        if not merge:
            self.crop_nodes.append({"cx": agent.x, "cy": agent.y, "intensity": 0.3})

        build_radius = min(3, 1 + local_humans // 2)
        for dy in range(-build_radius, build_radius + 1):
            for dx in range(-build_radius, build_radius + 1):
                if dx * dx + dy * dy <= build_radius * build_radius:
                    ny, nx = y_int + dy, x_int + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W and self.base_water_map[ny, nx] < 0.2:
                        self.grid.layers[ny, nx, LAYER_SETTLEMENT] = min(1.0, self.grid.layers[ny, nx, LAYER_SETTLEMENT] + 0.5)
                        self.base_settlement_map[ny, nx] = self.grid.layers[ny, nx, LAYER_SETTLEMENT]

    def _get_predictive_hotspots(self):
        if hasattr(self, 'mortality_events'): self.mortality_events = [m for m in self.mortality_events if self.tick - m["tick"] < 300]
        else: self.mortality_events = []
        return self.mortality_events

    def step(self):  # noqa: C901
        try:
            self.tick   += 1
            self.is_night = (self.tick % 800) > 400
            if self.tick % 6000 == 0:
                self.season = "dry" if getattr(self, "season", "wet") == "wet" else "wet"
            self.world.step()

            self._update_dynamic_resources()

            if self.tick % 10 == 0: self._update_analytics()

            # --- EXTERNAL AGENT INJECTION ---
            # Replaces the flawed reproduction/cloning mechanic.
            # Agents now wander in from the map boundaries to replenish the ecosystem
            # up to the baseline carrying capacity defined in species_counts.
            if self.tick % 150 == 0:
                new_agents = []
                current_counts = {}
                for a in self.agents:
                    if getattr(a, "alive", True):
                        current_counts[a.species] = current_counts.get(a.species, 0) + 1

                for species, target_count in self.species_counts.items():
                    if current_counts.get(species, 0) < target_count:
                        new_a = make_agent(self.agent_id_counter, self.world, species)
                        new_a.eco_mods = get_ecological_modifiers(species, self.topology_name)
                        new_a.weight = float(SPECIES_WEIGHTS_KG.get(species, 25.0))
                        
                        # Spawn at map boundary
                        edge = self.rng.choice(["top", "bottom", "left", "right"])
                        if edge == "top":      new_a.x, new_a.y = self.rng.uniform(10, self.W - 10), 2.0
                        elif edge == "bottom": new_a.x, new_a.y = self.rng.uniform(10, self.W - 10), self.H - 2.0
                        elif edge == "left":   new_a.x, new_a.y = 2.0, self.rng.uniform(10, self.H - 10)
                        else:                  new_a.x, new_a.y = self.W - 2.0, self.rng.uniform(10, self.H - 10)
                        
                        new_a.home_x, new_a.home_y = new_a.x, new_a.y
                        new_a.heading = self.rng.uniform(0, 2 * math.pi)
                        new_a.migrating = False
                        new_a.return_home_timer = 0
                        new_a.attack_cooldown = 0
                        
                        if species == "elephant":
                            new_a.herd_id = self.rng.choice([1, 2])
                            self.herd_map[new_a.id] = new_a.herd_id
                        elif species == "human":
                            new_a.group_id = 0
                            
                        self._snap_agent_to_graph(new_a)
                        self.agent_id_counter += 1
                        new_agents.append(new_a)
                
                self.agents.extend(new_agents)

            if self.tick % 50 == 0:
                self.world.grid.layers[:, :, LAYER_SETTLEMENT] -= 0.02
                self.world.grid.layers[:, :, LAYER_COVER]      += 0.01

                trampled_sett_mask = (self.base_settlement_map < 0.05) & (self.grid.layers[:, :, LAYER_SETTLEMENT] > 0.01)
                self.grid.layers[:, :, LAYER_SETTLEMENT][trampled_sett_mask] = 0.0
                np.clip(self.world.grid.layers[:, :, LAYER_SETTLEMENT], 0.0, 1.0, out=self.world.grid.layers[:, :, LAYER_SETTLEMENT])
                np.clip(self.world.grid.layers[:, :, LAYER_COVER], 0.0, 1.0, out=self.world.grid.layers[:, :, LAYER_COVER])

                # Maintain fence integrity against decay
                self._draw_fence()

            water_exists = any(w["active"] for w in self.water_nodes)
            crops_exist  = len(self.crop_nodes) > 0
            self._rebuild_spatial_index()

            for agent in self.agents:
                if not getattr(agent, "alive", True): continue
                if not hasattr(agent, "migrating"): agent.migrating = False
                if not hasattr(agent, "migration_target"):
                    target = self._nearest_water_target(agent.x, agent.y)
                    agent.migration_target = (target["cx"], target["cy"]) if target else self._snap_xy_to_graph(self.rng.uniform(20, self.W - 20), self.rng.uniform(20, self.H - 20))
                if not hasattr(agent, "heading"): agent.heading = self.rng.uniform(0, 2 * math.pi)
                if not hasattr(agent, "weight"): 
                    agent.weight = float(SPECIES_WEIGHTS_KG.get(agent.species, 25.0))

                agent.attack_cooldown = max(0, getattr(agent, "attack_cooldown", 0) - 1)
                agent.energy -= 0.002 if agent.species == "human" else 0.005
                agent.thirst += 0.02
                if self.topology_name == "drought": agent.thirst += 0.05
                agent.age += 1

                # Dynamic Weight Calculation
                if agent.energy < 40.0:
                    agent.weight -= (0.5 if agent.species == "elephant" else 0.05)
                
                dist_to_water = self._dist_to_nearest_water(agent.x, agent.y)
                dist_to_crop = min([math.hypot(agent.x - c["cx"], agent.y - c["cy"]) for c in self.crop_nodes], default=999.0)

                # SURVIVAL FAILSAFES
                if dist_to_water < 12.0: agent.thirst = 0.0
                if agent.species in ["human", "elephant"] and dist_to_crop < 12.0: 
                    agent.energy = min(100.0, agent.energy + 20.0)
                    agent.weight += (0.2 if agent.species == "elephant" else 0.02) # Gain weight from food

                if agent.thirst >= 100.0:
                    agent.alive = False
                    self.mortality_events.append({"id": f"thirst_{self.tick}_{agent.id}", "tick": self.tick, "type": "mortality", "cx": agent.x, "cy": agent.y, "radius": 12, "label": f"{agent.species.capitalize()} Died of Thirst"})
                    continue
                if agent.energy <= 0:
                    agent.alive = False
                    self.mortality_events.append({"id": f"starve_{self.tick}_{agent.id}", "tick": self.tick, "type": "mortality", "cx": agent.x, "cy": agent.y, "radius": 12, "label": f"{agent.species.capitalize()} Starved"})
                    continue

                habitat_collapsed = False
                if agent.species == "human":
                    if dist_to_water > 60.0 or (dist_to_crop > 30.0 and agent.energy < 40): habitat_collapsed = True
                elif agent.species == "elephant":
                    if dist_to_water > 80.0 or (self.grid.sample(LAYER_COVER, agent.x, agent.y) < 0.1 and dist_to_crop > 30.0): habitat_collapsed = True
                elif agent.species == "sloth_bear" and self.grid.sample(LAYER_COVER, agent.x, agent.y) < 0.2: habitat_collapsed = True
                elif agent.species == "leopard" and self.grid.sample(LAYER_SETTLEMENT, agent.x, agent.y) > 0.2: habitat_collapsed = True

                if habitat_collapsed and not agent.migrating:
                    agent.migrating = True
                    active_waters = [w for w in self.water_nodes if w["active"]]
                    if active_waters:
                        tgt = self.rng.choice(active_waters)
                        agent.migration_target = (tgt["cx"], tgt["cy"])
                    else: agent.migration_target = (self.rng.uniform(15, self.W - 15), self.rng.uniform(15, self.H - 15))
                    agent.mode = "MIGRATE"

                if agent.migrating:
                    agent.mode = "MIGRATE"
                    tx, ty = agent.migration_target
                    if math.hypot(agent.x - tx, agent.y - ty) < 20.0 or dist_to_water < 20.0:
                        agent.migrating = False
                        agent.mode = "WANDER"

                x_int, y_int = int(agent.x), int(agent.y)
                if 0 <= x_int < self.grid.W and 0 <= y_int < self.grid.H:
                    if agent.species == "elephant" and self.grid.sample(LAYER_COVER, agent.x, agent.y) > 0.1:
                        agent.energy = min(100.0, agent.energy + 0.5) 
                    elif agent.species == "human" and not agent.migrating:
                        local_humans = sum(1 for h in self.agents if h.species == "human" and getattr(h, "alive", True) and math.hypot(h.x - agent.x, h.y - agent.y) < 15.0)
                        
                        if getattr(agent, "fear_level", 0.0) > 0.7 or agent.mode == "BUILD_FENCE":
                            agent.mode = "BUILD_FENCE"
                            for dy in range(-2, 3):
                                for dx in range(-2, 3):
                                    ny, nx = int(agent.y) + dy, int(agent.x) + dx
                                    if 0 <= ny < self.H and 0 <= nx < self.W:
                                        self.grid.layers[ny, nx, LAYER_OBSTACLE] = 1.0
                                        self.grid.layers[ny, nx, LAYER_COVER] = 0.0

                        if agent.mode not in ["FLEE", "WATER", "RETURN_HOME", "MIGRATE", "BUILD_FENCE"]:
                            self._build_settlement_and_crops(agent, y_int, x_int, dist_to_water, local_humans)

                # UTILITIES
                u_water  = ((agent.thirst / 100.0) * agent.eco_mods["water_urgency"] * 3.0 if water_exists else 0.0)
                if agent.thirst > 80.0 and water_exists: u_water = 50.0

                u_food   = (((100.0 - agent.energy) / 100.0) * agent.eco_mods["food_urgency"] * 2.0 if crops_exist else 0.0)
                if agent.energy < 40.0 and crops_exist: u_food = 50.0
                if agent.species == "elephant" and agent.energy < 85.0 and crops_exist: u_food = 60.0

                u_wander = 0.5
                u_flee   = u_hunt = u_defend = 0.0
                u_return  = 8.0  if getattr(agent, "return_home_timer", 0) > 0 else 0.0
                u_migrate = 10.0 if getattr(agent, "migrating", False)      else 0.0
                
                if getattr(self, "season", "wet") == "dry" and agent.species == "elephant" and dist_to_water > 30.0:
                    u_migrate += 20.0
                    agent.migrating = True
                
                threat_dx = threat_dy = hunt_dx = hunt_dy = 0.0

                # Human-specific utilities
                u_patrol = 0.0
                if agent.species == "human":
                    # u_patrol: High reward for staying within settlement or crop areas
                    settlement_val = self.grid.sample(LAYER_SETTLEMENT, agent.x, agent.y)
                    crop_val = self.grid.sample(LAYER_CROP, agent.x, agent.y)
                    if settlement_val > 0.1 or crop_val > 0.1:
                        u_patrol = 15.0  # Strong reward for being in patrol zone
                    else:
                        u_patrol = -5.0  # Penalize wandering into deep forest

                for other in self._nearby_agents(agent, 72.0):
                    if not getattr(other, "alive", True) or other.id == agent.id: continue
                    dist = math.hypot(agent.x - other.x, agent.y - other.y)

                    if agent.species in ["leopard", "tiger", "lion"]:
                        if other.species == "elephant" and dist < 15.0:
                            u_flee = max(u_flee, 3.5); threat_dx += agent.x - other.x; threat_dy += agent.y - other.y
                        elif other.species == "human" and dist < 30.0:
                            friends = sum(1 for h in self.agents if h.species == "human" and getattr(h, "alive", True) and h.id != other.id and math.hypot(other.x - h.x, other.y - h.y) < 15.0)
                            if friends <= 2: 
                                u_hunt  = max(u_hunt, 40.0); hunt_dx = other.x - agent.x; hunt_dy = other.y - agent.y

                    if agent.species == "elephant":
                        if other.species == "human" and dist < 15.0:
                            human_invading = (self.grid.sample(LAYER_CROP, other.x, other.y) > 0.05 or self.grid.sample(LAYER_SETTLEMENT, other.x, other.y) > 0.05)
                            if agent.thirst > 70.0 or agent.energy < 30.0 or human_invading:
                                u_defend = max(u_defend, 20.0); hunt_dx = other.x - agent.x; hunt_dy = other.y - agent.y
                        
                        if other.species == "elephant" and getattr(agent, "herd_id", -1) != getattr(other, "herd_id", -2):
                            if getattr(other, "is_leader", False) and dist < 60.0 and self.tick > 600:
                                u_defend = max(u_defend, 40.0); hunt_dx = other.x - agent.x; hunt_dy = other.y - agent.y

                    if agent.species == "human" and other.species in ["leopard", "elephant", "sloth_bear", "tiger", "lion"]:
                        # u_defend: If wildlife enters nearby crop or settlement, move towards to intercept
                        other_in_crop = self.grid.sample(LAYER_CROP, other.x, other.y) > 0.05
                        other_in_settlement = self.grid.sample(LAYER_SETTLEMENT, other.x, other.y) > 0.05
                        if (other_in_crop or other_in_settlement) and dist < 25.0:
                            u_defend = max(u_defend, 35.0)
                            hunt_dx = other.x - agent.x
                            hunt_dy = other.y - agent.y

                        # u_flee: If predator gets within 1 cell and no other humans nearby, prioritize fleeing
                        if dist < 1.0:
                            nearby_humans = sum(1 for h in self.agents if h.species == "human" and getattr(h, "alive", True) and h.id != agent.id and math.hypot(agent.x - h.x, agent.y - h.y) < 10.0)
                            if nearby_humans <= 1:
                                u_flee = max(u_flee, 50.0)
                                threat_dx += agent.x - other.x
                                threat_dy += agent.y - other.y
                        elif dist < 20.0:
                            u_flee = max(u_flee, 40.0); threat_dx += agent.x - other.x; threat_dy += agent.y - other.y

                    # COMBAT EXECUTION 
                    if dist < 8.0 and getattr(agent, "attack_cooldown", 0) <= 0:
                        attack_landed = False
                        
                        if agent.species in ["leopard", "tiger", "lion"] and other.species == "human":
                            friends = sum(1 for h in self.agents if h.species == "human" and getattr(h, "alive", True) and h.id != other.id and math.hypot(other.x - h.x, other.y - h.y) < 15.0)
                            if friends <= 2 and self.rng.random() < 0.40: attack_landed = True
                            
                        elif agent.species == "elephant":
                            if other.species == "human":
                                human_invading = (self.grid.sample(LAYER_CROP, other.x, other.y) > 0.05 or self.grid.sample(LAYER_SETTLEMENT, other.x, other.y) > 0.05)
                                if (agent.thirst > 70.0 or agent.energy < 30.0 or human_invading) and self.rng.random() < 0.2: 
                                    attack_landed = True
                            elif other.species == "elephant":
                                if self.tick > 600 and getattr(agent, "herd_id", 1) != getattr(other, "herd_id", 2):
                                    if getattr(other, "is_leader", False):
                                        if self.rng.random() < 0.4: attack_landed = True
                                    else:
                                        if self.rng.random() < 0.05: attack_landed = True
                            
                        elif (agent.species == "human" and getattr(agent, "mode", "") == "DEFEND"):
                            if other.species in ["elephant", "leopard"]:
                                agent.energy -= 2.0
                                attack_landed = False 

                        if attack_landed:
                            if other.species == "elephant" and not getattr(other, "is_leader", False):
                                attack_landed = False
                            else:
                                other.alive = False
                                agent.attack_cooldown = 100
                                agent.energy = min(100.0, agent.energy + 50.0)
                                label = "Elephant Leader Assassinated!" if agent.species == "elephant" and other.species == "elephant" else f"{other.species.capitalize()} killed by {agent.species.capitalize()}"
                                self.mortality_events.append({"id": f"kill_{self.tick}_{other.id}", "tick": self.tick, "type": "mortality", "cx": other.x, "cy": other.y, "radius": 15, "label": label})

                best = u_wander
                agent.mode = "WANDER"
                if u_water   > best: best = u_water;   agent.mode = "WATER"
                if u_food    > best: best = u_food;    agent.mode = "FOOD"
                if u_hunt    > best: best = u_hunt;    agent.mode = "HUNT"
                if u_flee    > best: best = u_flee;    agent.mode = "FLEE"
                if u_defend  > best: best = u_defend;  agent.mode = "DEFEND"
                if u_patrol  > best: best = u_patrol;  agent.mode = "PATROL"
                if u_return  > best: best = u_return;  agent.mode = "RETURN_HOME"
                if u_migrate > best:                   agent.mode = "MIGRATE"

                if (agent.species == "elephant" and not getattr(agent, "is_leader", False) and agent.mode == "WANDER"):
                    agent.mode = "FOLLOW"

                dx = dy = 0.0
                target_x = target_y = None
                is_rl = (agent.species in ["elephant", "leopard", "sloth_bear"] and getattr(agent, "trainer", None) is not None)

                # --- ANIMAL BRIDGE SPATIAL MEMORY LOGIC ---
                if (not self._graph_enabled()) and agent.species == "elephant" and agent.mode in ["FOOD", "WATER", "MIGRATE", "RETURN_HOME"]:
                    if agent.mode == "MIGRATE": target_y = agent.migration_target[1]
                    elif agent.mode == "RETURN_HOME": target_y = agent.home_y
                    else: target_y = min(self.crop_nodes, key=lambda c: math.hypot(agent.x - c["cx"], agent.y - c["cy"]))["cy"] if self.crop_nodes else agent.y
                    
                    # If target is across the fence, override coordinates to the bridge
                    if (agent.y < self.fence_y and target_y > self.fence_y) or (agent.y > self.fence_y and target_y < self.fence_y):
                        target_x, target_y = self.bridge_center[0], self.bridge_center[1]

                if is_rl and agent.mode in ["FOOD", "WANDER", "RETURN_HOME", "MIGRATE", "WATER"]:
                    if agent.mode == "MIGRATE":
                        target_x, target_y = agent.migration_target
                    elif agent.mode == "RETURN_HOME":
                        target_x, target_y = agent.home_x, agent.home_y
                        agent.return_home_timer -= 1
                    elif agent.mode == "WATER":
                        nearest_water = self._nearest_water_target(agent.x, agent.y)
                        if nearest_water:
                            target_x, target_y = nearest_water["cx"], nearest_water["cy"]
                        else:
                            target_x, target_y = agent.x, agent.y
                    else:
                        active_crops = [c for c in self.crop_nodes]
                        if active_crops:
                            best = min(active_crops, key=lambda c: math.hypot(agent.x - c["cx"], agent.y - c["cy"]))
                            target_x, target_y = best["cx"], best["cy"]
                        else:
                            agent.heading += self.rng.uniform(-0.5, 0.5)
                            target_x, target_y = agent.x + math.cos(agent.heading) * 10, agent.y + math.sin(agent.heading) * 10

                    # Apply Bridge override again if needed for RL 
                    if (not self._graph_enabled()) and agent.species == "elephant" and ((agent.y < self.fence_y and target_y > self.fence_y) or (agent.y > self.fence_y and target_y < self.fence_y)):
                        target_x, target_y = self.bridge_center[0], self.bridge_center[1]

                    need_val = float(100.0 - agent.energy)
                    torch = _torch_module()
                    state = torch.tensor([need_val, float(target_x - agent.x), float(target_y - agent.y), 100.0, 100.0], dtype=torch.float32)
                    act_vec, log_prob, val = agent.trainer.choose_action(state)
                    
                    heading = float(act_vec[0])
                    velocity_scale = max(0.0, min(1.0, float(act_vec[1])))
                    spd = (1.5 if agent.species == "elephant" else 2.5) * velocity_scale
                    
                    g_mag = math.hypot(target_x - agent.x, target_y - agent.y)
                    if g_mag > 0:
                        dx += ((target_x - agent.x) / g_mag) * (1.0 if agent.species == "elephant" else 2.0)
                        dy += ((target_y - agent.y) / g_mag) * (1.0 if agent.species == "elephant" else 2.0)

                    dx += math.cos(heading) * spd
                    dy += math.sin(heading) * spd

                    if getattr(agent, "last_state", None) is not None:
                        rew = float(agent.rewards.get_reward(need_val, 100.0, bool(math.hypot(target_x - agent.x, target_y - agent.y) < 2.0))[0])
                        agent.trainer.remember(agent.last_state, agent.last_action, agent.last_log_prob, rew, agent.last_value, False)
                        if self.tick % 64 == 0:
                            agent.trainer.learn()
                            
                    agent.last_state = state
                    agent.last_action = act_vec
                    agent.last_log_prob = log_prob
                    agent.last_value = val

                else:
                    if agent.mode == "FLEE": dx, dy = threat_dx, threat_dy
                    elif agent.mode in ["HUNT", "DEFEND"]: dx, dy = hunt_dx, hunt_dy
                    elif agent.mode == "WATER":
                        nearest_water = self._nearest_water_target(agent.x, agent.y)
                        if nearest_water:
                            target_x, target_y = nearest_water["cx"], nearest_water["cy"]
                            dx, dy = target_x - agent.x, target_y - agent.y
                            mag = math.hypot(dx, dy)
                            if mag > 0: dx, dy = (dx / mag) * 2.0, (dy / mag) * 2.0
                    elif agent.mode == "MIGRATE":
                        target_x, target_y = agent.migration_target
                        dx, dy = (agent.migration_target[0] - agent.x, agent.migration_target[1] - agent.y)
                        mag = math.hypot(dx, dy)
                        if mag > 0: dx, dy = (dx / mag) * 2.5, (dy / mag) * 2.5
                    elif agent.mode == "RETURN_HOME":
                        target_x, target_y = agent.home_x, agent.home_y
                        dx, dy = agent.home_x - agent.x, agent.home_y - agent.y
                        agent.return_home_timer -= 1
                    elif agent.mode == "PATROL":
                        # Patrol mode: move towards home/settlement center to stay in protected zone
                        target_x, target_y = agent.home_x, agent.home_y
                        dx, dy = agent.home_x - agent.x, agent.home_y - agent.y
                        mag = math.hypot(dx, dy)
                        if mag > 0: dx, dy = (dx / mag) * 1.5, (dy / mag) * 1.5
                    elif agent.mode == "FOLLOW":
                        leader = next((a for a in self.agents if a.species == "elephant" and getattr(a, "is_leader", False) and getattr(a, "herd_id", None) == agent.herd_id and getattr(a, "alive", True)), None)
                        if leader and math.hypot(leader.x - agent.x, leader.y - agent.y) > 4.0:
                            target_x, target_y = leader.x, leader.y
                            dx, dy = leader.x - agent.x, leader.y - agent.y
                        else:
                            agent.heading += self.rng.uniform(-0.4, 0.4)
                            dx, dy = math.cos(agent.heading) * 1.5, math.sin(agent.heading) * 1.5
                    else:
                        if agent.species == "human":
                            ghx, ghy = getattr(agent, "home_x", agent.x), getattr(agent, "home_y", agent.y)
                            agent.home_x, agent.home_y = ghx * 0.99 + agent.x * 0.01, ghy * 0.99 + agent.y * 0.01
                            if math.hypot(agent.x - ghx, agent.y - ghy) > 40.0:
                                dx += (ghx - agent.x) * 0.5; dy += (ghy - agent.y) * 0.5
                            else:
                                agent.heading += self.rng.uniform(-0.8, 0.8)
                                dx += math.cos(agent.heading) * 1.5; dy += math.sin(agent.heading) * 1.5
                        else:
                            agent.heading += self.rng.uniform(-0.8, 0.8)
                            dx += math.cos(agent.heading) * 1.5; dy += math.sin(agent.heading) * 1.5

                dx += self.rng.uniform(-0.2, 0.2); dy += self.rng.uniform(-0.2, 0.2)
                mag = math.hypot(dx, dy)
                if mag > 0:
                    spd = (0.75 if agent.species == "elephant" else 2.0 if agent.species == "leopard" else 1.5)
                    spd *= agent.eco_mods["speed_mult"]
                    if agent.mode in ["HUNT", "DEFEND"]: spd *= 3.5 if agent.species == "elephant" else 2.0
                    elif agent.mode in ["FLEE", "RETURN_HOME", "MIGRATE"]: spd *= 2.0
                    elif agent.mode == "FOLLOW": spd *= 1.5
                    agent.vx, agent.vy = (dx / mag) * spd, (dy / mag) * spd

                # Free continuous movement: do not constrain wildlife to the OSM path graph.
                new_x = agent.x + agent.vx
                new_y = agent.y + agent.vy

                # --- AGENTS LEAVING THE MAP ---
                # If they cross the boundary, they leave the simulation instead of bouncing
                if new_x <= 0 or new_x >= self.grid.W or new_y <= 0 or new_y >= self.grid.H:
                    agent.alive = False
                    self.mortality_events.append({
                        "id": f"left_{self.tick}_{agent.id}", "tick": self.tick, 
                        "type": "migration", "cx": agent.x, "cy": agent.y, 
                        "radius": 15, "label": f"{agent.species.capitalize()} migrated off-map"
                    })
                    continue

                agent.x = new_x
                agent.y = new_y

                # Human-Wildlife Conflict (HWC) Resolution
                if agent.species in ["elephant", "leopard"] and getattr(agent, "alive", True):
                    for other in self._nearby_agents(agent, 2.0):
                        if other.species == "human" and getattr(other, "alive", True):
                            dist = math.hypot(agent.x - other.x, agent.y - other.y)
                            if dist <= 2.0:
                                # Massive negative reward for wild animal (noise/firecrackers)
                                if getattr(agent, "rewards", None):
                                    agent.rewards.add_conflict_penalty(50.0)

                                # Push wild animal directly away from human
                                push_dx = agent.x - other.x
                                push_dy = agent.y - other.y
                                push_mag = math.hypot(push_dx, push_dy)
                                if push_mag > 0:
                                    push_strength = 3.0
                                    agent.vx = (push_dx / push_mag) * push_strength
                                    agent.vy = (push_dy / push_mag) * push_strength
                                    # Apply pushback immediately (respecting boundaries)
                                    agent.x = max(0.0, min(float(self.W) - 1.0, agent.x + agent.vx))
                                    agent.y = max(0.0, min(float(self.H) - 1.0, agent.y + agent.vy))

                                # Rare mortality trigger (2% chance) on direct 1-cell collision
                                if dist < 1.0 and self.rng.random() < 0.02:
                                    victim = self.rng.choice([agent, other])
                                    victim.alive = False
                                    label = f"{victim.species.capitalize()} killed in severe HWC incident"
                                    self.mortality_events.append({"id": f"hwc_kill_{self.tick}_{victim.id}", "tick": self.tick, "type": "mortality", "cx": victim.x, "cy": victim.y, "radius": 12, "label": label})

            # --- LOGGER OUTPUT ---
            if self.tick % 20 == 0:
                with open(self.log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    for agent in self.agents:
                        if not getattr(agent, "alive", True): continue
                        
                        condition = "Normal"
                        if agent.thirst > 70.0: condition = "Severe Thirst"
                        elif agent.energy < 30.0: condition = "Starving"
                        elif agent.mode == "DEFEND": condition = "Stressed/Defending"
                        elif agent.mode == "HUNT": condition = "Aggressive/Hunting"
                        elif agent.mode == "MIGRATE": condition = "Migrating"
                        
                        writer.writerow([
                            self.tick, 
                            agent.id, 
                            agent.species, 
                            round(getattr(agent, "weight", 0.0), 2), 
                            round(agent.energy, 2), 
                            round(100.0 - agent.thirst, 2), 
                            agent.mode, 
                            condition
                        ])

            if self.tick % 10 == 0:
                raw_hotspots = compute_conflicts(self.zones, self.agents, self.herd_map)
                predictions  = self._get_predictive_hotspots()
                combined     = raw_hotspots + predictions

                parsed = []
                for h in combined:
                    if isinstance(h, dict): parsed.append(h)
                    else:
                        parsed.append({
                            "id":     getattr(h, "id",     "unknown"), "type":   getattr(h, "type",   "conflict"),
                            "cx":     getattr(h, "cx",     0), "cy":     getattr(h, "cy",     0),
                            "radius": getattr(h, "radius", 10), "score":  getattr(h, "score",  0),
                            "label":  getattr(h, "label",  "Conflict Zone")
                        })
                self.hotspots = parsed

        except Exception:
            print("\n[CRITICAL ERROR IN SIMULATION ENGINE]")
            traceback.print_exc()

    def get_state(self):
        return {
            "tick":      self.tick,
            "is_night":  self.is_night,
            "agents":    [a.to_dict() for a in self.agents],
            "world":     self.world.export_state(),
            "analytics": self.analytics_data,
            "hotspots":  self.hotspots,
            "water_basins": [{"cx": b["cx"], "cy": b["cy"], "lat": b.get("lat"), "lon": b.get("lon"), "label": b.get("label", "Water"), "kind": b.get("kind", "water"), "intensity": round(b["intensity"], 3), "active": b["active"], "radius": round(WATER_BASIN_MAX_RADIUS * b["intensity"], 2)} for b in getattr(self, "water_nodes", [])],
            "crop_zones": [{"cx": c["cx"], "cy": c["cy"], "intensity": round(c["intensity"], 3), "radius": round(WATER_BASIN_MAX_RADIUS * c["intensity"], 2)} for c in getattr(self, "crop_nodes", [])],
            "region": self.region_metadata,
            "graph": self.geo_graph.to_frontend(max_edges=2500) if self.geo_graph else None,
            "species_counts": dict(self.species_counts),
        }