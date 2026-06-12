import sys
import os
import asyncio
import random
from typing import Dict, List

# ---------------------------------------------------------
# PATH FIX: Force Python to see the root directory
# ---------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from simcore.rollout.multi_sim import MultiAgentSim
except ModuleNotFoundError:
    from simcore.rollout.multi_sim import MultiAgentSim

_SAFE_FALLBACK_COUNTS = {"elephant": 10, "leopard": 5, "sloth_bear": 5}


class SimManager:
    def __init__(self):
        self.sim: MultiAgentSim | None = None
        self.fps: int = 3
        self._running_task: asyncio.Task | None = None

        print("[SimManager] Initialized")
        # NOTE: The ML CSV Logger has been moved directly into multi_sim.py 
        # to capture advanced metrics like Dist_to_Water and Dist_to_Threat!

    # ==========================================
    # Reset Simulation
    # ==========================================
    async def reset_sim(self, payload: Dict | None = None):
        payload = payload or {}

        topology = payload.get("topology", "bannerghatta_osm")
        seed = payload.get("seed", 1)
        self.fps = payload.get("fps", 3 if topology == "bannerghatta_osm" else 20)
        
        # Never inject a hardcoded baseline from the backend. The UI controls population sizing.
        species_counts = payload.get("species_counts") or dict(_SAFE_FALLBACK_COUNTS)

        print(f"[SimManager] Resetting sim: topo={topology}, seed={seed}, counts={species_counts}")

        random.seed(seed)

        if self._running_task:
            self._running_task.cancel()
            self._running_task = None

        real_map = topology == "bannerghatta_osm"
        topology_config = {
            "name": topology,
            "width": 240 if real_map else 120,
            "height": 720 if real_map else 120
        }

        self.sim = MultiAgentSim(
            topology_config=topology_config,
            seed=seed,
            fps=self.fps,
            species_counts=species_counts 
        )

        print("[SimManager] Simulation reset complete")

        self._running_task = asyncio.create_task(self._run_loop())
        print("[SimManager] Started simulation loop")

    # ==========================================
    # Simulation Loop & Data Extraction
    # ==========================================
    async def _run_loop(self):
        while True:
            await asyncio.sleep(1.0 / self.fps)

            if self.sim:
                self.sim.step()

    # ==========================================
    # State Packet (for WebSocket)
    # ==========================================
    def get_state_packet(self) -> Dict:
        if not self.sim:
            return {"type": "state", "t": 0, "agents": [], "hotspots": []}

        return {
            "type": "state",
            "t": self.sim.tick, 
            "season": getattr(self.sim, "season", "wet"),
            "agents": [
                {
                    "id": a.id,
                    "species": a.species,
                    "x": a.x,
                    "y": a.y,
                    "energy": a.energy,
                    "thirst": a.thirst,
                    "mode": a.mode,
                    "weight": getattr(a, "weight", None),
                    "alive": getattr(a, "alive", True),
                    "herd_id": getattr(a, "herd_id", None),
                    "is_leader": getattr(a, "is_leader", False),
                    "target": getattr(a, "target", None),
                    "graph_node": getattr(a, "graph_node", None),
                }
                for a in self.sim.agents
            ],
            "hotspots": getattr(self.sim, "hotspots", []),
            "event_counts": getattr(self.sim, "event_counts", {}),
            # NEW: Analytics dashboard data
            "analytics": getattr(self.sim, "analytics_data", None),
            "region": getattr(self.sim, "region_metadata", None),
            "water_basins": [{"cx": b["cx"], "cy": b["cy"], "lat": b.get("lat"), "lon": b.get("lon"), "label": b.get("label", "Water"), "kind": b.get("kind", "water"), "intensity": round(b["intensity"], 3), "active": b["active"]} for b in getattr(self.sim, "water_nodes", [])],
            "crop_zones": [{"cx": c["cx"], "cy": c["cy"], "intensity": round(c["intensity"], 3)} for c in getattr(self.sim, "crop_nodes", [])],
            "species_counts": dict(getattr(self.sim, "species_counts", {})),
            # NEW: Dynamically shifting terraformed zones
            "zones": [{"cx": z.centroid_x, "cy": z.centroid_y, "radius": z.radius, "type": z.type} for z in self.sim.zones] if hasattr(self.sim, "zones") else []
        }

    # ==========================================
    # UI Config & Topologies
    # ==========================================
    def get_config(self) -> Dict:
        return {
            "defaults": {
                "topology": "bannerghatta_osm",
                "seed": 1,
                "fps": 3,
                "species_counts": {
                    **_SAFE_FALLBACK_COUNTS,
                },
            }
        }

    def get_topologies(self) -> List[Dict]:
        return [
            {"id": "bannerghatta_osm", "name": "Bannerghatta OSM Graph", "desc": "OSMnx roads, forest tracks, footways, and Suvarnamukhi water resources. Population size comes from UI species_counts."},
            {"id": "bannerghatta", "name": "Bannerghatta Proxy", "desc": "Bengaluru-edge dry forest, farms, villages, tanks, and corridor pressure."},
            {"id": "edge_farmland", "name": "Edge Farmland", "desc": "Forest meets cropland with human settlements."},
            {"id": "river_basin", "name": "River Basin", "desc": "Abundant central water, high vegetation, stable ecosystem."},
            {"id": "dense_forest", "name": "Dense Forest Core", "desc": "High tree cover, low human conflict, natural foraging."},
            {"id": "fragmented", "name": "Fragmented Settlement", "desc": "Scattered villages and crops break up the forest corridors."},
            {"id": "high_slope", "name": "Hilly / High Slope", "desc": "Steep terrain forces movement and resources into narrow valleys."},
            {"id": "drought", "name": "Drought Landscape", "desc": "Scarce water and dry crops drive extreme survival conflict."}
        ]

sim_manager = SimManager()
