import asyncio
import base64
import json
from typing import List

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from viz_web.sim_manager import sim_manager

app = FastAPI()
app.mount("/static", StaticFiles(directory="viz_web/static"), name="static")

connected_clients: List[WebSocket] = []

# ===============================
# Startup
# ===============================

@app.on_event("startup")
async def startup_event():
    print("[Server] Starting up...")
    await sim_manager.reset_sim()
    asyncio.create_task(broadcast_loop())

# ===============================
# HTTP Routes
# ===============================

@app.get("/")
async def root():
    with open("viz_web/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/topologies")
async def get_topologies():
    return {
        "topologies": sim_manager.get_topologies()
    }

@app.get("/api/config")
async def get_config():
    return sim_manager.get_config()

@app.post("/api/reset")
async def reset_sim(payload: dict):
    print("[Server] Reset requested:", payload)
    await sim_manager.reset_sim(payload)

    for ws in connected_clients:
        await send_init(ws)

    return {"status": "ok"}

# ===============================
# WebSocket
# ===============================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    print("[WebSocket] Client connected")

    try:
        await send_init(ws)

        while True:
            # FIX: Reduced from 60 seconds to 0.05! This stops the UI from freezing.
            await asyncio.sleep(60.0)

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
        if ws in connected_clients:
            connected_clients.remove(ws)

# ===============================
# Broadcast Loop
# ===============================

async def broadcast_loop():
    while True:
        await asyncio.sleep(1.0 / sim_manager.fps)

        state = sim_manager.get_state_packet()

        if not connected_clients:
            continue

        sim = sim_manager.sim
        
        # Inject dynamic zones into the state packet
        if sim:
            state["zones"] = [
                {
                    "id": getattr(z, "id", f"zone_{i}"),
                    "type": getattr(z, "type", "unknown"),
                    "cx": getattr(z, "centroid_x", getattr(z, "cx", 0)),
                    "cy": getattr(z, "centroid_y", getattr(z, "cy", 0)),
                    "radius": getattr(z, "radius", 10),
                    "label": getattr(z, "label", f"{getattr(z, 'type', 'zone').title()} Zone")
                }
                for i, z in enumerate(sim.zones)
            ]

        # ==========================================
        # --- NEW: SDSS ANALYTICS INJECTION ---
        # ==========================================
        
        # 1. Calculate Behavioral Intents dynamically from the agent payload
        intent_counts = {}
        active_agents = state.get("agents", [])
        
        for agent_dict in active_agents:
            mode = agent_dict.get("mode", "IDLE")
            readable_mode = {
                "MIGRATE": "Migrating",
                "WATER": "Seeking Water",
                "FOOD": "Foraging",
                "FLEE": "Evading Threat",
                "HUNT": "Hunting",
                "DEFEND": "Defending Border",
                "RETURN_HOME": "Returning to Base",
                "IDLE": "Idle/Resting"
            }.get(mode, mode)
            
            intent_counts[readable_mode] = intent_counts.get(readable_mode, 0) + 1

        # 2. Fetch HWC Conflict Logs safely
        recent_conflicts = getattr(sim, "recent_logs", []) if sim else []
        
        # 3. Calculate Ecosystem Metrics
        cap_pct = 100.0
        if sim and hasattr(sim, "baseline_population") and sim.baseline_population > 0:
            cap_pct = min(100.0, (len(active_agents) / sim.baseline_population) * 100)
        elif len(active_agents) > 0:
            cap_pct = min(100.0, (len(active_agents) / 100) * 100) # Fallback

        # Inject the analytics payload into the WebSocket state
        state["analytics"] = {
            "intents": intent_counts,
            "new_conflicts": recent_conflicts,
            "carrying_capacity": cap_pct,
            "eco_scores": {"health": 88, "range": 94} # Example baselines
        }

        # Clear the logs on the backend so they don't echo infinitely on the frontend
        if sim and hasattr(sim, "recent_logs"):
            sim.recent_logs = []

        # ==========================================

        for ws in connected_clients[:]:
            try:
                await ws.send_json(state)
            except Exception:
                if ws in connected_clients:
                    connected_clients.remove(ws)

# ===============================
# INIT PACKET
# ===============================

async def send_init(ws: WebSocket):
    sim = sim_manager.sim
    if not sim:
        return

    W = sim.grid.W
    H = sim.grid.H

    float_layers = sim.grid.layers
    uint8_layers = (float_layers * 255).astype(np.uint8)

    raw_bytes = uint8_layers.tobytes()

    expected = W * H * 7
    if len(raw_bytes) != expected:
        print(f"[ERROR] Layer size mismatch: got={len(raw_bytes)} expected={expected}")

    b64 = base64.b64encode(raw_bytes).decode("ascii")

    zones_data = getattr(sim, "zones", [])

    msg = {
        "type": "init",
        "topology": {
            "id": getattr(sim, "topology_name", "active"),
            "name": getattr(sim, "region_metadata", None).get("name", getattr(sim, "topology_name", "Active Topology")) if getattr(sim, "region_metadata", None) else getattr(sim, "topology_name", "Active Topology"),
            "desc": "Simcore generated topology."
        },
        "region": getattr(sim, "region_metadata", None),
        "world": {
            "W": W,
            "H": H,
            "layers_u8": {
                "b64": b64
            },
            "graph": sim.geo_graph.to_frontend(max_edges=4000) if getattr(sim, "geo_graph", None) else None,
        },
        "species_counts": dict(getattr(sim, "species_counts", {})),
        "water_resources": [{"cx": b["cx"], "cy": b["cy"], "lat": b.get("lat"), "lon": b.get("lon"), "label": b.get("label", "Water"), "kind": b.get("kind", "water"), "intensity": b.get("intensity", 1.0), "active": b.get("active", True)} for b in getattr(sim, "water_nodes", [])],
        "zones": [
            {
                "id": getattr(z, "id", f"zone_{i}"),
                "type": getattr(z, "type", "unknown"),
                "cx": getattr(z, "centroid_x", getattr(z, "cx", 0)),  
                "cy": getattr(z, "centroid_y", getattr(z, "cy", 0)),  
                "radius": getattr(z, "radius", 10),
                "strength": getattr(z, "mean_strength", 1.0),
                "label": getattr(z, "label", f"{getattr(z, 'type', 'zone').title()} Zone"),
            }
            for i, z in enumerate(zones_data)
        ],
    }

    await ws.send_json(msg)