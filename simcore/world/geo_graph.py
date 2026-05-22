from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Any

from simcore.data.bannerghatta_ecology import (
    BANNERGHATTA_BBOX,
    BANNERGHATTA_DECLARED_AREA_SQ_KM,
)
from simcore.data.bannerghatta_osm_features import (
    BANNERGHATTA_RELATION_ID,
    RECHARGE_PIT_COORDS,
    SAFARI_ZONE_COORDS,
    SUVARNA_MUKHI_ANCHOR_COORDS,
    SUVARNA_MUKHI_OSM_WAY_IDS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_CACHE = PROJECT_ROOT / "data" / "bannerghatta_graph.json"
OVERPASS_RAW_CACHE = PROJECT_ROOT / "data" / "bannerghatta_overpass_raw.json"

GRAPH_HIGHWAYS = {
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "service",
    "track",
    "path",
    "footway",
    "bridleway",
    "steps",
    "living_street",
}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371009.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def latlon_to_xy(lat: float, lon: float, bbox: dict[str, float], width: int, height: int) -> tuple[float, float]:
    x = (lon - bbox["west"]) / max(1e-12, bbox["east"] - bbox["west"]) * (width - 1)
    y = (bbox["north"] - lat) / max(1e-12, bbox["north"] - bbox["south"]) * (height - 1)
    return float(x), float(y)


def point_in_poly(x: float, y: float, pts: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(pts) - 1
    for i, (xi, yi) in enumerate(pts):
        xj, yj = pts[j]
        if (yi > y) != (yj > y):
            x_at_y = (xj - xi) * (y - yi) / max(1e-12, yj - yi) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def relation_rings(relation: dict[str, Any], ways: dict[int, dict[str, Any]]) -> list[list[int]]:
    segments: list[list[int]] = []
    for member in relation.get("members", []):
        if member.get("type") != "way" or member.get("role") not in {"outer", ""}:
            continue
        way = ways.get(int(member["ref"]))
        if way and len(way.get("nodes", [])) >= 2:
            segments.append(list(way["nodes"]))

    rings: list[list[int]] = []
    while segments:
        ring = segments.pop(0)
        changed = True
        while changed and ring[0] != ring[-1]:
            changed = False
            for i, seg in enumerate(segments):
                if ring[-1] == seg[0]:
                    ring.extend(seg[1:])
                elif ring[-1] == seg[-1]:
                    ring.extend(reversed(seg[:-1]))
                elif ring[0] == seg[-1]:
                    ring = seg[:-1] + ring
                elif ring[0] == seg[0]:
                    ring = list(reversed(seg[1:])) + ring
                else:
                    continue
                segments.pop(i)
                changed = True
                break
        if len(ring) >= 4 and ring[0] == ring[-1]:
            rings.append(ring)
    return rings


@dataclass
class GeoGraph:
    width: int
    height: int
    bbox: dict[str, float]
    nodes: dict[str, dict[str, float]]
    edges: list[dict[str, Any]]
    water_polylines: list[dict[str, Any]] = field(default_factory=list)
    water_points: list[dict[str, Any]] = field(default_factory=list)
    recharge_pits: list[dict[str, Any]] = field(default_factory=list)
    boundary_rings: list[list[tuple[float, float]]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.adj: dict[str, list[tuple[str, float]]] = {nid: [] for nid in self.nodes}
        for e in self.edges:
            u, v = str(e["u"]), str(e["v"])
            if u not in self.nodes or v not in self.nodes:
                continue
            length = float(e.get("length", 1.0))
            self.adj.setdefault(u, []).append((v, length))
            self.adj.setdefault(v, []).append((u, length))
        self._node_items = list(self.nodes.items())
        self._node_ids = list(self.nodes)
        self._bucket_size = max(4.0, min(float(self.width), float(self.height)) / 32.0)
        self._node_buckets: dict[tuple[int, int], list[tuple[str, dict[str, float]]]] = {}
        for nid, node in self._node_items:
            key = (int(float(node["x"]) // self._bucket_size), int(float(node["y"]) // self._bucket_size))
            self._node_buckets.setdefault(key, []).append((nid, node))

    @classmethod
    def empty(cls, width: int, height: int) -> "GeoGraph":
        return cls(width=width, height=height, bbox=dict(BANNERGHATTA_BBOX), nodes={}, edges=[])

    @classmethod
    def from_json(cls, payload: dict[str, Any], width: int, height: int) -> "GeoGraph":
        bbox = dict(payload.get("bbox") or BANNERGHATTA_BBOX)

        def scale_node(node: dict[str, Any]) -> dict[str, float]:
            lat, lon = float(node["lat"]), float(node["lon"])
            x, y = latlon_to_xy(lat, lon, bbox, width, height)
            return {"lat": lat, "lon": lon, "x": x, "y": y}

        nodes = {str(k): scale_node(v) for k, v in payload.get("nodes", {}).items()}
        water_polylines = _scale_polylines(payload.get("water_polylines", []), bbox, width, height)
        water_points = _scale_points(payload.get("water_points", []), bbox, width, height)
        recharge_pits = _scale_points(payload.get("recharge_pits", []), bbox, width, height)
        boundary = [
            [latlon_to_xy(float(p["lat"]), float(p["lon"]), bbox, width, height) for p in ring]
            for ring in payload.get("boundary_rings", [])
        ]
        return cls(
            width=width,
            height=height,
            bbox=bbox,
            nodes=nodes,
            edges=list(payload.get("edges", [])),
            water_polylines=water_polylines,
            water_points=water_points,
            recharge_pits=recharge_pits,
            boundary_rings=boundary,
            metadata=dict(payload.get("metadata", {})),
        )

    def to_frontend(self, max_edges: int | None = None) -> dict[str, Any]:
        edges = self.edges if max_edges is None else self.edges[:max_edges]
        return {
            "bbox": self.bbox,
            "declared_area_sq_km": BANNERGHATTA_DECLARED_AREA_SQ_KM,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": {
                nid: {"x": round(n["x"], 3), "y": round(n["y"], 3), "lat": n["lat"], "lon": n["lon"]}
                for nid, n in self.nodes.items()
            },
            "edges": [{"u": str(e["u"]), "v": str(e["v"]), "kind": e.get("kind", "path")} for e in edges],
            "water_polylines": self.water_polylines,
            "water_points": self.water_points,
            "recharge_pits": self.recharge_pits,
            "safari_zones": self._scaled_safari_zones(),
        }

    def coord_to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        return latlon_to_xy(lat, lon, self.bbox, self.width, self.height)

    def nearest_node_xy(self, x: float, y: float) -> str | None:
        if self._node_buckets:
            bx = int(float(x) // self._bucket_size)
            by = int(float(y) // self._bucket_size)
            best_id = None
            best_d2 = float("inf")
            max_ring = int(max(self.width, self.height) // self._bucket_size) + 2

            for ring in range(max_ring + 1):
                for gy in range(by - ring, by + ring + 1):
                    for gx in range(bx - ring, bx + ring + 1):
                        if ring and bx - ring < gx < bx + ring and by - ring < gy < by + ring:
                            continue
                        for nid, node in self._node_buckets.get((gx, gy), []):
                            d2 = (node["x"] - x) ** 2 + (node["y"] - y) ** 2
                            if d2 < best_d2:
                                best_d2 = d2
                                best_id = nid
                if best_id is not None and math.sqrt(best_d2) <= max(1.0, ring * self._bucket_size):
                    return best_id
            if best_id is not None:
                return best_id

        best_id = None
        best_d2 = float("inf")
        for nid, node in self._node_items:
            d2 = (node["x"] - x) ** 2 + (node["y"] - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_id = nid
        return best_id

    def snap_xy(
        self,
        x: float,
        y: float,
        map_width: int | None = None,
        map_height: int | None = None,
    ) -> tuple[float, float, str | None]:
        """
        Snap a point to the nearest valid graph node.

        - When `map_width/map_height` are omitted, `x/y` are assumed to already be
          in this graph's pixel space.
        - When provided, `x/y` are treated as coordinates in an arbitrary
          viewport/grid size and are rescaled into the graph's pixel space
          before snapping, then rescaled back.
        """
        if map_width is None or map_height is None:
            nid = self.nearest_node_xy(x, y)
            if nid is None:
                return x, y, None
            node = self.nodes[nid]
            return node["x"], node["y"], nid

        mw = max(2, int(map_width))
        mh = max(2, int(map_height))
        gw = max(2, int(self.width))
        gh = max(2, int(self.height))
        xg = float(x) / float(mw - 1) * float(gw - 1)
        yg = float(y) / float(mh - 1) * float(gh - 1)
        sx, sy, nid = self.snap_xy(xg, yg)
        xo = float(sx) / float(gw - 1) * float(mw - 1)
        yo = float(sy) / float(gh - 1) * float(mh - 1)
        return xo, yo, nid

    def sample_node(self, rng: random.Random, prefer: str | None = None) -> tuple[float, float, str | None]:
        if not self.nodes:
            return rng.uniform(1, self.width - 2), rng.uniform(1, self.height - 2), None
        candidates = self._node_ids
        if prefer:
            tagged = [nid for nid in candidates if self.nodes[nid].get("kind") == prefer]
            if tagged:
                candidates = tagged
        nid = rng.choice(candidates)
        node = self.nodes[nid]
        return node["x"], node["y"], nid

    def move_agent_on_graph(
        self,
        agent: Any,
        desired_dx: float,
        desired_dy: float,
        speed: float,
        rng: random.Random,
        target_xy: tuple[float, float] | None = None,
    ) -> bool:
        if not self.nodes:
            return False

        current = getattr(agent, "graph_node", None)
        if current not in self.nodes:
            agent.x, agent.y, current = self.snap_xy(float(agent.x), float(agent.y))
            agent.graph_node = current
            agent.graph_next_node = None
        if current is None:
            return False

        next_node = getattr(agent, "graph_next_node", None)
        if next_node not in self.nodes:
            next_node = self._choose_next_node(current, desired_dx, desired_dy, target_xy, rng)
            agent.graph_next_node = next_node

        if next_node is None:
            agent.vx = agent.vy = 0.0
            return True

        dest = self.nodes[next_node]
        dx = dest["x"] - float(agent.x)
        dy = dest["y"] - float(agent.y)
        dist = math.hypot(dx, dy)
        if dist <= max(0.001, speed):
            old_x, old_y = float(agent.x), float(agent.y)
            agent.x, agent.y = dest["x"], dest["y"]
            agent.graph_node = next_node
            agent.graph_next_node = None
            agent.vx, agent.vy = agent.x - old_x, agent.y - old_y
            return True

        old_x, old_y = float(agent.x), float(agent.y)
        agent.x += dx / dist * speed
        agent.y += dy / dist * speed
        agent.vx, agent.vy = agent.x - old_x, agent.y - old_y
        return True

    def paint_layers(self, layers: Any, layer_water: int, layer_obstacle: int, layer_cover: int) -> None:
        h, w = int(layers.shape[0]), int(layers.shape[1])
        for ring in self.boundary_rings:
            _fill_polygon(layers[:, :, layer_cover], ring, 0.55)

        for e in self.edges:
            u = self.nodes.get(str(e["u"]))
            v = self.nodes.get(str(e["v"]))
            if not u or not v:
                continue
            _draw_line(layers[:, :, layer_obstacle], [(u["x"], u["y"]), (v["x"], v["y"])], 0.65, 0.45)

        for line in self.water_polylines:
            pts = [(float(p["x"]), float(p["y"])) for p in line.get("points", [])]
            _draw_line(layers[:, :, layer_water], pts, 1.4, 1.0)
            _draw_line(layers[:, :, layer_cover], pts, 2.0, 0.0)

        for p in self.water_points + self.recharge_pits:
            _paint_circle(layers[:, :, layer_water], float(p["x"]), float(p["y"]), 3.0, 0.9)
            _paint_circle(layers[:, :, layer_cover], float(p["x"]), float(p["y"]), 4.0, 0.0)

        layers[:, :, layer_obstacle] = layers[:, :, layer_obstacle].clip(0.0, 1.0)
        layers[:, :, layer_water] = layers[:, :, layer_water].clip(0.0, 1.0)
        layers[:, :, layer_cover] = layers[:, :, layer_cover].clip(0.0, 1.0)

    def _choose_next_node(
        self,
        current: str,
        desired_dx: float,
        desired_dy: float,
        target_xy: tuple[float, float] | None,
        rng: random.Random,
    ) -> str | None:
        neighbors = self.adj.get(current, [])
        if not neighbors:
            return None
        if len(neighbors) == 1:
            return neighbors[0][0]

        cur = self.nodes[current]
        mag = math.hypot(desired_dx, desired_dy)
        tx, ty = target_xy if target_xy is not None else (cur["x"] + desired_dx, cur["y"] + desired_dy)
        current_target_dist = math.hypot(tx - cur["x"], ty - cur["y"])

        scored: list[tuple[float, str]] = []
        for nid, _ in neighbors:
            n = self.nodes[nid]
            vx = n["x"] - cur["x"]
            vy = n["y"] - cur["y"]
            vmag = math.hypot(vx, vy)
            align = 0.0 if mag <= 1e-9 or vmag <= 1e-9 else (vx * desired_dx + vy * desired_dy) / (vmag * mag)
            progress = current_target_dist - math.hypot(tx - n["x"], ty - n["y"])
            scored.append((align * 0.75 + progress * 0.025 + rng.random() * 0.01, nid))

        scored.sort(reverse=True)
        return scored[0][1]

    def _scaled_safari_zones(self) -> list[dict[str, Any]]:
        zones = []
        lat_m = 111_320.0
        lon_m = lat_m * math.cos(math.radians((self.bbox["north"] + self.bbox["south"]) * 0.5))
        x_units_per_m = (self.width - 1) / max(1.0, (self.bbox["east"] - self.bbox["west"]) * lon_m)
        y_units_per_m = (self.height - 1) / max(1.0, (self.bbox["north"] - self.bbox["south"]) * lat_m)
        units_per_m = (x_units_per_m + y_units_per_m) * 0.5
        for zid, zone in SAFARI_ZONE_COORDS.items():
            x, y = self.coord_to_xy(zone["center"]["lat"], zone["center"]["lon"])
            zones.append({
                "id": zid,
                "label": zone["label"],
                "cx": x,
                "cy": y,
                "radius": float(zone["radius_m"]) * units_per_m,
            })
        return zones


def load_bannerghatta_graph(width: int, height: int) -> GeoGraph:
    if GRAPH_CACHE.exists():
        try:
            return GeoGraph.from_json(json.loads(GRAPH_CACHE.read_text(encoding="utf-8")), width, height)
        except Exception as exc:
            print(f"[GeoGraph] Could not read {GRAPH_CACHE}: {exc}")

    if OVERPASS_RAW_CACHE.exists():
        try:
            payload = build_graph_payload_from_overpass(json.loads(OVERPASS_RAW_CACHE.read_text(encoding="utf-8")))
            return GeoGraph.from_json(payload, width, height)
        except Exception as exc:
            print(f"[GeoGraph] Could not build graph from {OVERPASS_RAW_CACHE}: {exc}")

    return GeoGraph.empty(width, height)


def build_graph_payload_from_overpass(osm: dict[str, Any]) -> dict[str, Any]:
    raw_nodes = {int(el["id"]): el for el in osm.get("elements", []) if el.get("type") == "node"}
    raw_ways = {int(el["id"]): el for el in osm.get("elements", []) if el.get("type") == "way"}
    relations = [el for el in osm.get("elements", []) if el.get("type") == "relation"]

    relation = next(
        (
            rel for rel in relations
            if int(rel.get("id", -1)) == BANNERGHATTA_RELATION_ID
            or rel.get("tags", {}).get("name") == "Bannerghatta National Park"
        ),
        None,
    )
    boundary_node_rings = relation_rings(relation, raw_ways) if relation else []
    boundary_lonlat = []
    for ring in boundary_node_rings:
        pts = []
        for nid in ring:
            n = raw_nodes.get(int(nid))
            if n:
                pts.append((float(n["lon"]), float(n["lat"])))
        if pts:
            boundary_lonlat.append(pts)

    def inside(nid: int) -> bool:
        n = raw_nodes.get(int(nid))
        if not n:
            return False
        if not boundary_lonlat:
            return True
        lon, lat = float(n["lon"]), float(n["lat"])
        return any(point_in_poly(lon, lat, ring) for ring in boundary_lonlat)

    graph_nodes: dict[str, dict[str, float]] = {}
    graph_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()

    for way_id, way in raw_ways.items():
        tags = way.get("tags", {})
        highway = tags.get("highway")
        if isinstance(highway, list):
            highway = highway[0] if highway else None
        if highway not in GRAPH_HIGHWAYS:
            continue

        way_nodes = [int(nid) for nid in way.get("nodes", []) if int(nid) in raw_nodes]
        for u, v in zip(way_nodes, way_nodes[1:]):
            if not inside(u) or not inside(v):
                continue
            a, b = raw_nodes[u], raw_nodes[v]
            su, sv = str(u), str(v)
            graph_nodes.setdefault(su, {"lat": float(a["lat"]), "lon": float(a["lon"])})
            graph_nodes.setdefault(sv, {"lat": float(b["lat"]), "lon": float(b["lon"])})
            key = tuple(sorted((su, sv)))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            graph_edges.append({
                "u": su,
                "v": sv,
                "osmid": way_id,
                "kind": str(highway),
                "length_m": round(haversine_m(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"])), 3),
            })

    bbox = _bbox_from_boundary(boundary_lonlat) or dict(BANNERGHATTA_BBOX)
    water_polylines = _extract_suvarnamukhi(raw_nodes, raw_ways)
    water_points = _extract_water_points(raw_nodes, raw_ways, boundary_lonlat)
    recharge = [{"lat": p["lat"], "lon": p["lon"], "label": p["label"], "type": "recharge_pit"} for p in RECHARGE_PIT_COORDS]
    boundary = [
        [{"lat": lat, "lon": lon} for lon, lat in ring]
        for ring in boundary_lonlat
    ]

    return {
        "source": "OpenStreetMap cached Overpass fallback; refresh with tools/fetch_bannerghatta_osm.py for OSMnx graph",
        "bbox": bbox,
        "nodes": graph_nodes,
        "edges": graph_edges,
        "water_polylines": water_polylines,
        "water_points": water_points,
        "recharge_pits": recharge,
        "boundary_rings": boundary,
        "metadata": {
            "osm_relation_id": BANNERGHATTA_RELATION_ID,
            "declared_area_sq_km": BANNERGHATTA_DECLARED_AREA_SQ_KM,
            "suvarnamukhi_way_ids": SUVARNA_MUKHI_OSM_WAY_IDS,
            "movement": "Agents are snapped to and advanced along OSM highway/path/track edges.",
        },
    }


def _bbox_from_boundary(boundary_lonlat: list[list[tuple[float, float]]]) -> dict[str, float] | None:
    pts = [p for ring in boundary_lonlat for p in ring]
    if not pts:
        return None
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return {"north": max(lats), "south": min(lats), "west": min(lons), "east": max(lons)}


def _extract_suvarnamukhi(raw_nodes: dict[int, dict[str, Any]], raw_ways: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    lines = []
    for way_id in SUVARNA_MUKHI_OSM_WAY_IDS:
        way = raw_ways.get(int(way_id))
        if not way:
            continue
        pts = []
        for nid in way.get("nodes", []):
            node = raw_nodes.get(int(nid))
            if node:
                pts.append({"lat": float(node["lat"]), "lon": float(node["lon"])})
        if len(pts) >= 2:
            lines.append({"id": str(way_id), "name": "Suvarnamukhi", "type": way.get("tags", {}).get("waterway", "stream"), "points": pts})

    if lines:
        return lines
    return [{"id": "suvarnamukhi_fallback", "name": "Suvarnamukhi", "type": "stream", "points": list(SUVARNA_MUKHI_ANCHOR_COORDS)}]


def _extract_water_points(
    raw_nodes: dict[int, dict[str, Any]],
    raw_ways: dict[int, dict[str, Any]],
    boundary_lonlat: list[list[tuple[float, float]]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for way_id, way in raw_ways.items():
        tags = way.get("tags", {})
        if tags.get("natural") != "water" and tags.get("landuse") not in {"reservoir", "basin"}:
            continue
        coords = []
        for nid in way.get("nodes", []):
            node = raw_nodes.get(int(nid))
            if node:
                coords.append((float(node["lat"]), float(node["lon"])))
        if not coords:
            continue
        lat = sum(p[0] for p in coords) / len(coords)
        lon = sum(p[1] for p in coords) / len(coords)
        if boundary_lonlat and not any(point_in_poly(lon, lat, ring) for ring in boundary_lonlat):
            continue
        label = tags.get("name") or tags.get("water") or "Mapped water body"
        points.append({"id": str(way_id), "lat": lat, "lon": lon, "label": label, "type": "lake"})
    return points[:40]


def _scale_polylines(lines: list[dict[str, Any]], bbox: dict[str, float], width: int, height: int) -> list[dict[str, Any]]:
    scaled = []
    for line in lines:
        pts = []
        for p in line.get("points", []):
            x, y = latlon_to_xy(float(p["lat"]), float(p["lon"]), bbox, width, height)
            pts.append({"lat": float(p["lat"]), "lon": float(p["lon"]), "x": x, "y": y})
        if pts:
            scaled.append({**line, "points": pts})
    return scaled


def _scale_points(points: list[dict[str, Any]], bbox: dict[str, float], width: int, height: int) -> list[dict[str, Any]]:
    scaled = []
    for p in points:
        x, y = latlon_to_xy(float(p["lat"]), float(p["lon"]), bbox, width, height)
        scaled.append({**p, "x": x, "y": y})
    return scaled


def _paint_circle(layer: Any, x: float, y: float, radius: float, value: float) -> None:
    h, w = int(layer.shape[0]), int(layer.shape[1])
    r = max(0.5, float(radius))
    x0 = max(0, int(math.floor(x - r)))
    x1 = min(w - 1, int(math.ceil(x + r)))
    y0 = max(0, int(math.floor(y - r)))
    y1 = min(h - 1, int(math.ceil(y + r)))
    rr = r * r
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            if (xx - x) ** 2 + (yy - y) ** 2 <= rr:
                layer[yy, xx] = max(float(layer[yy, xx]), value)


def _draw_line(layer: Any, pts: list[tuple[float, float]], width: float, value: float) -> None:
    if len(pts) < 2:
        return
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(dist * 2.0))
        for i in range(steps + 1):
            t = i / steps
            _paint_circle(layer, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, width, value)


def _fill_polygon(layer: Any, pts: list[tuple[float, float]], value: float) -> None:
    if len(pts) < 3:
        return
    h, w = int(layer.shape[0]), int(layer.shape[1])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = max(0, int(math.floor(min(xs))))
    x1 = min(w - 1, int(math.ceil(max(xs))))
    y0 = max(0, int(math.floor(min(ys))))
    y1 = min(h - 1, int(math.ceil(max(ys))))
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            if point_in_poly(xx + 0.5, yy + 0.5, pts):
                layer[yy, xx] = max(float(layer[yy, xx]), value)
