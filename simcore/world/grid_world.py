from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

LAYER_WATER = 0
LAYER_CROP = 1
LAYER_LIVESTOCK = 2
LAYER_SETTLEMENT = 3
LAYER_OBSTACLE = 4
LAYER_COVER = 5
LAYER_SLOPE = 6

N_LAYERS = 7


@dataclass
class GridWorld:
    W: int
    H: int
    layers: np.ndarray  # (H, W, N_LAYERS) float32 in [0,1]

    def in_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x < self.W and 0.0 <= y < self.H

    def sample(self, layer: int, x: float, y: float) -> float:
        ix = int(np.clip(x, 0, self.W - 1))
        iy = int(np.clip(y, 0, self.H - 1))
        return float(self.layers[iy, ix, layer])

    def is_obstacle(self, x: float, y: float) -> bool:
        return self.sample(LAYER_OBSTACLE, x, y) > 0.5

    def move_cost(self, x: float, y: float) -> float:
        slope = self.sample(LAYER_SLOPE, x, y)
        cover = self.sample(LAYER_COVER, x, y)
        return 1.0 + 1.5 * slope + 0.3 * (1.0 - cover)

    def nearest_peak(self, layer: int, x: float, y: float, r: int) -> tuple[float, float] | None:
        ix = int(np.clip(x, 0, self.W - 1))
        iy = int(np.clip(y, 0, self.H - 1))
        x0, x1 = max(0, ix - r), min(self.W - 1, ix + r)
        y0, y1 = max(0, iy - r), min(self.H - 1, iy + r)

        patch = self.layers[y0:y1 + 1, x0:x1 + 1, layer]
        if patch.size == 0:
            return None
        j = int(np.argmax(patch))
        py, px = divmod(j, patch.shape[1])
        if patch[py, px] <= 0.2:
            return None
        return float(x0 + px), float(y0 + py)


def _blur2d(a: np.ndarray, k: int = 7) -> np.ndarray:
    """Size-preserving box blur using cumulative sums."""
    if k <= 1:
        return a.astype(np.float32, copy=False)

    pad = k // 2
    H, W = a.shape
    a2 = np.pad(a, ((pad, pad), (pad, pad)), mode="reflect")

    c = np.cumsum(a2, axis=1)
    c = np.pad(c, ((0, 0), (1, 0)), mode="constant")
    h = (c[:, k:k + W] - c[:, 0:W]) / float(k)

    c2 = np.cumsum(h, axis=0)
    c2 = np.pad(c2, ((1, 0), (0, 0)), mode="constant")
    v = (c2[k:k + H, :] - c2[0:H, :]) / float(k)

    return v.astype(np.float32)


def _gaussian_blob(H: int, W: int, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    dx = (xs - cx) / max(1e-6, sx)
    dy = (ys - cy) / max(1e-6, sy)
    return np.exp(-(dx * dx + dy * dy) * 0.5).astype(np.float32)


def _resize_layers_nearest(src: np.ndarray, H: int, W: int) -> np.ndarray:
    if src.shape[0] == H and src.shape[1] == W:
        return src.astype(np.float32, copy=False)
    y_idx = np.linspace(0, src.shape[0] - 1, H).round().astype(np.int32)
    x_idx = np.linspace(0, src.shape[1] - 1, W).round().astype(np.int32)
    return src[y_idx][:, x_idx].astype(np.float32, copy=False)


def _load_cached_bannerghatta_osm(W: int, H: int) -> np.ndarray | None:
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "bannerghatta_osm_layers.npz"
    if not path.exists():
        print(f"[GridWorld] OSM cache missing: {path}. Falling back to proxy Bannerghatta.")
        print("[GridWorld] Run: python tools\\fetch_bannerghatta_osm.py")
        return None

    data = np.load(path, allow_pickle=False)
    cached = data["layers"].astype(np.float32)
    if cached.ndim != 3 or cached.shape[2] != N_LAYERS:
        raise ValueError(f"Invalid Bannerghatta OSM cache shape: {cached.shape}")
    cached = _resize_layers_nearest(cached, H, W)
    np.clip(cached, 0.0, 1.0, out=cached)
    return cached


def make_topology_layers(topology: str, W: int, H: int, rng: np.random.Generator) -> np.ndarray:
    """Returns (H, W, 7) float32 layers in [0,1]."""
    topo = (topology or "edge_farmland").lower()
    if topo == "bannerghatta_osm":
        cached = _load_cached_bannerghatta_osm(W, H)
        if cached is not None:
            return cached
        topo = "bannerghatta"

    layers = np.zeros((H, W, N_LAYERS), dtype=np.float32)

    cover = rng.random((H, W), dtype=np.float32)
    slope = rng.random((H, W), dtype=np.float32)

    cover = _blur2d(cover, 11)
    slope = _blur2d(slope, 15)

    cover = (cover - cover.min()) / (cover.max() - cover.min() + 1e-6)
    slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)

    layers[:, :, LAYER_COVER] = np.clip(cover * 0.9 + 0.05, 0, 1)
    layers[:, :, LAYER_SLOPE] = np.clip(slope * 0.7, 0, 1)

    if topo == "edge_farmland":
        x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        farm_band = np.exp(-(x / 0.22) ** 2).astype(np.float32)
        farm_band = np.repeat(farm_band, H, axis=0)

        crop = np.clip(_blur2d(farm_band + 0.15 * rng.random((H, W), dtype=np.float32), 9), 0, 1)
        settlement = np.clip(_blur2d(farm_band * 0.8 + 0.10 * rng.random((H, W), dtype=np.float32), 13), 0, 1)
        water = _gaussian_blob(H, W, cx=W * 0.55, cy=H * 0.50, sx=W * 0.09, sy=H * 0.10)

        layers[:, :, LAYER_CROP] = crop
        layers[:, :, LAYER_SETTLEMENT] = settlement * 0.8
        layers[:, :, LAYER_WATER] = water
        layers[:, :, LAYER_COVER] = np.clip(layers[:, :, LAYER_COVER] * (0.6 + 0.6 * (1 - farm_band)), 0, 1)

    elif topo in ("scarce_water", "drought"):
        water = np.zeros((H, W), dtype=np.float32)

        holes = []
        for _ in range(6):
            cx = float(rng.integers(int(W * 0.15), int(W * 0.85)))
            cy = float(rng.integers(int(H * 0.15), int(H * 0.85)))
            holes.append((cx, cy))

        dry_mask = rng.random(len(holes)) < 0.5 
        for i, (cx, cy) in enumerate(holes):
            size = float(rng.uniform(W * 0.030, W * 0.055))
            strength = float(rng.uniform(0.85, 1.00))
            if dry_mask[i]:
                strength *= float(rng.uniform(0.18, 0.35)) 
            water += _gaussian_blob(H, W, cx, cy, sx=size, sy=size) * strength

        water = np.clip(_blur2d(water, 5), 0, 1)

        base_veg = rng.random((H, W), dtype=np.float32)
        base_veg = _blur2d(base_veg, 17)
        base_veg = (base_veg - base_veg.min()) / (base_veg.max() - base_veg.min() + 1e-6)

        veg_patches = np.zeros((H, W), dtype=np.float32)
        for _ in range(int(rng.integers(5, 9))):
            cx = float(rng.uniform(W * 0.10, W * 0.90))
            cy = float(rng.uniform(H * 0.10, H * 0.90))
            sx = float(rng.uniform(W * 0.06, W * 0.12))
            sy = float(rng.uniform(H * 0.06, H * 0.12))
            veg_patches += _gaussian_blob(H, W, cx, cy, sx=sx, sy=sy) * float(rng.uniform(0.55, 0.95))

        crop = 0.45 * base_veg + 0.55 * _blur2d(veg_patches, 7)
        crop = np.clip(crop + 0.45 * water, 0, 1) 

        settlement = np.clip(_blur2d(rng.random((H, W), dtype=np.float32) * 0.10, 15), 0, 1)

        layers[:, :, LAYER_WATER] = water
        layers[:, :, LAYER_CROP] = crop
        layers[:, :, LAYER_SETTLEMENT] = settlement
        layers[:, :, LAYER_COVER] = np.clip(layers[:, :, LAYER_COVER] * 0.60, 0, 1)
        layers[:, :, LAYER_SLOPE] = np.clip(layers[:, :, LAYER_SLOPE] * 0.90 + 0.05, 0, 1)

    elif topo in ("river_valley", "river_basin"):
        x = np.arange(W, dtype=np.float32)
        centerline = (H * 0.5 + (H * 0.18) * np.sin(2 * np.pi * x / (W * 0.8))).astype(np.float32)

        water = np.zeros((H, W), dtype=np.float32)
        yy = np.arange(H, dtype=np.float32)
        for ix in range(W):
            cy = centerline[ix]
            water[:, ix] = np.exp(-((yy - cy) ** 2) / (2 * (H * 0.03) ** 2))

        water = np.clip(_blur2d(water, 5), 0, 1)
        crop = np.clip(_blur2d(water * 0.9 + 0.10 * rng.random((H, W), dtype=np.float32), 9), 0, 1)
        settlement = np.clip(_blur2d(water * 0.6 + 0.10 * rng.random((H, W), dtype=np.float32), 13), 0, 1)

        layers[:, :, LAYER_WATER] = water
        layers[:, :, LAYER_CROP] = crop
        layers[:, :, LAYER_SETTLEMENT] = settlement
        layers[:, :, LAYER_COVER] = np.clip(layers[:, :, LAYER_COVER] * (0.65 + 0.35 * (1 - water)), 0, 1)

    # ==========================================
    # ✅ NEW: DENSE FOREST CORE
    # ==========================================
    elif topo == "dense_forest":
        # Very high tree cover, flatter ground
        layers[:, :, LAYER_COVER] = np.clip(layers[:, :, LAYER_COVER] * 0.5 + 0.5, 0, 1) 
        layers[:, :, LAYER_SLOPE] = np.clip(layers[:, :, LAYER_SLOPE] * 0.5, 0, 1) 
        
        water = np.zeros((H, W), dtype=np.float32)
        for _ in range(4): # Scattered hidden forest ponds
            cx = float(rng.uniform(W * 0.2, W * 0.8))
            cy = float(rng.uniform(H * 0.2, H * 0.8))
            water += _gaussian_blob(H, W, cx, cy, sx=W*0.04, sy=H*0.04) * 0.9
        
        layers[:, :, LAYER_WATER] = np.clip(_blur2d(water, 5), 0, 1)
        layers[:, :, LAYER_CROP] = np.clip(_blur2d(rng.random((H, W), dtype=np.float32) * 0.05, 9), 0, 1) # Almost no crops
        layers[:, :, LAYER_SETTLEMENT] = np.clip(_blur2d(rng.random((H, W), dtype=np.float32) * 0.05, 11), 0, 1) # Almost no humans

    # ==========================================
    # ✅ NEW: FRAGMENTED SETTLEMENT
    # ==========================================
    elif topo == "fragmented":
        settlement = np.zeros((H, W), dtype=np.float32)
        crop = np.zeros((H, W), dtype=np.float32)
        
        # 5 scattered distinct villages breaking up the landscape
        for _ in range(5):
            cx = float(rng.uniform(W * 0.15, W * 0.85))
            cy = float(rng.uniform(H * 0.15, H * 0.85))
            settlement += _gaussian_blob(H, W, cx, cy, sx=W*0.06, sy=H*0.06) * 1.0
            crop += _gaussian_blob(H, W, cx + W*0.05, cy - H*0.05, sx=W*0.08, sy=H*0.08) * 0.9
            crop += _gaussian_blob(H, W, cx - W*0.05, cy + H*0.05, sx=W*0.08, sy=H*0.08) * 0.9
        
        layers[:, :, LAYER_SETTLEMENT] = np.clip(_blur2d(settlement, 7), 0, 1)
        layers[:, :, LAYER_CROP] = np.clip(_blur2d(crop, 9), 0, 1)
        
        # Deforestation directly around human settlements
        human_footprint = np.clip(layers[:, :, LAYER_SETTLEMENT] + layers[:, :, LAYER_CROP], 0, 1)
        layers[:, :, LAYER_COVER] = np.clip(layers[:, :, LAYER_COVER] - human_footprint, 0, 1)
        
        water = _gaussian_blob(H, W, cx=W*0.5, cy=H*0.8, sx=W*0.1, sy=H*0.05) * 0.8
        water += _gaussian_blob(H, W, cx=W*0.8, cy=H*0.2, sx=W*0.08, sy=H*0.08) * 0.8
        layers[:, :, LAYER_WATER] = np.clip(water, 0, 1)

    # ==========================================
    # ✅ NEW: HIGH SLOPE TERRAIN
    # ==========================================
    elif topo == "high_slope":
        slope = layers[:, :, LAYER_SLOPE]
        # Amplify slope mathematically to create sharp mountains
        layers[:, :, LAYER_SLOPE] = np.clip(slope * 2.0, 0, 1)
        
        # Resources are trapped in the low valleys
        valley_mask = 1.0 - layers[:, :, LAYER_SLOPE]
        
        water = np.clip(_blur2d(valley_mask * rng.random((H, W), dtype=np.float32), 15) * 2.5, 0, 1)
        layers[:, :, LAYER_WATER] = water
        
        crop = np.clip(_blur2d(valley_mask * rng.random((H, W), dtype=np.float32), 11) * 1.8, 0, 1)
        layers[:, :, LAYER_CROP] = crop
        
        settlement = np.clip(_blur2d(crop * rng.random((H, W), dtype=np.float32), 9) * 1.5, 0, 1)
        layers[:, :, LAYER_SETTLEMENT] = settlement
        
        # Trees grow heavily on the mountainsides
        layers[:, :, LAYER_COVER] = np.clip(layers[:, :, LAYER_COVER] + (layers[:, :, LAYER_SLOPE] * 0.5), 0, 1)

    elif topo in ("bannerghatta", "bannerghatta_bnp"):
        # Bannerghatta National Park is modelled as a long, narrow dry-forest
        # strip on Bengaluru's southern urban edge. The coordinates are not a
        # surveyed boundary; they are a deterministic ecological proxy tuned to
        # preserve the conflict drivers of the real landscape.
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        xn = xx / max(1.0, float(W - 1))
        yn = yy / max(1.0, float(H - 1))

        # North-south protected forest strip with a mild bend and variable width.
        center = 0.50 + 0.055 * np.sin(2 * np.pi * (yn * 1.15 + 0.10))
        width = 0.115 - 0.030 * yn + 0.020 * np.exp(-((yn - 0.70) / 0.22) ** 2)
        forest_core = np.exp(-((xn - center) / np.maximum(width, 0.035)) ** 4)

        # Bengaluru-side pressure: denser urban edge to the north and northeast,
        # with village clusters along both forest boundaries.
        north_urban = np.exp(-(yn / 0.24) ** 2) * (0.65 + 0.35 * xn)
        east_urban = np.exp(-((xn - 0.88) / 0.13) ** 2) * np.exp(-((yn - 0.28) / 0.34) ** 2)
        west_villages = np.zeros((H, W), dtype=np.float32)
        east_villages = np.zeros((H, W), dtype=np.float32)
        for cy, amp in [(0.18, 0.75), (0.36, 0.55), (0.58, 0.48), (0.78, 0.38)]:
            west_villages += _gaussian_blob(H, W, cx=W * 0.22, cy=H * cy, sx=W * 0.055, sy=H * 0.055) * amp
        for cy, amp in [(0.22, 0.75), (0.44, 0.58), (0.66, 0.46), (0.84, 0.35)]:
            east_villages += _gaussian_blob(H, W, cx=W * 0.78, cy=H * cy, sx=W * 0.060, sy=H * 0.055) * amp

        settlement = np.clip(_blur2d(north_urban + east_urban + west_villages + east_villages, 7), 0, 1)
        settlement *= np.clip(1.0 - 0.82 * forest_core, 0, 1)

        # Crops form an agricultural pressure ring along the protected-area edge.
        edge_band = np.exp(-((np.abs(xn - center) - (width + 0.050)) / 0.055) ** 2)
        crop = 0.68 * edge_band + 0.42 * (west_villages + east_villages)
        crop += 0.12 * rng.random((H, W), dtype=np.float32)
        crop = np.clip(_blur2d(crop, 9), 0, 1)
        crop *= np.clip(1.0 - 0.72 * forest_core, 0, 1)

        # Small tanks and valley water bodies; enough to drive dry-season movement
        # without making the landscape uniformly safe.
        water = np.zeros((H, W), dtype=np.float32)
        water_sites = [
            (0.43, 0.18, 0.035, 0.030, 0.85),
            (0.57, 0.31, 0.040, 0.032, 0.70),
            (0.45, 0.51, 0.036, 0.030, 0.78),
            (0.62, 0.70, 0.045, 0.035, 0.86),
            (0.38, 0.86, 0.034, 0.030, 0.70),
        ]
        for cx, cy, sx, sy, amp in water_sites:
            water += _gaussian_blob(H, W, cx=W * cx, cy=H * cy, sx=W * sx, sy=H * sy) * amp

        # Undulating rocky ridges through the park interior.
        ridge = np.exp(-((xn - (center + 0.040 * np.sin(yn * 12.0))) / 0.070) ** 2)
        layers[:, :, LAYER_SLOPE] = np.clip(layers[:, :, LAYER_SLOPE] * 0.45 + ridge * 0.65, 0, 1)

        cover_noise = _blur2d(rng.random((H, W), dtype=np.float32), 13)
        cover_noise = (cover_noise - cover_noise.min()) / (cover_noise.max() - cover_noise.min() + 1e-6)
        layers[:, :, LAYER_COVER] = np.clip(0.10 + 0.82 * forest_core + 0.16 * cover_noise - 0.50 * settlement - 0.28 * crop, 0, 1)
        layers[:, :, LAYER_WATER] = np.clip(_blur2d(water, 5), 0, 1)
        layers[:, :, LAYER_CROP] = crop
        layers[:, :, LAYER_SETTLEMENT] = settlement

        # Approximate Bannerghatta Road / NH-209 corridor pressure, with the
        # runtime bridge/fence layer later acting as the wildlife crossing.
        road_x = 0.40 + 0.12 * yn
        road = np.exp(-((xn - road_x) / 0.012) ** 2)
        layers[:, :, LAYER_OBSTACLE] = np.clip(road * (0.65 - 0.35 * forest_core), 0, 1)
        layers[:, :, LAYER_COVER] = np.clip(layers[:, :, LAYER_COVER] * (1.0 - 0.35 * road), 0, 1)

    else:
        water = _gaussian_blob(H, W, cx=W * 0.60, cy=H * 0.55, sx=W * 0.10, sy=H * 0.10)
        crop = _gaussian_blob(H, W, cx=W * 0.35, cy=H * 0.40, sx=W * 0.12, sy=H * 0.12)
        settlement = _gaussian_blob(H, W, cx=W * 0.40, cy=H * 0.55, sx=W * 0.10, sy=H * 0.10)

        layers[:, :, LAYER_WATER] = water
        layers[:, :, LAYER_CROP] = crop
        layers[:, :, LAYER_SETTLEMENT] = settlement

    layers[:, :, LAYER_LIVESTOCK] = 0.0
    np.clip(layers, 0.0, 1.0, out=layers)
    return layers
