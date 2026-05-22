from simcore.world.grid_world import (
    LAYER_WATER,
    LAYER_CROP,
)


class WorldRuntime:
    """
    Runtime ecological world.

    Wraps GridWorld and adds:
    - Resource consumption
    - Passive regeneration
    """

    # =====================================================
    # INIT
    # =====================================================

    def __init__(self, grid_world, rng):
        self.grid = grid_world
        self.rng = rng

        # Expose dimensions (agents rely on this)
        self.width = grid_world.W
        self.height = grid_world.H

    # =====================================================
    # STEP / REGENERATION
    # =====================================================

    def step(self):
        """
        Advance ecological simulation one tick.
        Applies small passive regeneration to water and crops.
        """

        # Water regenerates slowly
        self.grid.layers[:, :, LAYER_WATER] = (
            self.grid.layers[:, :, LAYER_WATER] + 0.0005
        ).clip(0.0, 1.0)

        # Crops regenerate slightly faster
        self.grid.layers[:, :, LAYER_CROP] = (
            self.grid.layers[:, :, LAYER_CROP] + 0.001
        ).clip(0.0, 1.0)

    # =====================================================
    # RESOURCE CONSUMPTION
    # =====================================================

    def consume_at(self, x: float, y: float, amount: float, kind: str) -> float:
        """
        Agent attempts to consume a resource at position (x, y).

        Returns actual amount consumed.
        """

        # Clamp position to valid grid cell
        ix = int(max(0, min(self.width - 1, x)))
        iy = int(max(0, min(self.height - 1, y)))

        if kind == "water":
            layer = LAYER_WATER
        elif kind == "crop":
            layer = LAYER_CROP
        else:
            return 0.0

        available = float(self.grid.layers[iy, ix, layer])
        consumed = min(amount, available)

        # Reduce resource in grid
        self.grid.layers[iy, ix, layer] -= consumed

        return consumed

    # =====================================================
    # EXPORT STATE
    # =====================================================

    def export_state(self):
        """
        Export resource layers for frontend visualization.
        """

        return {
            "width": self.width,
            "height": self.height,
            "water": self.grid.layers[:, :, LAYER_WATER].tolist(),
            "crop": self.grid.layers[:, :, LAYER_CROP].tolist(),
        }