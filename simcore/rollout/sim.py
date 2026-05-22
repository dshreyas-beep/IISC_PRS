# simcore/rollout/sim.py
from __future__ import annotations
import numpy as np
import math

from simcore.world.topology import make_topology
from simcore.agents.agent import Agent
from simcore.behavior.controller import choose_mode, pick_target
from simcore.behavior.movement import teacher_step


class SingleAgentSim:
    def __init__(self, seed: int, species: str, topology: str, W: int = 120, H: int = 80):
        self.seed = seed
        self.species = species
        self.topology = topology
        self.rng = np.random.default_rng(seed)
        self.world = make_topology(seed=seed, W=W, H=H, preset=topology)

        x = W * 0.25 + float(self.rng.integers(0, 10))
        y = H * 0.25 + float(self.rng.integers(0, 10))
        heading = float(self.rng.uniform(0, 2 * math.pi))
        self.agent = Agent.spawn(0, species, x=x, y=y, heading=heading)
        self.t = 0

    def step(self):
        a = self.agent
        a.mode = choose_mode(a, self.world)

        # Increased search radius (important for scarce resources/topology tests)
        # old: vision_radius * 0.3 capped to 40
        # new: vision_radius * 0.8 capped to 80 (more realistic sensing/goal targeting)
        search_r = int(max(10, min(80, a.traits.get("vision_radius") * 0.8)))

        target = pick_target(a, self.world, search_r)
        if target is not None:
            dx = target[0] - a.x
            dy = target[1] - a.y
            desired = math.atan2(dy, dx)

            alpha = 0.15 + 0.65 * a.traits.get("path_straightness")
            d = (desired - a.heading + math.pi) % (2 * math.pi) - math.pi
            a.heading = (a.heading + alpha * d) % (2 * math.pi)

        action = teacher_step(a, self.world, self.rng)
        self.t += 1
        return action

    def meta(self):
        return {"seed": self.seed, "species": self.species, "topology": self.topology}
