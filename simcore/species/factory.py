# simcore/species/factory.py
from __future__ import annotations
import numpy as np
import math

from simcore.agents.agent import Agent


def make_agent(species: str, world, rng: np.random.Generator) -> Agent:
    """
    Spawns an Agent and attaches species name as agent.species
    so conflict and logging modules can use it consistently.
    """

    x = float(rng.uniform(0, world.W - 1))
    y = float(rng.uniform(0, world.H - 1))
    heading = float(rng.uniform(0, 2 * math.pi))
    agent_id = int(rng.integers(1, 10**9))

    # Agent.spawn signature in your repo: Agent.spawn(agent_id, species_name, ...)
    a = Agent.spawn(agent_id, species, x=x, y=y, heading=heading)

    # ✅ attach species label for downstream modules
    a.species = species

    return a
