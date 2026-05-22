# simcore/rollout/rollout.py
from __future__ import annotations
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, Any

from simcore.world.grid_world import (
    LAYER_WATER, LAYER_CROP, LAYER_LIVESTOCK, LAYER_SETTLEMENT, LAYER_COVER, LAYER_SLOPE
)
from simcore.species.base import CANONICAL_TRAITS_ORDER

# Action scaling used in v5 stable dynamics
ACTION_DELTA_MAX = 0.9  # radians


def build_obs(agent, world) -> np.ndarray:
    """
    Observation vector = [traits_vector, state_vector, local_samples]

    traits_vector: canonical ordered trait values
    state_vector: x_norm, y_norm, sin(h), cos(h), energy_ratio, thirst_ratio, rest_debt
    local_samples: water,crop,livestock,settlement,cover,slope (at current cell)

    D = len(CANONICAL_TRAITS_ORDER) + 7 + 6
    """
    traits = np.array(agent.traits.vector(CANONICAL_TRAITS_ORDER), dtype=np.float32)

    x_norm = np.float32(agent.x / max(1.0, world.W))
    y_norm = np.float32(agent.y / max(1.0, world.H))
    sh = np.float32(math.sin(agent.heading))
    ch = np.float32(math.cos(agent.heading))
    er = np.float32(agent.energy_ratio())
    tr = np.float32(agent.thirst_ratio())
    rd = np.float32(agent.rest_debt)
    state = np.array([x_norm, y_norm, sh, ch, er, tr, rd], dtype=np.float32)

    local = np.array([
        world.sample(LAYER_WATER, agent.x, agent.y),
        world.sample(LAYER_CROP, agent.x, agent.y),
        world.sample(LAYER_LIVESTOCK, agent.x, agent.y),
        world.sample(LAYER_SETTLEMENT, agent.x, agent.y),
        world.sample(LAYER_COVER, agent.x, agent.y),
        world.sample(LAYER_SLOPE, agent.x, agent.y),
    ], dtype=np.float32)

    return np.concatenate([traits, state, local], axis=0)


@dataclass
class Rollout:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    meta: Dict[str, Any]


def generate_rollout(sim, T: int) -> Rollout:
    """
    Teacher-generated rollout (used for BC seed dataset).
    """
    obs_list, act_list, rew_list, done_list = [], [], [], []
    for _ in range(T):
        a = sim.agent
        obs = build_obs(a, sim.world)
        action = sim.step()  # teacher action already applied inside

        # Optional reward for RL compatibility (not used for BC)
        settle = sim.world.sample(LAYER_SETTLEMENT, a.x, a.y)
        r = 0.2 * a.energy_ratio() + 0.2 * a.thirst_ratio() - 0.2 * settle * a.traits.get("settlement_avoid")
        done = 1.0 if (a.energy <= 0.0 or a.thirst <= 0.0) else 0.0

        obs_list.append(obs)
        act_list.append(np.array(action, dtype=np.float32))
        rew_list.append(np.float32(r))
        done_list.append(np.float32(done))
        if done > 0:
            break

    return Rollout(
        obs=np.stack(obs_list, axis=0).astype(np.float32),
        actions=np.stack(act_list, axis=0).astype(np.float32),
        rewards=np.array(rew_list, dtype=np.float32),
        dones=np.array(done_list, dtype=np.float32),
        meta=sim.meta(),
    )
