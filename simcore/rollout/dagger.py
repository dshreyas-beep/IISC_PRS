# simcore/rollout/dagger.py
from __future__ import annotations
import copy
import numpy as np
import math

from simcore.rollout.rollout import build_obs
from simcore.behavior.controller import choose_mode, pick_target
from simcore.behavior.movement import teacher_step
from simcore.behavior.apply_action import apply_action


def _clone_agent(agent):
    # dataclass shallow copy is enough because fields are primitives + dict
    a2 = copy.copy(agent)
    a2.memory = dict(agent.memory) if hasattr(agent, "memory") else {}
    return a2


def teacher_action_for_state(agent, world, rng: np.random.Generator) -> tuple[float, float]:
    """
    Compute teacher action WITHOUT advancing the real agent by cloning state.
    """
    a = _clone_agent(agent)

    # same mode & target-blending used in sim
    a.mode = choose_mode(a, world)

    search_r = int(max(10, min(80, a.traits.get("vision_radius") * 0.8)))
    target = pick_target(a, world, search_r)
    if target is not None:
        dx = target[0] - a.x
        dy = target[1] - a.y
        desired = math.atan2(dy, dx)
        alpha = 0.15 + 0.65 * a.traits.get("path_straightness")
        d = (desired - a.heading + math.pi) % (2 * math.pi) - math.pi
        a.heading = (a.heading + alpha * d) % (2 * math.pi)

    # teacher_step both returns action and applies it on the CLONE
    return teacher_step(a, world, rng)


def collect_dagger_data(sim, policy, steps: int, seed: int = 0):
    """
    Run the *learned* policy to generate visited states.
    Label each visited state with teacher action.
    Returns arrays: obs, teacher_actions, dones
    """
    rng = np.random.default_rng(seed)
    obs_buf = []
    act_buf = []
    done_buf = []

    for _ in range(steps):
        a = sim.agent

        # choose mode + target heading blend exactly as teacher pipeline expects
        a.mode = choose_mode(a, sim.world)
        search_r = int(max(10, min(80, a.traits.get("vision_radius") * 0.8)))
        target = pick_target(a, sim.world, search_r)
        if target is not None:
            dx = target[0] - a.x
            dy = target[1] - a.y
            desired = math.atan2(dy, dx)
            alpha = 0.15 + 0.65 * a.traits.get("path_straightness")
            d = (desired - a.heading + math.pi) % (2 * math.pi) - math.pi
            a.heading = (a.heading + alpha * d) % (2 * math.pi)

        obs = build_obs(a, sim.world)

        # label with teacher for THIS state
        teacher_act = teacher_action_for_state(a, sim.world, rng)

        # step using ML policy action
        ml_act = policy.act(obs)
        apply_action(a, sim.world, ml_act)

        done = 1.0 if (a.energy <= 0.0 or a.thirst <= 0.0) else 0.0

        obs_buf.append(obs)
        act_buf.append(np.array(teacher_act, dtype=np.float32))
        done_buf.append(np.float32(done))

        if done > 0:
            break

    return (
        np.stack(obs_buf, axis=0).astype(np.float32),
        np.stack(act_buf, axis=0).astype(np.float32),
        np.array(done_buf, dtype=np.float32),
    )
