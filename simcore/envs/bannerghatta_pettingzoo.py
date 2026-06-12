from pettingzoo.utils import ParallelEnv
import numpy as np
from gymnasium import spaces
from simcore.rollout.multi_sim import MultiAgentSim
import math

class BannerghattaEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "bannerghatta_v0"}

    def __init__(self, topology_config=None, seed=1):
        if topology_config is None:
            topology_config = {"name": "bannerghatta_osm", "width": 240, "height": 720}
        self.sim = MultiAgentSim(topology_config, seed=seed)
        self.possible_agents = []
        for a in self.sim.agents:
            if getattr(a, "brain", None) is not None:
                self.possible_agents.append(f"{a.species}_{a.id}")
        self.agents = self.possible_agents[:]

        # 5 states: Hunger, dx_crops, dy_crops, dx_human, dy_human
        self.observation_spaces = {agent: spaces.Box(low=-1000, high=1000, shape=(5,), dtype=np.float32) for agent in self.possible_agents}
        
        # 2 actions: Heading (radians), Velocity (scale)
        self.action_spaces = {agent: spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32) for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.sim = MultiAgentSim({"name": "bannerghatta_osm", "width": 240, "height": 720}, seed=seed)
        self.agents = self.possible_agents[:]
        
        obs = {}
        for agent_id in self.agents:
            a = self._get_agent(agent_id)
            if a:
                obs[agent_id] = self._get_obs(a)
        return obs, {}

    def step(self, actions):
        # Apply actions to simulation
        for agent_id, action in actions.items():
            a = self._get_agent(agent_id)
            if a and a.alive:
                a.heading = float(action[0])
                a.vx = math.cos(a.heading) * max(0.0, float(action[1]))
                a.vy = math.sin(a.heading) * max(0.0, float(action[1]))

        # Step sim
        self.sim.step()

        # Gather results
        obs = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent_id in self.agents:
            a = self._get_agent(agent_id)
            if a:
                obs[agent_id] = self._get_obs(a)
                if a.rewards:
                    reward, done = a.rewards.get_reward(100.0 - a.energy, 999.0, False)
                    rewards[agent_id] = reward
                    terminations[agent_id] = not a.alive
                else:
                    rewards[agent_id] = 0.0
                    terminations[agent_id] = False
                truncations[agent_id] = False
                infos[agent_id] = {}

        # Remove dead agents
        self.agents = [agent for agent in self.agents if not terminations[agent]]

        return obs, rewards, terminations, truncations, infos

    def _get_agent(self, agent_id_str):
        id_int = int(agent_id_str.split("_")[1])
        for a in self.sim.agents:
            if a.id == id_int:
                return a
        return None

    def _get_obs(self, agent):
        hunger = 100.0 - agent.energy
        # Dummy values for crops/humans for now, these should be calculated
        return np.array([hunger, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
