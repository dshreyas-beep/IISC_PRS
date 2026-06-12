import random


RL_SPECIES = {"elephant", "leopard", "sloth_bear", "human"}
_ML_IMPORTS = None


def _load_ml_stack():
    global _ML_IMPORTS
    if _ML_IMPORTS is None:
        from models import AnimalBrain
        from trainer import AgentTrainer
        from rewards import RewardCalculator
        _ML_IMPORTS = (AnimalBrain, AgentTrainer, RewardCalculator)
    return _ML_IMPORTS

class Agent:
    """
    Biological agent with spatial awareness, internal state, and social structure.
    Now upgraded with Live Reinforcement Learning and Conflict Rules.
    """
    def __init__(self, agent_id: int, world, species: str = "elephant"):
        self.id = agent_id
        self.species = species
        
        # Position & Velocity
        self.x = random.uniform(0, world.width)
        self.y = random.uniform(0, world.height)
        self.vx = 0.0
        self.vy = 0.0
        
        # Biological State
        self.energy = random.uniform(80.0, 100.0)
        self.thirst = random.uniform(0.0, 20.0)
        self.age = 0
        self.alive = True
        self.mode = "WANDER"
        
        # Social Structure
        self.herd_id = None
        self.is_leader = False
        self.fear_level = 0.0

        # ==========================================
        # 🛡️ CONFLICT RULES & TIMERS (UPDATED)
        # ==========================================
        # 600 ticks = 30 seconds of pure peace to establish territory
        self.adaptation_period = 600  
        self.last_conflict_tick = -999 
        # 400 ticks = 20 seconds of cooldown between fights
        self.conflict_cooldown = 400  

        # ==========================================
        # 🧠 LIVE RL BRAIN INITIALIZATION
        # ==========================================
        # State: [Hunger, dx_crops, dy_crops, dx_human, dy_human].
        # Only focal RL species receive neural policies; the expanded
        # Bannerghatta biodiversity classes keep the shared lifecycle and
        # rule-based movement without thousands of unused model instances.
        self.brain = None
        self.trainer = None
        self.rewards = None

        if species in RL_SPECIES:
            AnimalBrain, AgentTrainer, RewardCalculator = _load_ml_stack()
            self.brain = AnimalBrain(input_size=5)
        
        if species == "elephant":
            self.trainer = AgentTrainer(brain=self.brain)
            self.rewards = RewardCalculator(max_hunger=150.0, conflict_threshold=3.0)
        elif species == "leopard":
            self.trainer = AgentTrainer(brain=self.brain)
            self.rewards = RewardCalculator(max_hunger=80.0, conflict_threshold=2.0)
        elif species == "sloth_bear":
            self.trainer = AgentTrainer(brain=self.brain)
            self.rewards = RewardCalculator(max_hunger=100.0, conflict_threshold=1.5)
        elif species == "human":
            # Humans now get a brain so they can learn the peace-treaty rewards
            self.trainer = AgentTrainer(brain=self.brain)
            self.rewards = RewardCalculator(max_hunger=100.0, conflict_threshold=1.0)
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None

    def can_attack(self, target_agent, current_tick: int) -> bool:
        """Evaluates if this agent is biologically allowed to attack the target."""
        
        # 1. The Adaptation Phase: No fighting while exploring the new environment
        if current_tick < self.adaptation_period:
            return False
            
        # 2. The Cooldown: Conflicts should be rare, not continuous
        if (current_tick - self.last_conflict_tick) < self.conflict_cooldown:
            return False

        # 3. Strict Rule: Humans never attack humans
        if self.species == "human" and target_agent.species == "human":
            return False

        # 4. Sloth Bear Logic: Prioritize wandering; only attack humans
        if self.species == "sloth_bear":
            if target_agent.species != "human":
                return False

        # 5. Elephant Dominance: Allow elephant-on-elephant conflict
        # (The cooldown ensures it is rare and not immediate)
        return True

    def to_dict(self):
        """Export state for the WebSocket frontend."""
        return {
            "id": self.id,
            "species": self.species,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "weight": getattr(self, "weight", None),
            "graph_node": getattr(self, "graph_node", None),
            "state": {
                "energy": self.energy,
                "thirst": self.thirst,
                "age": self.age,
                "alive": self.alive,
                "mode": self.mode,
                "fear_level": getattr(self, "fear_level", 0.0)
            },
            "herd_id": self.herd_id,
            "is_leader": self.is_leader
        }

def make_agent(agent_id: int, world, species: str = "elephant"):
    """Create a fully initialized agent."""
    return Agent(agent_id, world, species)
