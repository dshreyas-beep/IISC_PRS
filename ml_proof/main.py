"""
main.py
Multi-Species RL Training & Testing Loop
"""
import torch
import random
import math
from models import AnimalBrain 
from trainer import AgentTrainer
from logger import ConflictLogger
from rewards import RewardCalculator

NUM_TRAIN_EPISODES = 2000
NUM_TEST_EPISODES = 100
MAX_TICKS = 200

# --- Initialize AI for 3 distinct species ---
elephant_brain = AnimalBrain(5, 4)
elephant_trainer = AgentTrainer(elephant_brain, learning_rate=0.001)
elephant_rewards = RewardCalculator(max_hunger=150.0, conflict_threshold=3.0)

leopard_brain = AnimalBrain(5, 4)
leopard_trainer = AgentTrainer(leopard_brain, learning_rate=0.002)
leopard_rewards = RewardCalculator(max_hunger=80.0, conflict_threshold=2.0)

bear_brain = AnimalBrain(5, 4)
bear_trainer = AgentTrainer(bear_brain, learning_rate=0.0015)
bear_rewards = RewardCalculator(max_hunger=100.0, conflict_threshold=1.5)

logger = ConflictLogger("conflict_log.csv")

def dist(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_state(data, water_pos, crops_pos, human_pos):
    d_water = dist(data["pos"], water_pos)
    d_crops = dist(data["pos"], crops_pos)
    d_humans = dist(data["pos"], human_pos)
    human_dens = 0.8 if d_humans < 5.0 else 0.1 
    return torch.tensor([data["hunger"], data["stress"], d_water, d_crops, human_dens], dtype=torch.float32)

print("--- Phase 1: Multi-Species Training ---")
for ep in range(NUM_TRAIN_EPISODES):
    # Randomize map to prevent memorization
    h_pos = [random.uniform(4, 6), random.uniform(4, 6)]
    w_pos = [random.uniform(12, 18), random.uniform(12, 18)]
    c_pos = [random.uniform(1, 3), random.uniform(1, 3)]
    
    animals = {
        "Elephant": {"pos": [random.uniform(8, 12), random.uniform(8, 12)], "hunger": 0.0, "stress": 0.0, "done": False, "trainer": elephant_trainer, "rewards": elephant_rewards, "brain": elephant_brain},
        "Leopard": {"pos": [random.uniform(8, 12), random.uniform(8, 12)], "hunger": 0.0, "stress": 0.0, "done": False, "trainer": leopard_trainer, "rewards": leopard_rewards, "brain": leopard_brain},
        "Sloth_Bear": {"pos": [random.uniform(8, 12), random.uniform(8, 12)], "hunger": 0.0, "stress": 0.0, "done": False, "trainer": bear_trainer, "rewards": bear_rewards, "brain": bear_brain}
    }

    for tick in range(MAX_TICKS):
        if all(a["done"] for a in animals.values()): break
        
        for name, data in animals.items():
            if data["done"]: continue
            
            state = get_state(data, w_pos, c_pos, h_pos)
            action = data["trainer"].choose_action(state)
            
            if action == 0: data["pos"][1] += 1.0
            elif action == 1: data["pos"][1] -= 1.0
            elif action == 2: data["pos"][0] += 1.0
            elif action == 3: data["pos"][0] -= 1.0
            
            data["hunger"] += 1.0 
            
            d_human = dist(data["pos"], h_pos)
            found_food = dist(data["pos"], c_pos) < 1.0
            reward, done = data["rewards"].get_reward(data["hunger"], d_human, found_food)
            data["done"] = done
            
            next_state = get_state(data, w_pos, c_pos, h_pos)
            
            # Log exact moment of conflict
            if d_human < data["rewards"].conflict_threshold:
                with torch.no_grad():
                    q_vals = data["brain"](state).tolist()
                actions = ["North", "South", "East", "West"]
                logger.log_conflict(tick, name, data["pos"], state.tolist(), q_vals, actions[action])
                
            data["trainer"].learn(state, action, reward, next_state, done)

    if ep % 500 == 0:
        print(f"Ep {ep} | Elephant Epsilon: {elephant_trainer.epsilon:.3f}")

print("Training Complete. Generating Testing Data...")

print("\n--- Phase 2: Testing Generalization ---")
elephant_trainer.epsilon, leopard_trainer.epsilon, bear_trainer.epsilon = 0.0, 0.0, 0.0
elephant_brain.eval(); leopard_brain.eval(); bear_brain.eval()

results = {s: {"success": 0, "starved": 0} for s in ["Elephant", "Leopard", "Sloth_Bear"]}

for _ in range(NUM_TEST_EPISODES):
    h_pos, w_pos, c_pos = [random.uniform(2, 8), random.uniform(2, 8)], [random.uniform(10, 20), random.uniform(10, 20)], [random.uniform(0, 5), random.uniform(0, 5)]
    
    test_animals = {
        "Elephant": {"pos": [random.uniform(5, 15), random.uniform(5, 15)], "hunger": 0.0, "stress": 0.0, "done": False, "trainer": elephant_trainer, "rewards": elephant_rewards},
        "Leopard": {"pos": [random.uniform(5, 15), random.uniform(5, 15)], "hunger": 0.0, "stress": 0.0, "done": False, "trainer": leopard_trainer, "rewards": leopard_rewards},
        "Sloth_Bear": {"pos": [random.uniform(5, 15), random.uniform(5, 15)], "hunger": 0.0, "stress": 0.0, "done": False, "trainer": bear_trainer, "rewards": bear_rewards}
    }
    
    for _ in range(MAX_TICKS):
        if all(a["done"] for a in test_animals.values()): break
        for name, data in test_animals.items():
            if data["done"]: continue
            state = get_state(data, w_pos, c_pos, h_pos)
            with torch.no_grad(): action = data["trainer"].choose_action(state)
            
            if action == 0: data["pos"][1] += 1.0
            elif action == 1: data["pos"][1] -= 1.0
            elif action == 2: data["pos"][0] += 1.0
            elif action == 3: data["pos"][0] -= 1.0
            
            data["hunger"] += 1.0
            if dist(data["pos"], c_pos) < 1.0:
                results[name]["success"] += 1
                data["done"] = True
            elif data["hunger"] >= data["rewards"].max_hunger:
                results[name]["starved"] += 1
                data["done"] = True

for sp, counts in results.items():
    print(f"{sp} -> Survival: {(counts['success']/NUM_TEST_EPISODES)*100}% | Starved: {(counts['starved']/NUM_TEST_EPISODES)*100}%")