"""
rewards.py
Dynamic Reward Calculator
"""
class RewardCalculator:
    def __init__(self, max_hunger=100.0, conflict_threshold=2.0):
        self.max_hunger = max_hunger
        self.conflict_threshold = conflict_threshold

    def get_reward(self, hunger, dist_to_humans, found_food):
        """Standard RL Foraging Reward Function"""
        reward = -1.0  # Cost of living
        done = False

        # Apply any stored conflict penalty from HWC incidents
        if hasattr(self, '_conflict_penalty') and self._conflict_penalty > 0:
            reward -= self._conflict_penalty
            self._conflict_penalty = 0.0  # Reset after applying

        if hunger >= self.max_hunger:
            reward -= 1000.0  # Starvation penalty
            done = True
            return reward, done

        if found_food:
            reward += 500.0  # Survival success
            done = True
            return reward, done

        if dist_to_humans < self.conflict_threshold:
            # Dynamic stress penalty based on proximity
            proximity_factor = self.conflict_threshold - dist_to_humans
            reward -= (proximity_factor * 50.0)

        return reward, done

    def add_conflict_penalty(self, penalty_amount: float):
        """Apply an immediate conflict penalty (used for HWC incidents)"""
        # This method stores a penalty that will be applied in the next reward calculation
        # For now, we'll store it as an attribute that get_reward can use
        self._conflict_penalty = getattr(self, '_conflict_penalty', 0.0) + penalty_amount