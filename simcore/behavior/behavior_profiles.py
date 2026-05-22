# simcore/behavior/behavior_profiles.py

def get_ecological_modifiers(species: str, topology: str) -> dict:
    """
    Returns the biological and topological weight modifiers for a given species.
    This acts as the mathematical DNA for utility calculations.
    """
    # 1. BASELINE TRAITS (If topology is neutral)
    mods = {
        "water_urgency": 1.0,
        "food_urgency": 1.0,
        "fear_multiplier": 1.0,
        "slope_cost": 1.0,
        "speed_mult": 1.0,
    }

    # 2. SPECIES-SPECIFIC PHYSICAL TRAITS
    if species == "elephant":
        mods["slope_cost"] = 3.0  # Heavy, avoids steep elevation
    elif species == "leopard":
        mods["slope_cost"] = 0.5  # Agile climber
    elif species == "sloth_bear":
        mods["slope_cost"] = 1.2
    elif species == "human":
        mods["slope_cost"] = 1.5

    # 3. TOPOLOGY-DRIVEN BEHAVIOR SHIFTS
    if topology == "edge_farmland":
        if species == "elephant":
            mods["food_urgency"] = 1.4      # High crop attraction
            mods["fear_multiplier"] = 1.2   # Cautious of nearby settlements
        elif species == "leopard":
            mods["food_urgency"] = 1.6      # Hunts livestock near edges
        elif species == "human":
            mods["fear_multiplier"] = 1.5   # High guard/reactive state

    elif topology == "river_valley" or topology == "river_basin":
        if species == "elephant":
            mods["water_urgency"] = 0.5     # Abundant water, low panic
            mods["food_urgency"] = 0.4      # Less crop raiding needed
        elif species == "leopard":
            mods["speed_mult"] = 0.7        # Ambush predator behavior (moves less)

    elif topology == "scarce_water" or topology == "drought":
        # Conflict probability spikes significantly
        mods["water_urgency"] = 2.5
        if species == "elephant":
            mods["food_urgency"] = 1.5      # Forced to raid crops for moisture/food
            mods["fear_multiplier"] = 0.6   # Desperation overrides fear
        elif species == "leopard":
            mods["food_urgency"] = 2.0      # Prey scarcity drives aggressive hunting

    elif topology == "dense_forest" or topology == "forest_core":
        if species == "elephant":
            mods["fear_multiplier"] = 0.2   # Low conflict, feel safe
            mods["food_urgency"] = 0.5      # Natural foraging is sufficient

    elif topology == "corridor_fence" or topology == "fragmented":
        # Fragmented landscapes increase stress and speed
        mods["speed_mult"] = 1.2
        mods["fear_multiplier"] = 1.8       # High defensive reactivity

    elif topology == "bannerghatta" or topology == "bannerghatta_bnp":
        # Bengaluru-edge protected forest: narrow habitat, farms/villages on the
        # boundary, and dry-season pressure around tanks and corridor crossings.
        mods["water_urgency"] = 1.25
        mods["fear_multiplier"] = 1.35
        if species == "elephant":
            mods["food_urgency"] = 1.65
            mods["water_urgency"] = 1.55
            mods["fear_multiplier"] = 0.85  # crop pressure can override caution
        elif species == "leopard":
            mods["food_urgency"] = 1.45
            mods["speed_mult"] = 0.85       # more edge/ambush than long chases
        elif species == "sloth_bear":
            mods["fear_multiplier"] = 1.65  # surprise encounters escalate quickly
        elif species == "human":
            mods["fear_multiplier"] = 1.75
            mods["food_urgency"] = 1.20

    return mods
