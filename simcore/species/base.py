# sim/species/base.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any


# Canonical trait schema (v1): every species ultimately conforms to these keys.
# Values are floats. Most are normalized 0..1 except noted (e.g., speed, vision_radius, max_energy).
CANONICAL_TRAITS: Set[str] = {
    # 1) Locomotion & physical capability
    "speed",
    "turn_randomness",
    "stamina",
    "burst_power",
    "path_straightness",
    "obstacle_push_ability",
    "climb_ability",
    "slope_tolerance",

    # 2) Perception & awareness
    "vision_radius",
    "night_vision_bonus",
    "hearing_sensitivity",
    "smell_sensitivity",
    "stealth_factor",

    # 3) Energetics & physiology
    "max_energy",
    "max_thirst",
    "energy_cost",
    "thirst_cost",
    "digest_efficiency",
    "water_dependence",
    "heat_tolerance",
    "min_rest_need",

    # 4) Risk, fear & decision bias
    "risk_tolerance",
    "risk_assessment",
    "risk_avoid",
    "fear",
    "boldness",
    "escape_bias",
    "startle_reactivity",

    # 5) Space use & memory
    "territory_affinity",
    "site_fidelity",
    "social_memory",
    "boundary_crossing",

    # 6) Resource preference (topology-agnostic)
    "water_seek",
    "veg_seek",
    "wild_prey_seek",
    "crop_seek",
    "livestock_seek",
    "fruit_seek",
    "salt_seek",
    "carrion_use",
    "crop_reward_sensitivity",

    # 7) Social structure & interaction
    "group_affinity",
    "coordination_bias",
    "leadership_tendency",
    "aggression",
    "calf_protectiveness",
    "habituation_rate",
    "human_predictability_learning",

    # 8) Human landscape interaction
    "settlement_avoid",
    "open_area_avoid",
    "threat_sensitivity",
    "blockage_intolerance",

    # 9) Temporal activity bias
    "nocturnal_bias",
    "crepuscular_bias",
    "diurnal_bias",

    # 10) Foraging / hunting style bias
    "ambush_preference",
    "chase_preference",
    "prey_cache_tendency",
    "curiosity",
}


# Stable ordering for vectorization (important for ML)
CANONICAL_TRAITS_ORDER: List[str] = sorted(CANONICAL_TRAITS)


def _default_trait_values() -> Dict[str, float]:
    """
    Safe defaults so legacy species files (with fewer fields) won't crash.
    These are *neutral* defaults, not species-specific.
    """
    d: Dict[str, float] = {}

    # Non-normalized / scale-like defaults
    d["speed"] = 4.0
    d["vision_radius"] = 60.0
    d["max_energy"] = 100.0
    d["max_thirst"] = 100.0
    d["energy_cost"] = 0.6
    d["thirst_cost"] = 0.6

    # Normalized defaults (neutral)
    for k in CANONICAL_TRAITS:
        if k in d:
            continue
        d[k] = 0.5

    return d


DEFAULT_TRAITS: Dict[str, float] = _default_trait_values()


@dataclass(frozen=True)
class Traits:
    """
    Canonical trait container.
    - Accepts partial kwargs: missing keys are filled from DEFAULT_TRAITS.
    - Extra/unknown keys error out (prevents silent typos).
    - Provides a stable vector() method for ML.
    """
    values: Dict[str, float]

    def __init__(self, **kwargs: float):
        extra = set(kwargs.keys()) - CANONICAL_TRAITS
        if extra:
            raise ValueError(f"Unknown trait keys: {sorted(extra)}")

        merged = dict(DEFAULT_TRAITS)
        merged.update(kwargs)

        # Ensure every canonical key exists
        missing = CANONICAL_TRAITS - set(merged.keys())
        if missing:
            # This should never happen because we start with DEFAULT_TRAITS
            raise ValueError(f"Internal error: missing default traits: {sorted(missing)}")

        object.__setattr__(self, "values", merged)

    def get(self, key: str) -> float:
        if key not in CANONICAL_TRAITS:
            raise KeyError(f"Trait '{key}' is not in canonical schema.")
        return float(self.values[key])

    def as_dict(self) -> Dict[str, float]:
        return dict(self.values)

    def vector(self, order: Optional[List[str]] = None) -> List[float]:
        """
        Returns trait values in stable order for model input.
        """
        ord_ = order if order is not None else CANONICAL_TRAITS_ORDER
        # Validate custom order if provided
        unknown = set(ord_) - CANONICAL_TRAITS
        if unknown:
            raise ValueError(f"Order contains unknown traits: {sorted(unknown)}")
        return [float(self.values[k]) for k in ord_]


@dataclass(frozen=True)
class SpeciesProfile:
    kind: str
    traits: Traits
    notes: Optional[Dict[str, Any]] = None
