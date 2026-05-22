# sim/species/sloth_bear.py

from .base import SpeciesProfile, Traits


PROFILE = SpeciesProfile(
    kind="sloth_bear",
    traits=Traits(
        # 1) Locomotion & physical capability
        speed=4.0,                 # scaled later by world
        turn_randomness=0.85,
        stamina=0.4,
        burst_power=0.9,
        path_straightness=0.1,
        obstacle_push_ability=0.4,
        climb_ability=0.7,
        slope_tolerance=0.8,

        # 2) Perception & awareness
        vision_radius=30.0,        # poor vision (kept low)
        night_vision_bonus=0.7,
        hearing_sensitivity=0.6,
        smell_sensitivity=1.0,
        stealth_factor=0.4,

        # 3) Energetics & physiology
        max_energy=120.0,
        max_thirst=110.0,
        energy_cost=0.8,
        thirst_cost=0.5,
        digest_efficiency=0.8,
        water_dependence=0.6,
        heat_tolerance=0.3,
        min_rest_need=0.7,

        # 4) Risk, fear & decision bias
        risk_tolerance=0.3,
        risk_assessment=0.2,
        risk_avoid=0.8,
        fear=0.8,
        boldness=0.3,
        escape_bias=0.2,
        startle_reactivity=1.0,

        # 5) Space use & memory
        territory_affinity=0.2,
        site_fidelity=0.6,
        social_memory=0.4,
        boundary_crossing=0.5,

        # 6) Resource preference
        water_seek=0.6,
        veg_seek=0.2,
        wild_prey_seek=0.0,
        crop_seek=0.4,
        livestock_seek=0.0,
        fruit_seek=1.0,
        salt_seek=0.2,
        carrion_use=0.1,
        crop_reward_sensitivity=0.5,

        # 7) Social structure & interaction
        group_affinity=0.1,
        coordination_bias=0.0,
        leadership_tendency=0.0,
        aggression=0.9,
        calf_protectiveness=1.0,
        habituation_rate=0.1,
        human_predictability_learning=0.3,

        # 8) Human landscape interaction
        settlement_avoid=0.9,
        open_area_avoid=0.8,
        threat_sensitivity=0.9,
        blockage_intolerance=0.4,

        # 9) Temporal activity bias
        nocturnal_bias=0.9,
        crepuscular_bias=0.8,
        diurnal_bias=0.1,

        # 10) Foraging / hunting style bias
        ambush_preference=0.0,
        chase_preference=0.0,
        prey_cache_tendency=0.0,
        curiosity=0.4,
    ),
    notes={
        "india_conflict": "Human injury risk is high due to defensive aggression during surprise encounters.",
        "signature": "Olfaction-driven forager; low vision; high startle reactivity and defensive charge tendency.",
        "ecology": "Rugged-terrain tolerant, fruit/insect specialist; largely nocturnal with high cover preference.",
    },
)
