# sim/species/elephant.py

from .base import SpeciesProfile, Traits


PROFILE = SpeciesProfile(
    kind="elephant",
    traits=Traits(
        # 1) Locomotion & physical capability
        speed=4.0,                 # scaled later by world
        turn_randomness=0.2,
        stamina=0.8,
        burst_power=0.7,
        path_straightness=0.9,
        obstacle_push_ability=1.0,
        climb_ability=0.1,
        slope_tolerance=0.3,

        # 2) Perception & awareness
        vision_radius=60.0,        # moderate visual range
        night_vision_bonus=0.5,
        hearing_sensitivity=0.9,
        smell_sensitivity=1.0,
        stealth_factor=0.1,

        # 3) Energetics & physiology
        max_energy=140.0,
        max_thirst=160.0,
        energy_cost=0.7,
        thirst_cost=0.9,
        digest_efficiency=0.4,
        water_dependence=1.0,
        heat_tolerance=0.6,
        min_rest_need=0.3,

        # 4) Risk, fear & decision bias
        risk_tolerance=0.8,
        risk_assessment=0.9,
        risk_avoid=0.4,
        fear=0.3,
        boldness=0.7,
        escape_bias=0.4,
        startle_reactivity=0.5,

        # 5) Space use & memory
        territory_affinity=0.4,
        site_fidelity=0.9,
        social_memory=1.0,
        boundary_crossing=0.8,

        # 6) Resource preference
        water_seek=1.0,
        veg_seek=0.8,
        wild_prey_seek=0.0,
        crop_seek=1.0,
        livestock_seek=0.0,
        fruit_seek=0.9,
        salt_seek=0.6,
        carrion_use=0.0,
        crop_reward_sensitivity=1.0,

        # 7) Social structure & interaction
        group_affinity=0.9,
        coordination_bias=0.8,
        leadership_tendency=0.7,
        aggression=0.5,
        calf_protectiveness=1.0,
        habituation_rate=0.7,
        human_predictability_learning=0.9,

        # 8) Human landscape interaction
        settlement_avoid=0.6,
        open_area_avoid=0.5,
        threat_sensitivity=0.8,
        blockage_intolerance=0.9,

        # 9) Temporal activity bias
        nocturnal_bias=0.6,
        crepuscular_bias=0.9,
        diurnal_bias=0.3,

        # 10) Foraging / hunting style bias
        ambush_preference=0.0,
        chase_preference=0.0,
        prey_cache_tendency=0.0,
        curiosity=0.7,
    ),
    notes={
        "india_conflict": "High crop attraction; conflict escalates when traditional routes are blocked.",
        "signature": "Highly social, memory-driven megaherbivore with extreme water and crop dependence.",
        "ecology": "Movement tightly coupled to water, seasonal crops, and learned human patterns.",
    },
)
