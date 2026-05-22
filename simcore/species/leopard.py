# sim/species/leopard.py

from .base import SpeciesProfile, Traits


PROFILE = SpeciesProfile(
    kind="leopard",
    traits=Traits(
        # 1) Locomotion & physical capability
        speed=6.2,
        turn_randomness=0.60,
        stamina=0.55,
        burst_power=0.80,
        path_straightness=0.55,
        obstacle_push_ability=0.35,
        climb_ability=0.90,
        slope_tolerance=0.70,

        # 2) Perception & awareness
        vision_radius=80.0,
        night_vision_bonus=0.35,
        hearing_sensitivity=0.70,
        smell_sensitivity=0.60,
        stealth_factor=0.75,

        # 3) Energetics & physiology
        max_energy=110.0,
        max_thirst=100.0,
        energy_cost=0.70,
        thirst_cost=0.70,
        digest_efficiency=0.65,
        water_dependence=0.55,
        heat_tolerance=0.55,
        min_rest_need=0.45,

        # 4) Risk, fear & decision bias
        risk_tolerance=0.55,
        risk_assessment=0.70,
        risk_avoid=0.55,
        fear=0.35,
        boldness=0.50,
        escape_bias=0.65,
        startle_reactivity=0.45,

        # 5) Space use & memory
        territory_affinity=0.75,
        site_fidelity=0.45,
        social_memory=0.55,
        boundary_crossing=0.70,

        # 6) Resource preference (topology-agnostic)
        water_seek=0.70,
        veg_seek=0.10,
        wild_prey_seek=0.90,
        crop_seek=0.30,
        livestock_seek=0.85,
        fruit_seek=0.15,
        salt_seek=0.20,
        carrion_use=0.40,
        crop_reward_sensitivity=0.35,

        # 7) Social structure & interaction
        group_affinity=0.20,
        coordination_bias=0.20,
        leadership_tendency=0.40,
        aggression=0.55,
        calf_protectiveness=0.20,
        habituation_rate=0.30,
        human_predictability_learning=0.55,

        # 8) Human landscape interaction
        settlement_avoid=0.55,
        open_area_avoid=0.60,
        threat_sensitivity=0.55,
        blockage_intolerance=0.40,

        # 9) Temporal activity bias (bias only, not a rule)
        nocturnal_bias=0.75,
        crepuscular_bias=0.60,
        diurnal_bias=0.20,

        # 10) Foraging / hunting style bias
        ambush_preference=0.80,
        chase_preference=0.30,
        prey_cache_tendency=0.65,
        curiosity=0.45,
    ),
    notes={
        "india_conflict": "Highly adaptable edge predator; livestock depredation near villages is a major driver.",
        "signature": "Stealthy ambush hunter with strong climbing + prey caching advantage.",
        "ecology": "Broad habitat tolerance; boundary crossing and habituation are relatively high vs larger cats.",
    },
)
