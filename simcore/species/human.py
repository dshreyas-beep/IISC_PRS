from .base import SpeciesProfile, Traits

PROFILE = SpeciesProfile(
    kind="human",
    traits=Traits(
        speed=5.0,
        turn_randomness=0.35,
        vision_radius=60.0,

        max_energy=100.0,
        max_thirst=100.0,
        energy_cost=0.6,
        thirst_cost=0.7,

        risk_tolerance=0.25,
        territory_affinity=0.90,

        water_seek=0.6,
        veg_seek=0.0,
        crop_seek=0.7,
        livestock_seek=0.6,
        settlement_avoid=0.0,
        risk_avoid=0.3,

        aggression=0.35,
        fear=0.35,
        habituation_rate=0.25,
        site_fidelity=0.30,
    ),
    notes={
        "role_hint": "Baseline human. You can later add farmer/guard via per-agent role overrides.",
        "india_conflict": "Responds strongly near assets (crops/livestock).",
    },
)
