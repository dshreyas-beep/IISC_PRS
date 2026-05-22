from .base import SpeciesProfile, Traits

PROFILE = SpeciesProfile(
    kind="tiger",
    traits=Traits(
        speed=6.0,
        turn_randomness=0.45,
        vision_radius=85.0,

        max_energy=115.0,
        max_thirst=105.0,
        energy_cost=0.75,
        thirst_cost=0.70,

        risk_tolerance=0.35,
        territory_affinity=0.85,

        water_seek=0.8,
        veg_seek=0.1,
        crop_seek=0.10,
        livestock_seek=0.25,
        settlement_avoid=0.85,
        risk_avoid=0.7,

        aggression=0.75,
        fear=0.25,
        habituation_rate=0.12,
        site_fidelity=0.35,
    ),
    notes={
        "india_conflict": "Avoids settlements; conflict rarer but higher severity when triggered.",
        "signature": "High territoriality (handled via territory_affinity + later territory memory).",
    },
)
