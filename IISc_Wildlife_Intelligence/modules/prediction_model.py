import numpy as np

def calculate_habitat_suitability(row, season, species, demographic_profile="General"):
    score = 0
    
    # 🐻 SLOTH BEAR LOGIC
    if species == "Sloth Bear":
        if row['vegetation_index'] > 0.5: score += 20
        if demographic_profile == "Mother with Cubs":
            if row['proximity_to_rocky_outcrop'] < 300: score += 60 # Denning
            if row['proximity_to_village'] < 800: score -= 40       # Avoid humans
        else:
            if row['proximity_to_agriculture'] < 300: score += 30   # Crop Raiding
            if row['proximity_to_rocky_outcrop'] < 1000: score += 20

    # 🐅 TIGER LOGIC
    elif species == "Tiger":
        if row['vegetation_index'] > 0.7: score += 40
        if row['proximity_to_grassland'] < 500: score += 30
        if season == "Summer" and row['proximity_to_water'] < 500: score += 30

    # 🐆 LEOPARD LOGIC
    elif species == "Leopard":
        if row['vegetation_index'] > 0.4: score += 20
        if row['proximity_to_village'] < 500: score += 40           # Tolerates humans
        if row['proximity_to_rocky_outcrop'] < 500: score += 20

    # 🐘 ELEPHANT LOGIC
    elif species == "Elephant":
        if row['proximity_to_water'] < 1000: score += 50
        if row['slope_angle'] > 30: score -= 100                    # Cannot climb
        else: score += 20
        if season == "Winter" and row['proximity_to_agriculture'] < 200: score += 40

    # Normalize
    final_prob = max(0, min(score, 100)) / 100.0
    return final_prob