import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- CONFIGURATION ---
NUM_ROWS = 500  # Generating 500 rows for better density

# --- SCIENTIFIC HOTSPOTS (Where do these animals actually live?) ---
hotspots = {
    'Sloth Bear': [
        {'District': 'Koppal', 'State': 'Karnataka', 'Coords': [15.35, 76.15]},
        {'District': 'Banaskantha', 'State': 'Gujarat', 'Coords': [24.30, 72.20]},
        {'District': 'Balaghat', 'State': 'Madhya Pradesh', 'Coords': [21.81, 80.18]},
        {'District': 'Dhenkanal', 'State': 'Odisha', 'Coords': [20.64, 85.59]}
    ],
    'Tiger': [
        {'District': 'Mysuru', 'State': 'Karnataka', 'Coords': [12.29, 76.63]}, # Bandipur area
        {'District': 'Chandrapur', 'State': 'Maharashtra', 'Coords': [19.96, 79.29]}, # Tadoba area
        {'District': 'Nainital', 'State': 'Uttarakhand', 'Coords': [29.38, 79.46]}  # Corbett area
    ],
    'Elephant': [
        {'District': 'Kodagu', 'State': 'Karnataka', 'Coords': [12.42, 75.73]},
        {'District': 'Angul', 'State': 'Odisha', 'Coords': [20.83, 85.15]},
        {'District': 'Coimbatore', 'State': 'Tamil Nadu', 'Coords': [11.01, 76.95]}
    ],
    'Leopard': [
        {'District': 'Mumbai Suburban', 'State': 'Maharashtra', 'Coords': [19.07, 72.87]}, # Aarey Colony
        {'District': 'Mandya', 'State': 'Karnataka', 'Coords': [12.52, 76.89]},
        {'District': 'Guwahati', 'State': 'Assam', 'Coords': [26.14, 91.73]}
    ]
}

data = []

# --- GENERATION LOOP ---
for i in range(NUM_ROWS):
    # 1. Pick a Random Species
    species = random.choice(['Sloth Bear', 'Tiger', 'Elephant', 'Leopard'])
    
    # 2. Pick a Scientific Location for that Species
    # (Tigers won't appear in Gujarat, Bears won't appear in Mumbai)
    location = random.choice(hotspots[species])
    
    # 3. Generate Random Date (2020-2024)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    rand_days = random.randint(0, (end_date - start_date).days)
    date_obj = start_date + timedelta(days=rand_days)
    
    # 4. Generate Random Coordinates (District Center + Jitter)
    # Jitter spreads points by ~10km so they don't stack
    lat = location['Coords'][0] + np.random.uniform(-0.1, 0.1)
    lon = location['Coords'][1] + np.random.uniform(-0.1, 0.1)
    
    # 5. Determine Realistic Outcome
    # Elephants/Tigers cause more fatalities than Leopards
    if species in ['Tiger', 'Elephant']:
        outcome = np.random.choice(['Deceased', 'Injured'], p=[0.4, 0.6])
    else:
        outcome = np.random.choice(['Deceased', 'Injured'], p=[0.1, 0.9])

    data.append({
        'Incident-id': f"SIM-{i+1000}",
        'Date(dd/mm/yr)': date_obj.strftime("%d/%m/%Y"),
        'Animal': species,
        'village': 'Forest Fringe (Simulated)',
        'District': location['District'],
        'State': location['State'],
        'Victim age': random.randint(18, 70),
        'Victim outcome': outcome,
        'Incident details': 'Simulated data based on regional conflict probability',
        'lat': round(lat, 5),
        'lon': round(lon, 5)
    })

# --- SAVE ---
df = pd.DataFrame(data)
df.to_csv("data/bulk_data.csv", index=False)
print(f"✅ Generated {NUM_ROWS} multi-species rows in 'data/bulk_data.csv'")