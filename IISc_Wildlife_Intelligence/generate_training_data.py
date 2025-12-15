import pandas as pd
import numpy as np
from geopy.distance import distance

# Load your verified data
df_pos = pd.read_csv("data/verified_geocoded.csv")
df_pos['Target'] = 1  # 1 = Conflict occurred here

# Generate "Pseudo-Absence" points (0 = No conflict)
# We pick random points within 5-10km of the conflict sites
# representing "Nearby areas that were safe"
data_neg = []

for _, row in df_pos.iterrows():
    # Random offset (approx 5-10km)
    lat_offset = np.random.uniform(-0.05, 0.05)
    lon_offset = np.random.uniform(-0.05, 0.05)
    
    data_neg.append({
        'lat': row['lat'] + lat_offset,
        'lon': row['lon'] + lon_offset,
        'Animal': row['Animal'],
        'District': row['District'],
        'Target': 0  # 0 = Safe Zone
    })

df_neg = pd.DataFrame(data_neg)
df_final = pd.concat([df_pos, df_neg], ignore_index=True)

# Save
df_final.to_csv("data/model_training_data.csv", index=False)
print(f"✅ Generated Training Set: {len(df_final)} rows ({len(df_pos)} Attacks, {len(df_neg)} Safe Points)")