import pandas as pd
import numpy as np

# 1. LOAD DATA (With cleanup)
try:
    df = pd.read_csv("data/verified_incidents_2025.csv", encoding='ISO-8859-1')
    # CRITICAL FIX: Drop empty rows (from Excel) and rows missing District
    df = df.dropna(subset=['District']) 
    df = df[df['District'].astype(str).str.strip() != 'nan']
    print(f"✅ Loaded {len(df)} verified rows for processing.")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# 2. PRECISE DISTRICT COORDINATES - Add any missing districts from your data here!
# Add coordinates for any district not listed that is in your verified_incidents_2025.csv
district_coords = {
    # Karnataka
    'Koppal': [15.35, 76.15], 'Mysuru': [12.29, 76.63], 'Ballari': [15.13, 76.92],
    'Belagavi': [15.84, 74.49], 'Shivamogga': [13.92, 75.56], 'Kodagu': [12.42, 75.73],
    'Mandya': [12.52, 76.89], 'Saragur': [11.97, 76.43],
    # Madhya Pradesh
    'Balaghat': [21.81, 80.18], 'Singrauli': [24.19, 82.66], 'Seoni': [22.08, 79.54],
    # Gujarat
    'Chhota Udepur': [22.30, 74.01], 'Banaskantha': [24.30, 72.20],
    # Odisha
    'Dhenkanal': [20.64, 85.59], 'Angul': [20.83, 85.15], 'Jajpur': [20.85, 86.33],
    # Chhattisgarh
    'Korba': [22.35, 82.68], 'Kanker': [20.27, 81.49],
    # Tamil Nadu
    'Coimbatore': [11.01, 76.95], 'Tirupattur': [12.49, 78.57],
    # Andhra Pradesh
    'Srikakulam': [18.30, 83.89], 'Vizianagaram': [18.10, 83.39], 'Prakasam': [15.75, 80.00], # Added Prakasam
    # Uttar Pradesh
    'Bahraich': [27.57, 81.59], 'Pilibhit': [28.64, 79.80], 'Kheri': [27.94, 80.77],
    'Dharmapur': [27.56, 81.58], # Added based on your terminal output
    # Jammu & Kashmir
    'reasi': [33.08, 74.83], 'sujan Pur': [32.39, 75.87], # Added based on your terminal output
    # Maharashtra
    'chandrapur': [19.96, 79.29], 'chimur': [20.48, 79.35], 'akapur': [19.86, 79.31], # Added based on your terminal output
    # --- ADD ANY OTHER MISSING DISTRICTS HERE ---
    # Example: 'New_District': [Lat, Lon],
}

def get_coords(row):
    dist = str(row['District']).strip()
    center = district_coords.get(dist, [20.59, 78.96]) # Default to India Center if missing
    
    # Tiny random jitter (1km) for visual separation
    lat = center[0] + np.random.uniform(-0.01, 0.01)
    lon = center[1] + np.random.uniform(-0.01, 0.01)
    return pd.Series([lat, lon])

print("🚀 Starting Instant Geocoding...")
df[['lat', 'lon']] = df.apply(get_coords, axis=1)

output_file = "data/verified_geocoded.csv"
df.to_csv(output_file, index=False)
print(f"✅ Done! Saved {len(df)} rows to '{output_file}'")