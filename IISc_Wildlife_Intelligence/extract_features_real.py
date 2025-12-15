import pandas as pd
import requests
import numpy as np
import time
import math
import random

# --- CONFIGURATION ---
INPUT_FILE = "data/model_training_data.csv"
OUTPUT_FILE = "data/model_ready_data.csv"

# List of Overpass Mirrors (Load Balancing)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]

# --- 1. SETUP DATA ---
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"📂 Loaded {len(df)} rows for feature extraction.")
except FileNotFoundError:
    print("❌ Error: 'data/model_training_data.csv' not found.")
    exit()

# --- 2. ROBUST REQUEST FUNCTION ---
def make_overpass_request(query, max_retries=3):
    """Tries multiple servers with backoff logic to avoid errors."""
    for i in range(max_retries):
        # Pick a random server to spread the load
        endpoint = random.choice(OVERPASS_ENDPOINTS)
        try:
            response = requests.get(endpoint, params={'data': query}, timeout=20)
            
            # Check for Rate Limiting (429) or Server Errors (5xx)
            if response.status_code == 200:
                try:
                    return response.json()
                except:
                    # If response isn't JSON, it's likely an HTML error page
                    print(f"   ⚠️ Server returned invalid JSON. Retrying...")
            elif response.status_code == 429:
                print(f"   ⏳ Rate Limited (429). Sleeping 5s...")
                time.sleep(5)
            else:
                print(f"   ⚠️ Error {response.status_code} from {endpoint}")
                
        except Exception as e:
            print(f"   ⚠️ Connection Error: {e}")
            
        # Exponential Backoff (Wait 2s, then 4s, then 8s)
        time.sleep(2 * (i + 1))
    
    return None # Failed after all retries

# --- 3. EXTRACTOR FUNCTIONS ---

def get_elevation(lat, lon):
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        res = requests.get(url, timeout=10).json()
        return res['results'][0]['elevation']
    except:
        return 0 

def get_distance_to_nearest(lat, lon, feature_type):
    # Optimized Queries (Search smaller radius first for speed)
    radius = 5000 # 5km
    
    queries = {
        "water": f"""
            [out:json][timeout:25];
            (way["natural"="water"](around:{radius},{lat},{lon});
             relation["natural"="water"](around:{radius},{lat},{lon}););
            out center;
        """,
        "forest": f"""
            [out:json][timeout:25];
            (way["landuse"="forest"](around:{radius},{lat},{lon});
             way["natural"="wood"](around:{radius},{lat},{lon}););
            out center;
        """,
        "village": f"""
            [out:json][timeout:25];
            (node["place"="village"](around:{radius},{lat},{lon});
             way["landuse"="residential"](around:{radius},{lat},{lon}););
            out center;
        """
    }
    
    data = make_overpass_request(queries[feature_type])
    
    if not data or not data.get('elements'):
        return radius  # Assume max distance if nothing found
    
    # Calculate nearest distance
    min_dist = radius
    for element in data['elements']:
        if 'lat' in element:
            el_lat, el_lon = element['lat'], element['lon']
        elif 'center' in element:
            el_lat, el_lon = element['center']['lat'], element['center']['lon']
        else:
            continue
        
        # Fast Euclidean Calc
        d_lat = (el_lat - lat) * 111000
        d_lon = (el_lon - lon) * 111000 * math.cos(math.radians(lat))
        dist = math.sqrt(d_lat**2 + d_lon**2)
        
        if dist < min_dist:
            min_dist = dist
            
    return round(min_dist, 2)

# --- 4. EXECUTION LOOP ---
print("🚀 Starting Robust Extraction...")
print("   (Saving progress every 10 rows so you don't lose data)")

# Initialize columns
for col in ['elevation', 'dist_water', 'dist_forest', 'dist_village']:
    if col not in df.columns: df[col] = 0

for i, row in df.iterrows():
    # Skip rows that are already processed (if restarting script)
    if df.at[i, 'dist_water'] != 0 and df.at[i, 'dist_forest'] != 0:
        continue

    print(f"🌍 Row {i}/{len(df)}: {row['District']} ({row['Animal']})")
    
    # 1. Elevation
    df.at[i, 'elevation'] = get_elevation(row['lat'], row['lon'])
    
    # 2. Features (Water, Forest, Village)
    df.at[i, 'dist_water'] = get_distance_to_nearest(row['lat'], row['lon'], "water")
    df.at[i, 'dist_forest'] = get_distance_to_nearest(row['lat'], row['lon'], "forest")
    df.at[i, 'dist_village'] = get_distance_to_nearest(row['lat'], row['lon'], "village")
    
    # 3. SAVE PROGRESS EVERY 10 ROWS (Critical)
    if i % 10 == 0:
        df.to_csv(OUTPUT_FILE, index=False)
        print("   💾 Progress Saved.")
    
    # Sleep 1s to be nice to the server
    time.sleep(1)

# Final Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"🎉 DONE! All data saved to '{OUTPUT_FILE}'")