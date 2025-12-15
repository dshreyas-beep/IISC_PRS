import pandas as pd
import requests
import json
import time
import os
import sys

# --- CONFIGURATION ---
# Paste your working Google Cloud Key here. 
# If you don't have one working yet, the script will use the "Offline Backup" below.
API_KEY = "AIzaSyCAagQd9c2R-3ozXC2ck7Wb1fzv9PBDb30" 

# --- 1. SMART FILE FINDER (Fixes your "Not Found" error) ---
def find_input_file(filename="incidents.csv"):
    # Look in likely locations
    possible_paths = [
        filename,                       # Current folder
        f"data/{filename}",             # Data subfolder
        f"../{filename}",               # Parent folder
        r"D:\IISC_PRS\IISc_Wildlife_Intelligence\data\incidents.csv" # Absolute path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found file at: {path}")
            return path
            
    print(f"❌ Error: Could not find '{filename}' in any common folders.")
    print("   Please ensure the file is named exactly 'incidents.csv'")
    sys.exit()

CSV_PATH = find_input_file("incidents.csv")
JSON_PATH = "incidents.json"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# --- 2. OFFLINE BACKUP DICTIONARY (Plan B) ---
# This ensures you get points on the map even if the Google API denies your key.
DISTRICT_BACKUP = {
    'nashik': [19.9975, 73.7898],
    'nagina dehat': [29.4444, 78.4346], # Bijnor region
    'bijnor': [29.3724, 78.1368],
    'jalpaiguri': [26.5167, 88.7177],
    'chandrapur': [19.9615, 79.2961],
    'bahraich': [27.5705, 81.5977],
    'pilibhit': [28.6430, 79.8045],
    'mysuru': [12.2958, 76.6394],
    'srikakulam': [18.30, 83.89],
    'dhenkanal': [20.64, 85.59],
    'chhota udepur': [22.30, 74.01],
    'balaghat': [21.81, 80.18],
    'reasi': [33.08, 74.83],
    'sujan pur': [32.39, 75.87],
    'saragur': [11.97, 76.43],
    'bageshwar': [29.84, 79.77],
    'kheri': [27.95, 80.77],
    'lakhimpur kheri': [27.95, 80.77],
    'pune': [18.52, 73.85],
    'coimbatore': [11.01, 76.95]
}

# --- 3. HELPER FUNCTIONS (Tig-map Structure) ---

def canonicalize_state(name):
    """Standardizes state names."""
    if not isinstance(name, str): return ""
    name = name.strip().lower()
    mapping = {
        "maharastra": "Maharashtra",
        "uttar pradesh": "Uttar Pradesh",
        "west bengal": "West Bengal",
        "madhya pradesh": "Madhya Pradesh",
        "andhra pradesh": "Andhra Pradesh",
        "gujarat": "Gujarat",
        "karnataka": "Karnataka",
    }
    return mapping.get(name, name.title())

def clean_place_name(name):
    """Cleans up village/city names."""
    if not isinstance(name, str): return ""
    name = name.strip().title()
    return name

def build_address_levels(row):
    """Creates a priority list: Village -> District -> State"""
    state = canonicalize_state(row.get('State', ''))
    district = clean_place_name(row.get('District', ''))
    village = clean_place_name(row.get('village', ''))
    
    levels = []
    if village and district and state:
        levels.append(f"{village}, {district}, {state}, India")
    if district and state:
        levels.append(f"{district}, {state}, India")
    if state:
        levels.append(f"{state}, India")
    return levels

def get_lat_lng(address, district_name=None):
    """Tries Google API first, then falls back to Dictionary."""
    
    # 1. Try Google API
    params = {"address": address, "key": API_KEY}
    try:
        response = requests.get(GEOCODE_URL, params=params)
        data = response.json()
        
        if data["status"] == "OK":
            loc = data["results"][0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
        elif data["status"] == "REQUEST_DENIED":
            print(f"   ⚠️ API Denied. Using Offline Backup...")
        elif data["status"] == "ZERO_RESULTS":
            pass # Try next level
            
    except Exception as e:
        print(f"   ⚠️ Network Error: {e}")

    # 2. Offline Backup (Plan B)
    # If API fails, look up the district in our hardcoded list
    if district_name:
        clean_dist = district_name.strip().lower()
        if clean_dist in DISTRICT_BACKUP:
            return DISTRICT_BACKUP[clean_dist][0], DISTRICT_BACKUP[clean_dist][1]
            
    return None, None

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"📂 Loading data...")
    # Handle encoding errors common in Excel CSVs
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, encoding='ISO-8859-1')

    if "lat" not in df.columns: df["lat"] = None
    if "lon" not in df.columns: df["lon"] = None

    print(f"🚀 Processing {len(df)} incidents...")

    for i, row in df.iterrows():
        # Skip if already done
        if pd.notna(row["lat"]) and pd.notna(row["lon"]):
            continue

        address_levels = build_address_levels(row)
        district_name = str(row.get('District', ''))
        found = False
        
        for level_idx, address in enumerate(address_levels):
            # Pass district name only for backup lookup
            lat, lng = get_lat_lng(address, district_name if level_idx >= 0 else None)
            
            if lat is not None and lng is not None:
                df.at[i, "lat"] = lat
                df.at[i, "lon"] = lng
                print(f"   ✅ Found: {address[:30]}... -> {lat}, {lng}")
                found = True
                break
            
            time.sleep(0.1) 

        if not found:
            print(f"   ❌ Failed: {district_name}")

    # --- SAVE ---
    # Save CSV for debugging
    df.to_csv("data/incidents_geocoded.csv", index=False)
    print("💾 Saved CSV to 'data/incidents_geocoded.csv'")

    # Save JSON for Web App
    valid_data = df.dropna(subset=["lat", "lon"])
    # Handle NaN values for JSON
    valid_data = valid_data.where(pd.notnull(valid_data), None)
    
    records = valid_data.to_dict(orient="records")
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    
    print(f"🎉 Success! Exported {len(records)} incidents to '{JSON_PATH}'")