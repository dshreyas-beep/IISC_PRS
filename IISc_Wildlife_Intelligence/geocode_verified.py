import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# 1. SETUP
# We need a unique user_agent so OpenStreetMap doesn't block us
geolocator = Nominatim(user_agent="iisc_wildlife_project_student_v2")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

print("📂 Loading 'verified_incidents_2025.csv'...")

# --- THE FIX IS HERE ---
try:
    # We add encoding='ISO-8859-1' to handle Windows special characters (like smart quotes)
    df = pd.read_csv("data/verified_incidents_2025.csv", encoding='ISO-8859-1') 
    print("✅ File loaded successfully!")
except FileNotFoundError:
    print("❌ Error: File not found. Please make sure:")
    print("   1. You have a folder named 'data' inside 'IISc_Wildlife_Intelligence'")
    print("   2. Your file 'verified_incidents_2025.csv' is inside that 'data' folder.")
    exit()
except UnicodeDecodeError:
    # If ISO-8859-1 fails, try cp1252 (another common Windows format)
    print("⚠️ ISO-8859-1 failed, trying cp1252...")
    df = pd.read_csv("data/verified_incidents_2025.csv", encoding='cp1252')

# 2. DEFINE SEARCH LOGIC
def get_lat_lon(row):
    try:
        # Check if columns exist, otherwise skip
        village = str(row['village']) if 'village' in row else ""
        district = str(row['District']) if 'District' in row else ""
        state = str(row['State']) if 'State' in row else ""

        # Construct query
        query = f"{village}, {district}, {state}"
        print(f"🔍 Looking up: {query}...")
        location = geolocator.geocode(query)
        
        # Fallback
        if location is None:
            print(f"   ⚠️ Exact village not found. Trying District: {district}")
            query = f"{district}, {state}"
            location = geolocator.geocode(query)
            
        if location:
            return pd.Series([location.latitude, location.longitude])
        else:
            return pd.Series([None, None])
    except Exception as e:
        print(f"Error lookup: {e}")
        return pd.Series([None, None])

# 3. EXECUTE
print("🚀 Starting Geocoding (This might take 1-2 minutes)...")
df[['lat', 'lon']] = df.apply(get_lat_lon, axis=1)

# 4. SAVE
output_file = "data/verified_geocoded.csv"
df.to_csv(output_file, index=False)
print(f"✅ Done! Geocoded data saved to '{output_file}'")