import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# --- IMPORT CUSTOM MODULE ---
# We wrap this in a try-except block to prevent crashing if the file is missing
try:
    from modules.prediction_model import calculate_habitat_suitability
except ImportError:
    st.error("🚨 Critical Error: Could not import 'modules.prediction_model'. Ensure the file exists in the 'modules' folder.")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="IISc Wildlife Intelligence")

# --- DATA LOADING FUNCTION ---
@st.cache_data
def load_data():
    master_df = pd.DataFrame()

    # 1. LOAD VERIFIED DATA (The one you geocoded)
    try:
        # We try standard UTF-8 first, then fallback to Latin-1 if needed
        try:
            df_verified = pd.read_csv("data/verified_geocoded.csv")
        except UnicodeDecodeError:
            df_verified = pd.read_csv("data/verified_geocoded.csv", encoding='ISO-8859-1')
            
        # Only keep rows where Geocoding was successful
        df_verified = df_verified.dropna(subset=['lat', 'lon'])
        df_verified['Source_Type'] = 'Verified Incident'
        
        # Standardize Columns
        # Ensure 'Animal' column exists (default to Sloth Bear if missing in manual data)
        if 'Animal' not in df_verified.columns:
            df_verified['Animal'] = 'Sloth Bear'
            
        master_df = pd.concat([master_df, df_verified], ignore_index=True)
    except FileNotFoundError:
        st.warning("⚠️ 'data/verified_geocoded.csv' not found. (Did you run the geocoding script?)")

    # 2. LOAD BULK DATA (The Multi-Species Simulated Data)
    try:
        df_bulk = pd.read_csv("data/bulk_data.csv")
        df_bulk['Source_Type'] = 'Historical/Simulated'
        master_df = pd.concat([master_df, df_bulk], ignore_index=True)
    except FileNotFoundError:
        st.warning("⚠️ 'data/bulk_data.csv' not found. (Did you run the bulk generator script?)")

    if master_df.empty:
        st.error("❌ No data loaded. Please run your data setup scripts first!")
        return pd.DataFrame()

    # 3. SIMULATE ENVIRONMENTAL VARIABLES (Crucial for the Model)
    # Since your Excel/Bulk data doesn't have Satellite data, we simulate it for the demo.
    np.random.seed(42) # Fixed seed so results don't jump around
    rows = len(master_df)
    
    master_df['vegetation_index'] = np.random.uniform(0.1, 0.9, rows)        # NDVI (0-1)
    master_df['proximity_to_water'] = np.random.randint(50, 2500, rows)      # Meters
    master_df['proximity_to_village'] = np.random.randint(0, 2000, rows)     # Meters
    master_df['proximity_to_rocky_outcrop'] = np.random.randint(0, 3000, rows) # Meters (Important for Bears)
    master_df['proximity_to_grassland'] = np.random.randint(0, 2000, rows)   # Meters (Important for Tigers)
    master_df['slope_angle'] = np.random.uniform(0, 45, rows)                # Degrees (Important for Elephants)
    master_df['proximity_to_agriculture'] = np.random.randint(0, 1000, rows) # Meters

    return master_df

# Load the data once
raw_df = load_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🐾 IISc Wildlife System")
st.sidebar.markdown("---")
st.sidebar.header("🔍 Analysis Settings")

# 1. SPECIES SELECTOR
# We get the list of animals actually present in your data
if not raw_df.empty and 'Animal' in raw_df.columns:
    available_animals = raw_df['Animal'].dropna().unique().tolist()
else:
    available_animals = ["Sloth Bear", "Tiger", "Elephant", "Leopard"]

selected_species = st.sidebar.selectbox("Select Target Species", available_animals)

# 2. DEMOGRAPHIC SELECTOR
demographic_options = ["General", "Solitary Male"]
if selected_species in ["Sloth Bear", "Tiger", "Leopard"]:
    demographic_options.append("Mother with Cubs")
selected_demographic = st.sidebar.selectbox("Demographic Profile", demographic_options)

# 3. SEASON SELECTOR
selected_season = st.sidebar.select_slider("Select Season", options=["Summer", "Monsoon", "Winter"])

# 4. TOGGLES
show_prediction = st.sidebar.checkbox("Show Post-Incident Escape Path", value=False)
filter_map = st.sidebar.checkbox("Filter Map by Species", value=True, help="If checked, only shows points for the selected animal.")

# --- MAIN DASHBOARD ---
st.title(f"📍 {selected_species} Presence & Risk Model")
st.markdown(f"**Research Context:** Modeling ecological probability for **{selected_demographic}** during **{selected_season}**.")

# --- DATA FILTERING ---
if not raw_df.empty:
    # Creating a copy to avoid SettingWithCopy warnings
    df = raw_df.copy()
    
    # FILTER 1: Show only selected species?
    if filter_map:
        df = df[df['Animal'] == selected_species]
        if df.empty:
            st.warning(f"No recorded incidents found for {selected_species} in the dataset.")
            st.stop()
    
    # --- MODEL APPLICATION ---
    # Apply the logic from modules/prediction_model.py
    df['probability'] = df.apply(
        lambda row: calculate_habitat_suitability(
            row, 
            selected_season, 
            selected_species, 
            selected_demographic
        ), 
        axis=1
    )

    # --- VISUALIZATION ---
    
    # Layer 1: Heatmap (Risk Zones)
    layer_heatmap = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position='[lon, lat]',
        get_weight="probability",
        radiusPixels=50,
        intensity=1.5,
        threshold=0.1,
        opacity=0.6
    )

    # Layer 2: Scatterplot (Actual Incidents)
    layer_scatter = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_color='[200, 30, 0, 160]',
        get_radius=2000, # Large radius to be visible at country scale
        pickable=True
    )

    map_layers = [layer_heatmap, layer_scatter]

    # Layer 3: Escape Path Simulation (Optional)
    if show_prediction and not df.empty:
        # We take the first point in the filtered data as a "Demo Incident"
        demo_incident = df.iloc[0]
        incident_lat, incident_lon = demo_incident['lat'], demo_incident['lon']
        
        # Simulate a Safe Zone (just slightly offset for demo purposes)
        safe_lat = incident_lat + 0.05
        safe_lon = incident_lon - 0.05
        
        layer_path = pdk.Layer(
            "ArcLayer",
            data=[{'source': [incident_lon, incident_lat], 'target': [safe_lon, safe_lat]}],
            get_source_position='source',
            get_target_position='target',
            get_source_color=[255, 0, 0],   # Red (Danger)
            get_target_color=[0, 255, 0],   # Green (Safety)
            get_width=5,
            get_tilt=15
        )
        map_layers.append(layer_path)
        st.info(f"🚨 **Escape Simulation:** Showing projected retreat path for a {selected_species} incident.")

    # Render Map
    # Auto-center map based on data
    if not df.empty:
        mid_lat = df['lat'].mean()
        mid_lon = df['lon'].mean()
        zoom_level = 5
    else:
        mid_lat, mid_lon, zoom_level = 20.59, 78.96, 4

    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=zoom_level, pitch=45)
    
    # Tooltip for interactivity
    tooltip = {
        "html": "<b>Animal:</b> {Animal}<br/>"
                "<b>Location:</b> {District}<br/>"
                "<b>Risk Probability:</b> {probability}<br/>"
                "<b>Activity:</b> {Incident details}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    st.pydeck_chart(pdk.Deck(
        layers=map_layers, 
        initial_view_state=view_state,
        tooltip=tooltip
    ))

    # --- STATISTICS PANEL ---
    st.subheader("📊 Live Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Incidents", len(df))
    c2.metric("High Risk Zones (>50%)", len(df[df['probability'] > 0.5]))
    c3.metric("Selected Season", selected_season)
    c4.metric("Species Filter", selected_species)

else:
    st.info("👋 Welcome! Please ensure your data scripts have run successfully.")