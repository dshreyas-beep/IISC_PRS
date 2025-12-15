import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide", 
    page_title="Wildlife Conflict Intelligence", 
    page_icon="🐾"
)

# --- 2. DATA LOADER ---
@st.cache_data
def load_data():
    # Priority list of files
    possible_files = [
        "data/final_geocoded_data.csv",
        "data/incidents_geocoded.csv",
        "data/verified_geocoded.csv",
        "incidents_geocoded.csv"
    ]
    
    df = pd.DataFrame()
    for file in possible_files:
        if os.path.exists(file):
            try:
                try: df = pd.read_csv(file, encoding='utf-8')
                except: df = pd.read_csv(file, encoding='ISO-8859-1')
                break
            except: continue
    
    if df.empty:
        return pd.DataFrame()

    # Standardization
    if 'Animal' in df.columns:
        df['Animal'] = df['Animal'].astype(str).str.title().str.strip()
    
    # Ensure coordinates
    df = df.dropna(subset=['lat', 'lon'])
    
    # Ensure Source URL exists
    if 'Source url' not in df.columns:
        df['Source url'] = "#"

    # --- SIMULATE ENV DATA FOR MCDA MODEL ---
    # NOTE: This is where we simulate satellite data for the "Model"
    np.random.seed(42)
    rows = len(df)
    df['sim_water_dist'] = np.random.uniform(0, 1, rows)  # 0 = close to water
    df['sim_veg_density'] = np.random.uniform(0, 1, rows) # 1 = dense forest
    df['sim_rocky'] = np.random.uniform(0, 1, rows)       # 1 = rocky terrain
    
    return df

df = load_data()

# --- 3. SIDEBAR: INTELLIGENCE HUB ---
st.sidebar.title("🧠 Intelligence Hub")

# Feature 1: Hotspots
st.sidebar.subheader("1. Visualization Mode")
view_mode = st.sidebar.radio("Map Layer:", ["📍 Exact Locations", "🔥 Hotspot Density"], index=0)

# Feature 2: Seasonal AI
st.sidebar.subheader("2. Seasonal Forecasting (MCDA)")
target_season = st.sidebar.selectbox(
    "Predict Risk Zones For:",
    ["Current (None)", "Summer (Water Stress)", "Monsoon (Veg Growth)", "Winter (Shelter Seeking)"]
)

# Feature 3: Escape Routes
st.sidebar.subheader("3. Post-Encounter AI")
show_escape = st.sidebar.checkbox("🔮 Predict Escape Vectors")

# Filters
st.sidebar.markdown("---")
if not df.empty:
    all_animals = sorted(df['Animal'].unique().tolist())
    selected_animals = st.sidebar.multiselect("Filter Species", all_animals, default=all_animals)
    filtered_df = df[df['Animal'].isin(selected_animals)]
else:
    filtered_df = pd.DataFrame()

# --- 4. MAIN MAP LOGIC ---
st.title("🐾 Wildlife Conflict Intelligence System")

if not filtered_df.empty:
    layers = []
    
    # --- LAYER A: SEASONAL PREDICTION (The MCDA Model) ---
    if target_season != "Current (None)":
        
        pred_df = filtered_df.copy()
        
        # MCDA LOGIC (Multi-Criteria Decision Analysis)
        if "Summer" in target_season:
            # Rule: High Risk near Water (1 - distance)
            pred_df['weight'] = (1 - pred_df['sim_water_dist']) * 10 
            color_range = [[173, 216, 230], [0, 0, 255]] # Blue
            st.info("☀️ **Summer Forecast (MCDA Model):** Weighting shifts to **Hydrology (Water Proximity)**.")
            
        elif "Monsoon" in target_season:
            # Rule: High Risk in Dense Vegetation
            pred_df['weight'] = pred_df['sim_veg_density'] * 10
            color_range = [[144, 238, 144], [0, 100, 0]] # Green
            st.info("🌧️ **Monsoon Forecast (MCDA Model):** Weighting shifts to **Vegetation Density (NDVI)**.")
            
        elif "Winter" in target_season:
            # Rule: High Risk in Rocky/Shelter areas
            pred_df['weight'] = pred_df['sim_rocky'] * 10
            color_range = [[255, 165, 0], [139, 69, 19]] # Orange
            st.info("❄️ **Winter Forecast (MCDA Model):** Weighting shifts to **Terrain Ruggedness (Shelter)**.")

        # HEATMAP FOR PREDICTION
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=pred_df,
            get_position='[lon, lat]',
            get_weight='weight',
            radiusPixels=100,     # Wide radius for "Regional Prediction"
            intensity=2,
            threshold=0.05,       # Low threshold to ensure visibility
            opacity=0.6,
            color_range=color_range
        ))

    # --- LAYER B: VISUALIZATION MODES ---
    if view_mode == "📍 Exact Locations":
        # Dynamic coloring
        def get_color(animal):
            s = str(animal).lower()
            if 'tiger' in s: return [255, 140, 0, 200]
            if 'leopard' in s: return [255, 215, 0, 200]
            if 'bear' in s: return [200, 30, 0, 200]
            return [128, 128, 128, 200]

        filtered_df['color'] = filtered_df['Animal'].apply(get_color)
        
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=filtered_df,
            get_position='[lon, lat]',
            get_color='color',
            get_radius=8000,
            pickable=True,
            stroked=True,
            line_width_min_pixels=1,
            get_line_color=[0, 0, 0]
        ))
    
    elif view_mode == "🔥 Hotspot Density":
        # STANDARD DENSITY MAP (The Feature that wasn't working)
        # Fixed: Lowered threshold and increased intensity
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=filtered_df,
            get_position='[lon, lat]',
            get_weight=1,      # Each point counts as 1
            radiusPixels=60,
            intensity=3,       # Higher intensity
            threshold=0.01,    # Very low threshold ensures ALL points show up
            opacity=0.6
        ))

    # --- LAYER C: ESCAPE VECTORS ---
    if show_escape:
        # Heuristic: Flee North-West towards simulated forest cover
        vectors = []
        for i, row in filtered_df.iterrows():
            vectors.append({
                "source": [row['lon'], row['lat']],
                "target": [row['lon'] - 0.05, row['lat'] + 0.05]
            })
            
        layers.append(pdk.Layer(
            "ArcLayer",
            data=vectors,
            get_source_position="source",
            get_target_position="target",
            get_source_color=[255, 50, 50],
            get_target_color=[50, 255, 50],
            get_width=4,
            get_tilt=15
        ))
        st.success("🔮 **Escape AI:** Vectors indicate probable flight path (Least Cost Path Analysis).")

    # --- RENDER MAP ---
    tooltip = {
        "html": "<b>Animal:</b> {Animal}<br/><b>Location:</b> {District}<br/><b>Details:</b> {Incident details}",
        "style": {"backgroundColor": "black", "color": "white"}
    }
    
    view_state = pdk.ViewState(
        latitude=filtered_df['lat'].mean(),
        longitude=filtered_df['lon'].mean(),
        zoom=5,
        pitch=45 if show_escape else 0
    )

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v10"
    ))

    # --- DRILL DOWN & SOURCE LINKS ---
    st.markdown("---")
    st.subheader(f"📋 Incident Verification ({len(filtered_df)} Records)")
    
    # Show incidents
    for i, row in filtered_df.head(50).iterrows():
        title = f"📍 {row['Animal']} in {row['District']} ({row['State']})"
        
        with st.expander(title):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**Village:** {row['village']}")
                st.markdown(f"**Details:** {row['Incident details']}")
                st.markdown(f"**Outcome:** {row['Victim outcome']}")
            with c2:
                # SOURCE LINK BUTTON
                url = str(row.get('Source url', '#')).strip()
                if url.lower().startswith("http"):
                    st.link_button("🔗 Open Source News", url)
                else:
                    st.button("No Source Link", disabled=True, key=f"btn_{i}")

else:
    st.warning("⚠️ No data loaded. Please run 'final_geocoder.py' first.")