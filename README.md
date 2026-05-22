WildSim Unity ML Backend (MVP)
=============================

Generates multi-species, multi-topology rollouts for training a trait-conditioned policy.

Live Web Simulation
-------------------

Run the spectating dashboard:

  uvicorn viz_web.server:app --reload

The default topology is now `bannerghatta_osm`, which loads a cached
OpenStreetMap/OSMnx Bannerghatta National Park movement graph and a 240x720
raster perception layer. A calibrated `bannerghatta` proxy is kept as a fallback,
but the OSM mode constrains agents to real OSM roads, tracks, footways, and paths
clipped to the injected Bannerghatta bbox.

The Bannerghatta constants live in `simcore/data/bannerghatta_ecology.py`:

  BANNERGHATTA_BBOX = north 12.8229972, south 12.3443525, west 77.4828815, east 77.6368603
  BANNERGHATTA_DECLARED_AREA_SQ_KM = 260.51
  BANNERGHATTA_DECLARED_AREA_ACRES = 65127.5
  BANNERGHATTA_BASELINE_SPECIES_COUNTS = 100 species classes / 2300 animals

`simcore/rollout/multi_sim.py` injects those counts automatically when topology
is `bannerghatta_osm` and no override species_counts is passed. Suvarnamukhi
stream OSM way IDs, fallback stream coordinates, OSM water polygons, and BBP
recharge-pit clusters are converted into `water_nodes` and snapped to graph
nodes during initialization.

Real Bannerghatta Map
---------------------

Fetch and cache OpenStreetMap features for the Bannerghatta region:

  python tools\fetch_bannerghatta_osm.py --refresh

This writes:

  data/bannerghatta_osm_layers.npz
  data/bannerghatta_overpass_raw.json
  data/bannerghatta_osm_metadata.json
  data/bannerghatta_graph.json

The dashboard default is `bannerghatta_osm`. If the cache is missing, the sim
builds from the raw Overpass cache when available, otherwise it returns an empty
graph and keeps the raster/proxy surfaces alive. OpenStreetMap data requires
attribution: OpenStreetMap contributors.

Run:
  python generate_rollouts.py --out rollouts_multi_v1.npz --steps 600 --episodes_per_combo 10 --seed0 1

NPZ keys:
  obs, actions, rewards, dones, species, topology, episode
