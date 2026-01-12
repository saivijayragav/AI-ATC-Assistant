import os
from fastapi import FastAPI
from joblib import load
import pandas as pd
import numpy as np
import requests
from fastapi.middleware.cors import CORSMiddleware
import math


app = FastAPI(title="Air Traffic ML Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load models ONCE (very important)
# --------------------------------------------------
iso_forest = load("models/anomaly_detector.pkl")
scaler_anomaly = load("models/anomaly_scaler.pkl")

# --------------------------------------------------
# OpenSky fetch
# --------------------------------------------------
OPENSKY_URL = "https://opensky-network.org/api/states/all"


BACKUP_PATH = "backup_snapshot.csv"
rf_model = load("models/conflict_predictor.pkl")

EARTH_RADIUS_KM = 6371.0

from sklearn.neighbors import BallTree

def compute_conflict_risk(df):
    df = df.reset_index(drop=True)
    n = len(df)

    df['conflict_prob'] = 0.0

    if n < 2:
        return df
    
    # 1. Prepare coordinates for BallTree (radians)
    coords = np.deg2rad(df[['lat', 'lon']].values)
    
    # 2. Build Tree (Haversine metric)
    tree = BallTree(coords, metric='haversine')
    
    # 3. Query radius (15 km in radians)
    radius_rad = 15.0 / EARTH_RADIUS_KM
    
    # query_radius returns an array of arrays of indices
    indices_list = tree.query_radius(coords, r=radius_rad)

    for i, neighbors in enumerate(indices_list):
        f1 = df.iloc[i]
        max_prob = 0.0
        
        # neighbors includes the point itself (i), so we skip it
        for j in neighbors:
            if i == j:
                continue
            
            f2 = df.iloc[j]

            # Vertical pre-filter (still essential)
            v_dist = abs(f1['geoaltitude'] - f2['geoaltitude'])
            if v_dist > 800:
                continue

            # Calculate detailed features for the ML model
            # We re-calculate distances slightly redundantly but it's cheap for <10 neighbors
            # and safer for feature consistency
            
            # Re-use haversine func if needed, or just trust the tree gave us <15km.
            # We calculate h_dist for the feature vector regardless.
            h_dist = haversine_km(f1['lat'], f1['lon'], f2['lat'], f2['lon'])

            feature_row = pd.DataFrame([{
                'horizontal_dist': h_dist,
                'vertical_dist': v_dist,
                'relative_velocity': abs(f1['velocity'] - f2['velocity']),
                'relative_heading': abs(f1['heading'] - f2['heading']),
                'relative_vertrate': abs(f1['vertrate'] - f2['vertrate']),
                'avg_altitude': (f1['geoaltitude'] + f2['geoaltitude']) / 2,
                'avg_velocity': (f1['velocity'] + f2['velocity']) / 2,
                'converging': int(
                    min(
                        abs(f1['heading'] - f2['heading']),
                        360 - abs(f1['heading'] - f2['heading'])
                    ) > 150
                ),
                'heading_diff': min(
                    abs(f1['heading'] - f2['heading']),
                    360 - abs(f1['heading'] - f2['heading'])
                )
            }])

            prob = rf_model.predict_proba(feature_row)[0][1]
            max_prob = max(max_prob, prob)

        df.at[i, 'conflict_prob'] = max_prob

    return df


def haversine_km(lat1, lon1, lat2, lon2):
    # Keep helper for feature calculation
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))



# Global cache for persistence
LAST_VALID_DF = None


from fastapi import FastAPI, Response

# ... imports ...


# ... (rest of code) ...

def fetch_live_aircraft(bounds=None):
    global LAST_VALID_DF
    
    try:
        # Defaults to Europe if no bounds provided
        if bounds:
            lamin, lomin, lamax, lomax = bounds
        else:
            lamin, lomin, lamax, lomax = 35.0, -10.0, 70.0, 30.0

        params = {
            "lamin": lamin,
            "lomin": lomin,
            "lamax": lamax,
            "lomax": lomax
        }
        
        r = requests.get(OPENSKY_URL, params=params, timeout=5)
        r.raise_for_status()

        data = r.json()
        states = data.get("states", [])

        if not states:
            print(f"✓ OpenSky: 0 aircraft in region {bounds}")
            return pd.DataFrame(columns=[
                "icao24","callsign","origin_country","time_position",
                "last_contact","lon","lat","baroaltitude",
                "onground","velocity","heading","vertrate",
                "sensors","geoaltitude","squawk","spi","position_source"
            ]), None

        cols = [
            "icao24","callsign","origin_country","time_position",
            "last_contact","lon","lat","baroaltitude",
            "onground","velocity","heading","vertrate",
            "sensors","geoaltitude","squawk","spi","position_source"
        ]

        print(f"✓ Live OpenSky data used ({len(states)} aircraft)")
        new_df = pd.DataFrame(states, columns=cols)
        
        # Update cache
        LAST_VALID_DF = new_df
        return new_df, None

    except Exception as e:
        print(f"⚠️ OpenSky unavailable ({e})")
        
        if LAST_VALID_DF is not None:
             print(f"   -> Persisting last known valid data ({len(LAST_VALID_DF)} aircraft)")
             # Return cache + Error message
             return LAST_VALID_DF, str(e)

        print("   -> Fallback to backup snapshot")
        if os.path.exists(BACKUP_PATH):
            return pd.read_csv(BACKUP_PATH), str(e)
        else:
            raise RuntimeError("No backup snapshot available")


def clean_live_data(df, bounds=None):
    # HARD LIMIT for demo performance
    MAX_AIRCRAFT = 1200
    if len(df) > MAX_AIRCRAFT:
        df = df.sample(MAX_AIRCRAFT, random_state=42)

    df = df[
        (df['lat'].notna()) &
        (df['lon'].notna()) &
        (df['velocity'].notna()) &
        (df['heading'].notna()) &
        (df['vertrate'].notna()) &
        (df['geoaltitude'].notna()) &
        (df['onground'] == False)
    ]

    df['callsign'] = df['callsign'].fillna('UNKNOWN')

    # Filter by requested bounds if provided
    if bounds:
        lamin, lomin, lamax, lomax = bounds
        df = df[
            (df['lat'].between(lamin, lamax)) &
            (df['lon'].between(lomin, lomax))
        ]

    return df

@app.get("/airspace/snapshot")
def airspace_snapshot(response: Response, lamin: float = 35.0, lomin: float = -10.0, lamax: float = 70.0, lomax: float = 30.0):
    bounds = (lamin, lomin, lamax, lomax)
    
    df, error = fetch_live_aircraft(bounds)
    
    if error:
        # Signal to frontend that data might be stale
        response.headers["X-Error"] = error
        
    df = clean_live_data(df, bounds)

    if df.empty:
        return []

    # --- Anomaly ---
    # Ensure cols exist
    for col in ['velocity', 'geoaltitude', 'vertrate', 'heading']:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[['velocity', 'geoaltitude', 'vertrate', 'heading']]
    X_scaled = scaler_anomaly.transform(X)
    df['anomaly'] = iso_forest.predict(X_scaled)

    # --- Conflict risk ---
    df = compute_conflict_risk(df)

    return df[[
        'callsign','lat','lon','geoaltitude',
        'velocity','heading','anomaly','conflict_prob'
    ]].to_dict(orient="records")
