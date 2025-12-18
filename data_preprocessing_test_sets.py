import pandas as pd
import pickle
import os
import numpy as np
from tqdm import tqdm
from collections import namedtuple
from configs.data_paths import Stats_Data_Path

# Define a container for one training/test sample
Sample = namedtuple("Sample", ["features", "target", "meta"])


def get_zoned_df(appended_data):
    """
    Split the dataframe into four different ocean basins
    based on latitude and longitude.
    """

    # Arctic region
    zone_ARCTIC = appended_data.loc[appended_data['nav_lat'] > 70.0]
    zone_ARCTIC['zone'] = 'ARCTIC'
        
    # North Atlantic region
    zone_NORTH_ATLANTIC = appended_data.loc[
        (appended_data['nav_lon'] >= -75.0) & (appended_data['nav_lon'] <= 0.0)
    ]
    zone_NORTH_ATLANTIC = zone_NORTH_ATLANTIC.loc[
        (zone_NORTH_ATLANTIC['nav_lat'] >= 10) &
        (zone_NORTH_ATLANTIC['nav_lat'] <= 70)
    ]
    zone_NORTH_ATLANTIC['zone'] = 'NORTH_ATLANTIC'
    
    # Equatorial Pacific region
    zone_EQ = appended_data.loc[
        (appended_data['nav_lat'] >= -10.0) &
        (appended_data['nav_lat'] <= 10.0)
    ]
    zone_EQ_PACIFIC_1 = zone_EQ.loc[
        (zone_EQ['nav_lon'] >= 105.0) & (zone_EQ['nav_lon'] <= 180.0)
    ]
    zone_EQ_PACIFIC_2 = zone_EQ.loc[
        (zone_EQ['nav_lon'] >= -180.0) & (zone_EQ['nav_lon'] <= -80.0)
    ]
    zone_EQ_PACIFIC = pd.concat([zone_EQ_PACIFIC_1, zone_EQ_PACIFIC_2])
    zone_EQ_PACIFIC['zone'] = 'EQ_PACIFIC'
    
    # Southern Ocean region
    zone_SOUTHERN_OCEAN = appended_data.loc[appended_data['nav_lat'] <= -45]
    zone_SOUTHERN_OCEAN['zone'] = 'SOUTHERN_OCEAN'
    
    return zone_ARCTIC, zone_NORTH_ATLANTIC, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN


def preprocess_yearwise(
    year,
    frac=1,
    window_size=4,
    out_dir="/media/stu231428/1120 7818/Master_github/datasets/yearly",

    region='global',
    
    experiment_name="experiment_1"
):
    """
    Create samples for ONE specific year only.
    January automatically uses the last (window_size-1) months
    from the previous year.
    """

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------
    # 1) Load previous year + current year
    # -------------------------------------------------
    dfs = []

    for y in [year - 1, year]:
        if experiment_name == "experiment_1":
            file_path = f"/data/experiment_data/experiment_1/1/ORCA025.L46.LIM2vp.CFCSF6.MOPS.JRA.LP04-KLP002.hind_{y}_df.pkl"
        if experiment_name == "experiment_5":
            file_path = f"/data/experiment_data/experiment_5/ORCA025.L46.LIM2vp.CFCSF6.MOPS.JRA.LP04-KLP002.wind_{y}_df.pkl"

        print(f"Loading {y}")
        df = pd.read_pickle(file_path)

        # Keep only valid ocean points
        df = df[df["tmask"] == 1]

        # Drop unused columns
        df = df.drop(columns=['tmask', 'y', 'x', 'time_centered', 'e1t', 'e2t'])

        # Ensure identical spatial sampling for both years
        df_coords = df[['nav_lat', 'nav_lon']].drop_duplicates().reset_index(drop=True)
        df_coords["coord_id"] = df_coords.index
        df_coords_sampled = df_coords.sample(frac=frac, random_state=42)

        # Merge sampled coordinates back
        df = df.merge(df_coords_sampled, on=['nav_lat', 'nav_lon'], how='inner')
        df = df.sort_values(by=["coord_id", "time_counter"])

        # Select region
        if region == 'global':
            dfs.append(df)

        if region == 'Southern_Ocean':
            zone_ARCTIC, zone_NORTH_ATLANTIC, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN = get_zoned_df(df)
            dfs.append(zone_SOUTHERN_OCEAN)

        if region == 'North_Atlantic':
            zone_ARCTIC, zone_NORTH_ATLANTIC, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN = get_zoned_df(df)
            dfs.append(zone_NORTH_ATLANTIC)
        else:
            print("Region not defined properly. Please choose 'global', 'Southern_Ocean' or 'North_Atlantic'")

    # -------------------------------------------------
    # 2) Combine and sort time series
    # -------------------------------------------------
    ts = pd.concat(dfs)
    ts = ts.sort_values(by=["nav_lat", "nav_lon", "time_counter"])

    # -------------------------------------------------
    # 3) Cyclic encoding for latitude and longitude
    # -------------------------------------------------
    ts["lat_rad"] = np.deg2rad(ts["nav_lat"])
    ts["lon_rad"] = np.deg2rad(ts["nav_lon"])

    ts["sin_lat"] = np.sin(ts["lat_rad"])
    ts["cos_lat"] = np.cos(ts["lat_rad"])
    ts["sin_lon"] = np.sin(ts["lon_rad"])
    ts["cos_lon"] = np.cos(ts["lon_rad"])

    feature_columns = [
        'SST', 'SAL', 'ice_frac', 'mixed_layer_depth', 'heat_flux_down',
        'water_flux_up', 'stress_X', 'stress_Y', 'currents_X', 'currents_Y',
        'sin_lat', 'cos_lat', 'sin_lon', 'cos_lon'
    ]

    # -------------------------------------------------
    # 4) Global normalization (must come from training!)
    # -------------------------------------------------
    training_stats_dir = pickle.load(open("/data/training_set/training_stats.pkl", "rb"))
    feature_means = training_stats_dir["feature_mean"]
    feature_stds = training_stats_dir["feature_stds"]
    target_mean = training_stats_dir["target_mean"]
    target_std = training_stats_dir["target_stds"]

    ts[feature_columns] = (ts[feature_columns] - feature_means) / feature_stds
    ts['co2flux_pre'] = (ts['co2flux_pre'] - target_mean) / target_std

    # -------------------------------------------------
    # 5) Sliding window creation (only targets from current year!)
    # -------------------------------------------------
    samples = []
    grouped = ts.groupby(['nav_lat', 'nav_lon'])

    for (lat, lon), group in tqdm(grouped, desc=f"Processing {year}"):
        group = group.reset_index(drop=True)

        for i in range(window_size - 1, len(group)):
            row = group.iloc[i]

            # Ensure target belongs to the selected year
            if row["time_counter"].year != year:
                continue

            window = group.iloc[i - window_size + 1:i + 1]

            X = window[feature_columns].values
            y = row['co2flux_pre']

            meta = {
                "nav_lat": lat,
                "nav_lon": lon,
                "time_counter": row["time_counter"]
            }

            samples.append(Sample(X, y, meta))

    # -------------------------------------------------
    # 6) Save samples to disk
    # -------------------------------------------------
    out_file = f"{out_dir}/{region}_test_{year}_{experiment_name}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(samples, f)

    print(f"✅ Saved {len(samples)} samples for year {year} → {out_file}")




# -------------------------------------------------
# Configuration
# -------------------------------------------------
experiment_name = "experiment_1"
YEARS = range(2009, 2018)
REGION = "global"
out_dir = f"/data/test_sest/{experiment_name}/{REGION}"

# -------------------------------------------------
# Run preprocessing year by year
# -------------------------------------------------
for year in YEARS:
    preprocess_yearwise(
        year=year,
        out_dir=out_dir,
        experiment_name= experiment_name,
        region=REGION
    )
