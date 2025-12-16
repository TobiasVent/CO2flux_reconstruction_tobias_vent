import pandas as pd
import pickle
import os
import numpy as np
from tqdm import tqdm
from collections import namedtuple

Sample = namedtuple("Sample", ["features", "target", "meta"])


def get_zoned_df(appended_data):
    '''
    returns multiple dataframes corresponding to 4 different basins
    '''
    
    zone_ARCTIC = appended_data.loc[appended_data['nav_lat'] > 70.0]
    zone_ARCTIC['zone'] = 'ARCTIC'
        
    zone_NORTH_ATLANTIC= appended_data.loc[(appended_data['nav_lon'] >= -75.0) & (appended_data['nav_lon'] <= 0.0)]
    zone_NORTH_ATLANTIC = zone_NORTH_ATLANTIC.loc[(zone_NORTH_ATLANTIC['nav_lat'] >= 10) & (zone_NORTH_ATLANTIC['nav_lat'] <= 70)]
    zone_NORTH_ATLANTIC['zone'] = 'NORTH_ATLANTIC'
    
    zone_EQ= appended_data.loc[(appended_data['nav_lat'] >= -10.0) & (appended_data['nav_lat'] <= 10.0)]
    zone_EQ_PACIFIC_1 = zone_EQ.loc[(zone_EQ['nav_lon'] >= 105.0) & (zone_EQ['nav_lon'] <= 180.0)]
    zone_EQ_PACIFIC_2 = zone_EQ.loc[(zone_EQ['nav_lon'] >= -180.0) & (zone_EQ['nav_lon'] <= -80.0)]
    zone_EQ_PACIFIC = pd.concat([zone_EQ_PACIFIC_1, zone_EQ_PACIFIC_2])
    zone_EQ_PACIFIC['zone'] = 'EQ_PACIFIC'
    
    zone_SOUTHERN_OCEAN = appended_data.loc[appended_data['nav_lat'] <= -45]
    zone_SOUTHERN_OCEAN['zone'] = 'SOUTHERN_OCEAN'
    
    return zone_ARCTIC, zone_NORTH_ATLANTIC, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN

def preprocess_yearwise(
    year,
    frac=0.001,
    window_size=4,
    out_dir="/media/stu231428/1120 7818/Master_github/datasets/yearly",
    base_path="/media/stu231428/1120 7818/exp_5",
    feature_means = 0,
    feature_stds= 0,
    target_mean= 0,
    region = 'global',
    target_std= 0,
    experiment_name = "experiment_1"
):
    """
    Erstellt Samples NUR für ein Jahr.
    Januar nutzt automatisch die letzten (window_size-1) Monate aus dem Vorjahr.
    """

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------
    # 1) Lade Vorjahr + aktuelles Jahr
    # -------------------------------
    dfs = []

    for y in [year - 1, year]:
        file_path = f"/media/stu231428/1120 7818/Ocean Carbon/1/ORCA025.L46.LIM2vp.CFCSF6.MOPS.JRA.LP04-KLP002.hind_{y}_df.pkl"
        print(f"Loading {y}")
        df = pd.read_pickle(file_path)

        df = df[df["tmask"] == 1]
        df = df.drop(columns=['tmask', 'y', 'x', 'time_centered', 'e1t', 'e2t'])

        # gleiches Sampling für beide Jahre!
        df_coords = df[['nav_lat', 'nav_lon']].drop_duplicates().reset_index(drop=True)
        
        df_coords["coord_id"] = df_coords.index
        df_coords_sampled = df_coords.sample(frac=frac, random_state=42)

        df = df.merge(df_coords_sampled, on=['nav_lat', 'nav_lon'], how='inner')
        df = df.sort_values(by= ["coord_id", "time_counter"])
        if region == 'global':
            dfs.append(df)
        if region == 'Southern_Ocean':

            zone_ARCTIC, zone_NORTH_ATLANTIC, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN = get_zoned_df(df)
            dfs.append(zone_SOUTHERN_OCEAN)
        if region == 'North_Atlantic':
            
            zone_ARCTIC, zone_NORTH_ATLANTIC, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN = get_zoned_df(df)
            dfs.append(zone_NORTH_ATLANTIC)
        else:
          
        
            print("Region not defined properly. Please choose 'global', 'Southern_Ocean' or 'North_Atlantic' ")
            
     

    # -------------------------------
    # 2) Kombinieren & sortieren
    # -------------------------------
    ts = pd.concat(dfs)
    ts = ts.sort_values(by=["nav_lat", "nav_lon", "time_counter"])

    # -------------------------------
    # 3) Cyclic Encoding
    # -------------------------------
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

    # -------------------------------
    # 4) Normalisierung (global!)
    # -------------------------------
    # ⚠️ WICHTIG:
    # Diese Stats sollten aus dem TRAINING stammen
    # hier nur symbolisch


    ts[feature_columns] = (ts[feature_columns] - feature_means) / feature_stds


    ts['co2flux_pre'] = (ts['co2flux_pre'] - target_mean) / target_std

    # -------------------------------
    # 5) Sliding Windows (nur aktuelles Jahr!)
    # -------------------------------
    samples = []
    grouped = ts.groupby(['nav_lat', 'nav_lon'])

    for (lat, lon), group in tqdm(grouped, desc=f"Processing {year}"):
        group = group.reset_index(drop=True)

        for i in range(window_size - 1, len(group)):
            row = group.iloc[i]

            # 👉 Target MUSS im aktuellen Jahr liegen
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

    # -------------------------------
    # 6) Speichern
    # -------------------------------
    out_file = f"{out_dir}/{region}_test_{year}_{experiment_name}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(samples, f)

    print(f"✅ Saved {len(samples)} samples for year {year} → {out_file}")



file_path_stats = f'/data/stu231428/Master_Thesis/Data/stichpropen_nach_sample_from_coords/global_sample_1958_1988_0.01_percent_experiment_1.pkl'


df_for_stats = pd.read_pickle(file_path_stats)
df_for_stats["lat_rad"] = np.deg2rad(df_for_stats["nav_lat"])
df_for_stats["lon_rad"] = np.deg2rad(df_for_stats["nav_lon"])
df_for_stats["sin_lat"] = np.sin(df_for_stats["lat_rad"])
df_for_stats["cos_lat"] = np.cos(df_for_stats["lat_rad"])
df_for_stats["sin_lon"] = np.sin(df_for_stats["lon_rad"])
df_for_stats["cos_lon"] = np.cos(df_for_stats["lon_rad"])
# Define feature columns
feature_columns = [
    'SST', 'SAL', 'ice_frac', 'mixed_layer_depth', 'heat_flux_down',
    'water_flux_up', 'stress_X', 'stress_Y', 'currents_X', 'currents_Y',
    'sin_lat', 'cos_lat', 'sin_lon', 'cos_lon'  ### <<< NEW >>>
]

# # Load normalization statistics from training
# print("Loading normalization stats from training data...")
# norm_stats_path = "/data/stu231428/Transformed_data_LSTM/normalization_stats_global_0.005.pkl"
# with open(norm_stats_path, "rb") as f:
#     norm_stats = pickle.load(f)

# feature_means = pd.Series(norm_stats['mean'])
# feature_stds = pd.Series(norm_stats['std'])

# Calculate normalization statistics
print("Calculating normalization stats from data...")
feature_means = df_for_stats[feature_columns].mean()
feature_stds = df_for_stats[feature_columns].std()

# Apply normalization
print("Applying z-normalization using calculated statistics...")


target_mean = df_for_stats['co2flux_pre'].mean()
target_std = df_for_stats['co2flux_pre'].std()

print("target_mean:", target_mean)
print("target_std:", target_std)


YEARS = range(2018, 2008, -1)
REGION = "North_Atlantic"

for year in YEARS:
    preprocess_yearwise(
        year=year,
        feature_means=feature_means,
        feature_stds=feature_stds,
        target_mean=target_mean,
        target_std=target_std,
        region=REGION
    )

# preprocess_yearwise(year=2018,feature_means=feature_means,feature_stds=feature_stds,target_mean=target_mean,target_std=target_std)
# preprocess_yearwise(year=2017)
# preprocess_yearwise(year=2016)
# preprocess_yearwise(year=2015)
# preprocess_yearwise(year=2014)
# preprocess_yearwise(year=2013)