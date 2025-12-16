import pandas as pd
import pickle
import os
import numpy as np
from tqdm import tqdm
from collections import namedtuple

Sample = namedtuple("Sample", ["features", "target", "meta"])




def preprocess_data(feature_mean = 0,feature_stds =0,target_mean = 0,target_std = 0,  frac = 0.0001, out_dir = "/media/stu231428/1120 7818/Master_github/datasets",range_start=1958, range_end=1988,training=True):
    print("feature_mean", feature_mean, "feature_stds", feature_stds, "target_mean", target_mean, "target_std", target_std)

    time_series = pd.DataFrame()


    for i in range(range_start,range_end+1):
        print(i)
        file_path = f'/media/stu231428/1120 7818/exp_5/ORCA025.L46.LIM2vp.CFCSF6.MOPS.JRA.LP04-KLP002.wind_{i}_df.pkl'
        #file_path = f'/media/stu231428/1120 7818/exp_3/ORCA025.L46.LIM2vp.CFCSF6.MOPS.JRA.LP11-KLP009.clim_{i}_df.pkl'
        df = pd.read_pickle(file_path)
        df = df[df["tmask"]==1]
        df = df.drop(columns= ['tmask','y','x','time_centered','e1t','e2t'])
        df_coords = df[['nav_lat','nav_lon']].drop_duplicates().reset_index(drop=True)

        df_coords["coord_id"] = df_coords.index
        df_coords_sampled = df_coords.sample(frac=frac, random_state=42)
        df = df.merge(df_coords_sampled, on=['nav_lat', 'nav_lon'], how='inner')
        df = df.sort_values(by= ["coord_id", "time_counter"])
        time_series = pd.concat([time_series,df])



    time_series = time_series.sort_values(by=["nav_lat", "nav_lon", "time_counter"])

    print("Applying cyclic encoding for latitude and longitude...")
    time_series["lat_rad"] = np.deg2rad(time_series["nav_lat"])
    time_series["lon_rad"] = np.deg2rad(time_series["nav_lon"])

    # Compute sine and cosine
    time_series["sin_lat"] = np.sin(time_series["lat_rad"])
    time_series["cos_lat"] = np.cos(time_series["lat_rad"])
    time_series["sin_lon"] = np.sin(time_series["lon_rad"])
    time_series["cos_lon"] = np.cos(time_series["lon_rad"])

    feature_columns = [
        'SST', 'SAL', 'ice_frac', 'mixed_layer_depth', 'heat_flux_down',
        'water_flux_up', 'stress_X', 'stress_Y', 'currents_X', 'currents_Y',
        'sin_lat', 'cos_lat', 'sin_lon', 'cos_lon' 
    ]


    print("Calculating normalization stats from data...")
    if training:
        feature_mean = time_series[feature_columns].mean()
        feature_stds = time_series[feature_columns].std()
    print("Applying z-normalization using calculated statistics...")
    time_series[feature_columns] = (time_series[feature_columns] - feature_mean) / feature_stds
    if training:
        target_mean = time_series['co2flux_pre'].mean()
        target_std = time_series['co2flux_pre'].std()
    # Normalize target
    target_mean = time_series['co2flux_pre'].mean()
    target_std = time_series['co2flux_pre'].std()
    time_series['co2flux_pre'] = (time_series["co2flux_pre"] - target_mean) / target_std
    


    # --------------------------------------------------------------
    # 4️⃣ Generate sliding windows
    # --------------------------------------------------------------
    window_size = 4
   
    all_samples = []

    print("Generating features and targets from data...")
    grouped = time_series.groupby(['nav_lat', 'nav_lon'])

    for (lat, lon), group in tqdm(grouped, desc="Processing locations"):
        group = group.reset_index(drop=True)
        for i in range(window_size - 1, len(group)):
            window = group.iloc[i - window_size + 1:i + 1]

            X = window[feature_columns].values
            y = window.iloc[-1]['co2flux_pre']

            sample_info = {
                'nav_lat': lat,
                'nav_lon': lon,
                'time_counter': window.iloc[-1]['time_counter']
            }

            sample = Sample(X, y, sample_info)
            all_samples.append(sample)

    # --------------------------------------------------------------
    # 5️⃣ Save results
    # --------------------------------------------------------------
    print("Saving processed samples...")



    # os.makedirs(out_dir, exist_ok=True)
    # out_file = os.path.join(
    #     out_dir,
    #     "trainin_set.pkl"
    # )
    with open(out_dir, "wb") as f:
        pickle.dump(all_samples, f)
    print("✅ Done! Saved file:", out_dir)

    return feature_mean, feature_stds, target_mean, target_std




featre_mean, feature_stds, target_mean, target_std = preprocess_data(frac=0.0001, out_dir="/media/stu231428/1120 7818/Master_github/datasets/training_set.pkl",range_start=1958, range_end=1959,training=True)



featre_mean, feature_stds, target_mean, target_std = preprocess_data(featre_mean, feature_stds, target_mean, target_std, frac=0.0001, out_dir="/media/stu231428/1120 7818/Master_github/datasets/validation_set.pkl", range_start=1989, range_end=1989,training=False)



