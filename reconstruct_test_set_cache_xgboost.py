import os
import pickle
import pandas as pd
import numpy as np
from collections import namedtuple
import xgboost as xgb
from configs.data_paths import Stats_Data_Path
from configs.xgboost_config import DATA_PATHS_XGBoost
Sample = namedtuple("Sample", ["features", "target", "meta"])

# ==========================
# SETTINGS
# ==========================
# Define the year range to reconstruct
START_YEAR = 2012
END_YEAR = 2018

region = "global"
experiment = "experiment_1"

# Test set path pattern (per year)
TEST_PATH_PATTERN = (
    "/media/stu231428/1120 7818/Master_github/datasets/yearly/"
    "{region}_test_{year}_{experiment}.pkl"
)

# Output cache directory and filename pattern (per year)
CACHE_DIR = "/media/stu231428/1120 7818/Master_github/datasets/cache"
OUT_PATTERN = os.path.join(CACHE_DIR, "XGBoost_{region}_reconstruction_{year}_{experiment}.pkl")


def flatten_dyn_plus_static(x_window, n_dyn=10):
    """
    Flatten a sliding window into a single feature vector:
    - dynamic features (first n_dyn columns) are flattened across time
    - static features (remaining columns) are taken once (from the first timestep)
    """
    x_dyn = x_window[:, :n_dyn].reshape(-1)   # e.g. 4*10 = 40
    x_static = x_window[0, n_dyn:]            # e.g. 4 static features
    return np.concatenate([x_dyn, x_static]).astype(np.float32)


def reconstruct_one_year(year, target_mean, target_std):
    """
    Reconstruct CO2 flux for one specific year:
    - load year test samples
    - build X matrix
    - predict with XGBoost model
    - denormalize
    - save as a pickle dataframe
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = OUT_PATTERN.format(year=year, region=region, experiment=experiment)

    # ---------------------------
    # Load year-specific test samples
    # ---------------------------
    test_path = TEST_PATH_PATTERN.format(year=year, region=region, experiment=experiment)
    print("Using test_path:", test_path)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f" Test set for year={year} not found: {test_path}")

    print(f"Loading test samples for {year}: {test_path}")
    with open(test_path, "rb") as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples):,} samples")

    # ---------------------------
    # Build feature matrix
    # ---------------------------
    print(" Building feature matrix for XGBoost...")
    X = np.stack([flatten_dyn_plus_static(s.features) for s in samples], axis=0)
    y = np.array([s.target for s in samples], dtype=np.float32)
    metas = [s.meta for s in samples]
    print(f" X shape: {X.shape}, y shape: {y.shape}")

    # ---------------------------
    # Load model + predict
    # ---------------------------
    print("🤖 Loading XGBoost model...")
    with open(DATA_PATHS_XGBoost["model_out"], "rb") as f:
        model = pickle.load(f)
    print("Model loaded")

    print(" Predicting...")
    preds = model.predict(X).astype(np.float32)

    # Denormalize
    preds_denorm = preds * target_std + target_mean
    y_denorm = y * target_std + target_mean

    # ---------------------------
    # Build dataframe
    # ---------------------------
    print(" Building DataFrame...")
    rows = []
    n = len(samples)

    for i in range(n):
        rows.append({
            "nav_lat": metas[i]["nav_lat"],
            "nav_lon": metas[i]["nav_lon"],
            "time_counter": metas[i]["time_counter"],
            "co2flux_pre": float(y_denorm[i]),
            "reconstructed_co2flux_pre": float(preds_denorm[i]),
        })

        if i == 0 or (i + 1) % 200000 == 0 or (i + 1) == n:
            print(f"  → {i+1:,}/{n:,} rows")

    df = (pd.DataFrame(rows)
          .sort_values(["nav_lat", "nav_lon", "time_counter"])
          .reset_index(drop=True))

    print(f"📊 Final DataFrame shape: {df.shape}")
    print(df.head())

    # ---------------------------
    # Save
    # ---------------------------
    df.to_pickle(out_path)
    print(f"\n Saved reconstruction cache for {year} to:\n{out_path}")


def main():
    """
    Main entry point:
    - validate year range
    - load training normalization stats
    - loop over years and reconstruct each year
    """
    if END_YEAR < START_YEAR:
        raise ValueError(f"Invalid range: START_YEAR={START_YEAR} > END_YEAR={END_YEAR}")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # ---------------------------
    # Normalization stats (must match training!)
    # ---------------------------
    training_stats_dir = pickle.load(open(Stats_Data_Path["training_stats"], "rb"))
    target_mean = training_stats_dir["target_mean"]
    target_std = training_stats_dir["target_stds"]

    # target_mean = 0.1847209
    # target_std = 1.3368018
    print(f" target_mean={target_mean:.6f}, target_std={target_std:.6f}")

    # ---------------------------
    # Run reconstruction year by year
    # ---------------------------
    for year in range(START_YEAR, END_YEAR + 1):
        reconstruct_one_year(year, target_mean, target_std)


if __name__ == "__main__":
    main()
