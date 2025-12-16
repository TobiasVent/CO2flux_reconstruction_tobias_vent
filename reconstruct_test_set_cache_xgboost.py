import os
import pickle
import pandas as pd
import numpy as np
from collections import namedtuple
import xgboost as xgb

Sample = namedtuple("Sample", ["features", "target", "meta"])

# ==========================
# SETTINGS
# ==========================
YEAR = 2018

XGB_MODEL_PATH = "/data/stu231428/Master_Thesis/main/trained_models/xg_boost_with_pos_model.pkl"
CACHE_DIR = "/media/stu231428/1120 7818/Master_github/datasets/cache"

# ✅ FIX: benutze {year} (klein) und format(year=YEAR)
TEST_PATH_PATTERN = "/media/stu231428/1120 7818/Master_github/datasets/yearly/global_test_{year}_experiment_1.pkl"

# Normalization stats (target) – du setzt mean/std direkt
# NORM_STATS_PATH = "/data/stu231428/Master_Thesis/Data/normalization_stats.pkl"


def flatten_dyn_plus_static(x_window, n_dyn=10):
    x_dyn = x_window[:, :n_dyn].reshape(-1)   # 4*10 = 40
    x_static = x_window[0, n_dyn:]            # 4
    return np.concatenate([x_dyn, x_static]).astype(np.float32)


def main():
    # ---------------------------
    # 0) Output path (mit Jahr)
    # ---------------------------
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, f"XGBoost_global_reconstruction_{YEAR}_experiment_1.pkl")

    # ---------------------------
    # 1) Normalization stats
    # ---------------------------
    target_mean = 0.1847209
    target_std = 1.3368018
    print(f"✅ target_mean={target_mean:.6f}, target_std={target_std:.6f}")

    # ---------------------------
    # 2) Load year-specific test samples
    # ---------------------------
    test_path = TEST_PATH_PATTERN.format(year=YEAR)
    print("🔎 Using test_path:", test_path)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Test set for YEAR={YEAR} not found: {test_path}")

    print(f"📦 Loading test samples for {YEAR}: {test_path}")
    with open(test_path, "rb") as f:
        samples = pickle.load(f)

    print(f"✅ Loaded {len(samples):,} samples")

    # ---------------------------
    # 3) Build feature matrix
    # ---------------------------
    print("🧱 Building feature matrix for XGBoost...")
    X = np.stack([flatten_dyn_plus_static(s.features) for s in samples], axis=0)  # (N,44)
    y = np.array([s.target for s in samples], dtype=np.float32)                  # (N,)
    metas = [s.meta for s in samples]
    print(f"✅ X shape: {X.shape}, y shape: {y.shape}")

    # ---------------------------
    # 4) Load model + predict
    # ---------------------------
    print("🤖 Loading XGBoost model...")
    # XGBoost hinzufügen
   
    with open(XGB_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
   
    print("✅ Model loaded")

    print("🚀 Predicting...")
    preds = model.predict(X).astype(np.float32)

    # Denormalize
    preds_denorm = preds * target_std + target_mean
    y_denorm = y * target_std + target_mean

    # ---------------------------
    # 5) Build dataframe
    # ---------------------------
    print("📄 Building DataFrame...")
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
    # 6) Save
    # ---------------------------
    df.to_pickle(out_path)
    print(f"\n💾 Saved reconstruction cache for {YEAR} to:\n{out_path}")


if __name__ == "__main__":
    main()
