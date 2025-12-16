import torch
from torch.utils.data import DataLoader

import pickle
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

START_YEAR = 2018
END_YEAR = 2018
# ----------------
# -------------------------------------------------------------------------
# Metric-Funktion
# -------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

def flatten_sample_x_month(x_month, n_dyn=10, n_static=4):
    num_samples, window_size, num_features = x_month.shape
    flattened_samples = []
    for i in range(num_samples):
        x = x_month[i]  # shape: (window_size, num_features)
        x_dyn = x[:, :n_dyn].flatten()   # window_size × n_dyn
        x_static = x[0, n_dyn:]          # static features from the first time step
        flattened_sample = np.concatenate([x_dyn, x_static])
        flattened_samples.append(flattened_sample)
    return np.array(flattened_samples)
# -------------------------------------------------------------------------
# Device setup
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# Model definitions
# -------------------------------------------------------------------------
model_files = {
    "Simulation": ("dummy","dummy","dummy"),
    "LSTM": ("LSTMModel", "/data/stu231428/Master_Thesis/main/trained_models/lsmt_with_pos.pt", {"input_size": 14, "hidden_dim": 128, "dropout": 0, "num_layers": 3}),
    
    "Attention LSTM":("LSTMModelAttentionTemporal","/data/stu231428/Master_Thesis/main/trained_models/attention_temporal_lstm_with_pos.pt",{'input_size': 14,'hidden_dim': 256, 'dropout': 0.4}),
    "MLP": ("MLPModel", "/data/stu231428/Master_Thesis/main/trained_models/mlp_with_pos.pt", {"hidden_dims": [207, 248, 198], "input_dim": 44, "dropout": 0}),
    "xgboost": ("xgboost","xgb_model","dummy")
}

  # Platzhalter für Konsistenz

# -------------------------------------------------------------------------
# Region mapping
# -------------------------------------------------------------------------
region_map = {
    "EQ_Pacific": "Equatorial Pacific",
    "North_Atlantic": "North Atlantic",
    "Arctic": "Arctic Ocean",
    "Southern_Ocean": "Southern Ocean",
    "global": "Global"
}

simulation_map = {
    "experiment_1": "Simulation 1",
    "experiment_5": "Simulation 2"
}


# -------------------------------------------------------------------------
# Data paths
# -------------------------------------------------------------------------
data_paths = [
    #"North_Atlantic_experiment_1",
    #"Southern_Ocean_experiment_1",
    
    "global_experiment_1",
    #"North_Atlantic_5",
    #"Southern_Ocean_experiment_5",
    
    #"global_experiment_5"

]





# -------------------------------------------------------------------------
# Stats for rescaling predictions
# -------------------------------------------------------------------------

all_results = []
# -------------------------------------------------------------------------
# Main Loop
# -------------------------------------------------------------------------
for path in data_paths:

    


    # Region extrahieren
    region_key = [k for k in region_map if k in path][0]
    region_name = region_map[region_key]
    

    simulation_key = [k for k in simulation_map if k in path][0]
    simulation_name = simulation_map[simulation_key]

  

    for row_idx, (model_name, (ModelClass, model_path, model_kwargs)) in enumerate(model_files.items()):
       # Load cache

        base = "/media/stu231428/1120 7818/Master_github/datasets/cache"

        years = range(START_YEAR, END_YEAR + 1)

        files = []
        missing_years = []

        for year in years:
            path = f"{base}/{model_name}_{region_key}_reconstruction_{year}_experiment_1.pkl"
            if os.path.exists(path):
                files.append(path)
            else:
                missing_years.append(year)

        #  Fehler werfen, wenn Zeitraum nicht vollständig verfügbar
        if missing_years:
            raise FileNotFoundError(
                f" Reconstruction files missing for years: {missing_years}\n"
                f"Requested period: {START_YEAR}–{END_YEAR}\n"
                f"Base path: {base}\n"
                f"Model: {model_name}"
            )
        dfs = [pd.read_pickle(fp) for fp in files]
        df_cache = pd.concat(dfs, ignore_index=True)

        df_cache["time_counter"] = pd.to_datetime(df_cache["time_counter"])




        # Monat & Jahr extrahieren
        df_cache["year_month"] = df_cache["time_counter"].dt.to_period("M")

        # -----------------------------------------------------------------
        # Gesamte Metriken
        # -----------------------------------------------------------------
        mse, rmse, mae = compute_metrics(df_cache["reconstructed_co2flux_pre"], df_cache["co2flux_pre"])
        all_results.append({
            "Region": region_name,
            "Model": model_name,
            "Month": "Overall",
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae
        })

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join("/media/stu231428/1120 7818/Master_github/datasets/plots/metrics", f"model_metrics_monthly_and_total_{region_key}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Ergebnisse gespeichert unter: {csv_path}")
