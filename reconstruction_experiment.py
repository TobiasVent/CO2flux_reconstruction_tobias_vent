import torch
from torch.utils.data import DataLoader
from dataset import PointwiseSampleDatasetMonth, PointwiseSampleDatasetMonthMLPWithPos, MLPDataset
from models.mlp import MLPModel
from models.attention_lstm import LSTMModelAttentionTemporal
from models.lstm import LSTMModel
from sklearn.metrics import mean_squared_error

import pickle
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os





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
    "LSTM": (LSTMModel, "/data/stu231428/Master_Thesis/main/trained_models/lsmt_with_pos.pt", {"input_size": 14, "hidden_dim": 128, "dropout": 0, "num_layers": 3}),
    
    "Attention LSTM":(LSTMModelAttentionTemporal,"/data/stu231428/Master_Thesis/main/trained_models/attention_temporal_lstm_with_pos.pt",{'input_size': 14,'hidden_dim': 256, 'dropout': 0.4}),
    "MLP": (MLPModel, "/data/stu231428/Master_Thesis/main/trained_models/mlp_with_pos.pt", {"hidden_dims": [207, 248, 198], "input_dim": 44, "dropout": 0}),
    "xgboost": ("xgboost","xgb_model","dummy")
}

# XGBoost hinzufügen
# xgb_model_path = "/data/stu231428/Master_Thesis/main/trained_models/xg_boost_with_pos_model.pkl"
# with open(xgb_model_path, "rb") as f:
#     xgb_model = pickle.load(f)
# model_files["XGBoost"] = ("xgboost", xgb_model, None)  # Platzhalter für Konsistenz

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
     #"Southern_Ocean_january_experiment_1",
    #"Southern_Ocean_july_experiment_1",
     "global_experiment_1",
     #"North_Atlantic_experiment_5",
      #"Southern_Ocean_january_experiment_5",
      #"Southern_Ocean_july_experiment_5",
     #"global_experiment_5"

]

month_map = {
    1: "January",
    7: "July"
}



# -------------------------------------------------------------------------
# Stats for rescaling predictions
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Main Loop
# -------------------------------------------------------------------------
for path in data_paths:

    


    # Region extrahieren
    region_key = [k for k in region_map if k in path][0]
    region_name = region_map[region_key]
    

    simulation_key = [k for k in simulation_map if k in path][0]
    simulation_name = simulation_map[simulation_key]

    if region_key == "Southern_Ocean":
        month_map = {1:"January"} if "january" in path else {1:"July"}
        print(month_map)
        
        n_models = len(model_files)
        n_months = len(month_map)
        fig, axes = plt.subplots(
        n_models, n_months,
        figsize=(8, 1.2 * n_models),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,  # optional, wenn du gridspec benutzt, ggf. auskommentieren
        # horizontaler Abstand zwischen Spalten
        )

    if region_key == "global":
        month_map = {1:"January",7:"July"}
        print(month_map)
        n_models = len(model_files)
        n_months = len(month_map)
        fig, axes = plt.subplots(
        n_models, n_months,
        figsize=(8, 2.5 * n_models),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True
    )


    if region_key == "North_Atlantic":
        month_map = {1:"January",7:"July"}
        print(month_map)
        n_models = len(model_files)
        n_months = len(month_map)
        fig, axes = plt.subplots(
        n_models, n_months,
        figsize=(8, 3 * n_models),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True
        )    

    for row_idx, (real_model_name, (ModelClass, model_path, model_kwargs)) in enumerate(model_files.items()):
       # Load cache
        if real_model_name == "Simulation":
            model_name = "LSTM"
        else:
            model_name = real_model_name
        if region_key == "North_Atlantic" and simulation_key == "experiment_1":
            cache_file = f"/media/stu231428/1120 7818/prediction_caches/with_position/experiment_1/{region_key}/cache_{region_key}_{model_name}_with_position_experiment_1.pkl"
            df_cache = pd.read_pickle(cache_file)
    
        if region_key == "Southern_Ocean" and simulation_key == "experiment_1":   

            cache_file= f"//media/stu231428/1120 7818/prediction_caches/with_position/experiment_1/Southern_Ocean/cache_Southern_Ocean_2014_2018_{model_name}_with_position_experiment_1.pkl"
            df_cache = pd.read_pickle(cache_file)

        
        if region_key == "global" and simulation_key == "experiment_1":

            cache_file_2=  f"/media/stu231428/1120 7818/Master_github/datasets/cache/lstm_global_reconstruction.pkl"
            df_cache = pd.read_pickle(cache_file_2)
 
        
        if region_key == "North_Atlantic" and simulation_key == "experiment_5":
            cache_file = f"/media/stu231428/1120 7818/prediction_caches/with_position/experiment_5/North_Atlantic/cache_{model_name}_1958_2018_North_Atlantic_with_pos_exp_5.pkl"
            df_cache = pd.read_pickle(cache_file)
        
        if region_key == "Southern_Ocean" and simulation_key == "experiment_5":

            cache_file_3= f"/media/stu231428/1120 7818/prediction_caches/with_position/experiment_5/Southern_Ocean/cache_{model_name}_2004_2018_Southern_Ocean_with_pos_exp_5.pkl"
            df_cache = pd.read_pickle(cache_file_3)
        
        if region_key == "global" and simulation_key == "experiment_5":
            cache_file = f"/media/stu231428/1120 7818/prediction_caches/with_position/experiment_5/Global/cache_{model_name}_2018_Global_with_pos_exp_5.pkl"
            df_cache = pd.read_pickle(cache_file)
        for col_idx, (i, month_name) in enumerate(month_map.items()):
            df_cache["month"] = df_cache["time_counter"].dt.month
            df_cache["year"] = df_cache["time_counter"].dt.year
            df_month = df_cache[(df_cache["month"] == i) & (df_cache["year"] == 2018)]
            
            print(axes.ndim)
            if axes.ndim == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx] 
            # Dynamischen Ausschnitt aus Daten berechnen
            lon_min, lon_max = df_month["nav_lon"].min(), df_month["nav_lon"].max()
            lat_min, lat_max = df_month["nav_lat"].min(), df_month["nav_lat"].max()
            
            lon_pad = (lon_max - lon_min) * 0.01
            lat_pad = (lat_max - lat_min) * 0.01
            ax.set_extent([lon_min - lon_pad, lon_max + lon_pad,
                           lat_min - lat_pad, lat_max + lat_pad],
                          crs=ccrs.PlateCarree())     
            
            if region_name == "North Atlantic":
                projection = ccrs.PlateCarree()
            else:
                projection = ccrs.PlateCarree(central_longitude=180)

            # Features hinzufügen (gleiches Styling wie im Original)
            #ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='black')   
            if real_model_name == "Simulation":
                df_month["reconstructed_co2flux_pre"] = df_month["co2flux_pre"]
            # if real_model_name == "Simulation":
            #     df_month["predictions"] = df_month["targets"]
            print(df_month.head())
            mse = mean_squared_error(
                df_month["co2flux_pre"],
                df_month["reconstructed_co2flux_pre"]
            )

            print(
                f"📉 MSE | Model: {real_model_name:<15} "
                f"| Month: {month_name:<8} "
                f"| Region: {region_key:<15} "
                f"| MSE: {mse:.4f}"
)
            # Scatterplot
            sc = ax.scatter(
                df_month["nav_lon"], df_month["nav_lat"],
                c=df_month["reconstructed_co2flux_pre"], cmap='coolwarm', s=2, alpha=1,
                vmin=-5, vmax=5, transform=ccrs.PlateCarree()
            )
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0.5,
                alpha=0.5,
                linestyle='--'
            )
            gl.top_labels = False
            gl.right_labels = False
            # Nur ganz links Breitengrad-Labels anzeigen
            if col_idx != 0:
                gl.left_labels = False

            # Nur in der untersten Reihe Längengrad-Labels anzeigen
            if row_idx != n_models - 1:
                gl.bottom_labels = False

            # Titel für jedes Subplot
            #ax.set_title(f"{model_name} {month_name}", fontsize=11)
            if axes.ndim == 1:
                ax.set_title(f"{real_model_name}", fontsize=14)
            else:
                ax.set_title(f"{real_model_name} {month_name}", fontsize=14)

    # Gemeinsame Farbleiste unten
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(),
                        orientation='horizontal', shrink=0.6, pad=0.015, aspect = 30)
    cbar.set_label(r"$CO_{2}$ flux pre", fontsize=14)

    # Gesamttitel
    #fig.suptitle(f"Simulated and Reconstructed co2flux – {region_name} 2018", fontsize=14)
    #fig.suptitle(f"Simulated and Reconstructed co2flux – {region_name} 2018", fontsize=14)
    # Ausgabe speichern
    out_dir = "/media/stu231428/1120 7818/Master_github/datasets/plots/reconstruction/"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir,
                            f"combined_predicted_simulated_co2flux_pre_{region_key}_{month_name}_{simulation_key}.png")
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"✅ Saved combined plot: {out_file}")               

        

        
