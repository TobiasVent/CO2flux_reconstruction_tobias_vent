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
from models.mlp import MLPModel
from models.attention_lstm import LSTMModelAttentionTemporal
from models.lstm import LSTMModel


def flatten_sample_x_month(x_month, n_dyn=10, n_static=4):
    """
    Flatten a batch of windowed samples into 2D feature vectors.
    Dynamic features (first n_dyn columns) are flattened across the window.
    Static features are taken from the first timestep only.
    """
    num_samples, window_size, num_features = x_month.shape
    flattened_samples = []
    for i in range(num_samples):
        x = x_month[i]
        x_dyn = x[:, :n_dyn].flatten()
        x_static = x[0, n_dyn:]
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
model_files = ["LSTM", "Attention_LSTM,MLP,XGBoost"]

# Path to your cache directory
cache_dir = ""

#path to your output directory
out_dir = "/media/stu231428/1120 7818/Master_github/datasets/plots/differences"

# -------------------------------------------------------------------------
# Region mapping
# -------------------------------------------------------------------------
region_map = {
    "EQ_Pacific": "Equatorial Pacific",
    "North_Atlantic": "North Atlantic",
    "Arctic": "Arctic Ocean",
    "Southern_Ocean": "Southern Ocean",
    "global": "Global",
}

simulation_map = {
    "experiment_1": "Simulation 1",
    "experiment_5": "Simulation 2",
}

# -------------------------------------------------------------------------
# Data paths
# You need to compute the cache for 2018 for each region first
# -------------------------------------------------------------------------
data_paths = [
    "North_Atlantic_experiment_1",
    "Southern_Ocean_january_experiment_1",
    "Southern_Ocean_july_experiment_1",
    "global_experiment_1",
    "North_Atlantic_experiment_5",
    "Southern_Ocean_january_experiment_5",
    "Southern_Ocean_july_experiment_5",
    "global_experiment_5",
]

month_map = {
    1: "January",
    7: "July",
}

# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------
for path in data_paths:

    # Extract region and simulation from the path string
    region_key = [k for k in region_map if k in path][0]
    region_name = region_map[region_key]

    simulation_key = [k for k in simulation_map if k in path][0]
    simulation_name = simulation_map[simulation_key]

    if region_key == "Southern_Ocean":
        month_map = {1: "January"} if "january" in path else {1: "July"}
        print(month_map)

        n_models = len(model_files)
        n_months = len(month_map)
        fig, axes = plt.subplots(
            n_models,
            n_months,
            figsize=(8, 1.2 * n_models),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )

    if region_key == "global":
        month_map = {1: "January", 7: "July"}
        print(month_map)

        n_models = len(model_files)
        n_months = len(month_map)
        fig, axes = plt.subplots(
            n_models,
            n_months,
            figsize=(8, 2 * n_models),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )

    if region_key == "North_Atlantic":
        month_map = {1: "January", 7: "July"}
        print(month_map)

        n_models = len(model_files)
        n_months = len(month_map)
        fig, axes = plt.subplots(
            n_models,
            n_months,
            figsize=(8, 3 * n_models),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )

    for row_idx, model_name in enumerate(model_files):

        # Load cached reconstruction results
        cache_file = f"{cache_dir}/{model_name}_{region_name}_reconstruction_2018_{simulation_key}.pkl"
        df_cache = pd.read_pickle(cache_file)
        print(df_cache.head())

        for col_idx, (i, month_name) in enumerate(month_map.items()):
            df_cache["month"] = df_cache["time_counter"].dt.month
            df_cache["year"] = df_cache["time_counter"].dt.year

            df_month = df_cache[(df_cache["month"] == i) & (df_cache["year"] == 2018)]
            df_month["diff"] = df_month["reconstructed_co2flux_pre"] - df_month["co2flux_pre"]

            print(axes.ndim)

            if axes.ndim == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]

            # Compute dynamic extent from data bounds
            lon_min, lon_max = df_month["nav_lon"].min(), df_month["nav_lon"].max()
            lat_min, lat_max = df_month["nav_lat"].min(), df_month["nav_lat"].max()

            lon_pad = (lon_max - lon_min) * 0.01
            lat_pad = (lat_max - lat_min) * 0.01

            ax.set_extent(
                [
                    lon_min - lon_pad,
                    lon_max + lon_pad,
                    lat_min - lat_pad,
                    lat_max + lat_pad,
                ],
                crs=ccrs.PlateCarree(),
            )

            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0.5,
                alpha=0.5,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False

            # Show latitude labels only on the leftmost column
            if col_idx != 0:
                gl.left_labels = False

            # Show longitude labels only on the bottom row
            if row_idx != n_models - 1:
                gl.bottom_labels = False

            if region_name == "North Atlantic":
                projection = ccrs.PlateCarree()
            else:
                projection = ccrs.PlateCarree(central_longitude=180)

            # Add map features
            ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor="black")

            # Scatter plot of differences
            sc = ax.scatter(
                df_month["nav_lon"],
                df_month["nav_lat"],
                c=df_month["diff"],
                cmap="BrBG",
                s=2,
                alpha=1,
                vmin=-5,
                vmax=5,
                transform=ccrs.PlateCarree(),
            )

            # Title for each subplot
            if axes.ndim == 1:
                ax.set_title(f"{model_name}", fontsize=12)
            else:
                ax.set_title(f"{model_name} {month_name}", fontsize=12)

    # Shared colorbar at the bottom
    cbar = fig.colorbar(
        sc,
        ax=axes.ravel().tolist(),
        location="bottom",
        orientation="horizontal",
        shrink=0.6,
        pad=0.015,
        aspect=30,
    )
    cbar.set_label(
        r"Difference (Reconstructed $CO_{2}$ flux pre − Simulated $CO_{2}$ flux pre)",
        fontsize=14,
    )

    # Save output plot
    
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir, f"combined_co2flux_pre_{region_key}_{month_name}_{simulation_key}.png"
    )
    plt.savefig(out_file, dpi=250)
    plt.close(fig)
    print(f"Saved combined plot: {out_file}")
