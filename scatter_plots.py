import torch
from torch.utils.data import DataLoader

import pickle
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm

# Use viridis but render under-range values as white
white_viridis = plt.cm.viridis.copy()
white_viridis.set_under("white")

# Normalization for density coloring
norm = Normalize(vmin=1, vmax=1000)


def using_mpl_scatter_density(fig, x, y):
    """
    Create a scatter density plot using mpl_scatter_density and return
    the density artist, so it can be used for a colorbar.
    """
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    density = ax.scatter_density(x, y, cmap=white_viridis, dpi=150, norm=norm)
    return density


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
# Model selection
# -------------------------------------------------------------------------
model_files = ["LSTM", "Attention_LSTM", "MLP", "XGBoost"]

# Path to your cache directory
cache_dir = "/media/stu231428/1120 7818/Master_github/datasets/cache"

# Directory where you want to save the plots
out_dir = "/media/stu231428/1120 7818/Master_github/datasets/plots/scatter_plots"

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
# -------------------------------------------------------------------------
data_paths = [
    "North_Atlantic_experiment_1",
    "Southern_Ocean_experiment_1",
    "global_experiment_1",
    "North_Atlantic_experiment_5",
    "Southern_Ocean_experiment_5",
]

# -------------------------------------------------------------------------
# Time range for the scatter plot (inclusive)
# -------------------------------------------------------------------------
START_YEAR = 2018
END_YEAR = 2018

# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------
for path in data_paths:

    # Extract region and simulation from the path string
    region_key = [k for k in region_map if k in path][0]
    region_name = region_map[region_key]

    simulation_key = [k for k in simulation_map if k in path][0]
    simulation_name = simulation_map[simulation_key]

    for model_name, (ModelClass, model_obj, model_kwargs) in model_files.items():
        # Collect yearly cache files for the requested period
        years = range(START_YEAR, END_YEAR + 1)

        files = []
        missing_years = []

        for year in years:
            path = f"{cache_dir}/{model_name}_{region_key}_reconstruction_{year}_{simulation_key}.pkl"
            if os.path.exists(path):
                files.append(path)
            else:
                missing_years.append(year)

        # Raise an error if the requested period is not fully available
        if missing_years:
            raise FileNotFoundError(
                f"Reconstruction files missing for years: {missing_years}\n"
                f"Requested period: {START_YEAR}–{END_YEAR}\n"
                f"Base path: {cache_dir}\n"
                f"Model: {model_name}"
            )

        # Load and concatenate all yearly caches
        dfs = [pd.read_pickle(fp) for fp in files]
        df_cache = pd.concat(dfs, ignore_index=True)

        # Compute R² for simulated vs reconstructed values
        r2 = r2_score(df_cache["co2flux_pre"], df_cache["reconstructed_co2flux_pre"])

        # Create scatter density plot
        fig = plt.figure(figsize=(8, 6))
        density = using_mpl_scatter_density(
            fig,
            df_cache["co2flux_pre"],
            df_cache["reconstructed_co2flux_pre"],
        )

        # Add the 1:1 line (perfect reconstruction)
        min_val = min(
            df_cache["co2flux_pre"].min(),
            df_cache["reconstructed_co2flux_pre"].min(),
        )
        max_val = max(
            df_cache["co2flux_pre"].max(),
            df_cache["reconstructed_co2flux_pre"].max(),
        )
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Reconstruction")

        # Fit and plot a best-fit line
        reg = LinearRegression()
        reg.fit(
            df_cache["co2flux_pre"].values.reshape(-1, 1),
            df_cache["reconstructed_co2flux_pre"].values,
        )
        y_fit = reg.predict(np.array([min_val, max_val]).reshape(-1, 1))
        plt.plot(
            [min_val, max_val],
            y_fit,
            "b-",
            linewidth=2,
            alpha=0.5,
            label=f"Best fit: y={reg.coef_[0]:.2f}x{'+' if reg.intercept_ >= 0 else '-'}{abs(reg.intercept_):.2f}",
        )

        # Add R² annotation
        plt.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"),
        )

        # Add colorbar for point density
        cbar = fig.colorbar(density, shrink=0.7, label="Number of points per pixel")
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("Number of points per pixel", fontsize=18)

        # Set axes limits and labels
        plt.xlim(-30, 20)
        plt.ylim(-20, 20)
        plt.xlabel(r"Simulated $CO_{2}$ flux pre", fontsize=20)
        plt.ylabel(r"Reconstructed $CO_{2}$ flux pre", fontsize=20)
        plt.title(model_name, fontsize=22)
        plt.tick_params(axis="x", labelsize=20)
        plt.tick_params(axis="y", labelsize=20)

        # Place legend below the plot
        plt.legend(
            fontsize=18,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=3,
            frameon=False,
        )

        plt.grid(True)
        plt.tight_layout(rect=[0, 0.1, 1, 1])

        # Save output
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(
            out_dir,
            f"scatter_plot_with_density_{region_key}_{model_name}_with_position_{simulation_key}_same_scale.png",
        )

        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
