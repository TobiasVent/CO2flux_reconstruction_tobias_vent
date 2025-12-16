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
# "Viridis-like" colormap with white background
# white_viridis = LinearSegmentedColormap.from_list(
#     'white_viridis',
#     [
#         (0.0, '#ffffff'),   # exakt 0 -> weiß
#         (0.000001, '#440053'),  # alles >0 bekommt sofort Farbe
#         (0.2, '#404388'),
#         (0.4, '#2a788e'),
#         (0.6, '#21a784'),
#         (0.8, '#78d151'),
#         (1.0, '#fde624'),
#     ],
#     N=2048
# )
# "Viridis-like" colormap with white background
white_viridis = plt.cm.viridis.copy()
white_viridis.set_under('white')
norm = Normalize(vmin=1,vmax=1000)  
def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    #density = ax.scatter_density(x, y, cmap=white_viridis, vmax=1000)
    density = ax.scatter_density(x, y, cmap=white_viridis,dpi = 150, norm =norm) 
    # cbar = fig.colorbar(density, label='low density                     high density', ax=ax, orientation='horizontal', pad=0.15)
    # cbar.set_ticks(ticks=[], labels=[])

    # Ticks am unteren und oberen Ende setzen
    #cbar.set_ticks([vmin, vmax])
    #cbar.ax.set_yticklabels(['low density', 'high density'])
    # KEINE Colorbar:
    # cbar = fig.colorbar(density, ax=ax, orientation='horizontal', pad=0.15)
    # cbar.set_ticklabels(['low density', 'high density'])

    return density


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
    # "LSTM": (LSTMModel, "/data/stu231428/Master_Thesis/main/trained_models/lsmt_with_pos.pt", {"input_size": 14, "hidden_dim": 128, "dropout": 0, "num_layers": 3}),
    # "MLP": (MLPModel, "/data/stu231428/Master_Thesis/main/trained_models/mlp_with_pos.pt", {"hidden_dims": [207, 248, 198], "input_dim": 44, "dropout": 0}),
    # "Attention LSTM":(LSTMModelAttentionTemporal,"/data/stu231428/Master_Thesis/main/trained_models/attention_temporal_lstm_with_pos.pt",{'input_size': 14,'hidden_dim': 256, 'dropout': 0.4}),
}

# XGBoost hinzufügen
xgb_model_path = "/data/stu231428/Master_Thesis/main/trained_models/xg_boost_with_pos_model.pkl"
with open(xgb_model_path, "rb") as f:
    xgb_model = pickle.load(f)
model_files["XGBoost"] = ("xgboost", xgb_model, None)  # Platzhalter für Konsistenz

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
    #"North_Atlantic_experiment_5",
    #"Southern_Ocean_experiment_5",

]
# -------------------------------------------------------------------------
# Stats for rescaling predictions
# -------------------------------------------------------------------------

START_YEAR = 2018
END_YEAR = 2018
# -------------------------------------------------------------------------
# Main Loop
# -------------------------------------------------------------------------
for path in data_paths:



    # Region extrahieren
    region_key = [k for k in region_map if k in path][0]
    region_name = region_map[region_key]
    

    simulation_key = [k for k in simulation_map if k in path][0]
    simulation_name = simulation_map[simulation_key]


    for model_name, (ModelClass, model_obj, model_kwargs) in model_files.items():
       # Load cache

        base = "/media/stu231428/1120 7818/Master_github/datasets/cache"

        years = range(START_YEAR, END_YEAR + 1)

        files = []
        missing_years = []

        for year in years:
            path = f"{base}/{model_name}_{region_key}_reconstruction_{year}_{simulation_key}.pkl"
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
        # # Scatterplot aus Cache
        # r2 = r2_score(df_cache["targets"], df_cache["predictions"])
        # fig = plt.figure(figsize=(8, 6))

        # x = df_cache["targets"].values
        # y = df_cache["predictions"].values

        # ax = using_mpl_scatter_density(fig, x, y)

        # # Diagonale (perfekte Vorhersage)
        # min_val = min(x.min(), y.min())
        # max_val = max(x.max(), y.max())
        # ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect prediction")

        # # Best-fit line
        # reg = LinearRegression()
        # reg.fit(x.reshape(-1, 1), y)
        # y_fit = reg.predict(np.array([min_val, max_val]).reshape(-1, 1))
        # ax.plot(
        #     [min_val, max_val],
        #     y_fit,
        #     'b-',
        #     linewidth=2,
        #     label=f"Best fit: y={reg.coef_[0]:.2f}x{'+' if reg.intercept_ >= 0 else '-'}{abs(reg.intercept_):.2f}"
        # )

        # ax.text(
        #     0.05, 0.95, f"R² = {r2:.3f}", 
        #     transform=ax.transAxes,
        #     fontsize=14,
        #     va='top',
        #     bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray')
        # )

        # ax.set_xlabel("Simulated co2flux_pre", fontsize=14)
        # ax.set_ylabel("Reconstructed co2flux_pre", fontsize=14)
        # ax.legend(fontsize=14)
        # ax.grid(True)
        # fig.tight_layout()

        # out_dir = "/data/stu231428/Master_Thesis/final_main/plots/scatter_plots"
        # os.makedirs(out_dir, exist_ok=True)
        # out_file = os.path.join(
        #     out_dir,
        #     f"scatter_plot_with_density_{region_key}_{model_name}_with_position_{simulation_key}.png"
        # )
        # fig.savefig(out_file, dpi=150)
        # plt.close(fig)



        # Scatterplot aus Cache
        r2 = r2_score(df_cache["co2flux_pre"], df_cache["reconstructed_co2flux_pre"])
        fig = plt.figure(figsize=(8, 6))
        #plt.scatter(df_cache["targets"], df_cache["predictions"], alpha=0.3, s=5,color = "gray")
        density =  using_mpl_scatter_density(fig, df_cache["co2flux_pre"], df_cache["reconstructed_co2flux_pre"])

        # Diagonale (perfekte Vorhersage)
        min_val = min(df_cache["co2flux_pre"].min(), df_cache["reconstructed_co2flux_pre"].min())
        max_val = max(df_cache["co2flux_pre"].max(), df_cache["reconstructed_co2flux_pre"].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Reconstruction")
        # Best-fit line
        reg = LinearRegression()
        reg.fit(df_cache["co2flux_pre"].values.reshape(-1, 1), df_cache["reconstructed_co2flux_pre"].values)
        y_fit = reg.predict(np.array([min_val, max_val]).reshape(-1, 1))
        plt.plot([min_val, max_val], y_fit, 'b-', linewidth=2, alpha = 0.5, label=f"Best fit: y={reg.coef_[0]:.2f}x{'+' if reg.intercept_ >= 0 else '-'}{abs(reg.intercept_):.2f}")


        plt.text(
            0.05, 0.95, f"R² = {r2:.3f}", 
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray')
        ) 

        cbar = fig.colorbar(density, shrink = 0.7, label = "Number of points per pixel")
        cbar.ax.tick_params(labelsize=16)          # tick label font size
        cbar.set_label("Number of points per pixel", fontsize=18)  

        #cbar = fig.colorbar( density, shrink = 0.5,label='Low Density                                                  High Density', orientation='horizontal')
        #cbar.set_ticks(ticks=[], labels=[])
        # Best-fit line
        # lower = np.percentile(df_cache["targets"], 1)
        # upper = np.percentile(df_cache["targets"], 99)
        # if region_key == "Southern_Ocean":

        #     plt.xlim(-15, 15)
        #     plt.ylim(-15, 15)
        # if region_key == "North_Atlantic":

        #     plt.xlim(-15, 15)
        #     plt.ylim(-15, 15)
        
        # if region_key == "global":
        #     plt.xlim(-40, 20)
        #     plt.ylim(-30, 20)
        plt.xlim(-30, 20)
        plt.ylim(-20, 20)
        plt.xlabel(r"Simulated $CO_{2}$ flux pre", fontsize=20)
        plt.ylabel(r"Reconstructed $CO_{2}$ flux pre", fontsize=20)
        plt.title(model_name, fontsize=22)
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)

        # ✅ Legende unterhalb des Plots mit etwas mehr Abstand
        plt.legend(
            fontsize=18,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),  # ⬅️ weiter nach unten verschoben
            ncol=3,
            frameon=False
        )

        plt.grid(True)
        plt.tight_layout(rect=[0, 0.1, 1, 1])  # ⬅️ mehr Platz unten für die Legende

        out_dir = "/media/stu231428/1120 7818/Master_github/datasets/plots/scatter_plots"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(
            out_dir,
            f"scatter_plot_with_density_{region_key}_{model_name}_with_position_{simulation_key}_same_scale.png"
        )

        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()



