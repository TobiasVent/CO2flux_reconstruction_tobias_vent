import torch
from torch.utils.data import DataLoader

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from matplotlib.patches import Patch
import calendar
START_YEAR = 2012
END_YEAR = 2012



experiment_name = "experiment_1"
out_dir = "/data/results/annual_seasonal_plots/"

# -------------------------------------------------------------------------
# Device setup
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# Model definitions
# -------------------------------------------------------------------------


model_files = ["LSTM","Attention_LSTM","MLP","XGBoost"]



# -------------------------------------------------------------------------
# Helper: Daten laden und vorverarbeiten (mit Cache in Python, nicht Pickle)
# -------------------------------------------------------------------------
def load_df_cache(ocean, model_name, cache_dict):
    """
    Lädt und verarbeitet df_cache für eine (ocean, model_name)-Kombination.
    Nutzt ein dict cache_dict, damit die Daten nur einmal von Platte gelesen werden.
    """
    key = (ocean, model_name)
    if key in cache_dict:
        return cache_dict[key]
    
    else:
        

        years = range(START_YEAR, END_YEAR + 1)

        files = []
        missing_years = []
        cache_dir = f"data/reconstruction_cache/{experiment_name}/{ocean}"
        for year in years:
            path = f"{cache_dir}/{model_name}_{ocean}_reconstruction_{year}_{experiment_name}.pkl"
            if os.path.exists(path):
                files.append(path)
            else:
                missing_years.append(year)

        #  Fehler werfen, wenn Zeitraum nicht vollständig verfügbar
        if missing_years:
            raise FileNotFoundError(
                f" Reconstruction files missing for years: {missing_years}\n"
                f"Requested period: {START_YEAR}–{END_YEAR}\n"
                f"Base path: {cache_dir}\n"
                f"Model: {model_name}"
            )
        dfs = [pd.read_pickle(fp) for fp in files]
        df_cache = pd.concat(dfs, ignore_index=True)






    # Zeitspalte vorbereiten
    df_cache["time_counter"] = pd.to_datetime(df_cache["time_counter"])
    df_cache = df_cache[df_cache["time_counter"].dt.year >= 1959]

    # Monatsspalte für saisonale Statistiken
    df_cache["month"] = df_cache["time_counter"].dt.month
 # === DEBUG INFOS ===
    years = df_cache["time_counter"].dt.year
    year_min, year_max = years.min(), years.max()
    months_present = sorted(df_cache["month"].unique())
    missing_months = [m for m in range(1, 13) if m not in months_present]
    print(f"  Years range: {year_min}–{year_max}")
    print(f"  Total entries: {len(df_cache):,}")
    print(f"  Months present: {[calendar.month_abbr[m] for m in months_present]}")
    if missing_months:
        print(f"  ⚠️ Missing months: {[calendar.month_abbr[m] for m in missing_months]}")
    else:
        print("  ✅ All 12 months present")

    # Optional: Zeige, wie viele Einträge pro Jahr
    year_counts = years.value_counts().sort_index()
    print(f"  Entries per year (first 5): {year_counts.head().to_dict()}")
    print(f"  Entries per year (last 5):  {year_counts.tail().to_dict()}")
    cache_dict[key] = df_cache
    return df_cache


# -------------------------------------------------------------------------
# Plots vorbereiten
# -------------------------------------------------------------------------
#oceans = ["North Atlantic", "Southern Ocean"]
oceans = ["global", "global",]

fig_yearly, axes_yearly = plt.subplots(
    4, 2,
    figsize=(12, 4 * 4),
    constrained_layout=True
)

fig_seasonal, axes_seasonal = plt.subplots(
    4, 2,
    figsize=(12, 4 * 4),
    constrained_layout=True
)

# Cache für DataFrames (damit pro (ocean, model) nur einmal geladen wird)
df_cache_dict = {}

# -------------------------------------------------------------------------
# Main Loop: beide Figuren in einem Rutsch füllen
# -------------------------------------------------------------------------
for col_idx, ocean in enumerate(oceans):
    #for row_idx, (model_name, (ModelClass, model_path, model_kwargs)) in enumerate(model_files.items()):
    for row_idx, model_name in enumerate(model_files):

        df_cache = load_df_cache(ocean, model_name, df_cache_dict)

        # MSE nur einmal berechnen
        mse = mean_squared_error(df_cache["co2flux_pre"], df_cache["reconstructed_co2flux_pre"])
        print(f"{ocean} {model_name}: {mse}")

        # ===== Jahresmittel =====
        yearly = df_cache.groupby(df_cache["time_counter"].dt.year)
        yearly_avg_preds = yearly["reconstructed_co2flux_pre"].mean()
        yearly_avg_targets = yearly["co2flux_pre"].mean()
        # yearly_std_preds = yearly["predictions"].std()
        # yearly_std_targets = yearly["targets"].std()

        ax_y = axes_yearly[row_idx, col_idx]
        ax_y.plot(yearly_avg_preds.index, yearly_avg_preds.values, color="red", label="Reconstruction")
        ax_y.plot(yearly_avg_targets.index, yearly_avg_targets.values, color="black", label="Simulation")
        ax_y.text(
            0.98, 0.05,  # Position (relativ zu Achsenkoordinaten)
            f"MSE = {mse:.4f}",  # wissenschaftliche Notation (z. B. 1.23e-04)
            transform=ax_y.transAxes,  # Damit 0–1 = Achsenkoordinaten sind
            fontsize=10,
            color="black",
            ha="right",  # rechtsbündig
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
        )
        # Falls du wieder Std-Bänder willst, kannst du hier fill_between reaktivieren

        ax_y.set_xlabel("Year", fontsize=16)
        ax_y.set_ylabel(r"$CO_{2}$ flux pre", fontsize=16)
        if ocean == "Southern Ocean":

            ax_y.set_ylim(-1, 0.5)

        if ocean == "North Atlantic":

            ax_y.set_ylim(0, 1.5)
        ax_y.set_title(f"{ocean} ({model_name})", fontsize=18)
        ax_y.tick_params(axis='both', labelsize=12)
        ax_y.grid(True)

        # ===== Saisonale (monatliche) Mittel =====
        monthly = df_cache.groupby("month")
        monthly_avg_preds = monthly["reconstructed_co2flux_pre"].mean()
        monthly_avg_targets = monthly["co2flux_pre"].mean()
        monthly_std_preds = monthly["reconstructed_co2flux_pre"].std()
        monthly_std_targets = monthly["co2flux_pre"].std()

        ax_s = axes_seasonal[row_idx, col_idx]

        # x-Achse: vorhandene Monate (normalerweise 1–12)
        months = monthly_avg_preds.index.values
        month_labels = [calendar.month_abbr[m] for m in months]

        ax_s.plot(months, monthly_avg_preds.values, color="red", label="Reconstruction")
        ax_s.fill_between(
            months,
            (monthly_avg_preds - monthly_std_preds).values,
            (monthly_avg_preds + monthly_std_preds).values,
            color="red",
            alpha=0.2,
            label="Reconstruction ± std",
        )

        ax_s.plot(months, monthly_avg_targets.values, color="black", label="Simulation")
        ax_s.fill_between(
            months,
            (monthly_avg_targets - monthly_std_targets).values,
            (monthly_avg_targets + monthly_std_targets).values,
            color="black",
            alpha=0.2,
            label="Simulation ± std",
        )
        ax_s.text(
            0.98, 0.05,  # Position (relativ zu Achsenkoordinaten)
            f"MSE = {mse:.4f}",  # wissenschaftliche Notation (z. B. 1.23e-04)
            transform=ax_s.transAxes,  # Damit 0–1 = Achsenkoordinaten sind
            fontsize=10,
            color="black",
            ha="right",  # rechtsbündig
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
        )
        # 

        ax_s.set_xticks(months)
        ax_s.set_xticklabels(month_labels)
        ax_s.set_xlabel("Month",fontsize=16)
        ax_s.set_ylabel(r"$CO_{2}$ flux pre", fontsize=16)
        ax_s.tick_params(axis='both', labelsize=12)
        ocean_name = ocean
        if ocean == "Southern Ocean Simulation 1":
            ocean_name = "Southern Ocean"
        if ocean == "North Atlantic Simulation 1":
            ocean_name = "North Atlantic"
        ax_s.set_title(f"{ocean_name} ({model_name})", fontsize=18)
        ax_s.grid(True)


# -------------------------------------------------------------------------
# Gemeinsame Legenden & Speichern
# -------------------------------------------------------------------------
handles = [
    plt.Line2D([0], [0], color="red", label="Reconstruction"),
    Patch(facecolor="red", alpha=0.2, label="Reconstruction ± std"),
    plt.Line2D([0], [0], color="black", label="Simulation"),
    Patch(facecolor="black", alpha=0.2, label="Simulation ± std"),
]

# Jahresmittel-Plot
fig_yearly.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.0),
    ncol=2,
    fontsize=16,
)
fig_yearly.tight_layout(rect=[0, 0.05, 1, 1])


os.makedirs(out_dir, exist_ok=True)
fig_yearly.savefig(os.path.join(out_dir, f"{experiment_name}_annual_average_over_time.png"), dpi=150)
plt.close(fig_yearly)

# Saisonaler Plot
fig_seasonal.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=2,
    fontsize=16,
)
fig_seasonal.tight_layout(rect=[0, 0.05, 1, 1])
fig_seasonal.savefig(os.path.join(out_dir, f"{experiment_name}_monthly_seasonal_cycle.png"), dpi=150)
plt.close(fig_seasonal)
