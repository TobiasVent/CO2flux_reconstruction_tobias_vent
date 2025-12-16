import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import random
from models.attention_lstm import LSTMModelAttentionTemporalWithWeights
from collections import namedtuple

# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

Sample = namedtuple("Sample", ["features", "target", "meta"])

# -------------------------------------------------------------------------
# Device setup
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------------
# Helper function: robust month extraction from meta["time_counter"]
# -------------------------------------------------------------------------
def _to_month(x):
    """
    Converts different time formats (Timestamp, datetime64, string)
    into a calendar month.
    """
    return pd.to_datetime(x).month


def load_samples_by_month(pkl_path, months, window_size=None, input_size=None):
    """
    Loads a List[Sample] from a pickle file and groups samples by month.

    Returns a dictionary:
        month (int) -> (X, y)

    where:
        X has shape (N, T, D)
        y has shape (N,)
    """
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    buckets = {m: [] for m in months}

    for s in samples:
        # Sample contains:
        #   s.features -> (T, D)
        #   s.target
        #   s.meta["time_counter"]
        m = _to_month(s.meta["time_counter"])
        if m in buckets:
            buckets[m].append(s)

    out = {}
    for m, s_list in buckets.items():
        if len(s_list) == 0:
            out[m] = (np.empty((0, 0, 0)), np.empty((0,), dtype=np.float32))
            continue

        X = np.stack([ss.features for ss in s_list], axis=0)  # (N, T, D)
        y = np.array([ss.target for ss in s_list], dtype=np.float32)

        if window_size is not None and X.shape[1] != window_size:
            raise ValueError(
                f"Window mismatch for month={m}: got T={X.shape[1]}, expected {window_size}"
            )
        if input_size is not None and X.shape[2] != input_size:
            raise ValueError(
                f"Input mismatch for month={m}: got D={X.shape[2]}, expected {input_size}"
            )

        out[m] = (X, y)

    return out


# -------------------------------------------------------------------------
# Load trained attention-based LSTM model
# -------------------------------------------------------------------------
lstm_model_attention_temporal = LSTMModelAttentionTemporalWithWeights(
    input_size=14,      # 10 dynamic + 4 additional (e.g. positional / static features)
    hidden_dim=256,
    dropout=0.4
).to(device)

lstm_model_attention_temporal.load_state_dict(
    torch.load(
        "/data/stu231428/Master_Thesis/main/trained_models/attention_temporal_lstm_with_pos.pt",
        map_location=device
    )
)
lstm_model_attention_temporal.eval()


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
region_map = {
    "EQ_Pacific": "Equatorial Pacific",
    "North_Atlantic": "North Atlantic",
    "Arctic": "Arctic Ocean",
    "Southern_Ocean": "Southern Ocean",
    "Southern_ocean": "Southern Ocean"
}

# Months of interest (calendar months)
month_map = {
    1: "January",
    7: "July"
}

# Paths to yearly test datasets
data_paths = [
    ("/media/stu231428/1120 7818/Master_github/datasets/yearly/North_Atlantic_test_2018_experiment_1.pkl",
     "North_Atlantic"),
    ("/media/stu231428/1120 7818/Master_github/datasets/yearly/Southern_Ocean_test_2018_experiment_1.pkl",
     "Southern_Ocean"),
]

# Dynamic features (only these are visualized)
dynamic_features = [
    "SST", "SAL", "ice_frac", "mixed_layer_depth", "heat_flux_down",
    "water_flux_up", "stress_X", "stress_Y", "currents_X", "currents_Y"
]

# Human-readable labels for plots
label_map = {
    "SST": "SST",
    "SAL": "Salinity",
    "ice_frac": "Ice Fraction",
    "mixed_layer_depth": "Mixed Layer Depth",
    "heat_flux_down": "Heat Flux Down",
    "water_flux_up": "Water Flux Up",
    "stress_X": "Wind Stress X",
    "stress_Y": "Wind Stress Y",
    "currents_X": "Currents X",
    "currents_Y": "Currents Y",
}

dynamic_labels = [label_map.get(f, f) for f in dynamic_features]

# Indices of dynamic features in the 14D input vector
# Assumption: first 10 features are dynamic
dynamic_idx = list(range(10))

# Output directory
out_dir = "/media/stu231428/1120 7818/Master_github/datasets/plots/feature_importance/"
os.makedirs(out_dir, exist_ok=True)


# -------------------------------------------------------------------------
# Prepare figures:
# rows = regions, columns = selected months
# -------------------------------------------------------------------------
n_rows = len(data_paths)
n_cols = len(month_map)

fig_combined, axes_combined = plt.subplots(
    n_rows, n_cols, figsize=(12, 4 * n_rows), constrained_layout=True
)
fig_feat, axes_feat = plt.subplots(
    n_rows, n_cols, figsize=(12, 4 * n_rows), constrained_layout=True
)
fig_temp, axes_temp = plt.subplots(
    n_rows, n_cols, figsize=(14, 5 * n_rows), constrained_layout=True
)

# Ensure axes are always 2D arrays
if n_rows == 1 and n_cols == 1:
    axes_combined = np.array([[axes_combined]])
    axes_feat = np.array([[axes_feat]])
    axes_temp = np.array([[axes_temp]])
elif n_rows == 1:
    axes_combined = np.array([axes_combined])
    axes_feat = np.array([axes_feat])
    axes_temp = np.array([axes_temp])
elif n_cols == 1:
    axes_combined = np.array([[ax] for ax in axes_combined])
    axes_feat = np.array([[ax] for ax in axes_feat])
    axes_temp = np.array([[ax] for ax in axes_temp])


# -------------------------------------------------------------------------
# Loop over datasets and compute aggregated attention-based importances
# -------------------------------------------------------------------------
months_wanted = list(month_map.keys())
input_size = 14

for row_idx, (pkl_path, region_key) in enumerate(data_paths):
    region_name = region_map.get(region_key, region_key)

    print(f"\n=== Loading region: {region_name} ===")
    print(pkl_path)

    month_data = load_samples_by_month(
        pkl_path,
        months=months_wanted,
        window_size=None,
        input_size=input_size
    )

    for col_idx, (m, month_name) in enumerate(month_map.items()):
        X_month, y_month = month_data[m]

        ax_c = axes_combined[row_idx, col_idx]
        ax_f = axes_feat[row_idx, col_idx]
        ax_t = axes_temp[row_idx, col_idx]

        if X_month.shape[0] == 0:
            print(f"No samples for {region_name} - {month_name}")
            for ax in (ax_c, ax_f, ax_t):
                ax.set_title(f"{region_name}\n{month_name}\n(no data)")
                ax.axis("off")
            continue

        window_size = X_month.shape[1]
        print(f"Explaining {region_name} - {month_name}")
        print(f"X_month shape: {X_month.shape}")  # (N, T, 14)

        # -------------------------------------------------------------
        # Collect attention weights for randomly sampled instances
        # -------------------------------------------------------------
        n_available = X_month.shape[0]
        n_samples = min(300, n_available)
        sample_indices = np.random.choice(n_available, n_samples, replace=False)

        all_input_attn = []
        all_temp_attn = []

        with torch.no_grad():
            for instance_3d in X_month[sample_indices]:
                x_tensor = torch.tensor(
                    instance_3d, dtype=torch.float32, device=device
                ).unsqueeze(0)

                prediction, input_attention, temporal_attention = (
                    lstm_model_attention_temporal(x_tensor)
                )

                # Input attention: one tensor per timestep -> (T, D)
                input_attention_tensor = (
                    torch.stack(input_attention, dim=0).squeeze(1)
                )
                all_input_attn.append(input_attention_tensor)

                # Temporal attention: shape -> (T,)
                all_temp_attn.append(
                    temporal_attention.squeeze(0).squeeze(-1)
                )

        # -------------------------------------------------------------
        # Stack attention weights
        # -------------------------------------------------------------
        all_input_attn = torch.stack(all_input_attn, dim=0)  # (N, T, D)
        all_temp_attn = torch.stack(all_temp_attn, dim=0)    # (N, T)

        # -------------------------------------------------------------
        # Combine input-level and temporal attention
        # -------------------------------------------------------------
        combined_attn = all_input_attn * all_temp_attn.unsqueeze(-1)  # (N, T, D)

        # Average over samples -> (T, D)
        mean_combined_attn = combined_attn.mean(dim=0).cpu().numpy()

        # Keep only dynamic features
        mean_combined_attn_dyn = mean_combined_attn[:, dynamic_idx]  # (T, 10)

        # -------------------------------------------------------------
        # 1) Combined attention heatmap (dynamic features only)
        # -------------------------------------------------------------
        sns.heatmap(
            mean_combined_attn_dyn.T,
            cmap="Reds",
            xticklabels=[
                ("t" if k == window_size - 1 else f"t-{window_size - 1 - k}")
                for k in range(window_size)
            ],
            yticklabels=dynamic_labels,
            ax=ax_c,
            cbar=False,
            annot=True
        )
        ax_c.set_title(f"{region_name}\n{month_name}", fontsize=16)
        ax_c.set_xlabel("Time step", fontsize=16)
        ax_c.set_ylabel("Feature", fontsize=16)
        ax_c.tick_params(which="both", labelsize=14)

        # -------------------------------------------------------------
        # 2) Global feature importance (averaged over time)
        # -------------------------------------------------------------
        feature_importance_dyn = mean_combined_attn_dyn.mean(axis=0)

        ax_f.bar(range(len(dynamic_features)), feature_importance_dyn)
        ax_f.set_xticks(range(len(dynamic_features)))
        ax_f.set_xticklabels(dynamic_features, rotation=45, ha="right")
        ax_f.set_ylabel("Importance")
        ax_f.set_title(f"{region_name}\n{month_name}")

        # -------------------------------------------------------------
        # 3) Temporal importance (averaged over samples)
        # -------------------------------------------------------------
        mean_temp_attention = all_temp_attn.mean(dim=0).cpu().numpy()

        ax_t.bar(
            [f"t-{window_size - 1 - k}" for k in range(window_size)],
            mean_temp_attention
        )
        ax_t.set_ylabel("Importance")
        ax_t.set_xlabel("Time step")
        ax_t.set_title(f"{region_name}\n{month_name}")

    gc.collect()


# -------------------------------------------------------------------------
# Save figures
# -------------------------------------------------------------------------
# Shared colorbar for combined attention heatmaps
try:
    fig_combined.colorbar(
        axes_combined[0, 0].collections[0],
        ax=axes_combined,
        location="right",
        shrink=0.4
    )
except Exception as e:
    print(f"Colorbar skipped: {e}")

fig_combined.savefig(
    os.path.join(out_dir, "combined_attention_subplot_dynamic_only.png"),
    dpi=150,
    bbox_inches="tight"
)

fig_feat.suptitle(
    "Global feature importance per basin and month (dynamic features only)",
    fontsize=14
)
fig_feat.savefig(
    os.path.join(out_dir, "global_feature_importance_subplot_dynamic_only.png"),
    dpi=150,
    bbox_inches="tight"
)

fig_temp.suptitle(
    "Temporal importance per basin and month",
    fontsize=14
)
fig_temp.savefig(
    os.path.join(out_dir, "temporal_importance_subplot.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig_combined)
plt.close(fig_feat)
plt.close(fig_temp)

print("Plots saved.")
