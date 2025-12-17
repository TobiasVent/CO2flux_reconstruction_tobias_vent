import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from models.lstm import LSTMModelTimeShap
from timeshap.wrappers import TorchModelWrapper
from timeshap.utils import calc_avg_event
from timeshap.explainer import local_event, local_feat, local_cell_level
from timeshap.explainer import local_pruning
import os
from models.attention_lstm import LSTMModelAttentionTemporalTimeShap
import shap
from configs.lstm_config import DATA_PATHS_LSTM, HPARAMS_LSTM
from configs.attention_lstm_config import DATA_PATHS_Attention_LSTM, HPARAMS_Attention_LSTM
from timeshap.plot import (
    plot_temp_coalition_pruning,
    plot_event_heatmap,
    plot_feat_barplot,
    plot_cell_level,
)
from collections import namedtuple

# Set random seed for reproducibility
np.random.seed(42)

# Define a container for one sample
Sample = namedtuple("Sample", ["features", "target", "meta"])

# -------------------------------------------------------------------------
# Utility functions for loading and filtering data
# -------------------------------------------------------------------------

def load_yearly_samples(pkl_path):
    """
    Load yearly samples from a pickle file.
    The pickle file contains a list of Sample objects.
    Returns:
        X: feature tensor of shape (N, T, D)
        y: target array of shape (N,)
        meta: list of metadata dictionaries
    """
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    X = np.stack([s.features for s in samples], axis=0).astype(np.float32)
    y = np.array([s.target for s in samples], dtype=np.float32)
    meta = [s.meta for s in samples]
    return X, y, meta


def filter_by_month(X, y, meta, month_int):
    """
    Filter samples by a specific calendar month.
    """
    idx = [
        i for i, m in enumerate(meta)
        if pd.Timestamp(m["time_counter"]).month == month_int
    ]
    X_m = X[idx]
    y_m = y[idx]
    meta_m = [meta[i] for i in idx]
    return X_m, y_m, meta_m


# -------------------------------------------------------------------------
# Device setup
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# Helper functions for TimeSHAP processing
# -------------------------------------------------------------------------

def extract_cell_matrix(cell_data, feature_names, window_size):
    """
    Convert TimeSHAP cell-level output into a (T, D) matrix of Shapley values.

    Parameters:
        cell_data: DataFrame with columns ['Event', 'Feature', 'Shapley Value']
        feature_names: list of feature names
        window_size: number of timesteps

    Returns:
        Matrix of shape (T, D) with Shapley values
    """
    num_features = len(feature_names)
    mat = np.zeros((window_size, num_features))

    for _, row in cell_data.iterrows():
        # Extract timestep index (e.g. "Event -1")
        t = int(row["Event"].split()[-1])

        # Convert negative indexing used by TimeSHAP
        t = window_size + t

        # Map feature name to column index
        f = feature_names.index(row["Feature"])

        mat[t, f] = row["Shapley Value"]

    return mat


def plot_heatmap(mean_vals, feature_names, title="Mean SHAP heatmap", experiment_name="experiment_name"):
    """
    Plot and save a heatmap of mean SHAP values across timesteps and features.
    """
    plt.figure(figsize=(12, 6))
    x_labels = ["t -3", "t -2", "t -1", "t -0"]

    sns.heatmap(
        mean_vals.T,
        cmap="Reds",
        xticklabels=x_labels,
        yticklabels=feature_names,
        annot=True,
    )

    plt.xlabel("Time step")
    plt.ylabel("Feature")
    plt.title(title)

    out_dir = f"/data/stu231428/Master_Thesis/main/final_plots/feature_importance/{experiment_name}/"
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, title + experiment_name)
    plt.savefig(out_file, dpi=150)
    plt.close()


# -------------------------------------------------------------------------
# Main SHAP workflow for yearly data
# -------------------------------------------------------------------------

def run_shap_workflow_for_model_yearly(
    model,
    model_name,
    yearly_paths,
    feature_columns,
    dynamic_features,
    months_to_explain=(1, 7),
    n_samples=1,
    nsamples_shap=1000,
    seed=42,
):
    """
    Run a TimeSHAP workflow for a given model on yearly datasets.
    Computes cell-level SHAP values, aggregates them, and produces plots.
    """
    np.random.seed(seed)
    model.eval()
    wrapped_model = TorchModelWrapper(model)

    # Mapping for readable feature labels
    label_map = {
        "SST": "SST",
        "SAL": "SAL",
        "ice_frac": "Ice Fraction",
        "mixed_layer_depth": "Mixed Layer Depth",
        "heat_flux_down": "Heat Flux Down",
        "water_flux_up": "Water Flux Up",
        "stress_X": "Wind Stress X",
        "stress_Y": "Wind Stress Y",
        "currents_X": "Currents X",
        "currents_Y": "Currents Y",
        "co2flux_pre": r"$CO_{2}$ flux pre",
    }

    # Indices of dynamic features
    dyn_idx = [feature_columns.index(f) for f in dynamic_features]

    for pkl_path, region_name, experiment_name in yearly_paths:
        X_all, y_all, meta_all = load_yearly_samples(pkl_path)
        window_size = X_all.shape[1]

        for month_int in months_to_explain:
            month_name = {1: "January", 7: "July"}.get(month_int, str(month_int))

            X_month, y_month, meta_month = filter_by_month(
                X_all, y_all, meta_all, month_int
            )

            if len(X_month) == 0:
                print(f"No samples for {region_name} {month_name}")
                continue

            # Compute average baseline event from normalized inputs
            X_month_2d = X_month.reshape(-1, X_month.shape[2])
            X_month_2d_df = pd.DataFrame(X_month_2d, columns=feature_columns)

            average_event = calc_avg_event(
                X_month_2d_df,
                numerical_feats=list(range(len(feature_columns))),
                categorical_feats=[],
            )

            # Randomly select samples for explanation
            n_pick = min(n_samples, len(X_month))
            sample_indices = np.random.choice(len(X_month), n_pick, replace=False)

            cell_matrices = []

            for idx in sample_indices:
                instance_3d = X_month[idx : idx + 1]

                # Prune irrelevant timesteps
                pruning_dict = {"tol": 0}
                _, coal_prun_idx = local_pruning(
                    wrapped_model.predict_last_hs,
                    instance_3d,
                    pruning_dict=pruning_dict,
                    baseline=average_event,
                )

                pruned_idx = instance_3d.shape[1] + coal_prun_idx

                # Event-level SHAP
                event_data = local_event(
                    wrapped_model.predict_last_hs,
                    instance_3d,
                    event_dict={"rs": seed, "nsamples": nsamples_shap},
                    entity_col="entity",
                    entity_uuid=f"sample_{idx}",
                    baseline=average_event,
                    pruned_idx=pruned_idx,
                )

                # Feature-level SHAP
                feature_data = local_feat(
                    wrapped_model.predict_last_hs,
                    instance_3d,
                    {
                        "rs": seed,
                        "nsamples": nsamples_shap,
                        "feature_names": feature_columns,
                        "plot_features": {f: f for f in feature_columns},
                    },
                    entity_col="entity",
                    entity_uuid=f"sample_{idx}",
                    baseline=average_event,
                    pruned_idx=pruned_idx,
                )

                # Cell-level SHAP
                cell_data = local_cell_level(
                    wrapped_model.predict_last_hs,
                    instance_3d,
                    {
                        "rs": seed,
                        "nsamples": nsamples_shap,
                        "top_x_events": min(4, window_size),
                        "top_x_feats": len(feature_columns),
                    },
                    event_data,
                    feature_data,
                    entity_col="entity",
                    entity_uuid=f"sample_{idx}",
                    baseline=average_event,
                    pruned_idx=pruned_idx,
                )

                mat = extract_cell_matrix(cell_data, feature_columns, window_size)
                cell_matrices.append(mat)

            # Stack SHAP matrices across samples
            cell_matrices = np.stack(cell_matrices, axis=0)

            # Keep only dynamic features
            cell_matrices_dyn = cell_matrices[:, :, dyn_idx]
            cell_matrices_2d_dyn = cell_matrices_dyn.reshape(
                cell_matrices_dyn.shape[0], -1
            )

            # Corresponding input values
            X_dyn = X_month[sample_indices][:, :, dyn_idx]
            X_values_flat = X_dyn.reshape(X_dyn.shape[0], -1)

            # Build flattened feature names with time indices
            dyn_names_flat = [
                f"{label_map.get(feat, feat)} (t)" if t == 0 else f"{label_map.get(feat, feat)} (t-{t})"
                for t in range(window_size - 1, -1, -1)
                for feat in dynamic_features
            ]

            # Beeswarm plot
            shap.summary_plot(
                cell_matrices_2d_dyn,
                features=X_values_flat,
                feature_names=dyn_names_flat,
                max_display=15,
                show=False,
            )

            fig = plt.gcf()
            fig.suptitle(
                f"{region_name} {month_name} 2018 ({model_name})",
                fontsize=18,
            )

            out_png = (
                f"/media/stu231428/1120 7818/Master_github/datasets/plots/feature_importance/"
                f"beeswarm_{region_name}_{month_name}_{model_name}_{experiment_name}.png"
            )
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)


# -------------------------------------------------------------------------
# Load models


lstm_model = LSTMModelTimeShap(
    input_size=HPARAMS_LSTM["input_size"],
    hidden_dim=HPARAMS_LSTM["hidden_dim"],
    num_layers=HPARAMS_LSTM["num_layers"],
    dropout=HPARAMS_LSTM["dropout"],
)
lstm_model.load_state_dict(
    torch.load(DATA_PATHS_LSTM["model_out"])
)

lstm_model_attention_temporal = LSTMModelAttentionTemporalTimeShap(
    input_size=HPARAMS_Attention_LSTM["input_size"],
    hidden_dim=HPARAMS_Attention_LSTM["hidden_dim"],
    dropout=HPARAMS_Attention_LSTM["dropout"],
).to(device)
lstm_model_attention_temporal.load_state_dict(
    torch.load(DATA_PATHS_Attention_LSTM["model_out"])
)

models_list = [
    {"model": lstm_model, "name": "LSTM"},
    {"model": lstm_model_attention_temporal, "name": "Attention LSTM"},
]

# -------------------------------------------------------------------------
# Dataset and feature configuration
# -------------------------------------------------------------------------

yearly_paths = [
    (
        "/media/stu231428/1120 7818/Master_github/datasets/yearly/North_Atlantic_test_2018_experiment_1.pkl",
        "North Atlantic",
        "experiment_1",
    ),
    (
        "/media/stu231428/1120 7818/Master_github/datasets/yearly/Southern_Ocean_test_2018_experiment_1.pkl",
        "Southern Ocean",
        "experiment_1",
    ),
]

feature_columns = [
    "SST",
    "SAL",
    "ice_frac",
    "mixed_layer_depth",
    "heat_flux_down",
    "water_flux_up",
    "stress_X",
    "stress_Y",
    "currents_X",
    "currents_Y",
    "sin_lat",
    "cos_lat",
    "sin_lon",
    "cos_lon",
]

dynamic_features = [
    "SST",
    "SAL",
    "ice_frac",
    "mixed_layer_depth",
    "heat_flux_down",
    "water_flux_up",
    "stress_X",
    "stress_Y",
    "currents_X",
    "currents_Y",
]

# -------------------------------------------------------------------------
# Run SHAP analysis for all models
# -------------------------------------------------------------------------

for m in models_list:
    run_shap_workflow_for_model_yearly(
        model=m["model"],
        model_name=m["name"],
        yearly_paths=yearly_paths,
        feature_columns=feature_columns,
        dynamic_features=dynamic_features,
    )
