

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
from timeshap.plot import plot_temp_coalition_pruning, plot_event_heatmap, plot_feat_barplot, plot_cell_level
np.random.seed(42)
from collections import namedtuple


Sample = namedtuple("Sample", ["features", "target", "meta"])
#script to plot the average cellwise shapley values from a entire basin at one specific month


def load_yearly_samples(pkl_path):
    # pkl enthält: List[Sample(features, target, meta)]
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    X = np.stack([s.features for s in samples], axis=0).astype(np.float32)  # (N,T,D)
    y = np.array([s.target for s in samples], dtype=np.float32)            # (N,)
    meta = [s.meta for s in samples]                                       # list[dict]
    return X, y, meta

def filter_by_month(X, y, meta, month_int):
    idx = [i for i,m in enumerate(meta) if pd.Timestamp(m["time_counter"]).month == month_int]
    X_m = X[idx]
    y_m = y[idx]
    meta_m = [meta[i] for i in idx]
    return X_m, y_m, meta_m
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_cell_matrix(cell_data, feature_names, window_size):
    """
    Convert TimeSHAP cell_data to (T, D) NumPy matrix of Shapley values.
    - cell_data has columns: ['Event','Feature','Shapley Value']
    - Event is like 'Event -4' (string)
    - Feature is the feature name (string)
    """
    num_features = len(feature_names)
    mat = np.zeros((window_size, num_features))

    for _, row in cell_data.iterrows():
        # Parse event index (e.g., "Event -4" -> -4)
        t = int(row['Event'].split()[-1])

# -------------------------------------------
        # Convert negative index (TimeSHAP uses negative indexing from the end)
        t = window_size + t  # so "Event -1" becomes last timestep

        # Feature name -> column index
        f = feature_names.index(row['Feature'])

        mat[t, f] = row['Shapley Value']
    return mat

def plot_heatmap(mean_vals, feature_names, title="Mean SHAP heatmap",experiment_name = "experiment_name"):
    plt.figure(figsize=(12, 6))
    x_labels= ["t -3","t -2","t -1","t -0"]
    sns.heatmap(mean_vals.T, cmap="Reds",
                xticklabels=x_labels,
                yticklabels=feature_names, annot= True)
    plt.xlabel("Time step")
    plt.ylabel("Feature")
    plt.title(title)
    out_dir = f"/data/stu231428/Master_Thesis/main/final_plots/feature_importance/{experiment_name}/"
    
    out_file = os.path.join(out_dir, title + experiment_name)
    plt.savefig(out_file, dpi=150)

def run_shap_workflow_for_model_yearly(
    model, model_name,
    yearly_paths,                      # list of (pkl_path, region_name, experiment_name)
    feature_columns,
    dynamic_features,
    months_to_explain=(1, 7),           # (Jan, Jul)
    n_samples=1,
    nsamples_shap=1000,
    seed=42
):
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    wrapped_model = TorchModelWrapper(model)

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
        "co2flux_pre": r"$CO_{2}$ flux pre"
    }

    dyn_idx = [feature_columns.index(f) for f in dynamic_features]

    for pkl_path, region_name, experiment_name in yearly_paths:
        X_all, y_all, meta_all = load_yearly_samples(pkl_path)
        window_size = X_all.shape[1]
        assert window_size == 4, "Dein dyn_names_flat unten ist aktuell auf window=4 gebaut."

        for month_int in months_to_explain:
            month_name = {1:"January", 7:"July"}.get(month_int, str(month_int))
            X_month, y_month, meta_month = filter_by_month(X_all, y_all, meta_all, month_int)
            if len(X_month) == 0:
                print(f"[WARN] Keine Samples für {region_name} {month_name} in {pkl_path}")
                continue

            # --- Baseline passend zu deinem Preprocessing: aus NORMALISIERTEM X_month ---
            X_month_2d = X_month.reshape(-1, X_month.shape[2])
            X_month_2d_df = pd.DataFrame(X_month_2d, columns=feature_columns)
            average_event = calc_avg_event(
                X_month_2d_df,
                numerical_feats=list(range(len(feature_columns))),
                categorical_feats=[]
            )

            # --- sample indices ---
            n_pick = min(n_samples, len(X_month))
            sample_indices = np.random.choice(len(X_month), n_pick, replace=False)

            cell_matrices = []
            for idx in sample_indices:
                instance_3d = X_month[idx:idx+1]  # (1,T,D)

                pruning_dict = {'tol': 0}
                coal_plot_data, coal_prun_idx = local_pruning(
                    wrapped_model.predict_last_hs,
                    instance_3d,
                    pruning_dict=pruning_dict,
                    baseline=average_event
                )
                pruned_idx = instance_3d.shape[1] + coal_prun_idx

                event_data = local_event(
                    wrapped_model.predict_last_hs,
                    instance_3d,
                    event_dict={'rs':seed,'nsamples':nsamples_shap},
                    entity_col="entity",
                    entity_uuid=f"sample_{idx}",
                    baseline=average_event,
                    pruned_idx=pruned_idx
                )
                feature_data = local_feat(
                    wrapped_model.predict_last_hs,
                    instance_3d,
                    {
                        'rs':seed,
                        'nsamples':nsamples_shap,
                        'feature_names':feature_columns,
                        'plot_features':{f:f for f in feature_columns}
                    },
                    entity_col="entity",
                    entity_uuid=f"sample_{idx}",
                    baseline=average_event,
                    pruned_idx=pruned_idx
                )
                cell_data = local_cell_level(
                    wrapped_model.predict_last_hs,
                    instance_3d,
                    {
                        'rs':seed,
                        'nsamples':nsamples_shap,
                        'top_x_events':min(4, window_size),
                        'top_x_feats':len(feature_columns)
                    },
                    event_data,
                    feature_data,
                    entity_col="entity",
                    entity_uuid=f"sample_{idx}",
                    baseline=average_event,
                    pruned_idx=pruned_idx
                )

                mat = extract_cell_matrix(cell_data, feature_columns, window_size)  # (T,D)
                cell_matrices.append(mat)

            cell_matrices = np.stack(cell_matrices, axis=0)  # (N,T,D)

            # ===== NUR DYNAMICS für Beeswarm =====
            cell_matrices_dyn = cell_matrices[:, :, dyn_idx]                 # (N,T,D_dyn)
            cell_matrices_2d_dyn = cell_matrices_dyn.reshape(cell_matrices_dyn.shape[0], -1)

            # Eingabewerte passend zu diesen SHAPs (WICHTIG: gleiche Indizes!)
            X_dyn = X_month[sample_indices][:, :, dyn_idx]
            X_values_flat = X_dyn.reshape(X_dyn.shape[0], -1)

            dyn_names_flat = [
                f"{label_map.get(feat, feat)} (t)" if t == 0 else f"{label_map.get(feat, feat)} (t-{t})"
                for t in range(window_size - 1, -1, -1)
                for feat in dynamic_features
            ]

            # --- Beeswarm ---
            shap.summary_plot(
                cell_matrices_2d_dyn,
                features=X_values_flat,
                feature_names=dyn_names_flat,
                max_display=15,
                show=False
            )
            fig = plt.gcf()
            
            if len(fig.axes) > 1:
                cbar_ax = fig.axes[1]
                cbar_ax.set_ylabel("Feature value", fontsize=16)
                cbar_ax.tick_params(labelsize=16)
            fig.axes[0].tick_params(axis='both', labelsize=16)
            fig.axes[0].set_xlabel("Shapley value", fontsize=16)
            fig.suptitle(f"{region_name} {month_name} 2018 ({model_name})", fontsize=18)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            out_png = (
                f"/media/stu231428/1120 7818/Master_github/datasets/plots/feature_importance/"
                
                f"beeswarm_{region_name}_{month_name}_{model_name}_{experiment_name}.png"
            )
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # --- Optional: Dependence Plot für SST(t) (korrekter Name!) ---
            target_feature = "SST (t)"
            if target_feature in dyn_names_flat:
                idx_sst_t0 = dyn_names_flat.index(target_feature)
                shap.dependence_plot(
                    ind=idx_sst_t0,
                    shap_values=cell_matrices_2d_dyn,
                    features=X_values_flat,
                    feature_names=dyn_names_flat,
                    interaction_index=None,
                    show=False
                )
                out_dep_png = (
                    f"/media/stu231428/1120 7818/Master_github/datasets/plots/feature_importance/"
                    f"dependency_SST_t0_{region_name}_{month_name}_{model_name}_{experiment_name}.png"
                )
                plt.title(f"SHAP Dependency Plot for {target_feature} ({region_name}, {month_name}, {model_name})")
                plt.savefig(out_dep_png, dpi=150, bbox_inches="tight")
                plt.close()


# Modelle laden
# -------------------------------------------------------------------------
#lstm_model = LSTMModelTimeShap(input_size=10, hidden_dim=128, dropout=0, num_layers=3).to(device)
lstm_model = LSTMModelTimeShap(input_size=14, hidden_dim=128, dropout=0, num_layers=3).to(device)
lstm_model.load_state_dict(torch.load('/data/stu231428/Master_Thesis/main/trained_models/lsmt_with_pos.pt'))
lstm_model_attention_temporal = LSTMModelAttentionTemporalTimeShap(input_size=14, hidden_dim=256, dropout=0.4).to(device)
lstm_model_attention_temporal.load_state_dict(torch.load('/data/stu231428/Master_Thesis/main/trained_models/attention_temporal_lstm_with_pos.pt'))

models_list = [
    {"model": lstm_model, "name": "LSTM"},
    {"model": lstm_model_attention_temporal, "name": "Attention LSTM"}
]

region_map = {
    #"EQ_Pacific": "Equatorial Pacific",
    "North_Atlantic": "North Atlantic",
    "Arctic": "Arctic Ocean",
    "Southern_Ocean": "Southern Ocean",
    "Southern_ocean": "Southern Ocean"
}
month_map_experiment_1 = {
    105: "January",
    111: "July"
}
yearly_paths = [
    ("/media/stu231428/1120 7818/Master_github/datasets/yearly/North_Atlantic_test_2018_experiment_1.pkl",
     "North Atlantic", "experiment_1"),
    ("/media/stu231428/1120 7818/Master_github/datasets/yearly/Southern_Ocean_test_2018_experiment_1.pkl",
     "Southern Ocean", "experiment_1"),
]
month_map = {17: "January", 18: "July"}

# feature_columns = ['SST', 'SAL', 'ice_frac', 'mixed_layer_depth', 'heat_flux_down',
#                    'water_flux_up', 'stress_X', 'stress_Y', 'currents_X', 'currents_Y']


feature_columns = [
    'SST', 'SAL', 'ice_frac', 'mixed_layer_depth', 'heat_flux_down',
    'water_flux_up', 'stress_X', 'stress_Y', 'currents_X', 'currents_Y',
    'sin_lat', 'cos_lat', 'sin_lon', 'cos_lon'  ### <<< NEW >>>
]

# -------------------------------------------------------------------------
# Loop over datasets and compute aggregated SHAP
# -------------------------------------------------------------------------

dynamic_features = [
    'SST', 'SAL', 'ice_frac', 'mixed_layer_depth', 'heat_flux_down',
    'water_flux_up', 'stress_X', 'stress_Y', 'currents_X', 'currents_Y'  ### <<< NEW >>>
]

for m in models_list:
    run_shap_workflow_for_model_yearly(
        m["model"], m["name"],
        yearly_paths=yearly_paths,
        feature_columns=feature_columns,
        dynamic_features=dynamic_features
    )
