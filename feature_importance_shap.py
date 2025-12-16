import torch
from dataset import PointwiseSampleDatasetMonthMLP
from models.lstm import LSTMModel, LSTMModelTimeShap
from models.mlp import MLPModel
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

from collections import namedtuple
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
# -------------------------------------------------------------------------
# Device setup
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Keep your Sample and LSTMSampleDataset classes
Sample = namedtuple("Sample", ["features", "target", "meta"])

def flatten_sample(sample, n_dyn=10, n_static=4, window_size=4):
    x = sample.features  # shape: (4, 14)
    x_dyn = x[:, :n_dyn].flatten()   # 4×10 = 40
    x_static = x[0, n_dyn:]          # 4 static features
    return np.concatenate([x_dyn, x_static])


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
# Function: Combined SHAP Beeswarm Plot
# -------------------------------------------------------------------------
def plot_combined_shap_beeswarm(shap_values_dict, X_dict, feature_columns, region_name, out_path_png):
    """
    Creates one figure with all models (rows) and two months (columns: January, July).
    shap_values_dict: {model_name: {month_name: shap_values}}
    X_dict: {month_name: X_month_2d[sample_indices]}
    feature_columns: list of features
    region_name: str
    """
    months = ["January", "July"]
    models = list(shap_values_dict.keys())
    n_rows = len(models)
    n_cols = len(months)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False)

    for i, model_name in enumerate(models):
        for j, month in enumerate(months):
            shap_values = shap_values_dict[model_name][month]
            X_month = X_dict[month]

            shap.summary_plot(
                shap_values, 
                features=X_month, 
                feature_names=[f"{feat}_t-{t}" for t in range(X_month.shape[1]//len(feature_columns)-1,-1,-1) for feat in feature_columns],
                show=False, 
                plot_type="dot",
                
            )
            axes[i, j].set_title(f"{model_name} - {month}")

    plt.suptitle(f"SHAP Beeswarm — {region_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    plt.savefig(out_path_png, dpi=150)
    plt.close()
    print(f"Combined SHAP plot saved: {out_path_png}")

# -------------------------------------------------------------------------
# Model prediction wrapper for MLP
# -------------------------------------------------------------------------
def model_predict(x):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = MLP(x_tensor).cpu().numpy()
    return np.atleast_1d(preds)

# -------------------------------------------------------------------------
# Main workflow to run SHAP and create combined plots
# -------------------------------------------------------------------------

def run_shap_workflow_for_model_yearly(
    model, model_name,
    yearly_paths,                      # list of (pkl_path, region_name, experiment_name)
    feature_columns,
    dynamic_features,
    months_to_explain=(1, 7),           # (Jan, Jul)
    n_samples=1,
    
    seed=42
):
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "MLP":
        model.eval()
    

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


            X_month_2d = flatten_sample_x_month(X_month)
            if model_name == "MLP":
                explainer = shap.KernelExplainer(model_predict, shap.sample(X_month_2d, 2000))
                model.eval()
            elif model_name == "XGBoost":
                explainer = shap.TreeExplainer(model, shap.sample(X_month_2d, 2000))
            #X_month_2d = X_month.reshape(X_month.shape[0], -1)
            sample_indices = np.random.choice(X_month_2d.shape[0], n_samples, replace=False)

            

            X_sampled = X_month_2d[sample_indices]
            shap_values = explainer.shap_values(X_sampled)
            dyn_idx = [feature_columns.index(f) for f in dynamic_features]
            shap_values = shap_values[ :, :40]
            

            

            # dyn_names_flat = [
            #     f"{feat}   (t)" if t == 0 else f"{feat} (t-{t})"
            #     for t in range(4 - 1, -1, -1)
            #     for feat in dynamic_features
            # ]
            dyn_names_flat = [
                f"{label_map.get(feat, feat)} (t)" if t == 0 else f"{label_map.get(feat, feat)} (t-{t})"
                for t in range(4 - 1, -1, -1)
                for feat in dynamic_features
            ]


            
            #dyn_names_flat = [f"{feat}_t-{t}" for t in range(4 - 1, -1, -1) for feat in dynamic_features]



            X_values_flat = X_month[sample_indices][:, :, :len(dynamic_features)]
            X_values_flat = X_values_flat.reshape(X_values_flat.shape[0], -1)
            out_png = f"/media/stu231428/1120 7818/Master_github/datasets/plots/feature_importance/local_baseline_{model_name}_{month_name}_shap_{region_name}.png"
            shap.summary_plot(
                shap_values,
                features=X_values_flat,
                feature_names=dyn_names_flat,
                max_display=15,
                show=False
                
            )
            fig = plt.gcf()
            ax = plt.gca()
            # === DEBUG PRINT ===
            print("\n[DEBUG] Axes in current SHAP figure:")
            for idx, ax in enumerate(fig.axes):
                print(f"  Axis {idx}: {ax} -> title='{ax.get_title()}' xlabel='{ax.get_xlabel()}' ylabel='{ax.get_ylabel()}'")
            
            cbar_ax = fig.axes[1]
            cbar_ax.tick_params(labelsize=16)   # adjust font size
            cbar_ax.set_ylabel("Feature value", fontsize=16)
            fig.axes[0].tick_params(axis='both', labelsize=16) 
            fig.axes[0].set_xlabel("Shapley value", fontsize=16)
            #ax.tick_params(axis='both', labelsize=16)
            fig.suptitle(
            f"{region_name} {month_name} 2018 ({model_name})",
            fontsize=18
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # <-- Platz für Titel schaffen
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)







# def run_shap_workflow_combined(models_list, data_paths, feature_columns, region_map):
#     label_map = {
#         "SST": "SST",
#         "SAL": "SAL",
#         "ice_frac": "Ice Fraction",
#         "mixed_layer_depth": "Mixed Layer Depth",
#         "heat_flux_down": "Heat Flux Down",
#         "water_flux_up": "Water Flux Up",
#         "stress_X": "Wind Stress X",
#         "stress_Y": "Wind Stress Y",
#         "currents_X": "Currents X",
#         "currents_Y": "Currents Y",
#         "co2flux_pre": r"$CO_{2}$ flux pre"
#     }

#     dynamic_features = [
#         'SST', 'SAL', 'ice_frac', 'mixed_layer_depth', 'heat_flux_down',
#         'water_flux_up', 'stress_X', 'stress_Y', 'currents_X', 'currents_Y'  ### <<< NEW >>>
#     ]
#     for path_data, path_avg, month_map, experiment_name in data_paths:
#         # Load data
#         with open(path_data, "rb") as f:
#             data = pickle.load(f)
#         with open(path_avg, "rb") as f:
#             train_data = pickle.load(f)
#         X_train = np.array([flatten_sample(s) for s in train_data])
#         #X_train = np.array([s.features.flatten() for s in train_data])

#         # Determine region
#         region_key = [k for k in region_map if k in path_data][0]
#         region_name = region_map[region_key]

#         shap_values_dict = {}
#         X_dict = {}

#         for model_entry in models_list:
#             model = model_entry["model"]
#             model_name = model_entry["name"]



#             shap_values_dict[model_name] = {}

#             for i, month_name in month_map.items():
#                 X_month = data["X"][i]
#                 X_month_2d = flatten_sample_x_month(X_month)
#                 if model_name == "MLP":
#                     explainer = shap.KernelExplainer(model_predict, shap.sample(X_month_2d, 2000))
#                     model.eval()
#                 elif model_name == "XGBoost":
#                     explainer = shap.TreeExplainer(model, shap.sample(X_month_2d, 2000))
#                 #X_month_2d = X_month.reshape(X_month.shape[0], -1)
#                 sample_indices = np.random.choice(X_month_2d.shape[0], 300, replace=False)

#                 X_sampled = X_month_2d[sample_indices]
#                 shap_values = explainer.shap_values(X_sampled)
#                 dyn_idx = [feature_columns.index(f) for f in dynamic_features]
#                 shap_values = shap_values[ :, :40]
#                 shap_values_dict[model_name][month_name] = shap_values
#                 if month_name not in X_dict:
#                     X_dict[month_name] = X_sampled
                

#                 # dyn_names_flat = [
#                 #     f"{feat}   (t)" if t == 0 else f"{feat} (t-{t})"
#                 #     for t in range(4 - 1, -1, -1)
#                 #     for feat in dynamic_features
#                 # ]
#                 dyn_names_flat = [
#                     f"{label_map.get(feat, feat)} (t)" if t == 0 else f"{label_map.get(feat, feat)} (t-{t})"
#                     for t in range(4 - 1, -1, -1)
#                     for feat in dynamic_features
#                 ]


                
#                 #dyn_names_flat = [f"{feat}_t-{t}" for t in range(4 - 1, -1, -1) for feat in dynamic_features]



#                 X_values_flat = X_month[sample_indices][:, :, :len(dynamic_features)]
#                 X_values_flat = X_values_flat.reshape(X_values_flat.shape[0], -1)
#                 out_png = f"/data/stu231428/Master_Thesis/final_main/plots/feature_importance_feature_names/local_baseline_{model_name}_{month_name}_shap_{region_key}.png"
#                 shap.summary_plot(
#                     shap_values,
#                     features=X_values_flat,
#                     feature_names=dyn_names_flat,
#                     max_display=15,
#                     show=False
                    
#                 )
#                 fig = plt.gcf()
#                 ax = plt.gca()
#                 # === DEBUG PRINT ===
#                 print("\n[DEBUG] Axes in current SHAP figure:")
#                 for idx, ax in enumerate(fig.axes):
#                     print(f"  Axis {idx}: {ax} -> title='{ax.get_title()}' xlabel='{ax.get_xlabel()}' ylabel='{ax.get_ylabel()}'")
                
#                 cbar_ax = fig.axes[1]
#                 cbar_ax.tick_params(labelsize=16)   # adjust font size
#                 cbar_ax.set_ylabel("Feature value", fontsize=16)
#                 fig.axes[0].tick_params(axis='both', labelsize=16) 
#                 fig.axes[0].set_xlabel("Shapley value", fontsize=16)
#                 #ax.tick_params(axis='both', labelsize=16)
#                 fig.suptitle(
#                 f"{region_name} {month_name} 2018 ({model_name})",
#                 fontsize=18
#                 )
#                 fig.tight_layout(rect=[0, 0, 1, 0.95])  # <-- Platz für Titel schaffen
#                 fig.savefig(out_png, dpi=150, bbox_inches="tight")
#                 plt.close(fig)

        # # Save combined beeswarm plot
        # out_png = f"/data/stu231428/Master_Thesis/final_main/plots/feature_importance/_shap_{region_key}.png"
        # plot_combined_shap_beeswarm(shap_values_dict, X_dict, feature_columns, region_name, out_png)

# -------------------------------------------------------------------------
# Load models
# -------------------------------------------------------------------------
MLP = MLPModel(hidden_dims=[207, 248, 198], input_dim=44, dropout=0).to(device)
MLP.load_state_dict(torch.load('/data/stu231428/Master_Thesis/main/trained_models/mlp_with_pos.pt'))

XGBoost = pickle.load(open("/data/stu231428/Master_Thesis/main/trained_models/xg_boost_with_pos_model.pkl", "rb"))

models_list = [
    {"model": MLP, "name": "MLP"},
    {"model": XGBoost, "name": "XGBoost"}
]

# -------------------------------------------------------------------------
# Dataset info
# -------------------------------------------------------------------------
region_map = {
    "EQ_Pacific": "Equatorial Pacific",
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

feature_columns = ['SST', 'SAL', 'ice_frac', 'mixed_layer_depth', 'heat_flux_down',
                   'water_flux_up', 'stress_X', 'stress_Y', 'currents_X', 'currents_Y']

# -------------------------------------------------------------------------
# Run combined SHAP workflow
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
