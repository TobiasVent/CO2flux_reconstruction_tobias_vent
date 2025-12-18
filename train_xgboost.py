# scripts/train_xgboost.py
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np

from dataset import MLPDatasetWithPos
from models.mlp import MLPModel
from training import train_model
from configs.xgboost_config import DATA_PATHS_XGBoost, HPARAMS_XGBoost
from collections import namedtuple

# Define a container for one sample
Sample = namedtuple("Sample", ["features", "target", "meta"])


def main():
    # ================= Load data =================
    with open("/data/training_set/training_set.pkl", "rb") as f:
        train_samples = pickle.load(f)

    with open("/data/training_set/validation_set.pkl", "rb") as f:
        val_samples = pickle.load(f)

    # ================= Feature flattening =================
    def flatten_sample(sample, n_dyn=10, n_static=4, window_size=4):
        """
        Flatten one temporal sample into a single feature vector.
        Dynamic features are flattened across the time window.
        Static features are taken from the first timestep only.
        """
        x = sample.features                     # shape: (window_size, num_features)
        x_dyn = x[:, :n_dyn].flatten()          # window_size × n_dyn
        x_static = x[0, n_dyn:]                 # static features
        return np.concatenate([x_dyn, x_static])

    # Build training and validation feature matrices
    X_train = np.array([flatten_sample(s) for s in train_samples])
    y_train = np.array([s.target for s in train_samples]).reshape(-1, 1)

    X_val = np.array([flatten_sample(s) for s in val_samples])
    y_val = np.array([s.target for s in val_samples]).reshape(-1, 1)

    # ================= Model =================
    model = xgb.XGBRegressor(
        booster=HPARAMS_XGBoost["booster"],
        n_estimators=HPARAMS_XGBoost["n_estimators"],
        max_depth=HPARAMS_XGBoost["max_depth"],
        learning_rate=HPARAMS_XGBoost["learning_rate"],
        subsample=HPARAMS_XGBoost["subsample"],
        colsample_bytree=HPARAMS_XGBoost["colsample_bytree"],
        gamma=HPARAMS_XGBoost["gamma"],
        reg_lambda=HPARAMS_XGBoost["reg_lambda"],
        reg_alpha=HPARAMS_XGBoost["reg_alpha"],
        min_child_weight=HPARAMS_XGBoost["min_child_weight"],
    )

    # ================= Training =================
    model.fit(X_train, y_train.ravel(), verbose=True)

    # ================= Evaluation =================
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    print(f"Validation MSE: {mse:.6f}")

    # ================= Save model =================
    pickle.dump(model, open(DATA_PATHS_XGBoost["model_out"], "wb"))


if __name__ == "__main__":
    main()
