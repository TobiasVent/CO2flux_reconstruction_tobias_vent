# scripts/train_lstm.py
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from dataset import MLPDatasetWithPos
from models.mlp import MLPModel
from training import train_model
from configs.xgboost_config import DATA_PATHS, HPARAMS
from collections import namedtuple
Sample = namedtuple("Sample", ["features", "target", "meta"])
import numpy as np

def main():
    # ============== Daten laden =====================
    with open(DATA_PATHS["train_samples"], "rb") as f:
        train_samples = pickle.load(f)
    with open(DATA_PATHS["val_samples"], "rb") as f:
        val_samples = pickle.load(f)
    def flatten_sample(sample, n_dyn=10, n_static=4, window_size=4):
        x = sample.features  # shape: (4, 14)
        x_dyn = x[:, :n_dyn].flatten()   # 4×10 = 40
        x_static = x[0, n_dyn:]          # 4 static features
        return np.concatenate([x_dyn, x_static])


    X_train = np.array([flatten_sample(s) for s in train_samples])
    y_train = np.array([s.target for s in train_samples]).reshape(-1, 1)

    X_val = np.array([flatten_sample(s) for s in val_samples])
    y_val = np.array([s.target for s in val_samples]).reshape(-1, 1)

        # ============== Modell =====================
    model = xgb.XGBRegressor(
        booster=HPARAMS["booster"],
        n_estimators=HPARAMS["n_estimators"],
        max_depth=HPARAMS["max_depth"],
        learning_rate=HPARAMS["learning_rate"],
        subsample=HPARAMS["subsample"],
        colsample_bytree=HPARAMS["colsample_bytree"],
        gamma=HPARAMS["gamma"],
        reg_lambda=HPARAMS["reg_lambda"],
        reg_alpha=HPARAMS["reg_alpha"],
        min_child_weight=HPARAMS["min_child_weight"]
    )

    model.fit(X_train, y_train.ravel(), verbose=True)

    # --- Evaluate ---
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    print(f"Validation MSE: {mse:.6f}")

    # bestes Modell speichern


    # Save the model
    #model_filename = "/data/stu231428/Master_Thesis/main/trained_models/xg_boost_with_pos_model.pkl"
    pickle.dump(model, open(DATA_PATHS['model_out'], 'wb'))



    


if __name__ == "__main__":
    main()
