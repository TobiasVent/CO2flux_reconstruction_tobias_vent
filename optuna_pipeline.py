
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time
from collections import namedtuple
Sample = namedtuple("Sample", ["features", "target", "meta"])


import torch
from torch.utils.data import DataLoader, TensorDataset
from dataset import PointwiseSampleDataset ,MLPDataset # Beispiel
from models.mlp import MLPModel
from models.attention_lstm import LSTMModelAttention, LSTMModelAttentionTemporal
#from models.lstm_with_attention_2_layer import LSTMModelAttention2Layer#
from models.lstm import LSTMModel, GRUModel
#from models.da_rnn_encoder import InputAttentionEncoder
import pickle            # Beispiel
import optuna
from optuna.trial import TrialState
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
# Daten laden
from collections import namedtuple
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import configs.data_paths as Test_Data_PATHS
# Keep your Sample and LSTMSampleDataset classes
Sample = namedtuple("Sample", ["features", "target", "meta"])




import xgboost as xgb
from xgboost.callback import EarlyStopping
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def objective_xgboost(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "reg:squarederror",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "random_state": 42,
        "n_jobs": -1

    }

    model = xgb.XGBRegressor(**params)

    # Einfaches Training ohne Early Stopping
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse

def objective_input_attention_encoder(trial, train_dataset, val_dataset, device):
    # Beispiel: gleiche Hyperparameter-Ideen wie bei den anderen LSTMs
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # N und T automatisch aus Dataset bestimmen
    sample_x, sample_y = train_dataset[0]      # sample_x: (T, N)
    T = sample_x.shape[0]
    N = sample_x.shape[1]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = InputAttentionEncoder(
        N=N,
        M=hidden_dim,
        T=T,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = 20

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # xb: (batch, T, N)
            outputs = model(xb)                # (batch,)
            loss = criterion(outputs, yb.squeeze())
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb.squeeze())
                val_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)
        val_loss /= n_samples

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_val_loss


# Optuna objective function
def objective_lstm(trial, train_dataset, val_dataset, device):





    # 🔍 Hyperparameter-Sampling
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = 0
    if num_layers > 1:

        dropout = trial.suggest_categorical("dropout", [0.0,0.1,0.2,0.3,0.4, 0.5])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-5, 1e-2,log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)

    model = LSTMModel(input_size=14, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)





    # Simple training loop for 5 epochs (for speed)
    patience = 10  # how many epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs =20  # or whatever max epochs you want

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.squeeze())
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb.squeeze())
                val_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)
        val_loss /= n_samples

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break  # stop training early

    return best_val_loss


def objective_gru(trial, train_dataset, val_dataset, device):





    # 🔍 Hyperparameter-Sampling
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = 0
    if num_layers > 1:

        dropout = trial.suggest_categorical("dropout", [0.0,0.1,0.2,0.3,0.4, 0.5])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-5, 1e-2,log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)

    model = GRUModel(input_size=14, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)





    # Simple training loop for 5 epochs (for speed)
    patience = 10  # how many epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs =20  # or whatever max epochs you want

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.squeeze())
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb.squeeze())
                val_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)
        val_loss /= n_samples

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break  # stop training early

    return best_val_loss


def objective_lstm_attention(trial, train_dataset, val_dataset, device):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])

    dropout = trial.suggest_categorical("dropout", [0.0,0.1,0.2,0.3,0.4, 0.5])
    lr = trial.suggest_float("lr", 1e-5, 1e-2,log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    # Initialize model with suggested hyperparams
    model = LSTMModelAttention(
        input_size=14,   # Adjust input size as needed
        hidden_dim=hidden_dim,
            # You can optimize this too if you want
        dropout=dropout
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train for a few epochs (adjust as needed)
    patience = 10  # how many epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = 20  # or whatever max epochs you want

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.squeeze())
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb.squeeze())
                val_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)
        val_loss /= n_samples

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break  # stop training early

    return best_val_loss





def objective_lstm_attention_temporal(trial, train_dataset, val_dataset, device):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])

    dropout = trial.suggest_categorical("dropout", [0.0,0.1,0.2,0.3,0.4, 0.5])
    lr = trial.suggest_float("lr", 1e-5, 1e-2,log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    # Initialize model with suggested hyperparams
    model = LSTMModelAttentionTemporal(
        input_size=14,   # Adjust input size as needed
        hidden_dim=hidden_dim,
            # You can optimize this too if you want
        dropout=dropout
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train for a few epochs (adjust as needed)
    patience = 10  # how many epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = 20  # or whatever max epochs you want

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.squeeze())
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb.squeeze())
                val_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)
        val_loss /= n_samples

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break  # stop training early

    return best_val_loss


def objective_lstm_2layerattention(trial, train_dataset, val_dataset, device):
    # Suggest hyperparameters
    hidden_dim_1 = trial.suggest_categorical("hidden_dim_1", [32, 64, 128, 256])
    hidden_dim_2 = trial.suggest_categorical("hidden_dim_2", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_categorical("dropout", [0.0,0.1,0.2,0.3,0.4, 0.5])
    lr = trial.suggest_float("lr", 1e-5, 1e-2,log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
    # Initialize model with suggested hyperparams
    model = LSTMModelAttention2Layer(
        input_size=14,   # Adjust input size as needed
        hidden_dim_1 =hidden_dim_1,
        hidden_dim_2 = hidden_dim_2,
        num_layers = num_layers,
            # You can optimize this too if you want
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train for a few epochs (adjust as needed)
    patience = 10  # how many epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = 20  # or whatever max epochs you want

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.squeeze())
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb.squeeze())
                val_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)
        val_loss /= n_samples

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break  # stop training early

    return best_val_loss



def objective_mlp(trial, train_dataset, val_dataset, device):
    # Suggest hyperparameters for MLPModel
# Sample hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dims = []
    for i in range(n_layers):
        hidden_dims.append(trial.suggest_int(f"hidden_dim_{i}", 32, 256))
    dropout = trial.suggest_categorical("dropout", [0.0,0.1,0.2,0.3,0.4, 0.5])
    lr = trial.suggest_float("lr", 1e-5, 1e-2,log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_loader_mpl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
    val_loader_mlp = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
    model = MLPModel(input_dim=40, hidden_dims=hidden_dims, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    patience = 10  # how many epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = 20  # or whatever max epochs you want

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader_mpl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.squeeze())
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader_mlp:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb.squeeze())
                val_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)
        val_loss /= n_samples

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break  # stop training early

    return best_val_loss



def objective_tabnet(trial, X_train,y_train,X_val,y_val, device):

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_d = trial.suggest_int("n_d", 8, 64, step=8)
    n_a = trial.suggest_int("n_a", 8, 64, step=8)
    n_steps = trial.suggest_int("n_steps", 3, 10)
    gamma = trial.suggest_float("gamma", 1.0, 2.0, step=0.1)
    lambda_sparse = trial.suggest_float("lambda_sparse", 0.00001, 0.001, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)




    model = TabNetRegressor(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_params=dict(lr=learning_rate),
        verbose=0
    )

    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["mse"],
        patience=10,
        max_epochs=20,
        batch_size=batch_size,

        num_workers=2,
    )


    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse



def objective_random_forest(trial, X_train, y_train, X_val, y_val):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train.ravel())  # .ravel() falls y_train (N,1) ist
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse

train_path = Test_Data_PATHS["training_samples"]
val_path = Test_Data_PATHS["validation_samples"]

with open(train_path, "rb") as f:
    train_samples = pickle.load(f)


with open(val_path, "rb") as f:
    val_samples = pickle.load(f)

# Set seeds for reproducibility
def seed_all(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Datasets / Dataloaders laden

train_dataset = PointwiseSampleDataset(train_samples)
val_dataset = PointwiseSampleDataset(val_samples)



train_dataset_mlp = MLPDataset(train_samples)
val_dataset_mlp = MLPDataset(val_samples)

sampler = optuna.samplers.TPESampler(seed=42)

study_lstm = optuna.create_study(direction="minimize",storage="sqlite:///hyperparameter_LSTM.db",study_name="lstm_study",sampler=sampler)
print("Starting optimization for LSTMModel...")
study_lstm.optimize(lambda trial: objective_lstm(trial, train_dataset, val_dataset, device), n_trials=50)
print(f"Best LSTMModel trial: {study_lstm.best_trial.params}")
study_gru = optuna.create_study(direction="minimize",storage="sqlite:///hyperparameter_GRU.db",study_name="gru_study",sampler=sampler)
print("Starting optimization for GRUModel...")
study_gru.optimize(lambda trial: objective_gru(trial, train_dataset, val_dataset, device), n_trials=50)
print(f"Best LSTMModel trial: {study_gru.best_trial.params}")
study_temporal_attention_lstm = optuna.create_study(direction="minimize",storage="sqlite:///hyperparameter_attention_lstm_temporal.db",study_name="attention_lstm_temporal_study",sampler=sampler)
print("Starting optimization for LSTM_attention_temporal...")
study_temporal_attention_lstm.optimize(lambda trial: objective_lstm_attention_temporal(trial, train_dataset, val_dataset, device), n_trials=50)
print(f"Best LSTMModel trial: {study_temporal_attention_lstm.best_trial.params}")

study_mlp = optuna.create_study(direction="minimize",storage="sqlite:///hyperparameter_MLP.db",study_name="MLP_study",sampler=sampler)
print("Starting optimization for MLPModel...")
study_mlp.optimize(lambda trial: objective_mlp(trial, train_dataset_mlp, val_dataset_mlp, device), n_trials=50)
print(f"Best MLPModel trial: {study_mlp.best_trial.params}")

# study_attention_lstm = optuna.create_study(direction="minimize",storage="sqlite:///hyperparameter_lstm_attention.db",study_name="lstm_attention_study",sampler=sampler)
# print("Starting optimization for LSTM_Attention Model...")
# study_attention_lstm.optimize(lambda trial: objective_lstm_attention(trial, train_dataset, val_dataset, device), n_trials=50)
# print(f"Best LSTM_Attention Model trial: {study_attention_lstm.best_trial.params}")
# study_input_att = optuna.create_study(
#     direction="minimize",
#     storage="sqlite:///hyperparameter_input_attention_encoder.db",
#     study_name="input_attention_encoder_study",
#     sampler=sampler
# )
# print("Starting optimization for InputAttentionEncoder...")
# study_input_att.optimize(
#     lambda trial: objective_input_attention_encoder(trial, train_dataset, val_dataset, device),
#     n_trials=50
# )
# print(f"Best InputAttentionEncoder trial: {study_input_att.best_trial.params}")



del train_dataset
del val_dataset
del train_dataset_mlp
del val_dataset_mlp


def flatten_sample(sample, n_dyn=10, n_static=4, window_size=4):
    x = sample.features  # shape: (4, 14)
    x_dyn = x[:, :n_dyn].flatten()   # 4×10 = 40
    x_static = x[0, n_dyn:]          # 4 static features
    return np.concatenate([x_dyn, x_static])
X_train = np.array([flatten_sample(s) for s in train_samples])
y_train = np.array([s.target for s in train_samples]).reshape(-1, 1)

X_val = np.array([flatten_sample(s) for s in val_samples])
y_val = np.array([s.target for s in val_samples]).reshape(-1, 1)



study_xgb = optuna.create_study(
    direction="minimize",
    storage="sqlite:///hyperparameter_xgboost.db",
    study_name="xgboost_study",
    sampler=sampler
)
print("Starting optimization for XGBoost...")
study_xgb.optimize(lambda trial: objective_xgboost(trial, X_train, y_train, X_val, y_val), n_trials=50)
print(f"Best XGBoost trial: {study_xgb.best_trial.params}")


study_rf = optuna.create_study(direction="minimize", storage="sqlite:///hyperparameter_rf.db", study_name="rf_study", sampler=sampler)
print("Starting optimization for RandomForestRegressor...")
study_rf.optimize(lambda trial: objective_random_forest(trial, X_train, y_train, X_val, y_val), n_trials=50)
print(f"Best Random Forest trial: {study_rf.best_trial.params}")

# study_attention_2layerlstm = optuna.create_study(direction="minimize",storage="sqlite:///hyperparameter_lstm_attention_2layer.db",study_name="lstm_attention2layer_study",sampler=sampler)
# print("Starting optimization for LSTM_Attention2layer Model...")
# study_attention_2layerlstm.optimize(lambda trial: objective_lstm_2layerattention(trial, train_dataset, val_dataset, device), n_trials=50)
# print(f"Best LSTM_Attention2layer trial: {study_attention_2layerlstm.best_trial.params}")

# study_attention_tabnet = optuna.create_study(direction="minimize",storage="sqlite:///hyperparameter_tabnet.db",study_name="lstm_tabnet_study",sampler=sampler,load_if_exists=True)
# print("Starting optimization for tabnet Model...")
# study_attention_tabnet.optimize(lambda trial: objective_tabnet(trial, X_train, y_train,X_val,y_val, device), n_trials=50)
# print(f"Best tabnet trial: {study_attention_tabnet.best_trial.params}")