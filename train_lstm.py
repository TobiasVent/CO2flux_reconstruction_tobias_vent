# scripts/train_lstm.py
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import PointwiseSampleDataset
from models.lstm import LSTMModel
from training import train_model
from configs.lstm_config import DATA_PATHS, HPARAMS
from collections import namedtuple
Sample = namedtuple("Sample", ["features", "target", "meta"])

def main():
    # ============== Daten laden =====================
    with open(DATA_PATHS["train_samples"], "rb") as f:
        train_samples = pickle.load(f)
    with open(DATA_PATHS["val_samples"], "rb") as f:
        val_samples = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PointwiseSampleDataset(train_samples)
    val_dataset = PointwiseSampleDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=HPARAMS["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=HPARAMS["batch_size"],
        shuffle=False,
    )

    # ============== Modell =====================
    model = LSTMModel(
        input_size=HPARAMS["input_size"],
        hidden_dim=HPARAMS["hidden_dim"],
        num_layers=HPARAMS["num_layers"],
        dropout=HPARAMS["dropout"],
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"],
    )

    # ============== Training =====================
    train_losses, val_losses, best_state_dict = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=HPARAMS["num_epochs"],
        patience=HPARAMS["patience"],
        device=device,
    )

    # bestes Modell speichern
    model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), DATA_PATHS["model_out"])
    print(f"\n📁 Bestes Modell gespeichert unter: {DATA_PATHS['model_out']}")

    # ============== Plot =====================
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation MSE Loss (LSTM)")
    plt.legend()
    plt.grid(True)
    plt.savefig(DATA_PATHS["plot_out"])

    # ============== Verluste speichern =====================
    losses_dict = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open(DATA_PATHS["losses_out"], "wb") as f:
        pickle.dump(losses_dict, f)

    print(f"📂 Train/Val Losses gespeichert unter: {DATA_PATHS['losses_out']}")


if __name__ == "__main__":
    main()
