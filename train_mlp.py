# scripts/train_mlp.py
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import MLPDatasetWithPos
from models.mlp import MLPModel
from training import train_model
from configs.mlp_config import DATA_PATHS_MLP, HPARAMS_MLP
from collections import namedtuple

# Define a container for one sample
Sample = namedtuple("Sample", ["features", "target", "meta"])


def main():
    # ================= Load data =================
    with open(DATA_PATHS_MLP["train_samples"], "rb") as f:
        train_samples = pickle.load(f)

    with open(DATA_PATHS_MLP["val_samples"], "rb") as f:
        val_samples = pickle.load(f)

    # Select device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets
    train_dataset = MLPDatasetWithPos(train_samples)
    val_dataset = MLPDatasetWithPos(val_samples)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=HPARAMS_MLP["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=HPARAMS_MLP["batch_size"],
        shuffle=False,
    )

    # ================= Model =================
    model = MLPModel(
        input_dim=HPARAMS_MLP["input_dim"],
        hidden_dims=HPARAMS_MLP["hidden_dims"],
        dropout=HPARAMS_MLP["dropout"],
    )

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=HPARAMS_MLP["lr"],
        weight_decay=HPARAMS_MLP["weight_decay"],
    )

    # ================= Training =================
    train_losses, val_losses, best_state_dict = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=HPARAMS_MLP["num_epochs"],
        patience=HPARAMS_MLP["patience"],
        device=device,
    )

    # Load and save the best model
    model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), DATA_PATHS_MLP["model_out"])
    print(f"\nBest model saved to: {DATA_PATHS_MLP['model_out']}")

    # ================= Plot losses =================
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation MSE Loss (MLP)")
    plt.legend()
    plt.grid(True)
    plt.savefig(DATA_PATHS_MLP["plot_out"])

    # ================= Save losses =================
    losses_dict = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open(DATA_PATHS_MLP["losses_out"], "wb") as f:
        pickle.dump(losses_dict, f)

    print(f"Train/Validation losses saved to: {DATA_PATHS_MLP['losses_out']}")


if __name__ == "__main__":
    main()
