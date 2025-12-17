# scripts/train_attention_lstm.py
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import PointwiseSampleDataset
from models.attention_lstm import LSTMModelAttentionTemporal
from training import train_model
from configs.attention_lstm_config import DATA_PATHS_Attention_LSTM, HPARAMS_Attention_LSTM
from collections import namedtuple

# Define a container for one sample
Sample = namedtuple("Sample", ["features", "target", "meta"])


def main():
    # ================= Load data =================
    with open(DATA_PATHS_Attention_LSTM["train_samples"], "rb") as f:
        train_samples = pickle.load(f)

    with open(DATA_PATHS_Attention_LSTM["val_samples"], "rb") as f:
        val_samples = pickle.load(f)

    # Select device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets
    train_dataset = PointwiseSampleDataset(train_samples)
    val_dataset = PointwiseSampleDataset(val_samples)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=HPARAMS_Attention_LSTM["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=HPARAMS_Attention_LSTM["batch_size"],
        shuffle=False,
    )

    # ================= Model =================
    model = LSTMModelAttentionTemporal(
        input_size=HPARAMS_Attention_LSTM["input_size"],
        hidden_dim=HPARAMS_Attention_LSTM["hidden_dim"],
        dropout=HPARAMS_Attention_LSTM["dropout"],
    )

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=HPARAMS_Attention_LSTM["lr"],
        weight_decay=HPARAMS_Attention_LSTM["weight_decay"],
    )

    # ================= Training =================
    train_losses, val_losses, best_state_dict = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=HPARAMS_Attention_LSTM["num_epochs"],
        patience=HPARAMS_Attention_LSTM["patience"],
        device=device,
    )

    # Load and save the best model
    model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), DATA_PATHS_Attention_LSTM["model_out"])
    print(f"\nBest model saved to: {DATA_PATHS_Attention_LSTM['model_out']}")

    # ================= Plot losses =================
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation MSE Loss (Attention LSTM)")
    plt.legend()
    plt.grid(True)
    plt.savefig(DATA_PATHS_Attention_LSTM["plot_out"])

    # ================= Save losses =================
    losses_dict = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open(DATA_PATHS_Attention_LSTM["losses_out"], "wb") as f:
        pickle.dump(losses_dict, f)

    print(f"Train/Validation losses saved to: {DATA_PATHS_Attention_LSTM['losses_out']}")


if __name__ == "__main__":
    main()
