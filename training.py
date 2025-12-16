from typing import List, Dict, Tuple
import torch
from torch.utils.data import DataLoader


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    patience,
    device):
    """
    Trainiert ein Modell mit Early Stopping und gibt Trainings- und Validierungsverluste zurück.
    """

    model.to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb.squeeze())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * xb.size(0)
            n_train += xb.size(0)

        avg_train_loss = total_train_loss / n_train
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb.squeeze())
                total_val_loss += loss.item() * xb.size(0)
                n_val += xb.size(0)

        avg_val_loss = total_val_loss / n_val
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"✅ Neues bestes Modell (Val Loss: {best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"⚠️ Keine Verbesserung seit {epochs_no_improve} Epoche(n)")

        if epochs_no_improve >= patience:
            print(f"\n⏹ Early Stopping nach {epoch+1} Epochen.")
            break

    return train_losses, val_losses, best_state_dict
