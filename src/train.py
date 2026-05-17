"""
train.py
--------
Training loop for signal RECONSTRUCTION (MSE loss).

Shared by FCNet, RNNNet, and LSTMNet.

Features
--------
- MSELoss (assignment requirement)
- Adam optimizer + ReduceLROnPlateau scheduler
- Gradient clipping (max_norm=1.0, critical for RNN stability)
- Best-model checkpoint: results/<model_name>_best.pt
- Returns history dict with train_loss and val_loss per epoch
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mse_on_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute mean MSE over all batches in loader (no gradient)."""
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for x_flat, x_seq, y, _ in loader:
            x_flat = x_flat.to(device)
            x_seq  = x_seq.to(device)
            y      = y.to(device)
            pred   = model(x_flat, x_seq)     # [batch, 10]
            loss   = criterion(pred, y)
            total_loss += loss.item() * x_flat.size(0)
            total_n    += x_flat.size(0)
    return total_loss / total_n


# ── Main training function ────────────────────────────────────────────────────

def train_model(
    model:        nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    n_epochs:     int   = 50,
    patience:     int   = 15,
    lr:           float = 1e-3,
    device:       torch.device | None = None,
    model_name:   str   = "model",
    verbose:      bool  = True,
) -> dict:
    """
    Train model with MSE loss and return a history dict.

    Returns
    -------
    history : dict
        "train_loss" : list[float]  — MSE per epoch on train set
        "val_loss"   : list[float]  — MSE per epoch on val set
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history: dict[str, list] = {"train_loss": [], "val_loss": []}
    best_val_loss    = float("inf")
    best_epoch       = 1
    patience_counter = 0
    checkpoint_path  = RESULTS_DIR / f"{model_name}_best.pt"

    for epoch in range(1, n_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        run_loss, total_n = 0.0, 0
        for x_flat, x_seq, y, _ in train_loader:
            x_flat = x_flat.to(device)
            x_seq  = x_seq.to(device)
            y      = y.to(device)

            optimizer.zero_grad()
            pred = model(x_flat, x_seq)       # [batch, 10]
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            run_loss += loss.item() * x_flat.size(0)
            total_n  += x_flat.size(0)

        train_loss = run_loss / total_n
        val_loss   = _mse_on_loader(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  [{model_name}] Early stop at epoch {epoch}"
                          f" (no improvement for {patience} epochs)")
                break

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"  [{model_name}] Epoch {epoch:3d}/{n_epochs}"
                f" | Train MSE {train_loss:.6f}"
                f" | Val MSE {val_loss:.6f}"
            )

    if verbose:
        print(f"  Best val MSE: {best_val_loss:.6f}  (epoch {best_epoch})"
              f"  →  saved {checkpoint_path.name}")

    history["best_epoch"]    = best_epoch
    history["best_val_loss"] = best_val_loss
    return history

