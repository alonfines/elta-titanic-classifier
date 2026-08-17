"""Train the Titanic survival classifier and write all artifacts needed for inference to disk.

Standalone and reproducible: loads `data/train.csv`, splits it into train/validation, fits
preprocessing on the training split only, trains a small PyTorch MLP with early stopping, and
saves everything `ds_app.py` needs — model weights, the fitted preprocessor, model config, the
raw validation split (for the Streamlit evaluation page), and a per-epoch training log (for
reviewing/tuning hyperparameters later).

Usage:
    python train.py
    python train.py --epochs 300 --patience 20 --lr 5e-4

Outputs (all under --output-dir, default `models/`), regenerated fresh on every run:
    titanic_model.pt  - best-validation-loss model state_dict
    model_config.json - hyperparams + feature names, needed to reconstruct the model
    preprocessor.pkl  - fitted TitanicPreprocessor (identical logic used at inference)
    val_split.csv     - raw held-out validation rows (incl. Survived), for evaluation
    training_log.csv  - per-epoch train/val loss and metrics
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import DROPOUT, HIDDEN_DIM, TitanicNet
from preprocessing import TitanicPreprocessor

DEFAULT_DATA_PATH = Path("data/train.csv")
DEFAULT_OUTPUT_DIR = Path("models")
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to train.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write artifacts")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--epochs", type=int, default=300, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience (epochs)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_raw_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise SystemExit(
            f"{data_path} not found. Run `python fetch_data.py` first to download it from Kaggle."
        )
    return pd.read_csv(data_path)


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Sweep decision thresholds on validation probabilities and return the one maximizing F1.

    BCEWithLogitsLoss optimizes log-loss, not F1, so the model was never trained to make 0.5 a
    good cutoff — and 0.5 is an arbitrary choice for a ~38%-positive target anyway. This is
    disclosed as validation-set-tuned (not free-lunch): `train.py` reports both the default-0.5
    and tuned-threshold metrics side by side rather than only the tuned, better-looking number.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    # precision_recall_curve appends a final (precision=1, recall=0) pair with no matching
    # threshold (used for plotting the curve's end) — drop it before searching.
    precisions, recalls = precisions[:-1], recalls[:-1]
    denom = precisions + recalls
    f1s = np.divide(2 * precisions * recalls, denom, out=np.zeros_like(denom), where=denom > 0)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


@torch.no_grad()
def evaluate(model: nn.Module, loss_fn: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    logits = model(X)
    loss = loss_fn(logits, y).item()
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    y_np = y.numpy().astype(int)
    return {
        "loss": loss,
        "accuracy": accuracy_score(y_np, preds),
        "f1": f1_score(y_np, preds),
        "roc_auc": roc_auc_score(y_np, probs),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(args.data_path)
    train_df, val_df = train_test_split(
        df, test_size=args.val_size, stratify=df["Survived"], random_state=args.seed
    )
    print(
        f"Split {len(df)} rows -> {len(train_df)} train / {len(val_df)} val "
        f"(survival rate: train={train_df['Survived'].mean():.3f}, val={val_df['Survived'].mean():.3f})"
    )
    # Persisted so ds_app.py evaluates on exactly this held-out set, without re-deriving the split.
    val_df.to_csv(args.output_dir / "val_split.csv", index=False)

    preprocessor = TitanicPreprocessor()
    X_train = preprocessor.fit_transform(train_df)  # fit on the training split only
    X_val = preprocessor.transform(val_df)
    y_train = train_df["Survived"].to_numpy(dtype=np.float32)
    y_val = val_df["Survived"].to_numpy(dtype=np.float32)
    preprocessor.save(args.output_dir / "preprocessor.pkl")
    print(f"Preprocessed to {preprocessor.n_features} features: {preprocessor.feature_names_}")

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True
    )

    model = TitanicNet(n_features=preprocessor.n_features, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(xb)
            n_seen += len(xb)
        train_loss = epoch_train_loss / n_seen

        val_metrics = evaluate(model, loss_fn, X_val_t, y_val_t)
        epoch_row = {"epoch": epoch, "train_loss": train_loss}
        epoch_row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(epoch_row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:>3}  train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
                f"val_acc={val_metrics['accuracy']:.3f}  val_f1={val_metrics['f1']:.3f}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} (no val_loss improvement for {args.patience} epochs).")
            break

    model.load_state_dict(best_state)
    default_metrics = evaluate(model, loss_fn, X_val_t, y_val_t)  # decision threshold = 0.5
    print(
        f"\nBest epoch: {best_epoch}  (default threshold 0.5)  "
        f"val_loss={default_metrics['loss']:.4f}  val_acc={default_metrics['accuracy']:.3f}  "
        f"val_f1={default_metrics['f1']:.3f}  val_roc_auc={default_metrics['roc_auc']:.3f}"
    )

    # Threshold tuning: pick the cutoff that maximizes F1 on the validation probabilities,
    # rather than defaulting to 0.5. Explicitly disclosed as val-set-tuned — both metric sets
    # are saved and reported, not just the tuned (better-looking) one.
    model.eval()  # explicit, not just inherited from the evaluate() call above — don't rely on call order
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_t)).numpy()
    threshold, _ = find_best_threshold(y_val.astype(int), val_probs)
    tuned_preds = (val_probs >= threshold).astype(int)
    tuned_metrics = {
        "loss": default_metrics["loss"],  # loss and ROC-AUC don't depend on the decision cutoff
        "accuracy": accuracy_score(y_val.astype(int), tuned_preds),
        "f1": f1_score(y_val.astype(int), tuned_preds),
        "roc_auc": default_metrics["roc_auc"],
    }
    print(
        f"Tuned threshold={threshold:.3f}  "
        f"val_acc={tuned_metrics['accuracy']:.3f}  val_f1={tuned_metrics['f1']:.3f}  "
        f"(default-0.5 val_f1 was {default_metrics['f1']:.3f})"
    )

    torch.save(model.state_dict(), args.output_dir / "titanic_model.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "training_log.csv", index=False)
    with open(args.output_dir / "model_config.json", "w") as f:
        json.dump(
            {
                "n_features": preprocessor.n_features,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
                "feature_names": preprocessor.feature_names_,
                "best_epoch": best_epoch,
                "seed": args.seed,
                "val_size": args.val_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "decision_threshold": threshold,
                "val_metrics_default_threshold": default_metrics,
                "val_metrics_tuned_threshold": tuned_metrics,
            },
            f,
            indent=2,
        )
    print(f"\nSaved model, preprocessor, config, val split, and training log to {args.output_dir}/")


if __name__ == "__main__":
    train(parse_args())
