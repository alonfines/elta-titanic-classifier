"""Bonus: compare the shipped PyTorch model against a scikit-learn RandomForest baseline.

Not part of the required deliverable — `train.py` is what's graded. This is extra context for
the README's "Design choices" section, answering "does the model architecture matter here, or
would a standard tabular baseline do just as well?"

For a fair, honest comparison, both models see the exact same inputs: the same train/val split
`train.py` used (read back from `model_config.json` so this never drifts out of sync), and the
same already-fitted `TitanicPreprocessor` — reused via `.transform()`, not refit. The Random
Forest gets no tuning (sklearn defaults, fixed seed) and both models are scored at the default
0.5 threshold, so neither gets an advantage the other didn't also get.

Requires `python train.py` to have been run first (needs its saved preprocessor and config).

Usage:
    python compare_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from preprocessing import TitanicPreprocessor

DATA_PATH = Path("data/train.csv")
MODELS_DIR = Path("models")


def main() -> None:
    config_path = MODELS_DIR / "model_config.json"
    if not (config_path.exists() and (MODELS_DIR / "preprocessor.pkl").exists()):
        raise SystemExit("Run `python train.py` first — this script reuses its split and preprocessor.")

    with open(config_path) as f:
        config = json.load(f)

    df = pd.read_csv(DATA_PATH)
    # Same split train.py used, read back from its own saved config so this can't drift out of sync.
    train_df, val_df = train_test_split(
        df, test_size=config["val_size"], stratify=df["Survived"], random_state=config["seed"]
    )

    preprocessor = TitanicPreprocessor.load(MODELS_DIR / "preprocessor.pkl")
    X_train = preprocessor.transform(train_df)  # already fit by train.py — never refit here
    X_val = preprocessor.transform(val_df)
    y_train = train_df["Survived"].to_numpy()
    y_val = val_df["Survived"].to_numpy()

    rf = RandomForestClassifier(random_state=config["seed"])  # sklearn defaults, no tuning
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_val)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)

    mlp_metrics = config["val_metrics_default_threshold"]  # already computed by train.py, same split
    results = pd.DataFrame(
        [
            {
                "model": "PyTorch MLP (this project)",
                "accuracy": mlp_metrics["accuracy"],
                "f1": mlp_metrics["f1"],
                "roc_auc": mlp_metrics["roc_auc"],
            },
            {
                "model": "RandomForestClassifier (baseline)",
                "accuracy": accuracy_score(y_val, rf_preds),
                "f1": f1_score(y_val, rf_preds),
                "roc_auc": roc_auc_score(y_val, rf_probs),
            },
        ]
    ).round(3)

    print(f"Both models scored at the default 0.5 threshold, on the same {len(val_df)}-row validation split:\n")
    print(results.to_string(index=False))

    results.to_csv(MODELS_DIR / "baseline_comparison.csv", index=False)
    print(f"\nSaved to {MODELS_DIR / 'baseline_comparison.csv'}")


if __name__ == "__main__":
    main()
