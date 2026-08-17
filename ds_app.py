"""Streamlit app: validation results + inference UI for the Titanic PyTorch classifier.

Single page, top to bottom:
- Validation results: metrics/plots on the held-out split `train.py` wrote to
  `models/val_split.csv`, evaluated at the F1-tuned decision threshold.
- Run inference: point at any CSV with the raw Titanic schema, get per-row predictions; if the
  CSV also has a `Survived` column, the same evaluation plots/metrics are shown against it.

Reuses `preprocessing.TitanicPreprocessor` and `model.TitanicNet` as-is — the exact objects
`train.py` fit and saved — so inference here can never drift from what was trained.

Run with: streamlit run ds_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from model import TitanicNet
from preprocessing import TitanicPreprocessor

MODELS_DIR = Path("models")

# Fixed categorical order (never cycled): slot 1 = Died, slot 2 = Survived — same two colors,
# same assignment, in every chart in this app. Blue/orange rather than red/green: colorblind-safe
# and doesn't imply "bad/good" the way a traffic-light pair would.
COLOR_DIED = "#2a78d6"
COLOR_SURVIVED = "#eb6834"
COLOR_MUTED = "#898781"
COLOR_TEXT = "#0b0b0b"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": COLOR_TEXT,
        "text.color": COLOR_TEXT,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "grid.color": "#e1e0d9",
        "axes.grid": True,
        "grid.linewidth": 0.6,
    }
)


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


@st.cache_resource
def load_artifacts() -> dict | None:
    """Load everything train.py produced. Returns None if the model hasn't been trained yet."""
    config_path = MODELS_DIR / "model_config.json"
    model_path = MODELS_DIR / "titanic_model.pt"
    preproc_path = MODELS_DIR / "preprocessor.pkl"
    if not (config_path.exists() and model_path.exists() and preproc_path.exists()):
        return None

    with open(config_path) as f:
        config = json.load(f)

    preprocessor = TitanicPreprocessor.load(preproc_path)

    model = TitanicNet(n_features=config["n_features"], hidden_dim=config["hidden_dim"], dropout=config["dropout"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return {"config": config, "preprocessor": preprocessor, "model": model}


@st.cache_data
def load_val_split(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def predict(df: pd.DataFrame, artifacts: dict) -> tuple[np.ndarray, np.ndarray]:
    """Raw DataFrame -> (survival probabilities, predicted labels at the model's tuned threshold).

    Raises ValueError (via the preprocessor) if `df` is missing required raw columns.
    """
    X = artifacts["preprocessor"].transform(df)
    with torch.no_grad():
        probs = torch.sigmoid(artifacts["model"](torch.tensor(X))).numpy()
    threshold = artifacts["config"]["decision_threshold"]
    preds = (probs >= threshold).astype(int)
    return probs, preds


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def render_metrics_row(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> None:
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{accuracy_score(y_true, preds):.3f}")
    cols[1].metric("Precision", f"{precision_score(y_true, preds, zero_division=0):.3f}")
    cols[2].metric("Recall", f"{recall_score(y_true, preds, zero_division=0):.3f}")
    cols[3].metric("F1", f"{f1_score(y_true, preds, zero_division=0):.3f}")
    cols[4].metric("ROC-AUC", f"{roc_auc_score(y_true, probs):.3f}")


def plot_confusion_matrix(y_true: np.ndarray, preds: np.ndarray) -> plt.Figure:
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    labels = ["Died", "Survived"]

    fig, ax = plt.subplots(figsize=(4, 3.6))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", fontsize=13, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else COLOR_TEXT,
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()
    return fig


def plot_roc(y_true: np.ndarray, probs: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)

    fig, ax = plt.subplots(figsize=(4.3, 4))
    ax.plot(fpr, tpr, color=COLOR_DIED, linewidth=2.2, label=f"Model (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color=COLOR_MUTED, linestyle="--", linewidth=1.3, label="Random baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def plot_precision_recall(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> plt.Figure:
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)

    fig, ax = plt.subplots(figsize=(4.3, 4))
    ax.plot(recalls, precisions, color=COLOR_DIED, linewidth=2.2, label="Precision-recall")
    if len(thresholds):
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        ax.scatter(
            [recalls[idx]], [precisions[idx]], color=COLOR_SURVIVED, s=70, zorder=5,
            label=f"Chosen cutoff ({threshold:.2f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    return fig


def render_evaluation(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray, threshold: float) -> None:
    render_metrics_row(y_true, probs, preds)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(plot_confusion_matrix(y_true, preds))
    with c2:
        st.pyplot(plot_roc(y_true, probs))
    with c3:
        st.pyplot(plot_precision_recall(y_true, probs, threshold))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Titanic Survival Classifier — Alon Fainshtein", page_icon="🚢", layout="wide")
# Streamlit's default st.caption() size (used throughout for body/description text, as opposed
# to titles/headers) reads a bit small — bump it up without touching title/header sizes.
st.markdown(
    '<style>[data-testid="stCaptionContainer"] { font-size: 1.05rem; }</style>',
    unsafe_allow_html=True,
)
st.title("🚢 Titanic Survival Classifier")
st.caption(
    "By Alon Fainshtein · PyTorch MLP trained on the Kaggle Titanic dataset — "
    "held-out validation results and inference."
)

artifacts = load_artifacts()
if artifacts is None:
    st.error(
        "No trained model found under `models/`. Run `python train.py` first — it writes the "
        "model weights, preprocessor, config, validation split, and training log this app reads."
    )
    st.stop()

config = artifacts["config"]
tuned_threshold = float(config["decision_threshold"])

val_path = MODELS_DIR / "val_split.csv"
if not val_path.exists():
    st.error("`models/val_split.csv` not found — re-run `python train.py`.")
else:
    val_df = load_val_split(str(val_path))
    probs, preds = predict(val_df, artifacts)  # preds already thresholded at tuned_threshold below
    y_true = val_df["Survived"].to_numpy(dtype=int)

    st.subheader("Held-out validation performance")
    st.caption(
        f"{len(val_df)} rows, split off before any preprocessing was fit and never used in "
        f"training. Survival rate here: {y_true.mean():.1%}."
    )

    st.caption(
        "The model outputs a survival *probability* per passenger, not a yes/no verdict — "
        f"the metrics and plots below use a cutoff of {tuned_threshold:.3f} (not the usual "
        "0.5), the value that maximized F1 on this validation set. See the comparison table "
        "below for what that tuning bought over the default 0.5 cutoff."
    )
    render_evaluation(y_true, probs, preds, tuned_threshold)

    with st.expander("Default cutoff (0.5) vs. tuned cutoff — side by side", expanded=True):
        st.caption(
            "Both are reported explicitly because the tuned cutoff was chosen on this same "
            "validation set — the comparison shows exactly what that tuning bought."
        )
        comparison = pd.DataFrame(
            {
                "default (0.5)": config["val_metrics_default_threshold"],
                f"tuned ({tuned_threshold:.3f})": config["val_metrics_tuned_threshold"],
            }
        ).round(3)
        st.table(comparison)

st.divider()

st.subheader("Run inference on a CSV")
st.caption(
    "Path to a CSV with the raw Titanic schema: `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, "
    "`Parch`, `Fare`, `Cabin`, `Embarked`. `Survived` is optional — if present, evaluation "
    "plots and metrics are shown against it too, exactly as above."
)
csv_path_str = st.text_input(
    "CSV path",
    value=str(val_path),
    placeholder="e.g. data/sample_train.csv, or any CSV with the raw Titanic schema",
)
run = st.button("Run inference", type="primary")

if run:
    if not csv_path_str.strip():
        st.warning("Enter a CSV path above first.")
    else:
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            st.error(f"File not found: {csv_path}")
        else:
            try:
                infer_df = pd.read_csv(csv_path)
            except Exception as exc:
                st.error(f"Could not read `{csv_path}` as a CSV: {exc}")
                infer_df = None

            if infer_df is not None:
                try:
                    probs, preds = predict(infer_df, artifacts)
                except ValueError as exc:
                    # preprocessing.py's own input checks — missing columns, no rows,
                    # negative Fare, or any other invalid value that would otherwise
                    # silently turn into a NaN feature and a bogus-looking prediction.
                    st.error(str(exc))
                except Exception as exc:
                    # Right columns, wrong contents (e.g. Age as text) — surface a plain
                    # message instead of a raw Streamlit traceback.
                    st.error(
                        "Could not run inference on this CSV. Check that columns have the "
                        "expected types (e.g. `Age` and `Fare` numeric, not text).\n\n"
                        f"Details: {exc}"
                    )
                else:
                    st.success(f"Ran inference on {len(infer_df)} rows.")

                    results = infer_df.copy()
                    results["predicted_survived"] = preds
                    results["survival_probability"] = probs.round(3)
                    lead_cols = [c for c in ["PassengerId", "Name"] if c in results.columns]
                    lead_cols += ["predicted_survived", "survival_probability"]
                    other_cols = [c for c in results.columns if c not in lead_cols]
                    st.dataframe(results[lead_cols + other_cols], use_container_width=True, hide_index=True)

                    st.metric("Predicted survival rate", f"{preds.mean():.1%}")

                    # Evaluate on whatever subset actually has a label, rather than requiring
                    # every row to be labeled before showing anything. Match the column
                    # case/whitespace-insensitively so a near-miss header doesn't silently
                    # drop evaluation with no explanation.
                    survived_col = next(
                        (c for c in infer_df.columns if c.strip().lower() == "survived"), None
                    )
                    if survived_col is None:
                        st.caption(
                            "No `Survived` column in this CSV — showing predictions only. "
                            "There's no ground truth to evaluate against."
                        )
                    else:
                        labeled_mask = infer_df[survived_col].notna().to_numpy()
                        n_labeled = int(labeled_mask.sum())
                        if n_labeled == 0:
                            st.caption(
                                f"`{survived_col}` column is present but every value is blank — "
                                "showing predictions only, nothing to evaluate against."
                            )
                        else:
                            y_true = infer_df.loc[labeled_mask, survived_col].to_numpy(dtype=int)
                            st.subheader("Evaluation against provided ground truth")
                            if n_labeled < len(infer_df):
                                st.caption(
                                    f"{n_labeled} of {len(infer_df)} rows have a `{survived_col}` "
                                    "label — evaluating on those only."
                                )
                            if len(set(y_true)) < 2:
                                # Precision/recall/ROC-AUC and the ROC/PR curves are all
                                # undefined with only one class present — say so plainly
                                # instead of rendering a silent NaN or an sklearn warning.
                                acc = (preds[labeled_mask] == y_true).mean()
                                st.caption(
                                    f"All {n_labeled} labeled rows share the same "
                                    f"`{survived_col}` value, so precision/recall/ROC-AUC "
                                    f"aren't defined. Accuracy: {acc:.3f}."
                                )
                            else:
                                render_evaluation(y_true, probs[labeled_mask], preds[labeled_mask], tuned_threshold)
