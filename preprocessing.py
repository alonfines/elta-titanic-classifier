"""Shared preprocessing for the Titanic classification pipeline.

`TitanicPreprocessor` is fit once, on the *training split only*, inside `train.py`. The fitted
object is pickled to disk (`models/preprocessor.pkl`) alongside the model weights and reloaded
as-is by `ds_app.py` for inference — so training and inference always apply identical logic, and
nothing is ever re-fit on validation or new data.

Feature engineering follows the findings in `eda.ipynb` (§6 summary):
- Drop `PassengerId`, `Name`, `Ticket`, `Cabin`, `SibSp`, `Parch` in raw form — `SibSp`/`Parch`
  are consumed only to derive `FamilySize` below, not kept as separate model inputs (the three
  are collinear by construction: `FamilySize = SibSp + Parch + 1`).
- Engineer `Title` (from `Name`, rare titles bucketed), `FamilySize` (`SibSp + Parch + 1`),
  `HasCabin` (from `Cabin` non-null), `IsChild` (`Age < 16`).
- Impute `Age` by `Pclass`/`Title` group median (falls back to the global median for an unseen
  group combination); `Embarked` by mode; `Fare` by median (only relevant for inference CSVs —
  `train.csv` itself has no missing `Fare`).
- Encode `Sex`, `Embarked`, `Title` via one-hot; keep `Pclass` as its natural ordinal integer.
- Log-transform `Fare` (right-skewed with extreme outliers) and standardize all numeric columns.

Expected input schema (raw Kaggle Titanic columns): `Pclass`, `Name`, `Sex`, `Age`, `SibSp`,
`Parch`, `Fare`, `Cabin`, `Embarked`. `Survived` (the target) and `PassengerId`/`Ticket` are
ignored if present and are not required for `transform()`.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RARE_TITLE_MIN_COUNT = 10
CHILD_AGE_CUTOFF = 16

RAW_REQUIRED_COLS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
CATEGORICAL_COLS = ["Sex", "Embarked", "Title"]
NUMERIC_COLS = ["Age", "FareLog", "FamilySize"]
PASSTHROUGH_COLS = ["HasCabin", "IsChild", "Pclass"]


class TitanicPreprocessor:
    """Fits on a raw training DataFrame and transforms raw DataFrames into a model-ready
    float32 numpy array. Call `fit`/`fit_transform` on the training split only; `transform`
    is safe to call repeatedly on validation data or new inference CSVs.
    """

    def __init__(self) -> None:
        self._fitted = False
        self.embarked_mode_: str | None = None
        self.age_group_medians_: pd.Series | None = None
        self.age_global_median_: float | None = None
        self.fare_median_: float | None = None
        self.rare_titles_: set[str] = set()
        self.encoder_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] | None = None

    # ---- feature engineering, shared by fit and transform ----

    @staticmethod
    def _extract_title(df: pd.DataFrame) -> pd.Series:
        title = df["Name"].str.extract(r",\s*([^.]+)\.")[0].str.strip()
        return title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    def _engineer(self, df: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        missing = set(RAW_REQUIRED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")

        df = df.copy()

        df["Title"] = self._extract_title(df)
        if fitting:
            counts = df["Title"].value_counts()
            self.rare_titles_ = set(counts[counts < RARE_TITLE_MIN_COUNT].index)
        df["Title"] = df["Title"].where(~df["Title"].isin(self.rare_titles_), "Rare")
        if not fitting:
            # Titles never seen during fit (or already-bucketed rare ones) collapse to "Rare"
            # so the encoder never sees an unseen category outside the "ignore" path below.
            known_titles = set(self.encoder_.categories_[CATEGORICAL_COLS.index("Title")])
            df["Title"] = df["Title"].where(df["Title"].isin(known_titles), "Rare")

        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["HasCabin"] = df["Cabin"].notna().astype(int)

        if fitting:
            self.embarked_mode_ = df["Embarked"].mode(dropna=True).iloc[0]
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)

        if fitting:
            self.age_group_medians_ = df.groupby(["Pclass", "Title"])["Age"].median()
            self.age_global_median_ = df["Age"].median()
        df["Age"] = df.apply(
            lambda row: row["Age"]
            if pd.notna(row["Age"])
            else self.age_group_medians_.get((row["Pclass"], row["Title"]), np.nan),
            axis=1,
        )
        df["Age"] = df["Age"].fillna(self.age_global_median_)  # group had no training data at all
        df["IsChild"] = (df["Age"] < CHILD_AGE_CUTOFF).astype(int)

        if fitting:
            self.fare_median_ = df["Fare"].median()
        df["Fare"] = df["Fare"].fillna(self.fare_median_)
        df["FareLog"] = np.log1p(df["Fare"])

        return df

    # ---- public API ----

    def fit(self, df: pd.DataFrame) -> "TitanicPreprocessor":
        self.fit_transform(df)
        return self

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        eng = self._engineer(df, fitting=True)

        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_arr = self.encoder_.fit_transform(eng[CATEGORICAL_COLS])

        self.scaler_ = StandardScaler()
        num_arr = self.scaler_.fit_transform(eng[NUMERIC_COLS])

        pass_arr = eng[PASSTHROUGH_COLS].to_numpy(dtype=np.float32)

        cat_names = list(self.encoder_.get_feature_names_out(CATEGORICAL_COLS))
        self.feature_names_ = cat_names + NUMERIC_COLS + PASSTHROUGH_COLS
        self._fitted = True

        return np.hstack([cat_arr, num_arr, pass_arr]).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TitanicPreprocessor.transform() called before fit().")
        eng = self._engineer(df, fitting=False)
        cat_arr = self.encoder_.transform(eng[CATEGORICAL_COLS])
        num_arr = self.scaler_.transform(eng[NUMERIC_COLS])
        pass_arr = eng[PASSTHROUGH_COLS].to_numpy(dtype=np.float32)
        return np.hstack([cat_arr, num_arr, pass_arr]).astype(np.float32)

    @property
    def n_features(self) -> int:
        if self.feature_names_ is None:
            raise RuntimeError("Preprocessor has not been fit yet.")
        return len(self.feature_names_)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "TitanicPreprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, TitanicPreprocessor):
            raise TypeError(f"{path} does not contain a TitanicPreprocessor")
        return obj
