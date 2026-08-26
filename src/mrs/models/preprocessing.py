"""Preprocessing for the Logistic Regression baseline: median imputation with explicit
missingness flags, then standard scaling.

Fit only ever on the train split (Dev Plan Sec 5/33.6) -- global statistics computed
across splits would leak future information into the imputer/scaler even though the
underlying features are already leakage-safe per row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MedianImputerWithFlags(BaseEstimator, TransformerMixin):
    """Median-impute NaNs and append a `<col>_was_missing` indicator per imputed column.

    Only columns that actually contain a NaN in the *fitted* (training) data get a flag
    column, keeping cold-start visibility (Dev Plan Sec 33.7) without doubling the width
    of features that are never missing. Medians and the flagged-column set are frozen at
    fit time; transform never recomputes anything from the data it is applied to, so
    validation/test rows can never influence the imputation values.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "MedianImputerWithFlags":
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        medians = X.median(axis=0, skipna=True)
        # A column with zero non-null training values has no meaningful median; fall
        # back to 0.0 rather than propagating NaN into every row at inference.
        self.medians_ = medians.fillna(0.0)
        self.flagged_columns_ = [c for c in X.columns if X[c].isna().any()]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        missing_cols = [c for c in self.feature_names_in_ if c not in X.columns]
        if missing_cols:
            raise ValueError(f"MedianImputerWithFlags.transform: missing columns {missing_cols}")
        X = X[self.feature_names_in_]

        out = X.fillna(self.medians_)
        for col in self.flagged_columns_:
            out[f"{col}_was_missing"] = X[col].isna().astype(np.float64)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(self.feature_names_in_) + [f"{c}_was_missing" for c in self.flagged_columns_]
        return np.array(names)


def build_preprocessing_pipeline() -> Pipeline:
    """Impute (with missingness flags) then standard-scale. Fit on train only."""
    return Pipeline(
        [
            ("impute_flag", MedianImputerWithFlags()),
            ("scale", StandardScaler()),
        ]
    )
