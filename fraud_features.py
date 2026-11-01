"""Shared feature engineering and preprocessing for the mobile-wallet fraud project.
Imported by BOTH the notebook and the Streamlit app, so training and serving use
exactly the same feature structure. Nothing in this file is fitted on data.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

IN_SCOPE_TYPES = ["CASH_OUT", "TRANSFER"]          # the only types that contain fraud in the training data
RAW_FEATURES = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
BALANCE_FEATURES = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
ENGINEERED = ["delta_orig", "delta_dest", "err_orig", "err_dest", "amount_to_old_orig",
              "orig_drained", "orig_emptied", "dest_empty"]
MONEY_COLS = BALANCE_FEATURES + ["delta_orig", "delta_dest", "err_orig", "err_dest"]
FLAG_COLS = ["orig_drained", "orig_emptied", "dest_empty"]


def add_features(X):
    """Row-wise balance-change features (no fitting, so no leakage)."""
    X = pd.DataFrame(X).copy()
    for c in BALANCE_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X["delta_orig"] = X["oldbalanceOrg"] - X["newbalanceOrig"]            # money that left the origin account
    X["delta_dest"] = X["newbalanceDest"] - X["oldbalanceDest"]           # money that arrived at the destination
    X["err_orig"] = X["oldbalanceOrg"] - X["amount"] - X["newbalanceOrig"]  # 0 when the ledger reconciles
    X["err_dest"] = X["oldbalanceDest"] + X["amount"] - X["newbalanceDest"]
    ratio = np.where(X["oldbalanceOrg"] > 0, X["amount"] / X["oldbalanceOrg"].where(X["oldbalanceOrg"] > 0), 0.0)
    X["amount_to_old_orig"] = np.clip(np.nan_to_num(ratio, nan=0.0), 0, 10)
    X["orig_drained"] = ((X["oldbalanceOrg"] == X["amount"]) & (X["amount"] > 0)).astype(int)
    X["orig_emptied"] = (X["newbalanceOrig"] == 0).astype(int)
    X["dest_empty"] = ((X["oldbalanceDest"] == 0) & (X["newbalanceDest"] == 0) & (X["amount"] > 0)).astype(int)
    return X


def signed_log1p(X):
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def _names_add_features(_, input_features):
    return np.array(list(input_features) + [c for c in ENGINEERED if c not in input_features], dtype=object)


def make_preprocessor(scale=True, feature_set="all"):
    """Imputer + signed-log + (optional) scaler + one-hot, wrapped in a ColumnTransformer.

    feature_set='all' uses the raw columns plus the engineered balance-change features;
    feature_set='raw' (used only for the ablation study) uses the original columns only.
    """
    if feature_set not in {"all", "raw"}:
        raise ValueError("feature_set must be 'all' or 'raw'")
    raw_only = feature_set == "raw"
    money = [("imp", SimpleImputer(strategy="median")),
             ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one"))]
    if scale:
        money.append(("sc", StandardScaler()))
    parts = [("money", Pipeline(money), BALANCE_FEATURES if raw_only else MONEY_COLS)]
    if not raw_only:
        other = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
        flags = [("imp", SimpleImputer(strategy="constant", fill_value=0))]
        parts += [("ratio", Pipeline(other), ["amount_to_old_orig"]), ("flags", Pipeline(flags), FLAG_COLS)]
    parts.append(("type", OneHotEncoder(categories=[IN_SCOPE_TYPES], handle_unknown="ignore"), ["type"]))
    ct = ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("features", FunctionTransformer(add_features, feature_names_out=_names_add_features)),
                     ("columns", ct)])


class ResampledClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a classifier so that (random) resampling happens ONLY inside fit(), on the data it is given.

    Because the resampling lives inside fit(), cross-validation validation folds and the
    validation/test sets always keep their natural fraud prevalence.
    strategy: 'none' | 'under' | 'over'. ratio = target fraud:legitimate ratio after resampling.
    """
    def __init__(self, estimator, strategy="none", ratio=0.1, random_state=42):
        self.estimator = estimator
        self.strategy = strategy
        self.ratio = ratio
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y must contain only binary labels 0 and 1")
        if self.strategy not in {"none", "under", "over"}:
            raise ValueError("strategy must be 'none', 'under', or 'over'")
        if self.strategy != "none":
            if not np.isfinite(self.ratio) or self.ratio <= 0:
                raise ValueError("ratio must be finite and positive for under- or oversampling")
        rng = np.random.default_rng(self.random_state)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if self.strategy != "none" and (len(pos) == 0 or len(neg) == 0):
            raise ValueError("resampling requires at least one example from each class")
        if self.strategy == "under":
            n_neg = min(len(neg), int(len(pos) / self.ratio))
            idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
        elif self.strategy == "over":
            n_pos = max(len(pos), int(np.ceil(len(neg) * self.ratio)))
            idx = np.concatenate([neg, rng.choice(pos, n_pos, replace=True)])
        else:
            idx = np.arange(len(y))
        Xr = X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X)[idx]
        self.n_fit_rows_, self.n_fit_frauds_ = len(idx), int(y[idx].sum())
        self.estimator_ = clone(self.estimator).fit(Xr, y[idx])
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)
