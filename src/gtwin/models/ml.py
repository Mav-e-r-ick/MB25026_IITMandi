"""Statistical / ML forecasters sharing one interface.

Each model produces, for every example row:
  * point forecasts  pred_{15,30,60,90,120}
  * an event score   p_event  (probability of any reading >= threshold in the next 120 min)
Probabilities from ML classifiers are re-calibrated on held-out calibration participants
(Platt scaling); mechanistic probabilities come from the state-space model itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gtwin.config import SEED, TASK

HORIZONS = TASK.forecast_points_min


def _xy(df, feats, h):
    m = df[f"y_{h}"].notna()
    return df.loc[m, feats], df.loc[m, f"y_{h}"]


def _ev(df):
    return df["event_valid"] & (df["event"] >= 0)


class Platt:
    """Logistic calibration of a raw score on held-out data."""

    def fit(self, score, y):
        s = np.asarray(score, float).reshape(-1, 1)
        self.lr = LogisticRegression(C=1e3, max_iter=1000).fit(s, y) if len(np.unique(y)) > 1 else None
        return self

    def __call__(self, score):
        s = np.asarray(score, float).reshape(-1, 1)
        return self.lr.predict_proba(s)[:, 1] if self.lr is not None else np.full(len(s), 0.0)


class Persistence:
    name = "Persistence"

    def fit(self, tr, cal, feats):
        # event score: linear trend extrapolation over 2 h (max of last value and extrapolated value)
        self.platt = Platt().fit(self._score(cal[_ev(cal)]), cal.loc[_ev(cal), "event"])
        return self

    @staticmethod
    def _score(df):
        slope = df["d_15"].fillna(0) / 15.0
        return np.maximum(df["g0"], df["g0"] + slope * 120)

    def predict(self, df):
        out = pd.DataFrame(index=df.index)
        for h in HORIZONS:
            out[f"pred_{h}"] = df["g0"]
        out["p_event"] = self.platt(self._score(df))
        return out


class RidgeAR:
    name = "Ridge AR"

    def fit(self, tr, cal, feats):
        self.feats = feats
        self.reg = {}
        for h in HORIZONS:
            X, y = _xy(tr, feats, h)
            self.reg[h] = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)).fit(X, y - X["g0"])
        e = _ev(tr)
        self.clf = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(
            tr.loc[e, feats], tr.loc[e, "event"])
        ec = _ev(cal)
        self.platt = Platt().fit(self.clf.decision_function(cal.loc[ec, feats]), cal.loc[ec, "event"])
        return self

    def predict(self, df):
        out = pd.DataFrame(index=df.index)
        for h in HORIZONS:
            out[f"pred_{h}"] = df["g0"] + self.reg[h].predict(df[self.feats])
        out["p_event"] = self.platt(self.clf.decision_function(df[self.feats]))
        return out


def _hgb_reg():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)


def _hgb_clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)


class GBM:
    """Gradient boosting on engineered features. Targets are changes from the current reading."""
    name = "Gradient boosting"

    def __init__(self, base_col: str = "g0", extra_feats: tuple[str, ...] = (), name: str | None = None,
                 feats_override: list[str] | None = None):
        self.base_col, self.extra, self.override = base_col, list(extra_feats), feats_override
        if name:
            self.name = name

    def _base(self, df, h):
        return df[self.base_col if self.base_col == "g0" else f"{self.base_col}_{h}"]

    def fit(self, tr, cal, feats):
        self.feats = (list(self.override) if self.override is not None else list(feats)) + self.extra
        self.reg = {}
        for h in HORIZONS:
            m = tr[f"y_{h}"].notna()
            X = tr.loc[m, self.feats]
            self.reg[h] = _hgb_reg().fit(X, tr.loc[m, f"y_{h}"] - self._base(tr.loc[m], h))
        e = _ev(tr)
        self.clf = _hgb_clf().fit(tr.loc[e, self.feats], tr.loc[e, "event"])
        ec = _ev(cal)
        self.platt = Platt().fit(self._logit(cal.loc[ec]), cal.loc[ec, "event"])
        return self

    def _logit(self, df):
        p = np.clip(self.clf.predict_proba(df[self.feats])[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict(self, df):
        out = pd.DataFrame(index=df.index)
        for h in HORIZONS:
            out[f"pred_{h}"] = self._base(df, h) + self.reg[h].predict(df[self.feats])
        out["p_event"] = self.platt(self._logit(df))
        return out

    def feature_importance(self, df, n_repeats: int = 3, h: int = 60, max_rows: int = 3000):
        """Permutation importance of the h-minute forecast (MAE increase)."""
        rng = np.random.default_rng(SEED)
        d = df[df[f"y_{h}"].notna()]
        d = d.sample(min(max_rows, len(d)), random_state=SEED)
        base = np.abs(self.predict(d)[f"pred_{h}"] - d[f"y_{h}"]).mean()
        imp = {}
        for f in self.feats:
            vals = []
            for _ in range(n_repeats):
                dd = d.copy()
                dd[f] = rng.permutation(dd[f].to_numpy())
                vals.append(np.abs(self.predict(dd)[f"pred_{h}"] - d[f"y_{h}"]).mean() - base)
            imp[f] = float(np.mean(vals))
        return pd.Series(imp).sort_values(ascending=False)


class Mechanistic:
    """Wraps precomputed state-space outputs (columns mech_{h}, mech_sd_{h}, mech_p_event)."""
    name = "Mechanistic (Kalman)"

    def fit(self, tr, cal, feats):
        return self

    def predict(self, df):
        out = pd.DataFrame(index=df.index)
        for h in HORIZONS:
            out[f"pred_{h}"] = df[f"mech_{h}"]
        out["p_event"] = df["mech_p_event"]
        return out


MECH_FEATS = tuple([f"mech_{h}" for h in HORIZONS] + [f"mech_sd_{h}" for h in HORIZONS]
                   + ["mech_p_event", "mech_R", "mech_B"])


def make_models():
    return [
        Persistence(),
        RidgeAR(),
        GBM(),
        Mechanistic(),
        GBM(base_col="mech", extra_feats=MECH_FEATS, name="Hybrid (Kalman + GB residual)"),
    ]
