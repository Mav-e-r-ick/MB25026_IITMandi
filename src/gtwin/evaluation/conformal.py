"""Split-conformal prediction intervals, calibrated on held-out *participants*.

Guarantee (under exchangeability of participants): ~(1-alpha) marginal coverage for a new
participant drawn from the same population. It is NOT a guarantee for a different population
(e.g. Indian patients) or for any single individual - stated in the report.

Two variants:
  * absolute:   interval = pred +/- q,          q = quantile of |y - pred|
  * normalised: interval = pred +/- q * sd,     q = quantile of |y - pred| / sd   (sd from the Kalman twin)
The normalised version gives wider intervals when the twin is uncertain (after meals, after gaps).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtwin.config import TASK

HORIZONS = TASK.forecast_points_min


class Conformal:
    """`by` (optional column, e.g. diabetes group) gives Mondrian / group-conditional calibration:
    a separate quantile per group, so coverage holds within each group rather than only on average."""

    def __init__(self, alpha: float = 0.1, normalised: bool = False, by: str | None = None):
        self.alpha, self.normalised, self.by = alpha, normalised, by

    def _q(self, df, pred, h):
        m = df[f"y_{h}"].notna()
        s = np.abs(df.loc[m, f"y_{h}"] - pred.loc[m, f"pred_{h}"])
        if self.normalised:
            s = s / df.loc[m, f"mech_sd_{h}"]
        n = len(s)
        if n == 0:
            return np.nan
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        return float(np.quantile(s, level))

    def fit(self, df: pd.DataFrame, pred: pd.DataFrame):
        self.q = {h: self._q(df, pred, h) for h in HORIZONS}
        self.qg = {}
        if self.by:
            for g, d in df.groupby(self.by):
                self.qg[g] = {h: self._q(d, pred.loc[d.index], h) for h in HORIZONS}
        return self

    def interval(self, df: pd.DataFrame, pred: pd.DataFrame):
        lo, hi = pd.DataFrame(index=df.index), pd.DataFrame(index=df.index)
        for h in HORIZONS:
            q = self.q[h]
            if self.by:
                q = df[self.by].map(lambda g: self.qg.get(g, self.q).get(h, self.q[h])).astype(float)
            w = q * (df[f"mech_sd_{h}"] if self.normalised else 1.0)
            lo[f"lo_{h}"] = pred[f"pred_{h}"] - w
            hi[f"hi_{h}"] = pred[f"pred_{h}"] + w
        return lo, hi
