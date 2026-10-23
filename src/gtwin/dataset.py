"""Build prediction examples from a regular CGM grid.

Leakage rules (enforced here and tested in tests/test_dataset.py):
  * every feature at prediction time t uses only readings at slots <= t;
  * labels use only slots t+1 .. t+H;
  * a prediction is only made if the reading at t is present (not stale) and history coverage >= 80%;
  * an event label is only valid if >= 80% of future readings exist (otherwise excluded, not 'negative');
  * the event task is only scored where G(t) < threshold (otherwise the answer is trivially 'yes').
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtwin.config import TASK, TaskConfig

LAG_MIN = (5, 10, 15, 20, 30, 45, 60, 90, 120, 180)
WIN_MIN = (30, 60, 180)


def _locf(x: np.ndarray) -> np.ndarray:
    """Last observation carried forward (uses only the past)."""
    return pd.Series(x).ffill().to_numpy(float)


def participant_examples(ts: pd.Series, g: np.ndarray, cfg: TaskConfig = TASK, stride: int = 3,
                         g_label: np.ndarray | None = None, t_list: list[int] | None = None,
                         require_quality: bool = True) -> pd.DataFrame:
    """Features + targets for one participant. `g` is the glucose grid (NaN = missing).

    `g_label` (optional) supplies the clean series for targets when `g` has been deliberately
    corrupted (missingness / noise robustness experiments)."""
    g = np.asarray(g, float)
    gl = g if g_label is None else np.asarray(g_label, float)
    n, H, L, s = len(g), cfg.n_hor, cfg.n_hist, cfg.step_min
    obs = ~np.isnan(g)
    gf = _locf(g)
    csum = np.concatenate([[0], np.cumsum(obs)])
    rows = []
    ts_iter = range(L, n - 1, stride) if t_list is None else t_list
    for t in ts_iter:
        lo = max(0, t + 1 - L)
        hist_cov = (csum[t + 1] - csum[lo]) / L
        if require_quality and (not obs[t] or hist_cov < cfg.min_history_coverage):
            continue
        # g0 = latest available reading (equals g[t] when the reading at t exists; stale otherwise)
        r = {"t_idx": t, "timestamp": ts.iloc[t], "g0": gf[t], "hist_cov": hist_cov}
        for lm in LAG_MIN:
            k = lm // s
            r[f"lag_{lm}"] = gf[t - k] if t - k >= 0 else np.nan
            r[f"d_{lm}"] = gf[t] - r[f"lag_{lm}"]
        with np.errstate(all="ignore"):
            for wm in WIN_MIN:
                w = g[max(0, t - wm // s + 1): t + 1]
                ok = w[~np.isnan(w)]
                r[f"mean_{wm}"] = ok.mean() if len(ok) else np.nan
                r[f"std_{wm}"] = ok.std() if len(ok) else np.nan
                r[f"min_{wm}"] = ok.min() if len(ok) else np.nan
                r[f"max_{wm}"] = ok.max() if len(ok) else np.nan
            hist = g[max(0, t - 1440 // s + 1): t + 1]  # last 24 h
            hist = hist[~np.isnan(hist)]
            r["frac_above_24h"] = float((hist >= cfg.threshold_mgdl).mean()) if len(hist) else np.nan
            r["mean_24h"] = hist.mean() if len(hist) else np.nan
        tod = ts.iloc[t].hour * 60 + ts.iloc[t].minute
        r["tod_sin"], r["tod_cos"] = np.sin(2 * np.pi * tod / 1440), np.cos(2 * np.pi * tod / 1440)
        # ---- targets (future only)
        fut = gl[t + 1: t + 1 + H]
        if len(fut) < H:
            fut = np.concatenate([fut, np.full(H - len(fut), np.nan)])
        cov = np.mean(~np.isnan(fut))
        r["fut_cov"] = cov
        for pm in cfg.forecast_points_min:
            r[f"y_{pm}"] = fut[pm // s - 1]
        r["y_max"] = np.nanmax(fut) if cov > 0 else np.nan
        valid = cov >= cfg.min_future_coverage
        r["event_valid"] = bool(valid and gl[t] < cfg.threshold_mgdl and g[t] < cfg.threshold_mgdl)
        r["event"] = int(valid and np.nanmax(fut) >= cfg.threshold_mgdl) if valid else -1
        # minutes until first crossing (lead time analysis)
        cross = np.where(fut >= cfg.threshold_mgdl)[0]
        r["t_to_event_min"] = (cross[0] + 1) * s if len(cross) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def build_examples(cgm: pd.DataFrame, profile: pd.DataFrame, cfg: TaskConfig = TASK, stride: int = 3,
                   feature_col: str = "glucose", label_col: str = "glucose") -> pd.DataFrame:
    out = []
    for pid, d in cgm.groupby("participant_id", sort=True):
        d = d.sort_values("timestamp").reset_index(drop=True)
        ex = participant_examples(d["timestamp"], d[feature_col].to_numpy(float), cfg, stride,
                                  g_label=d[label_col].to_numpy(float))
        if len(ex):
            ex.insert(0, "participant_id", pid)
            out.append(ex)
    ex = pd.concat(out, ignore_index=True)
    keep = [c for c in ("participant_id", "group", "age", "gender", "hba1c", "bmi", "fasting_glucose",
                        "fasting_insulin", "homa_ir", "triglycerides") if c in profile]
    p = profile[keep].copy()
    if "gender" in p:
        p["male"] = p["gender"].astype(str).str.lower().str.startswith("m").astype(int)
        p = p.drop(columns="gender")
    return ex.merge(p, on="participant_id", how="left")


SENSOR_FEATURES = (["g0", "hist_cov"] + [f"lag_{m}" for m in LAG_MIN] + [f"d_{m}" for m in LAG_MIN]
                   + [f"{a}_{w}" for w in WIN_MIN for a in ("mean", "std", "min", "max")]
                   + ["frac_above_24h", "mean_24h", "tod_sin", "tod_cos"])
STATIC_FEATURES = ["age", "male", "hba1c"]  # extended automatically when PhysioNet bio.csv is present
OPTIONAL_STATIC = ["bmi", "fasting_glucose", "fasting_insulin", "homa_ir", "triglycerides"]


def static_features(df: pd.DataFrame) -> list[str]:
    return STATIC_FEATURES + [c for c in OPTIONAL_STATIC if c in df and df[c].notna().any()]
