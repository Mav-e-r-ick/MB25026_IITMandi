"""Metrics. Every summary reports the number of unique participants and events behind it."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from gtwin.config import SEED, TASK

HORIZONS = TASK.forecast_points_min


def forecast_errors(df: pd.DataFrame, pred: pd.DataFrame) -> dict:
    out = {}
    for h in HORIZONS:
        m = df[f"y_{h}"].notna()
        e = pred.loc[m, f"pred_{h}"] - df.loc[m, f"y_{h}"]
        out[f"MAE_{h}"] = float(np.abs(e).mean())
        out[f"RMSE_{h}"] = float(np.sqrt((e ** 2).mean()))
    return out


def event_metrics(df: pd.DataFrame, p: pd.Series) -> dict:
    m = df["event_valid"] & (df["event"] >= 0)
    y, s = df.loc[m, "event"].to_numpy(), p[m].to_numpy()
    if len(np.unique(y)) < 2:
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan, "prevalence": float(y.mean()) if len(y) else np.nan,
                "n_points": int(len(y)), "n_events": int(y.sum())}
    return {"AUROC": float(roc_auc_score(y, s)), "AUPRC": float(average_precision_score(y, s)),
            "Brier": float(brier_score_loss(y, s)), "prevalence": float(y.mean()),
            "n_points": int(len(y)), "n_events": int(y.sum())}


def calibration_table(df, p, bins=10) -> pd.DataFrame:
    m = df["event_valid"] & (df["event"] >= 0)
    d = pd.DataFrame({"p": p[m], "y": df.loc[m, "event"]})
    d["bin"] = pd.cut(d["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return d.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size")).reset_index()


def alert_burden(df: pd.DataFrame, p: pd.Series, threshold: float, refractory_min: int = 60,
                 step_min: int = 15) -> dict:
    """Simulate an alerting policy: alert when p >= threshold; then silence for `refractory_min`.

    Counts per participant-day of monitoring:
      * alerts, true alerts (event follows within 120 min), false alerts
      * episode recall: fraction of excursion episodes (first crossing after >= 60 min below threshold)
        preceded by an alert in the 120 min before the crossing, and the median lead time.
    """
    d = df.assign(p=p).sort_values(["participant_id", "timestamp"])
    days = 0.0
    n_alert = n_true = 0
    ep_total = ep_caught = 0
    leads = []
    for pid, g in d.groupby("participant_id"):
        g = g[g["event"] >= 0]
        if g.empty:
            continue
        days += (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 86400
        last_alert = None
        alert_times = []
        for ts, pv, ev, valid in zip(g["timestamp"], g["p"], g["event"], g["event_valid"]):
            if not valid:
                continue
            if pv >= threshold and (last_alert is None or (ts - last_alert).total_seconds() / 60 >= refractory_min):
                last_alert = ts
                alert_times.append(ts)
                n_alert += 1
                n_true += int(ev == 1)
        # excursion episodes: first crossing time after a below-threshold run
        crosses = g.loc[g["t_to_event_min"].notna() & g["event_valid"]]
        onset = (crosses["timestamp"] + pd.to_timedelta(crosses["t_to_event_min"], unit="min"))
        onset = onset.sort_values()
        episodes, prev = [], None
        for o in onset:
            if prev is None or (o - prev).total_seconds() / 60 > 60:
                episodes.append(o)
            prev = o
        at = np.array(alert_times, dtype="datetime64[ns]")
        for o in episodes:
            ep_total += 1
            o64 = np.datetime64(o)
            w = at[(at <= o64) & (at >= o64 - np.timedelta64(120, "m"))]
            if len(w):
                ep_caught += 1
                leads.append((o64 - w.min()) / np.timedelta64(1, "m"))
    days = max(days, 1e-9)
    return {"threshold": threshold, "alerts_per_day": n_alert / days, "false_alerts_per_day": (n_alert - n_true) / days,
            "alert_precision": n_true / n_alert if n_alert else np.nan,
            "episode_recall": ep_caught / ep_total if ep_total else np.nan,
            "median_lead_min": float(np.median(leads)) if leads else np.nan, "episodes": ep_total,
            "monitored_days": days}


def interval_metrics(df, lo: pd.DataFrame, hi: pd.DataFrame) -> dict:
    out = {}
    for h in HORIZONS:
        m = df[f"y_{h}"].notna()
        y = df.loc[m, f"y_{h}"]
        out[f"cover_{h}"] = float(((y >= lo.loc[m, f"lo_{h}"]) & (y <= hi.loc[m, f"hi_{h}"])).mean())
        out[f"width_{h}"] = float((hi.loc[m, f"hi_{h}"] - lo.loc[m, f"lo_{h}"]).mean())
    return out


def participant_bootstrap(df: pd.DataFrame, stat_fn, n_boot: int = 500, seed: int = SEED) -> tuple[float, float]:
    """95% CI by resampling participants (the independent unit), not time points."""
    rng = np.random.default_rng(seed)
    pids = df["participant_id"].unique()
    groups = {p: g for p, g in df.groupby("participant_id")}
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(pids, size=len(pids), replace=True)
        d = pd.concat([groups[p] for p in samp], ignore_index=True)
        try:
            v = stat_fn(d)
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)
