"""Summarise out-of-fold predictions into report tables (reports/results/summary_*.csv)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gtwin.config import REPORTS, TASK  # noqa: E402
from gtwin.evaluation.metrics import (alert_burden, calibration_table, event_metrics,  # noqa: E402
                                      forecast_errors, interval_metrics, participant_bootstrap)

OUT = REPORTS / "results"
HZ = TASK.forecast_points_min
MAIN = ["Persistence", "Ridge AR", "Gradient boosting", "Mechanistic (Kalman)", "Hybrid (Kalman + GB residual)"]


def load():
    oof = pd.read_csv(OUT / "oof_predictions.csv.gz")
    idx = pd.read_csv(OUT / "examples_index.csv.gz", parse_dates=["timestamp"])
    df = oof.merge(idx, on=["participant_id", "t_idx"], how="left")
    return df


def table_main(df, n_boot=300):
    rows = []
    c = df[df.scenario == "clean"]
    for name, d in c.groupby("model", sort=False):
        d = d.reset_index(drop=True)
        r = {"model": name, "participants": d.participant_id.nunique()}
        r.update(forecast_errors(d, d))
        r.update(event_metrics(d, d["p_event"]))
        r["MAE_60_CI"] = participant_bootstrap(d, lambda x: np.abs(x["pred_60"] - x["y_60"]).mean(), n_boot)
        r["MAE_120_CI"] = participant_bootstrap(d, lambda x: np.abs(x["pred_120"] - x["y_120"]).mean(), n_boot)

        def auprc(x):
            return event_metrics(x, x["p_event"])["AUPRC"]
        r["AUPRC_CI"] = participant_bootstrap(d, auprc, n_boot)
        if "lo_60" in d and d["lo_60"].notna().any():
            r.update(interval_metrics(d, d, d))
        rows.append(r)
    return pd.DataFrame(rows)


def table_intervals(df):
    """Coverage/width of marginal vs group-conditional (Mondrian) conformal intervals, by group."""
    rows = []
    c = df[(df.scenario == "clean") & df.model.isin(MAIN[1:])]
    for (name, grp), d in c.groupby(["model", "group"], sort=False):
        d = d.reset_index(drop=True)
        for variant, sfx in (("marginal", ""), ("group-conditional", "_grp")):
            lo = d[[f"lo_{h}{sfx}" for h in HZ]].set_axis([f"lo_{h}" for h in HZ], axis=1)
            hi = d[[f"hi_{h}{sfx}" for h in HZ]].set_axis([f"hi_{h}" for h in HZ], axis=1)
            rows.append({"model": name, "group": grp, "variant": variant, **interval_metrics(d, lo, hi)})
    return pd.DataFrame(rows)


def table_by_group(df):
    rows = []
    c = df[(df.scenario == "clean") & df.model.isin(MAIN)]
    for (name, grp), d in c.groupby(["model", "group"], sort=False):
        d = d.reset_index(drop=True)
        r = {"model": name, "group": grp, "participants": d.participant_id.nunique()}
        r.update({k: v for k, v in forecast_errors(d, d).items() if k.startswith("MAE")})
        r.update(event_metrics(d, d["p_event"]))
        if "lo_60" in d and d["lo_60"].notna().any():
            r.update({k: v for k, v in interval_metrics(d, d, d).items() if k.startswith("cover")})
        rows.append(r)
    return pd.DataFrame(rows)


def table_paired_diff(df, a="Hybrid (Kalman + GB residual)", b="Gradient boosting", n_boot=1000):
    """Participant-level paired comparison: mean of per-participant MAE differences, bootstrap CI."""
    c = df[df.scenario == "clean"]
    out = []
    for h in (30, 60, 120):
        pa = c[c.model == a].groupby("participant_id").apply(lambda x: np.abs(x[f"pred_{h}"] - x[f"y_{h}"]).mean())
        pb = c[c.model == b].groupby("participant_id").apply(lambda x: np.abs(x[f"pred_{h}"] - x[f"y_{h}"]).mean())
        dlt = (pa - pb).dropna()
        rng = np.random.default_rng(0)
        bs = [rng.choice(dlt.values, len(dlt)).mean() for _ in range(n_boot)]
        out.append({"comparison": f"{a} minus {b}", "horizon_min": h, "mean_MAE_diff": dlt.mean(),
                    "CI_low": np.percentile(bs, 2.5), "CI_high": np.percentile(bs, 97.5),
                    "participants_better": int((dlt < 0).sum()), "participants": len(dlt)})
    return pd.DataFrame(out)


def table_personalisation(df):
    rows = []
    pers = df[df.scenario == "personalised_after_day3"]
    keys = pers[["participant_id", "t_idx"]].drop_duplicates()
    for name in ["Mechanistic (Kalman)", "Hybrid (Kalman + GB residual)", "Gradient boosting"]:
        d = df[(df.scenario == "clean") & (df.model == name)].merge(keys, on=["participant_id", "t_idx"])
        rows.append({"model": name, "variant": "population", **forecast_errors(d, d), **event_metrics(d, d["p_event"])})
    for name, d in pers.groupby("model"):
        rows.append({"model": name.replace(" (personalised)", ""), "variant": "personalised (first 3 days)",
                     **forecast_errors(d, d), **event_metrics(d, d["p_event"])})
    return pd.DataFrame(rows)


def table_robustness(df):
    rows = []
    sel = df.model.isin(MAIN) | df.model.str.contains("noise model")
    for (scen, name), d in df[df.scenario.isin(["clean", "dropout30", "noise10"]) & sel].groupby(["scenario", "model"]):
        if scen == "clean":
            keys = df[df.scenario == "dropout30"][["participant_id", "t_idx"]].drop_duplicates()
            d = d.merge(keys, on=["participant_id", "t_idx"])
        rows.append({"scenario": scen, "model": name, **{k: v for k, v in forecast_errors(d, d).items() if k.startswith("MAE")},
                     **event_metrics(d, d["p_event"])})
    return pd.DataFrame(rows)


def table_alerts(df):
    rows = []
    c = df[(df.scenario == "clean") & df.model.isin(MAIN)]
    for name, d in c.groupby("model", sort=False):
        for label, thr_fn in [("threshold from calibration (<=1 false alert/day)", None),
                              ("fixed p>=0.5", 0.5), ("fixed p>=0.7", 0.7)]:
            if thr_fn is None:  # per-fold thresholds chosen on calibration participants
                parts = [alert_burden(g, g["p_event"], float(g["alert_thr"].iloc[0])) for _, g in d.groupby("fold")]
                tot_days = sum(p["monitored_days"] for p in parts)
                r = {"alerts_per_day": sum(p["alerts_per_day"] * p["monitored_days"] for p in parts) / tot_days,
                     "false_alerts_per_day": sum(p["false_alerts_per_day"] * p["monitored_days"] for p in parts) / tot_days,
                     "episode_recall": np.nansum([p["episode_recall"] * p["episodes"] for p in parts]) / max(sum(p["episodes"] for p in parts), 1),
                     "median_lead_min": np.nanmedian([p["median_lead_min"] for p in parts]),
                     "episodes": sum(p["episodes"] for p in parts), "monitored_days": tot_days,
                     "threshold": "per fold"}
            else:
                r = alert_burden(d, d["p_event"], thr_fn)
            rows.append({"model": name, "policy": label, **r})
    return pd.DataFrame(rows)


def table_alert_curve(df):
    rows = []
    c = df[(df.scenario == "clean") & df.model.isin(["Gradient boosting", "Mechanistic (Kalman)", "Hybrid (Kalman + GB residual)"])]
    for name, d in c.groupby("model"):
        for thr in np.round(np.arange(0.1, 0.96, 0.05), 2):
            rows.append({"model": name, **alert_burden(d, d["p_event"], float(thr))})
    return pd.DataFrame(rows)


def table_external():
    p = OUT / "external_shanghai_predictions.csv.gz"
    if not p.exists():
        return None
    e = pd.read_csv(p)
    rows = []
    for (ds, name), d in e.groupby(["dataset", "model"], sort=False):
        d = d.reset_index(drop=True)
        hz = [h for h in HZ if f"pred_{h}" in d and d[f"y_{h}"].notna().any()]
        r = {"dataset": ds, "model": name, "participants": d.participant_id.nunique()}
        for h in hz:
            m = d[f"y_{h}"].notna()
            r[f"MAE_{h}"] = float(np.abs(d.loc[m, f"pred_{h}"] - d.loc[m, f"y_{h}"]).mean())
        r.update(event_metrics(d, d["p_event"]))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    df = load()
    tabs = {"main": table_main(df), "by_group": table_by_group(df), "paired_hybrid_vs_gb": table_paired_diff(df),
            "paired_hybrid_vs_mech": table_paired_diff(df, b="Mechanistic (Kalman)"),
            "personalisation": table_personalisation(df), "robustness": table_robustness(df),
            "intervals": table_intervals(df),
            "alerts": table_alerts(df), "alert_curve": table_alert_curve(df), "external": table_external()}
    for k in ["Gradient boosting", "Mechanistic (Kalman)", "Hybrid (Kalman + GB residual)"]:
        d = df[(df.scenario == "clean") & (df.model == k)]
        tabs[f"calibration_{k.split()[0].lower()}"] = calibration_table(d, d["p_event"])
    for k, t in tabs.items():
        if t is not None:
            t.to_csv(OUT / f"summary_{k}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for k in ["main", "by_group", "paired_hybrid_vs_gb", "paired_hybrid_vs_mech", "personalisation", "robustness",
              "intervals", "alerts", "external"]:
        t = tabs[k]
        if t is None:
            continue
        print(f"\n=== {k} ===")
        cols = [c for c in t.columns if not c.startswith(("RMSE", "width", "n_points"))]
        print(t[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
