"""End-to-end evaluation: participant-level 5-fold CV on CGMacros + external test on ShanghaiT2DM.

Usage:  python scripts/run_experiments.py [--quick]

Outputs (reports/results/):  oof_predictions.csv.gz, summary_*.csv, run_log.json
Design:
  * participants (not time points) are the unit of splitting; folds stratified by diabetes group
  * inside each training fold, 20% of participants are held out for calibration
    (conformal intervals, Platt scaling, alert-threshold choice) - the test fold is never used for tuning
  * all models see identical examples
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gtwin.config import REPORTS, SEED, TASK  # noqa: E402
from gtwin.data.loaders import load_cgmacros, load_shanghai_t2dm  # noqa: E402
from gtwin.dataset import SENSOR_FEATURES, build_examples, static_features  # noqa: E402
from gtwin.evaluation.conformal import Conformal  # noqa: E402
from gtwin.models.mechanistic import MechParams, fit_forecast_params  # noqa: E402
from gtwin.models.ml import MECH_FEATS, GBM, make_models  # noqa: E402
from gtwin.models.state_features import mech_columns  # noqa: E402

OUT = REPORTS / "results"
HZ = TASK.forecast_points_min
PERSONAL_DAYS = 3.0


def series_dict(cgm, col="glucose"):
    return {p: d.sort_values("timestamp")[col].to_numpy(float) for p, d in cgm.groupby("participant_id")}


def corrupt(cgm: pd.DataFrame, kind: str, rng) -> pd.DataFrame:
    """Test-time robustness scenarios. Labels always come from the clean series."""
    c = cgm.copy()
    g = c["glucose"].to_numpy(float).copy()
    if kind == "dropout30":
        g[rng.random(len(g)) < 0.30] = np.nan
    elif kind == "noise10":
        g = g + rng.normal(0, 10, len(g))
    c["glucose_feat"] = g
    return c


def choose_alert_threshold(e_cal, p_cal, max_false_per_day: float = 1.0) -> float:
    from gtwin.evaluation.metrics import alert_burden
    for thr in np.round(np.arange(0.2, 0.96, 0.05), 2):
        r = alert_burden(e_cal, p_cal, float(thr))
        if r["false_alerts_per_day"] <= max_false_per_day:
            return float(thr)
    return 0.95


def fit_predict_fold(models, ex_fit, ex_cal, ex_test, feats):
    preds, cal_preds = {}, {}
    for m in models:
        m.fit(ex_fit, ex_cal, feats)
        preds[m.name] = m.predict(ex_test)
        cal_preds[m.name] = m.predict(ex_cal)
    return preds, cal_preds


def stack(preds: dict, ex: pd.DataFrame, scenario: str, fold: int, extra: dict | None = None):
    rows = []
    for name, p in preds.items():
        d = p.copy()
        d["model"], d["scenario"], d["fold"] = name, scenario, fold
        d["participant_id"] = ex["participant_id"].values
        d["t_idx"] = ex["t_idx"].values
        if extra and name in extra:
            for k, v in extra[name].items():
                d[k] = v.values
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main(quick: bool = False, external_only: bool = False):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cgm, prof, qc = load_cgmacros()
    if external_only:
        ext = external_shanghai(cgm, prof, None, None, None, quick)
        ext.to_csv(OUT / "external_shanghai_predictions.csv.gz", index=False)
        print(f"external done in {time.time()-t0:.0f}s")
        return
    qc.to_csv(OUT / "cgmacros_qc.csv", index=False)
    ex = build_examples(cgm, prof)
    feats = SENSOR_FEATURES + static_features(ex)
    sers = series_dict(cgm)
    pids = prof.set_index("participant_id")["group"]
    print(f"examples={len(ex)} participants={ex.participant_id.nunique()} feats={len(feats)}", flush=True)
    ex[["participant_id", "t_idx", "timestamp", "group", "g0", "event", "event_valid", "t_to_event_min"]
       + [f"y_{h}" for h in HZ]].to_csv(OUT / "examples_index.csv.gz", index=False)

    # corrupted copies (features only)
    scen_cgm = {k: corrupt(cgm, k, np.random.default_rng(SEED + i)) for i, k in enumerate(["dropout30", "noise10"])}
    from dataclasses import replace
    cfg_rob = replace(TASK, min_history_coverage=0.4)
    import gtwin.dataset as dsmod
    scen_ex = {}
    for k, c in scen_cgm.items():
        e = dsmod.build_examples(c, prof, cfg_rob, feature_col="glucose_feat", label_col="glucose")
        e = e.merge(ex[["participant_id", "t_idx"]], on=["participant_id", "t_idx"])  # same points as clean
        scen_ex[k] = e
    scen_sers = {k: series_dict(c, "glucose_feat") for k, c in scen_cgm.items()}

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    upid = np.asarray(pids.index, dtype=object)
    all_rows, fold_log = [], []
    for fold, (tr_i, te_i) in enumerate(sgkf.split(upid, pids.values, groups=upid)):
        tr_p, te_p = upid[tr_i], upid[te_i]
        fit_p, cal_p = train_test_split(tr_p, test_size=0.2, stratify=pids[tr_p].values, random_state=SEED + fold)
        print(f"fold {fold}: fit={len(fit_p)} cal={len(cal_p)} test={len(te_p)}", flush=True)
        # 1) system identification on fit participants only
        prm = fit_forecast_params([sers[p] for p in fit_p], MechParams(r=1.0), maxiter=120 if quick else 300)
        fold_log.append({"fold": fold, "test": list(te_p), "cal": list(cal_p), "params": {k: float(v) for k, v in vars(prm).items() if k != "use_meals"}})
        # 2) mechanistic columns for every example
        exf = ex.copy()
        mc = mech_columns(sers, exf, prm)
        exf = exf.join(mc)
        e_fit, e_cal, e_te = (exf[exf.participant_id.isin(s)] for s in (fit_p, cal_p, te_p))
        models = make_models() + [
            GBM(name="GB sensor-only", feats_override=SENSOR_FEATURES),
            GBM(name="GB profile + current reading", feats_override=["g0"] + static_features(ex)),
        ]
        preds, cal_preds = fit_predict_fold(models, e_fit, e_cal, e_te, feats)
        # 3) conformal intervals (calibrated on cal participants)
        extra = {}
        for name in preds:
            norm = name.startswith(("Mechanistic", "Hybrid"))
            cf = Conformal(alpha=0.1, normalised=norm).fit(e_cal, cal_preds[name])
            lo, hi = cf.interval(e_te, preds[name])
            cfg_ = Conformal(alpha=0.1, normalised=norm, by="group").fit(e_cal, cal_preds[name])
            lo_g, hi_g = cfg_.interval(e_te, preds[name])
            # alert threshold chosen on calibration participants only:
            # lowest threshold whose false-alert rate is <= 1 per patient-day (operational budget)
            thr = choose_alert_threshold(e_cal, cal_preds[name]["p_event"])
            extra[name] = {**{c: lo[c] for c in lo}, **{c: hi[c] for c in hi},
                           **{c + "_grp": lo_g[c] for c in lo_g}, **{c + "_grp": hi_g[c] for c in hi_g},
                           "alert_thr": pd.Series(thr, index=e_te.index)}
        all_rows.append(stack(preds, e_te, "clean", fold, extra))

        # 4) personalisation ablation: refit (p, tau) per test participant on first 3 days (MAP, prior = population)
        pers = {}
        for p in te_p:
            d = cgm[cgm.participant_id == p].sort_values("timestamp")
            n_cal = int(PERSONAL_DAYS * 288)
            if d["glucose"].iloc[:n_cal].notna().sum() < 288:
                pers[p] = prm
                continue
            pers[p] = fit_forecast_params([sers[p][:n_cal]], init=prm, prior=prm, prior_sd=0.5, free=(0, 1),
                                          maxiter=60 if quick else 120)
        late = e_te[e_te["t_idx"] >= int(PERSONAL_DAYS * 288)]
        mcp = mech_columns(sers, ex.loc[late.index], pers, seed=fold)
        late_p = ex.loc[late.index].join(mcp)
        hyb = next(m for m in models if m.name.startswith("Hybrid"))
        mech = next(m for m in models if m.name.startswith("Mechanistic"))
        pp = {"Mechanistic (personalised)": mech.predict(late_p), "Hybrid (personalised)": hyb.predict(late_p)}
        all_rows.append(stack(pp, late_p, "personalised_after_day3", fold))
        fold_log[-1]["personal_params"] = {p: {"p": float(v.p), "tau": float(v.tau)} for p, v in pers.items()}

        # 5) robustness scenarios (same fitted models, corrupted inputs)
        for k, e in scen_ex.items():
            et = e[e.participant_id.isin(te_p)].copy()
            et = et.join(mech_columns(scen_sers[k], et, prm, seed=fold))
            pr = {m.name: m.predict(et) for m in models if not m.name.startswith("GB ")}
            if k == "noise10":
                # structural advantage test: tell the twin the sensor is noisier (r += 10^2), no retraining
                from dataclasses import replace as _rep
                prm_n = _rep(prm, r=prm.r + 100.0)
                et_n = et.drop(columns=[c for c in et.columns if c.startswith("mech_")]).join(
                    mech_columns(scen_sers[k], et, prm_n, seed=fold))
                mech = next(m for m in models if m.name.startswith("Mechanistic"))
                hyb = next(m for m in models if m.name.startswith("Hybrid"))
                pr["Mechanistic (noise model updated)"] = mech.predict(et_n)
                pr["Hybrid (noise model updated)"] = hyb.predict(et_n)
            all_rows.append(stack(pr, et, k, fold))

        # feature importance of the fused GB model (fold 0 only, for the dashboard/report)
        if fold == 0:
            gb = next(m for m in models if m.name == "Gradient boosting")
            gb.feature_importance(e_te).to_csv(OUT / "feature_importance_gb_60min.csv", header=["mae_increase"])
            hy = next(m for m in models if m.name.startswith("Hybrid"))
            hy.feature_importance(e_te).to_csv(OUT / "feature_importance_hybrid_60min.csv", header=["mae_increase"])
        print(f"  done fold {fold} at {time.time()-t0:.0f}s", flush=True)

    oof = pd.concat(all_rows, ignore_index=True)
    oof.to_csv(OUT / "oof_predictions.csv.gz", index=False)

    # 6) external test: ShanghaiT2DM, glucose-only, 15-min resolution, models refit on ALL CGMacros
    ext = external_shanghai(cgm, prof, ex, feats, sers, quick)
    ext.to_csv(OUT / "external_shanghai_predictions.csv.gz", index=False)
    json.dump({"folds": fold_log, "seconds": time.time() - t0, "n_examples": len(ex), "features": feats},
              open(OUT / "run_log.json", "w"), indent=1, default=str)
    print(f"finished in {time.time()-t0:.0f}s", flush=True)


def external_shanghai(cgm, prof, ex, feats, sers, quick):
    """Downsample CGMacros to 15 min, train sensor-only models, test on Shanghai T2DM (never pooled)."""
    from dataclasses import replace

    import gtwin.dataset as dsmod
    cfg15 = replace(TASK, step_min=15)
    c15 = cgm.copy()
    c15 = c15[(c15["timestamp"].dt.minute % 15) == (c15.groupby("participant_id")["timestamp"].transform("min").dt.minute % 15)]
    # LAG/WIN definitions are in minutes, so the same features apply at 15-min resolution
    e15 = dsmod.build_examples(c15, prof, cfg15, stride=1)
    sh, shp = load_shanghai_t2dm()
    es = dsmod.build_examples(sh, shp, cfg15, stride=1)
    sfeat = [f for f in SENSOR_FEATURES if f in es and es[f].notna().any() and e15[f].notna().any()]
    s15 = series_dict(c15)
    prm = fit_forecast_params([s15[p] for p in s15], MechParams(r=1.0), maxiter=120 if quick else 250, dt=15.0)
    # refit at dt=15 by passing a 15-min KF through the generic code path
    from gtwin.models import state_features as sf
    old = sf.TASK
    sf.TASK = cfg15
    try:
        e15 = e15.join(sf.mech_columns(s15, e15, prm))
        es = es.join(sf.mech_columns(series_dict(sh), es, prm))
    finally:
        sf.TASK = old
    tr_p, cal_p = train_test_split(np.asarray(e15.participant_id.unique(), dtype=object), test_size=0.2, random_state=SEED)
    e_fit, e_cal = e15[e15.participant_id.isin(tr_p)], e15[e15.participant_id.isin(cal_p)]
    from gtwin.models.ml import GBM, Mechanistic, Persistence, RidgeAR
    models = [Persistence(), RidgeAR(), GBM(), Mechanistic(), GBM(base_col="mech", extra_feats=MECH_FEATS, name="Hybrid (Kalman + GB residual)")]
    rows = []
    for m in models:
        m.fit(e_fit, e_cal, sfeat)
        for nm, d in (("CGMacros-T2D-insample-15min", e15[e15.group == "T2D"]), ("ShanghaiT2DM", es)):
            p = m.predict(d)
            p["model"], p["dataset"] = m.name, nm
            p["participant_id"] = d["participant_id"].values
            for c in ["event", "event_valid", "timestamp", "t_to_event_min"] + [f"y_{h}" for h in HZ]:
                p[c] = d[c].values
            rows.append(p)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--external-only", action="store_true")
    a = ap.parse_args()
    main(a.quick, a.external_only)
