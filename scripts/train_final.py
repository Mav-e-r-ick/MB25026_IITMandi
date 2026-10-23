"""Train the deployable artefacts used by the dashboard (after evaluation has been done by CV).

  models/twin_params.json     population Kalman parameters (5-min resolution)
  models/hybrid.joblib        hybrid residual + event model (scikit-learn), Platt-calibrated
  models/serving.json         conformal quantiles (marginal + per group) and alert threshold
  data/demo/                  synthetic cohort (+ small CC BY-NC-SA CGMacros slice for local replay)

Licence note: models trained on CGMacros are treated as adaptations of CC BY-NC-SA 4.0 data -
non-commercial, attribution, share-alike (see DATA_LICENSES.md).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gtwin.config import MODELS_DIR, ROOT as R, SEED, TASK  # noqa: E402
from gtwin.data.loaders import load_cgmacros  # noqa: E402
from gtwin.dataset import SENSOR_FEATURES, build_examples, static_features  # noqa: E402
from gtwin.evaluation.conformal import Conformal  # noqa: E402
from gtwin.models.mechanistic import MechParams, fit_forecast_params  # noqa: E402
from gtwin.models.ml import MECH_FEATS, GBM  # noqa: E402
from gtwin.models.state_features import mech_columns  # noqa: E402
from gtwin.simulate import make_cohort  # noqa: E402
from run_experiments import choose_alert_threshold, series_dict  # noqa: E402

DEMO = R / "data" / "demo"


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    DEMO.mkdir(parents=True, exist_ok=True)
    cgm, prof, _ = load_cgmacros()
    ex = build_examples(cgm, prof)
    feats = SENSOR_FEATURES + static_features(ex)
    sers = series_dict(cgm)
    pids = prof.set_index("participant_id")["group"]
    fit_p, cal_p = train_test_split(np.asarray(pids.index, dtype=object), test_size=0.2,
                                    stratify=pids.values, random_state=SEED)
    prm = fit_forecast_params([sers[p] for p in fit_p], MechParams(r=1.0), maxiter=300)
    print("params", prm)
    ex = ex.join(mech_columns(sers, ex, prm))
    e_fit, e_cal = ex[ex.participant_id.isin(fit_p)], ex[ex.participant_id.isin(cal_p)]
    hyb = GBM(base_col="mech", extra_feats=MECH_FEATS, name="Hybrid (Kalman + GB residual)").fit(e_fit, e_cal, feats)
    cal_pred = hyb.predict(e_cal)
    cf = Conformal(alpha=0.1, normalised=True, by="group").fit(e_cal, cal_pred)
    thr = choose_alert_threshold(e_cal, cal_pred["p_event"])
    joblib.dump(hyb, MODELS_DIR / "hybrid.joblib", compress=3)
    p = {k: (float(v) if not isinstance(v, bool) else v) for k, v in asdict(prm).items()}
    json.dump(p, open(MODELS_DIR / "twin_params.json", "w"), indent=1)
    json.dump({"conformal_alpha": 0.1, "conformal_q": cf.q, "conformal_q_by_group": cf.qg, "alert_threshold": thr,
               "alert_policy": "lowest threshold with <=1 false alert per patient-day on calibration participants",
               "fit_participants": list(fit_p), "calibration_participants": list(cal_p), "features": hyb.feats,
               "trained_on": "CGMacros v1.0.0 (CC BY-NC-SA 4.0)"},
              open(MODELS_DIR / "serving.json", "w"), indent=1, default=float)
    print("alert threshold", thr, "conformal q", {k: round(v, 2) for k, v in cf.q.items()})

    # ---- demo data: synthetic cohort (always shipped)
    rows, profs, meals = [], [], []
    for p in make_cohort():
        rows.append(pd.DataFrame({"patient_id": p["patient_id"], "timestamp": p["timestamps"], "glucose": p["glucose"]}))
        profs.append({"patient_id": p["patient_id"], **p["profile"], "source": p["source"]})
        if len(p["meals"]):
            meals.append(p["meals"].assign(patient_id=p["patient_id"]))
    pd.concat(rows).to_csv(DEMO / "synthetic_cgm.csv", index=False)
    pd.DataFrame(profs).to_csv(DEMO / "synthetic_profiles.csv", index=False)
    pd.concat(meals).to_csv(DEMO / "synthetic_meals.csv", index=False)

    # ---- small real slice for replay (4 participants x 4 days, CC BY-NC-SA 4.0, attribution in README)
    pick = []
    for grp, k in (("T2D", 2), ("Prediabetes", 1), ("Healthy", 1)):
        # calibration participants (not used to fit the model) with the fewest missing readings in their first 4 days
        miss = {p: cgm[cgm.participant_id == p].sort_values("timestamp")["glucose"].iloc[: 4 * 288].isna().mean() for p in cal_p}
        cand = sorted([p for p in cal_p if pids[p] == grp], key=lambda p: miss[p])[:k]
        pick += cand
    sl = []
    for p in pick:
        d = cgm[cgm.participant_id == p].sort_values("timestamp").iloc[: 4 * 288]
        sl.append(d)
    pd.concat(sl).to_csv(DEMO / "cgmacros_slice.csv", index=False)
    pr = prof[prof.participant_id.isin(pick)].copy()
    pr["source"] = "CGMacros v1.0.0 (real, de-identified; CC BY-NC-SA 4.0)"
    pr.to_csv(DEMO / "cgmacros_slice_profiles.csv", index=False)
    print("demo slice participants", pick)


if __name__ == "__main__":
    main()
