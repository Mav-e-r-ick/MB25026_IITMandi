"""Precompute a full replay of every demo patient with the trained twin, for the web prototype.

Writes prototype/replay.json. Every number in the web prototype comes from this file (or from the evaluation
summaries) - the page itself does not invent predictions. Run after scripts/train_final.py.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gtwin.config import MODELS_DIR, REPORTS, TASK  # noqa: E402
from gtwin.models.mechanistic import MechParams  # noqa: E402
from gtwin.twin.engine import PatientRecord, TwinEngine  # noqa: E402

DEMO = ROOT / "data" / "demo"
OUT = ROOT / "prototype"
QCODES = {"insufficient": "short_history", "history coverage": "sparse_history", "stale": "stale",
          "no reading": "missing", "implausible": "implausible", "duplicate": "duplicate", "model error": "model_error"}


def qcode(q: str) -> str:
    for k, v in QCODES.items():
        if q.startswith(k):
            return v
    return "other"


def records():
    out = []
    for cgm_f, prof_f, meals_f, idc in [("synthetic_cgm.csv", "synthetic_profiles.csv", "synthetic_meals.csv", "patient_id"),
                                       ("cgmacros_slice.csv", "cgmacros_slice_profiles.csv", None, "participant_id")]:
        cgm = pd.read_csv(DEMO / cgm_f, parse_dates=["timestamp"]).rename(columns={idc: "patient_id"})
        prof = pd.read_csv(DEMO / prof_f).rename(columns={idc: "patient_id"})
        meals = pd.read_csv(DEMO / meals_f, parse_dates=["timestamp"]) if meals_f else None
        for pid, d in cgm.groupby("patient_id", sort=False):
            p = prof[prof.patient_id == pid].iloc[0].to_dict()
            src = p.pop("source")
            p.pop("patient_id", None)
            m = meals[meals.patient_id == pid] if meals is not None else None
            out.append(PatientRecord(str(pid), p, d["timestamp"].reset_index(drop=True), d["glucose"].to_numpy(float),
                                     src, m))
    return out


def main():
    prm = MechParams(**json.load(open(MODELS_DIR / "twin_params.json")))
    serving = json.load(open(MODELS_DIR / "serving.json"))
    hybrid = joblib.load(MODELS_DIR / "hybrid.joblib")
    patients = []
    for rec in records():
        grp = rec.profile.get("group")
        q = serving["conformal_q_by_group"].get(grp) or serving["conformal_q"]
        eng = TwinEngine(rec, prm, hybrid=hybrid, conformal_q={int(k): float(v) for k, v in q.items()},
                         alert_threshold=float(serving["alert_threshold"]))
        steps = []
        for t in range(len(rec.glucose)):
            fc = eng.advance(1)
            if fc is None:  # no reading yet to initialise the twin
                steps.append({"r": None, "u": 0, "q": ["short_history"]})
                continue
            y = eng.obs[t]
            s = {"r": None if np.isnan(y) else int(round(y)), "u": int(fc.usable),
                 "q": sorted({qcode(x) for x in fc.quality})}
            if fc.in_excursion:
                s["e"] = 1
            if fc.usable:
                s["p"] = round(float(fc.p_event), 3)
                s["m"] = [int(round(v)) for v in fc.mean]
                s["lo"] = [int(round(v)) for v in fc.lo]
                s["hi"] = [int(round(v)) for v in fc.hi]
            s["x"] = [round(float(v), 3) for v in eng.state_m]  # twin state: glucose, inflow, baseline
            steps.append(s)
        alerts = eng.outcome_of_alerts()
        al = []
        for _, a in alerts.iterrows():
            al.append({"i": int(rec.timestamps[rec.timestamps == a["time"]].index[0]), "p": round(float(a["p_event"]), 3),
                       "peak": int(round(a["predicted_peak"])), "outcome": a["outcome"]})
        prof = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in rec.profile.items()}
        meals = []
        if rec.meals is not None and len(rec.meals):
            for _, r in rec.meals.iterrows():
                idx = int(((r["timestamp"] - rec.timestamps.iloc[0]).total_seconds() // 60) // TASK.step_min)
                meals.append({"i": idx, "name": r["meal"], "carbs": int(r["carbs_g"])})
        patients.append({"id": rec.patient_id, "source": rec.source, "synthetic": "SYNTH" in rec.source.upper(),
                         "profile": prof, "t0": rec.timestamps.iloc[0].isoformat(), "steps": steps, "alerts": al,
                         "meals": meals})
        n_use = sum(s["u"] for s in steps)
        print(f"{rec.patient_id}: {len(steps)} steps, {n_use} usable, {len(al)} alerts", flush=True)

    res = REPORTS / "results"
    main_t = pd.read_csv(res / "summary_main.csv")
    keep = ["Persistence", "Ridge AR", "Gradient boosting", "Mechanistic (Kalman)", "Hybrid (Kalman + GB residual)"]
    main_t = main_t[main_t.model.isin(keep)][["model", "MAE_30", "MAE_60", "MAE_120", "AUROC", "AUPRC"]]
    curve = pd.read_csv(res / "summary_alert_curve.csv")
    curve = curve[curve.model == "Hybrid (Kalman + GB residual)"][
        ["threshold", "alerts_per_day", "false_alerts_per_day", "episode_recall", "median_lead_min"]]
    imp = pd.read_csv(res / "feature_importance_hybrid_60min.csv", index_col=0).head(8)
    meta = {"step_min": TASK.step_min, "horizon_steps": TASK.n_hor, "threshold": TASK.threshold_mgdl,
            "history_steps": TASK.n_hist, "alert_threshold": float(serving["alert_threshold"]),
            "params": {k: float(v) for k, v in vars(prm).items() if k != "use_meals"},
            "results": main_t.round(3).to_dict("records"), "alert_curve": curve.round(3).to_dict("records"),
            "importance": [{"feature": k, "mae_increase": round(float(v), 2)} for k, v in imp["mae_increase"].items()]}
    OUT.mkdir(exist_ok=True)
    def clean(o):  # JSON has no NaN: write null instead
        if isinstance(o, float) and np.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    with open(OUT / "replay.json", "w") as f:
        json.dump(clean({"meta": meta, "patients": patients}), f, separators=(",", ":"), allow_nan=False)
    print("wrote", OUT / "replay.json", round((OUT / "replay.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
