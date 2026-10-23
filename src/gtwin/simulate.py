"""Synthetic virtual patients for the public demo (clearly labelled SYNTHETIC everywhere).

Generated from the same state-space structure as the twin, in meal mode, with:
  * meal times and carbohydrate loads loosely reflecting a typical urban Indian day
    (rice/roti-based lunch and dinner, evening snack) - illustrative assumptions, not survey data;
  * group-specific basal level and clearance (healthy / prediabetes / T2D) chosen so that mean glucose
    and time >180 mg/dL resemble the corresponding CGMacros groups;
  * HbA1c derived from simulated mean glucose with the ADAG relation (HbA1c% = (mean + 46.7) / 28.7).

Because the generator shares the twin's model family, accuracy on synthetic patients is an
optimistic upper bound and is never reported as a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtwin.models.mechanistic import GlucoseKF, MechParams

GROUP_PRIORS = {  # basal mg/dL, clearance 1/min, carb effect gain
    "Healthy": dict(basal=(92, 6), p=(0.035, 0.006), gain=(1.1, 0.2)),
    "Prediabetes": dict(basal=(102, 6), p=(0.026, 0.005), gain=(1.4, 0.25)),
    "T2D": dict(basal=(125, 12), p=(0.02, 0.004), gain=(1.9, 0.35)),
}
MEALS = [  # (name, hour mean, hour sd, carbs mean g, carbs sd g, probability)
    ("Breakfast", 8.5, 0.6, 55, 15, 0.95),
    ("Lunch", 13.5, 0.7, 85, 20, 0.95),
    ("Snack", 17.5, 0.6, 30, 10, 0.6),
    ("Dinner", 21.0, 0.7, 80, 20, 0.95),
]


def make_patient(pid: str, group: str, days: int = 5, seed: int = 0, start="2026-01-05 00:00") -> dict:
    rng = np.random.default_rng(seed)
    pr = GROUP_PRIORS[group]
    basal = rng.normal(*pr["basal"])
    prm = MechParams(p=max(0.006, rng.normal(*pr["p"])), tau=20.0, q_g=0.4, q_r=0.002, q_b=0.002, r=9.0,
                     k_abs=0.025, c_meal=max(0.5, rng.normal(*pr["gain"])), use_meals=True)
    kf = GlucoseKF(prm, dt=5)
    n = days * 288
    ts = pd.date_range(start, periods=n, freq="5min")
    u = np.zeros(n)
    meals = []
    for d in range(days):
        for name, h, hs, c, cs, prob in MEALS:
            if rng.random() > prob:
                continue
            k = int(np.clip(d * 288 + rng.normal(h, hs) * 12, 0, n - 1))
            carbs = float(np.clip(rng.normal(c, cs), 10, 160))
            u[k] += carbs
            meals.append({"timestamp": ts[k], "meal": name, "carbs_g": round(carbs)})
    x = np.zeros(kf.nx)
    x[0] = x[2] = basal
    LQ = np.linalg.cholesky(kf.Q + 1e-12 * np.eye(kf.nx))
    g = np.empty(n)
    for t in range(n):
        x = kf.A @ x + kf.Bu * u[t] + LQ @ rng.standard_normal(kf.nx)
        x[0] = max(x[0], 45.0)
        g[t] = x[0] + rng.normal(0, np.sqrt(prm.r))
    g = np.round(np.clip(g, 40, 400))
    # realistic missingness: a few short sensor dropouts
    for _ in range(rng.integers(1, 4)):
        s = rng.integers(0, n - 12)
        g[s:s + rng.integers(2, 10)] = np.nan
    mean_g = np.nanmean(g)
    sex = "male" if rng.random() < 0.5 else "female"
    age = int(np.clip(rng.normal({"Healthy": 35, "Prediabetes": 45, "T2D": 55}[group], 8), 22, 75))
    profile = {"age": age, "gender": sex, "group": group, "hba1c": round((mean_g + 46.7) / 28.7, 1),
               "bmi": round(float(np.clip(rng.normal({"Healthy": 23, "Prediabetes": 26, "T2D": 28}[group], 3), 18, 40)), 1),
               "diagnoses": {"Healthy": "None recorded", "Prediabetes": "Prediabetes (HbA1c 5.7-6.4%)",
                             "T2D": "Type 2 diabetes"}[group],
               "medications": {"Healthy": "None", "Prediabetes": "None", "T2D": "Metformin (synthetic record)"}[group]}
    return {"patient_id": pid, "profile": profile, "timestamps": pd.Series(ts), "glucose": g,
            "meals": pd.DataFrame(meals), "source": "SYNTHETIC (generated, not a real person)"}


def make_cohort(seed: int = 2026) -> list[dict]:
    spec = [("S-01", "T2D"), ("S-02", "T2D"), ("S-03", "Prediabetes"), ("S-04", "T2D"), ("S-05", "Healthy"),
            ("S-06", "Prediabetes")]
    return [make_patient(pid, grp, seed=seed + i) for i, (pid, grp) in enumerate(spec)]
