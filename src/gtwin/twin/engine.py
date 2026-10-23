"""Digital Twin state engine: a virtual patient whose state updates as each new reading arrives.

Separation of concerns
  * `PatientRecord`  - static profile + recorded stream (replayed, never invented)
  * `TwinEngine`     - per-patient live state: Kalman state, observation history, quality flags,
                       forecasts, alert history. One engine instance per patient: switching patient
                       creates a new engine, so no state can carry over.
  * models           - trained artefacts loaded read-only (population Kalman parameters + optional
                       hybrid residual / event model)

`advance()` = the "Advance 5 minutes" button: consume the next recorded slot, validate it,
update the state, and produce a new forecast (or an explicit refusal when data quality is poor).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gtwin.config import TASK
from gtwin.data.cgm import GLU_MAX, GLU_MIN
from gtwin.dataset import participant_examples
from gtwin.models.mechanistic import GlucoseKF, MechParams

HZ = TASK.forecast_points_min


@dataclass
class PatientRecord:
    patient_id: str
    profile: dict
    timestamps: pd.Series           # regular 5-min grid
    glucose: np.ndarray             # NaN = missing reading
    source: str = "unknown"         # e.g. "CGMacros (real, de-identified)" or "Synthetic"
    meals: pd.DataFrame | None = None  # optional: timestamp, carbs (context display only)


@dataclass
class Forecast:
    time: pd.Timestamp
    mean: np.ndarray                # per 5-min step, 1..H
    lo: np.ndarray
    hi: np.ndarray
    points: dict                    # minute -> (mean, lo, hi) at reporting horizons
    p_event: float | None
    model: str
    quality: list[str]
    usable: bool
    in_excursion: bool = False      # latest reading already >= threshold: an excursion is under way,
                                    # so this is not a *prediction* and no alert is raised (matches evaluation)


@dataclass
class TwinEngine:
    record: PatientRecord
    params: MechParams
    hybrid: object | None = None          # fitted GBM(base_col="mech") or None
    conformal_q: dict | None = None       # minute -> normalised conformal quantile
    alert_threshold: float = 0.5
    refractory_min: int = 60
    cursor: int = -1
    state_m: np.ndarray | None = None
    state_P: np.ndarray | None = None
    history: list = field(default_factory=list)
    alerts: list = field(default_factory=list)
    _seen_ts: set = field(default_factory=set)

    def __post_init__(self):
        self.kf = GlucoseKF(self.params, dt=TASK.step_min)
        self.n = len(self.record.glucose)
        self.obs = np.full(self.n, np.nan)   # what the twin has actually received so far

    # ------------------------------------------------------------------ input validation
    def _validate(self, t: int) -> tuple[float, list[str]]:
        flags = []
        ts = self.record.timestamps.iloc[t]
        y = float(self.record.glucose[t])
        if ts in self._seen_ts:
            flags.append("duplicate timestamp ignored")
            return np.nan, flags
        self._seen_ts.add(ts)
        if np.isnan(y):
            flags.append("no reading this interval")
        elif not (GLU_MIN <= y <= GLU_MAX):
            flags.append(f"implausible value {y:.0f} mg/dL rejected")
            y = np.nan
        return y, flags

    def quality_status(self) -> list[str]:
        t = self.cursor
        issues = []
        L = TASK.n_hist
        lo = max(0, t + 1 - L)
        cov = np.mean(~np.isnan(self.obs[lo:t + 1])) if t >= 0 else 0.0
        if t + 1 < L:
            issues.append(f"insufficient history ({(t + 1) * TASK.step_min} of {TASK.history_min} min)")
        elif cov < TASK.min_history_coverage:
            issues.append(f"history coverage {cov:.0%} < {TASK.min_history_coverage:.0%}")
        last = np.where(~np.isnan(self.obs[:t + 1]))[0]
        if len(last) == 0:
            issues.append("no readings yet")
        else:
            stale = (t - last[-1]) * TASK.step_min
            if stale > TASK.max_staleness_min:
                issues.append(f"stale data: last reading {stale} min ago")
        return issues

    # ------------------------------------------------------------------ main step
    def advance(self, steps: int = 1) -> Forecast | None:
        fc = None
        for _ in range(steps):
            if self.cursor + 1 >= self.n:
                self.history.append({"time": None, "event": "end of replay stream"})
                return fc
            self.cursor += 1
            t = self.cursor
            y, flags = self._validate(t)
            self.obs[t] = y
            if self.state_m is None:
                if np.isnan(y):
                    continue
                self.state_m, self.state_P = self.kf.init_state(y)
            else:
                self.state_m, self.state_P, _, _ = self.kf.step(self.state_m, self.state_P, y)
            fc = self.forecast(flags)
            self._log(fc, y)
        return fc

    def forecast(self, flags: list[str] | None = None) -> Forecast:
        t = self.cursor
        ts = self.record.timestamps.iloc[t]
        quality = (flags or []) + self.quality_status()
        usable = not any(q.startswith(("insufficient", "history coverage", "stale", "no readings")) for q in quality)
        H = TASK.n_hor
        try:
            mean, sd, pev = self.kf.forecast_batch(self.state_m[None], self.state_P[None], H, n_samples=400,
                                                   threshold=TASK.threshold_mgdl,
                                                   rng=np.random.default_rng(t))
            mean, sd, p_event, model = mean[0], sd[0], float(pev[0]), "Kalman twin"
            if self.hybrid is not None and usable:
                mean, p_event = self._hybrid_adjust(mean, sd, p_event, t)
                model = "Hybrid (Kalman + GB residual)"
        except Exception as exc:  # model failure must never crash the dashboard
            quality.append(f"model error: {exc.__class__.__name__}")
            mean = np.full(H, np.nan)
            sd = np.full(H, np.nan)
            p_event, model, usable = None, "unavailable", False
        k = np.arange(1, H + 1) * TASK.step_min
        q = np.interp(k, list(self.conformal_q.keys()), list(self.conformal_q.values())) if self.conformal_q else np.full(H, 1.645)
        lo, hi = mean - q * sd, mean + q * sd
        pts = {h: (float(mean[h // TASK.step_min - 1]), float(lo[h // TASK.step_min - 1]), float(hi[h // TASK.step_min - 1])) for h in HZ}
        seen = self.obs[:t + 1]
        seen = seen[~np.isnan(seen)]
        in_exc = bool(len(seen) and seen[-1] >= TASK.threshold_mgdl)
        return Forecast(ts, mean, lo, hi, pts, p_event if usable else None, model, quality, usable, in_exc)

    def _hybrid_adjust(self, mean, sd, p_event, t):
        """Apply the learned residual correction at the reporting horizons, interpolate between them."""
        ex = participant_examples(self.record.timestamps, self.obs, TASK, t_list=[t], require_quality=False)
        if ex.empty:
            return mean, p_event
        for c, v in self.record.profile.items():
            ex[c] = v
        ex["male"] = int(str(self.record.profile.get("gender", "")).lower().startswith("m"))
        for h in HZ:
            ex[f"mech_{h}"] = mean[h // TASK.step_min - 1]
            ex[f"mech_sd_{h}"] = sd[h // TASK.step_min - 1]
        ex["mech_p_event"] = p_event
        ex["mech_R"], ex["mech_B"] = self.state_m[1], self.state_m[2]
        for f in self.hybrid.feats:
            if f not in ex:
                ex[f] = np.nan
        pr = self.hybrid.predict(ex)
        knots_k = [0] + [h // TASK.step_min for h in HZ]
        resid = [0.0] + [float(pr[f"pred_{h}"].iloc[0] - mean[h // TASK.step_min - 1]) for h in HZ]
        corr = np.interp(np.arange(1, len(mean) + 1), knots_k, resid)
        return mean + corr, float(pr["p_event"].iloc[0])

    def _log(self, fc: Forecast | None, y: float):
        if fc is None:
            return
        rec = {"time": fc.time, "reading": y, "p_event": fc.p_event, "usable": fc.usable, "in_excursion": fc.in_excursion,
               "quality": "; ".join(fc.quality), **{f"pred_{h}": fc.points[h][0] for h in HZ}}
        self.history.append(rec)
        if fc.p_event is not None and not fc.in_excursion and fc.p_event >= self.alert_threshold:
            last = self.alerts[-1]["time"] if self.alerts else None
            if last is None or (fc.time - last).total_seconds() / 60 >= self.refractory_min:
                self.alerts.append({"time": fc.time, "p_event": fc.p_event,
                                    "predicted_peak": float(np.nanmax(fc.mean)), "status": "open"})

    # ------------------------------------------------------------------ helpers for the UI
    def observed_so_far(self) -> pd.DataFrame:
        t = self.cursor + 1
        return pd.DataFrame({"timestamp": self.record.timestamps.iloc[:t].values, "glucose": self.obs[:t]})

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([h for h in self.history if h.get("time") is not None])

    def outcome_of_alerts(self) -> pd.DataFrame:
        """Retrospective: did a reading >= threshold follow within 120 min? (only once that time has passed)"""
        rows = []
        for a in self.alerts:
            t0 = a["time"]
            end = t0 + pd.Timedelta(minutes=TASK.horizon_min)
            seen = self.observed_so_far()
            if seen["timestamp"].max() < end:
                outcome = "pending"
            else:
                w = seen[(seen.timestamp > t0) & (seen.timestamp <= end)]["glucose"]
                outcome = "excursion followed" if (w >= TASK.threshold_mgdl).any() else "no excursion (false alert)"
            rows.append({**a, "outcome": outcome})
        return pd.DataFrame(rows)
