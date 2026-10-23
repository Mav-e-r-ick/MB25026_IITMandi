import numpy as np
import pandas as pd
import pytest

from gtwin.config import TASK
from gtwin.data.cgm import clean_values, deinterpolate_1min, to_regular_grid
from gtwin.dataset import participant_examples
from gtwin.models.mechanistic import GlucoseKF, MechParams, multistep_errors, run_filter
from gtwin.simulate import make_patient
from gtwin.twin.engine import PatientRecord, TwinEngine


def _interp_1min(readings, start="2024-01-01 08:00"):
    """Mimic the CGMacros release: real 5-min integer readings, linearly filled at 1 min."""
    t5 = pd.date_range(start, periods=len(readings), freq="5min")
    s = pd.Series(readings, index=t5, dtype=float).resample("1min").interpolate()
    return pd.DataFrame({"timestamp": s.index, "glucose": s.values})


# ------------------------------------------------------------------ data
def test_deinterpolation_recovers_real_readings():
    real = [100, 104, 111, 119, 125, 128, 126, 120]
    out = deinterpolate_1min(_interp_1min(real))
    assert out["glucose"].tolist() == real
    assert out["observed"].all()


def test_deinterpolation_with_phase_offset():
    df = _interp_1min([100, 107, 113, 118, 120]).iloc[2:]  # block starts between readings
    out = deinterpolate_1min(df)
    assert out["glucose"].tolist() == [107, 113, 118, 120]


def test_grid_keeps_gaps_as_missing():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:20"]),
                       "glucose": [100.0, 101.0, 110.0]})
    g = to_regular_grid(df, 5)
    assert len(g) == 5 and g["glucose"].isna().sum() == 2


def test_clean_values_flags():
    g, f = clean_values(pd.Series([100, 30, 105, 180, 106, 500.0]))
    assert f["out_of_range"] == 2 and np.isnan(g[1]) and np.isnan(g[5])


# ------------------------------------------------------------------ leakage
def test_features_do_not_use_future():
    rng = np.random.default_rng(0)
    n = 200
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="5min"))
    g = 120 + np.cumsum(rng.normal(0, 2, n))
    a = participant_examples(ts, g, stride=1)
    g2 = g.copy()
    g2[100:] += 80  # change the future only
    b = participant_examples(ts, g2, stride=1)
    feat_cols = [c for c in a.columns if not c.startswith(("y_", "event", "fut_", "t_to_"))]
    ra, rb = a[a.t_idx == 99][feat_cols], b[b.t_idx == 99][feat_cols]
    pd.testing.assert_frame_equal(ra.reset_index(drop=True), rb.reset_index(drop=True))
    assert a.loc[a.t_idx == 99, "y_60"].item() != b.loc[b.t_idx == 99, "y_60"].item()


def test_event_label_requires_future_coverage():
    n = 120
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="5min"))
    g = np.full(n, 120.0)
    g[61:80] = np.nan  # most of the next 2 h missing for t=60
    ex = participant_examples(ts, g, stride=1)
    row = ex[ex.t_idx == 60].iloc[0]
    assert row["event"] == -1 and not row["event_valid"]


def test_event_label_definition():
    n = 120
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="5min"))
    g = np.full(n, 150.0)
    g[70] = TASK.threshold_mgdl  # exactly at threshold counts as an excursion (>=)
    ex = participant_examples(ts, g, stride=1)
    assert ex.loc[ex.t_idx == 60, "event"].item() == 1
    assert ex.loc[ex.t_idx == 60, "t_to_event_min"].item() == 50
    assert ex.loc[ex.t_idx == 75, "event"].item() == 0


# ------------------------------------------------------------------ model
def test_kalman_steady_state_and_missing():
    kf = GlucoseKF(MechParams())
    y = np.full(300, 110.0)
    y[100:120] = np.nan
    m, P = run_filter(kf, y)
    assert abs(m[-1, 0] - 110) < 1 and abs(m[-1, 2] - 110) < 3
    assert P[119, 0, 0] > P[99, 0, 0]  # uncertainty grows during a gap


def test_forecast_batch_shapes_and_probability_bounds():
    kf = GlucoseKF(MechParams())
    m, P = run_filter(kf, np.linspace(100, 170, 100))
    mean, sd, p = kf.forecast_batch(m[[-1]], P[[-1]], TASK.n_hor, n_samples=300)
    assert mean.shape == (1, TASK.n_hor) and (sd > 0).all() and 0 <= p[0] <= 1
    _, _, p_hi = kf.forecast_batch(m[[-1]], P[[-1]], TASK.n_hor, threshold=400)
    assert p_hi[0] <= p[0]


def test_multistep_errors_resolution_independent():
    kf = GlucoseKF(MechParams(), dt=15)
    e, v = multistep_errors(kf, np.full(200, 100.0))
    assert e.shape[1] == 5 and np.allclose(e, 0, atol=1e-6)


# ------------------------------------------------------------------ twin engine
def _record(pid="S-T", group="T2D", seed=1):
    p = make_patient(pid, group, days=2, seed=seed)
    return PatientRecord(p["patient_id"], p["profile"], p["timestamps"], p["glucose"], p["source"], p["meals"])


def test_engine_refuses_until_enough_history_then_forecasts():
    eng = TwinEngine(_record(), MechParams())
    fc = eng.advance(5)
    assert fc is not None and not fc.usable and fc.p_event is None
    fc = eng.advance(TASK.n_hist)
    assert fc.usable and fc.p_event is not None and np.isfinite(fc.mean).all()


def test_engine_flags_stale_implausible_duplicate_and_end():
    rec = _record()
    rec.glucose[40:46] = np.nan
    rec.glucose[50] = 900
    eng = TwinEngine(rec, MechParams())
    eng.advance(46)
    assert any("stale" in q for q in eng.forecast().quality)
    eng.advance(5)
    assert any("implausible" in h["quality"] for h in eng.history if h.get("time") is not None)
    eng._seen_ts.add(rec.timestamps.iloc[eng.cursor + 1])
    eng.advance(1)
    assert "duplicate" in eng.history[-1]["quality"]
    eng.advance(len(rec.glucose))
    assert eng.history[-1].get("event") == "end of replay stream"


def test_switching_patients_does_not_carry_state():
    a = TwinEngine(_record("A", "T2D", 1), MechParams())
    a.advance(100)
    b = TwinEngine(_record("B", "Healthy", 2), MechParams())
    assert b.cursor == -1 and b.state_m is None and not b.alerts and not b.history


def test_no_alert_while_already_above_threshold():
    rec = _record()
    rec.glucose[:] = 250.0  # persistently high: an excursion is under way, nothing to predict
    eng = TwinEngine(rec, MechParams(), alert_threshold=0.0)
    eng.advance(80)
    assert eng.forecast().in_excursion and not eng.alerts
