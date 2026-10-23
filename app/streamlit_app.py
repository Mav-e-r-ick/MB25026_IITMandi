"""Clinician dashboard for the glucose-excursion Digital Twin.

Run:  streamlit run app/streamlit_app.py
Research prototype - not a medical device, not clinically validated.
"""
from __future__ import annotations

import os

# One thread per prediction: scikit-learn's OpenMP start-up costs ~6 ms per tree when predicting a
# single row, which made each 5-minute update take seconds. Must be set before sklearn is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gtwin.config import MODELS_DIR, REPORTS, TASK  # noqa: E402
from gtwin.models.mechanistic import GlucoseKF, MechParams  # noqa: E402
from gtwin.twin.engine import PatientRecord, TwinEngine  # noqa: E402

APP_TITLE = "Glucose Excursion Twin"  # working title - rename in one place
DEMO = ROOT / "data" / "demo"
INK, MUTED, TEAL, CRIMSON, PAPER, ZONE = "#1F2A30", "#66737B", "#0F6E74", "#B3261E", "#F5F7F6", "#F6E3DD"
WARMUP_STEPS = 48  # start the ward with 4 h of history per patient

st.set_page_config(page_title=APP_TITLE, page_icon="🩸", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, .stSelectbox, .stTabs, input {{
  font-family: 'Atkinson Hyperlegible', system-ui, sans-serif !important; }}
.block-container {{ padding-top: 3.2rem; max-width: 1280px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.banner {{ border-left: 4px solid {CRIMSON}; background: #FBEDEA; padding: .55rem .9rem; color: {INK};
  font-size: .92rem; margin-bottom: .8rem; }}
.pt-head {{ font-size: 1.55rem; font-weight: 700; color: {INK}; margin: 0; }}
.pt-sub {{ color: {MUTED}; font-size: .95rem; margin-top: .1rem; }}
.src-real {{ color: {TEAL}; font-weight: 700; }}
.src-syn {{ color: #7A5B00; font-weight: 700; }}
.now-num {{ font-size: 2.6rem; font-weight: 700; color: {INK}; font-variant-numeric: tabular-nums; line-height: 1; }}
.now-unit {{ color: {MUTED}; font-size: .95rem; }}
.risk {{ font-size: 2.0rem; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.flag {{ color: {CRIMSON}; font-size: .92rem; margin: .15rem 0; }}
.ok {{ color: {TEAL}; font-size: .92rem; margin: .15rem 0; }}
.side-note {{ color: {MUTED}; font-size: .85rem; }}
.q-row {{ display: flex; justify-content: space-between; padding: .35rem .2rem; border-bottom: 1px solid #E3E8E6;
  font-variant-numeric: tabular-nums; }}
.q-row.hot {{ color: {CRIMSON}; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_models():
    prm = MechParams(**{k: v for k, v in json.load(open(MODELS_DIR / "twin_params.json")).items()})
    serving = json.load(open(MODELS_DIR / "serving.json"))
    hybrid = joblib.load(MODELS_DIR / "hybrid.joblib")
    return prm, hybrid, serving


@st.cache_data
def load_cohort():
    recs = []

    def add(cgm_csv, prof_csv, meals_csv=None, id_col="patient_id"):
        if not Path(cgm_csv).exists():
            return
        cgm = pd.read_csv(cgm_csv, parse_dates=["timestamp"])
        prof = pd.read_csv(prof_csv)
        if id_col != "patient_id":
            cgm = cgm.rename(columns={id_col: "patient_id"})
            prof = prof.rename(columns={id_col: "patient_id"})
        meals = pd.read_csv(meals_csv, parse_dates=["timestamp"]) if meals_csv and Path(meals_csv).exists() else None
        for pid, d in cgm.groupby("patient_id", sort=False):
            p = prof[prof.patient_id == pid].iloc[0].to_dict()
            src = p.pop("source", "unknown")
            m = meals[meals.patient_id == pid] if meals is not None else None
            recs.append(dict(patient_id=str(pid), profile=p, timestamps=d["timestamp"].reset_index(drop=True),
                             glucose=d["glucose"].to_numpy(float), source=src, meals=m))

    add(DEMO / "synthetic_cgm.csv", DEMO / "synthetic_profiles.csv", DEMO / "synthetic_meals.csv")
    add(DEMO / "cgmacros_slice.csv", DEMO / "cgmacros_slice_profiles.csv", id_col="participant_id")
    return recs


def group_q(serving, group):
    q = serving["conformal_q_by_group"].get(group) or serving["conformal_q"]
    return {int(k): float(v) for k, v in q.items()}


def init_ward():
    prm, hybrid, serving = load_models()
    engines = {}
    for r in load_cohort():
        rec = PatientRecord(r["patient_id"], r["profile"], r["timestamps"], r["glucose"].copy(), r["source"], r["meals"])
        eng = TwinEngine(rec, prm, hybrid=hybrid, conformal_q=group_q(serving, r["profile"].get("group")),
                         alert_threshold=float(serving["alert_threshold"]))
        eng.advance(WARMUP_STEPS)
        engines[rec.patient_id] = eng
    st.session_state.engines = engines
    st.session_state.clock = WARMUP_STEPS
    st.session_state.reviewed = set()


def advance_all(steps: int):
    for eng in st.session_state.engines.values():
        eng.advance(steps)
    st.session_state.clock += steps


# ------------------------------------------------------------------ guard: artefacts present
if not (MODELS_DIR / "hybrid.joblib").exists():
    st.error("Model files are missing. Run `python scripts/train_final.py` first, then reload this page.")
    st.stop()
if "engines" not in st.session_state:
    with st.spinner("Starting the virtual ward (4 hours of history per patient)…"):
        init_ward()
prm, hybrid, serving = load_models()
engines: dict[str, TwinEngine] = st.session_state.engines


# ------------------------------------------------------------------ sidebar: review queue
def latest(eng):
    return eng.history_frame().iloc[-1] if len(eng.history_frame()) else None


with st.sidebar:
    st.markdown(f"### {APP_TITLE}")
    st.caption(f"Replay time +{st.session_state.clock * TASK.step_min // 60} h "
               f"{st.session_state.clock * TASK.step_min % 60:02d} min since monitoring started")
    c1, c2 = st.columns(2)
    if c1.button("Advance 5 min", width="stretch", type="primary"):
        advance_all(1)
    if c2.button("Advance 1 h", width="stretch"):
        advance_all(12)
    if st.button("Restart replay", width="stretch"):
        init_ward()
        st.rerun()

    st.markdown("#### Review queue")
    st.caption("Sorted by the twin's 2-hour excursion risk. Patients without a usable forecast are listed last.")
    queue = []
    for pid, eng in engines.items():
        h = latest(eng)
        p = h["p_event"] if h is not None and pd.notna(h["p_event"]) else -1.0
        if h is not None and h.get("in_excursion"):
            p = 2.0  # already above threshold: listed first
        queue.append((pid, p, eng))
    queue.sort(key=lambda x: -x[1])
    thr = float(serving["alert_threshold"])
    for pid, p, eng in queue:
        label = "no forecast" if p < 0 else (f"≥{TASK.threshold_mgdl:.0f} now" if p > 1 else f"{p:.0%}")
        hot = "hot" if p >= thr else ""
        kind = "synthetic" if "SYNTH" in eng.record.source.upper() else "real"
        st.markdown(f"<div class='q-row {hot}'><span>{pid} <span class='side-note'>({kind})</span></span>"
                    f"<span>{label}</span></div>", unsafe_allow_html=True)
    ids = [q[0] for q in queue]
    sel = st.selectbox("Open patient", ids, index=0)
    st.markdown(f"<p class='side-note'>Alert rule: risk ≥ {thr:.0%}, then silent for 60 min. "
                "Threshold chosen on calibration patients to keep false alerts to about 1 per patient-day.</p>",
                unsafe_allow_html=True)

st.markdown("<div class='banner'><b>Research prototype.</b> Not a medical device and not clinically validated. "
            "Forecasts are model scenarios for discussion, never treatment instructions.</div>", unsafe_allow_html=True)

eng = engines[sel]
rec = eng.record
prof = rec.profile
fc = eng.forecast()
is_syn = "SYNTH" in rec.source.upper()

# ------------------------------------------------------------------ header
h1, h2 = st.columns([3, 2])
with h1:
    st.markdown(f"<p class='pt-head'>Patient {rec.patient_id}</p>", unsafe_allow_html=True)
    sex = str(prof.get("gender", "")).capitalize()
    bits = [f"{prof.get('group', '')}", f"{int(prof['age'])} y" if pd.notna(prof.get("age")) else "", sex,
            f"HbA1c {prof['hba1c']}%" if pd.notna(prof.get("hba1c")) else ""]
    st.markdown(f"<p class='pt-sub'>{', '.join(b for b in bits if b)}</p>", unsafe_allow_html=True)
    cls = "src-syn" if is_syn else "src-real"
    st.markdown(f"<p class='pt-sub'>Data: <span class='{cls}'>{rec.source}</span></p>", unsafe_allow_html=True)

seen = eng.observed_so_far()
last_ok = seen.dropna()
with h2:
    a, b = st.columns(2)
    with a:
        if len(last_ok):
            lr = last_ok.iloc[-1]
            mins = int((seen["timestamp"].iloc[-1] - lr["timestamp"]).total_seconds() // 60)
            st.markdown(f"<div class='now-num'>{lr['glucose']:.0f}</div><div class='now-unit'>mg/dL, "
                        f"{'just now' if mins == 0 else f'{mins} min ago'}</div>", unsafe_allow_html=True)
    with b:
        if fc.in_excursion:
            st.markdown(f"<div class='risk' style='color:{CRIMSON}'>Above</div><div class='now-unit'>glucose is ≥ "
                        f"{TASK.threshold_mgdl:.0f} mg/dL now</div>", unsafe_allow_html=True)
        elif fc.p_event is None:
            st.markdown(f"<div class='risk' style='color:{MUTED}'>—</div><div class='now-unit'>no usable forecast</div>",
                        unsafe_allow_html=True)
        else:
            col = CRIMSON if fc.p_event >= thr else INK
            st.markdown(f"<div class='risk' style='color:{col}'>{fc.p_event:.0%}</div>"
                        f"<div class='now-unit'>risk of ≥{TASK.threshold_mgdl:.0f} mg/dL in next 2 h</div>",
                        unsafe_allow_html=True)


# ------------------------------------------------------------------ signature chart
def chart(eng: TwinEngine, fc, whatif=None):
    seen = eng.observed_so_far()
    now = seen["timestamp"].iloc[-1]
    win = seen[seen["timestamp"] >= now - pd.Timedelta(hours=8)]
    fut_t = [now + pd.Timedelta(minutes=TASK.step_min * k) for k in range(1, TASK.n_hor + 1)]
    fig = go.Figure()
    fig.add_hrect(y0=TASK.threshold_mgdl, y1=400, fillcolor=ZONE, opacity=0.6, line_width=0, layer="below")
    fig.add_hline(y=TASK.threshold_mgdl, line=dict(color=CRIMSON, width=1, dash="dot"))
    fig.add_annotation(x=win["timestamp"].iloc[0], y=TASK.threshold_mgdl, text=f"{TASK.threshold_mgdl:.0f} mg/dL",
                       showarrow=False, yshift=9, xanchor="left", font=dict(color=CRIMSON, size=12))
    fig.add_trace(go.Scatter(x=win["timestamp"], y=win["glucose"], mode="lines+markers", name="Measured",
                             line=dict(color=INK, width=2), marker=dict(size=3), connectgaps=False))
    if fc.usable and np.isfinite(fc.mean).all():
        fig.add_trace(go.Scatter(x=fut_t + fut_t[::-1], y=list(fc.hi) + list(fc.lo[::-1]), fill="toself",
                                 fillcolor="rgba(15,110,116,0.16)", line=dict(width=0), name="90% interval",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[now] + fut_t, y=[win["glucose"].dropna().iloc[-1]] + list(fc.mean),
                                 mode="lines", name="Twin forecast", line=dict(color=TEAL, width=3)))
        if whatif is not None:
            fig.add_trace(go.Scatter(x=[now] + fut_t, y=[fc.mean[0]] + list(whatif), mode="lines",
                                     name="Scenario", line=dict(color="#7A5B00", width=2, dash="dash")))
    if rec.meals is not None and len(rec.meals):
        m = rec.meals[(rec.meals.timestamp >= win["timestamp"].iloc[0]) & (rec.meals.timestamp <= now)]
        for _, r in m.iterrows():
            fig.add_annotation(x=r["timestamp"], y=60, text=f"{r['meal']} {int(r['carbs_g'])} g", showarrow=False,
                               font=dict(size=11, color=MUTED), textangle=-90, yanchor="bottom")
    fig.add_vline(x=now, line=dict(color=MUTED, width=1))
    fig.add_annotation(x=now, y=395, text="now", showarrow=False, xanchor="left", xshift=4, font=dict(color=MUTED))
    for a in eng.alerts:
        if a["time"] >= win["timestamp"].iloc[0]:
            fig.add_trace(go.Scatter(x=[a["time"]], y=[388], mode="markers", marker=dict(symbol="triangle-down",
                                     size=11, color=CRIMSON), name="Alert", showlegend=False,
                                     hovertext=f"Alert, risk {a['p_event']:.0%}"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Atkinson Hyperlegible, sans-serif", color=INK),
                      yaxis=dict(range=[40, 400], title="Glucose (mg/dL)", gridcolor="#EEF1F0"),
                      xaxis=dict(gridcolor="#EEF1F0"), legend=dict(orientation="h", y=-0.12))
    return fig


left, right = st.columns([3, 1])
with right:
    st.markdown("**Data quality**")
    if fc.quality:
        for q in fc.quality:
            st.markdown(f"<p class='flag'>{q}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='ok'>Complete, current readings</p>", unsafe_allow_html=True)
    if fc.usable:
        st.markdown("**Forecast**")
        rows = "".join(f"<div class='q-row'><span>+{h} min</span><span>{m:.0f} "
                       f"<span class='side-note'>({lo:.0f}–{hi:.0f})</span></span></div>"
                       for h, (m, lo, hi) in fc.points.items())
        st.markdown(rows, unsafe_allow_html=True)
        st.caption(f"Model: {fc.model}. Interval: 90% conformal, calibrated for the {prof.get('group', '')} group.")
    st.markdown("**Twin state**")
    m = eng.state_m
    if m is not None:
        slope = (seen["glucose"].ffill().iloc[-1] - seen["glucose"].ffill().iloc[-4]) / 15 if len(seen) > 4 else np.nan
        st.markdown(f"<div class='q-row'><span>Trend, last 15 min</span><span>{slope:+.1f} mg/dL/min</span></div>"
                    f"<div class='q-row'><span>Personal baseline</span><span>{m[2]:.0f} mg/dL</span></div>"
                    f"<div class='q-row'><span>Unexplained glucose inflow</span><span>{m[1]:+.2f} mg/dL/min</span></div>",
                    unsafe_allow_html=True)
        st.caption("Baseline and inflow are estimated model states, not laboratory measurements. Inflow bundles "
                   "meals, liver output and stress, which a CGM alone cannot separate.")

tab_live, tab_profile, tab_alerts, tab_whatif, tab_load, tab_model = st.tabs(
    ["Forecast", "Clinical profile", "Alerts and outcomes", "What-if", "Alert workload", "About the model"])

with tab_live:
    with left:
        st.plotly_chart(chart(eng, fc), width="stretch", config={"displayModeBar": False})
    hist = eng.history_frame()
    if len(hist):
        st.markdown("**Prediction history** (each row is one 5-minute update)")
        show = hist.tail(24).iloc[::-1].copy()
        show["time"] = pd.to_datetime(show["time"]).dt.strftime("%d %b %H:%M")
        show["p_event"] = show["p_event"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        st.dataframe(show[["time", "reading", "p_event", "pred_60", "pred_120", "quality"]].rename(columns={
            "reading": "reading (mg/dL)", "p_event": "2-h risk", "pred_60": "+60 min", "pred_120": "+120 min"}),
            hide_index=True, width="stretch")

with tab_profile:
    st.markdown("**Static record**")
    fields = {k: v for k, v in prof.items() if k not in ("patient_id",) and pd.notna(v)}
    st.table(pd.DataFrame({"field": list(fields.keys()), "value": [str(v) for v in fields.values()]}))
    if is_syn:
        st.caption("Every field above is synthetic. HbA1c is derived from simulated glucose with the ADAG formula.")
    else:
        st.caption("Real de-identified fields from CGMacros (age, sex, HbA1c, diabetes status). Dates are shifted by the "
                   "dataset creators. Diagnoses and medications are not part of this release.")
    if rec.meals is not None and len(rec.meals):
        st.markdown("**Logged meals so far**")
        now = seen["timestamp"].iloc[-1]
        st.dataframe(rec.meals[rec.meals.timestamp <= now].tail(10), hide_index=True)

with tab_alerts:
    out = eng.outcome_of_alerts()
    if out.empty:
        st.info("No alerts for this patient yet. Advance the replay to see alerts and their outcomes.")
    else:
        out["time"] = pd.to_datetime(out["time"]).dt.strftime("%d %b %H:%M")
        out["p_event"] = out["p_event"].map("{:.0%}".format)
        out["predicted_peak"] = out["predicted_peak"].round(0)
        st.dataframe(out[["time", "p_event", "predicted_peak", "outcome"]].rename(
            columns={"p_event": "risk at alert", "predicted_peak": "forecast peak (mg/dL)"}),
            hide_index=True, width="stretch")
        n_done = (out.outcome != "pending").sum()
        n_true = (out.outcome == "excursion followed").sum()
        if n_done:
            st.caption(f"{n_true} of {n_done} resolved alerts were followed by a reading ≥ {TASK.threshold_mgdl:.0f} "
                       "mg/dL within 2 h.")

with tab_whatif:
    st.markdown("Explore how the forecast could change if this patient ate now. This is a model scenario with "
                "assumed meal-response parameters, not a prediction of what will happen, and not advice.")
    carbs = st.slider("Carbohydrate eaten now (g)", 0, 120, 60, step=10)
    gain = {"Healthy": 1.1, "Prediabetes": 1.4, "T2D": 1.9}.get(prof.get("group"), 1.4)
    if fc.usable and eng.state_m is not None:
        mp = MechParams(**{**{k: v for k, v in vars(prm).items()}, "use_meals": True, "k_abs": 0.025, "c_meal": gain})
        kfm = GlucoseKF(mp)
        x = np.zeros(5)
        x[:3] = eng.state_m
        u = np.zeros(TASK.n_hor)
        u[0] = carbs
        base_mech = kfm.forecast_mean(x, TASK.n_hor, np.zeros(TASK.n_hor))
        meal_mech = kfm.forecast_mean(x, TASK.n_hor, u)
        scen = fc.mean + (meal_mech - base_mech)
        st.plotly_chart(chart(eng, fc, whatif=scen), width="stretch", config={"displayModeBar": False})
        st.caption(f"Assumptions: two-compartment gut absorption (rate 0.025/min) and a carbohydrate effect of {gain} "
                   f"for the {prof.get('group')} group, taken from the synthetic generator, not fitted to this patient. "
                   "Real meal responses vary widely between people.")
    else:
        st.warning("The scenario needs a usable forecast. Advance the replay until data quality is adequate.")

with tab_load:
    curve_p = REPORTS / "results" / "summary_alert_curve.csv"
    if curve_p.exists():
        cur = pd.read_csv(curve_p)
        cur = cur[cur.model == "Hybrid (Kalman + GB residual)"]
        st.markdown("How many alerts would a clinic team handle? Results come from cross-validated replay on 45 "
                    "CGMacros participants. Workload numbers are illustrative and use the assumptions you set below.")
        c1, c2, c3 = st.columns(3)
        n_pat = c1.number_input("Patients monitored", 10, 5000, 200, step=10)
        mins = c2.number_input("Minutes to review one alert", 1, 30, 3)
        thr_sel = c3.select_slider("Alert threshold", options=sorted(cur.threshold.round(2).unique()),
                                   value=min(cur.threshold.round(2).unique(), key=lambda t: abs(t - thr)))
        r = cur.loc[(cur.threshold.round(2) - thr_sel).abs().idxmin()]
        a, b, c, d = st.columns(4)
        a.metric("Alerts per day", f"{r.alerts_per_day * n_pat:,.0f}")
        b.metric("False alerts per day", f"{r.false_alerts_per_day * n_pat:,.0f}")
        c.metric("Excursions warned", f"{r.episode_recall:.0%}")
        d.metric("Review hours per day", f"{r.alerts_per_day * n_pat * mins / 60:,.1f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cur.alerts_per_day, y=cur.episode_recall, mode="lines+markers",
                                 line=dict(color=TEAL), text=cur.threshold, name="Hybrid"))
        fig.add_trace(go.Scatter(x=[r.alerts_per_day], y=[r.episode_recall], mode="markers",
                                 marker=dict(size=13, color=CRIMSON), name="Selected"))
        fig.update_layout(height=330, xaxis_title="Alerts per patient-day", yaxis_title="Share of excursions warned",
                          plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                          font=dict(family="Atkinson Hyperlegible, sans-serif", color=INK))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("An excursion counts as warned if an alert fired in the 2 hours before glucose first reached "
                   f"{TASK.threshold_mgdl:.0f} mg/dL. Assumes the same population as CGMacros, which may not hold.")
    else:
        st.info("Run `python scripts/summarize.py` to generate the workload curve.")

with tab_model:
    st.markdown(f"""
**What the twin is.** A state-space model of sensor glucose with three hidden states: current glucose, an
unexplained glucose inflow (meals, liver output, stress), and a slowly drifting personal baseline. A Kalman
filter updates these states with every new reading. A gradient-boosting model then corrects the twin's forecast
using recent glucose patterns and the static profile, and estimates the risk of reaching
{TASK.threshold_mgdl:.0f} mg/dL in the next 2 hours.

**What it was trained on.** CGMacros (PhysioNet, CC BY-NC-SA 4.0): 45 adults in California, of whom 14 have
type 2 diabetes. It has not been trained or validated on Indian patients.

**What it cannot do.** Without insulin, medication or meal timing in real time, the model cannot tell why glucose
rises. Its estimated states are modelling quantities, not measured physiology. It must not be used for diagnosis or
treatment.

**Event definition.** Any CGM reading ≥ {TASK.threshold_mgdl:.0f} mg/dL within 120 minutes, scored only when the
current reading is below that level and at least 80% of future readings exist. The threshold is a research setting,
not a treatment target.
""")
    imp = REPORTS / "results" / "feature_importance_hybrid_60min.csv"
    if imp.exists():
        fi = pd.read_csv(imp, index_col=0).head(10)
        st.markdown("**Inputs that matter most** (60-minute forecast, permutation importance on held-out patients)")
        fig = go.Figure(go.Bar(x=fi["mae_increase"][::-1], y=fi.index[::-1], orientation="h", marker_color=TEAL))
        fig.update_layout(height=320, xaxis_title="Increase in error when shuffled (mg/dL)", plot_bgcolor="white",
                          margin=dict(l=10, r=10, t=10, b=10), font=dict(family="Atkinson Hyperlegible, sans-serif"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
