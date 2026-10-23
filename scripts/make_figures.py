"""Report figures from reports/results/summary_*.csv -> reports/figures/*.png"""
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "reports" / "results", ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID, SURF = "#1F2A30", "#66737B", "#E6EAE8", "#FCFCFB"
COL = {  # validated categorical order (dataviz validator: all checks pass, light surface)
    "Hybrid (Kalman + GB residual)": "#2a78d6", "Gradient boosting": "#eb6834",
    "Mechanistic (Kalman)": "#1baf7a", "Ridge AR": "#eda100", "Persistence": "#8a8986"}
SHORT = {"Hybrid (Kalman + GB residual)": "Hybrid twin", "Gradient boosting": "Gradient boosting",
         "Mechanistic (Kalman)": "Kalman twin", "Ridge AR": "Ridge AR", "Persistence": "Persistence"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8})


def fig_mae():
    m = pd.read_csv(RES / "summary_main.csv").set_index("model")
    hz = [15, 30, 60, 90, 120]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ends = []
    for name, c in COL.items():
        y = [m.loc[name, f"MAE_{h}"] for h in hz]
        ax.plot(hz, y, color=c, lw=2, marker="o", ms=5, markeredgecolor=SURF, markeredgewidth=1.5)
        ends.append((y[-1], name))
    ends.sort()
    last = -1e9
    for y, name in ends:  # direct labels, nudged apart
        yy = max(y, last + 0.9)
        ax.text(123, yy, SHORT[name], color=INK, va="center", fontsize=10)
        last = yy
    ax.set_xlim(10, 150)
    ax.set_xticks(hz)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Mean absolute error (mg/dL)")
    ax.set_title("Forecast error grows with horizon; learned models beat persistence after 30 min",
                 loc="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "45 CGMacros participants, 5-fold participant-level cross-validation.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "mae_by_horizon.png", dpi=160)


def fig_auprc():
    m = pd.read_csv(RES / "summary_main.csv")
    m = m[m.model.isin(COL)].copy()
    m["ci"] = m["AUPRC_CI"].map(ast.literal_eval)
    m = m.sort_values("AUPRC")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, r in enumerate(m.itertuples()):
        lo, hi = r.ci
        ax.plot([lo, hi], [i, i], color=COL[r.model], lw=2, solid_capstyle="round")
        ax.plot(r.AUPRC, i, "o", color=COL[r.model], ms=9, markeredgecolor=SURF, markeredgewidth=2)
        ax.text(hi + 0.01, i, f"{r.AUPRC:.2f}", va="center", color=INK, fontsize=10)
    ax.axvline(m["prevalence"].iloc[0], color=MUTED, lw=1, ls=":")
    ax.text(m["prevalence"].iloc[0] + 0.005, -0.6, "chance level (event rate 16%)", color=MUTED, fontsize=9)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([SHORT[x] for x in m.model], color=INK)
    ax.set_xlim(0.1, 0.62)
    ax.set_ylim(-0.9, len(m) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Area under precision-recall curve (higher is better)")
    ax.set_title("Predicting a ≥180 mg/dL excursion in the next 2 hours", loc="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "Lines are 95% confidence intervals from resampling participants.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "auprc.png", dpi=160)


def fig_noise():
    r = pd.read_csv(RES / "summary_robustness.csv")
    clean = r[r.scenario == "clean"].set_index("model")["MAE_15"]
    noisy = r[r.scenario == "noise10"].set_index("model")["MAE_15"]
    rows = [("Persistence", "Persistence"), ("Gradient boosting", "Gradient boosting"),
            ("Hybrid twin", "Hybrid (Kalman + GB residual)"), ("Kalman twin", "Mechanistic (Kalman)"),
            ("Kalman twin, sensor-noise\nparameter updated", "Mechanistic (noise model updated)")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (lab, key) in enumerate(rows):
        hl = "noise model" in key
        ax.barh(i, noisy[key], color="#2a78d6" if hl else "#B9C3C8", height=0.6)
        ax.text(noisy[key] + 0.2, i, f"{noisy[key]:.1f}", va="center", color=INK, fontsize=10)
    base = clean["Gradient boosting"]
    ax.axvline(base, color=MUTED, ls=":", lw=1)
    ax.text(base + 0.15, -0.55, f"error on clean data, {base:.1f}", color=MUTED, fontsize=9)
    ax.set_ylim(-0.8, len(rows) - 0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("15-minute forecast error with a noisy sensor (MAE, mg/dL)")
    fig.suptitle("A noise-aware twin recovers accuracy without retraining", x=0.01, ha="left", fontsize=12,
                 color=INK)
    fig.text(0.01, 0.01, "Test inputs corrupted with Gaussian noise (SD 10 mg/dL); labels from clean readings.",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "noise_robustness.png", dpi=160)


def fig_alerts():
    c = pd.read_csv(RES / "summary_alert_curve.csv")
    a = pd.read_csv(RES / "summary_alerts.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name in ["Hybrid (Kalman + GB residual)", "Gradient boosting", "Mechanistic (Kalman)"]:
        d = c[c.model == name].sort_values("alerts_per_day")
        ax.plot(d.alerts_per_day, d.episode_recall, color=COL[name], lw=2, label=SHORT[name])
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)
    op = a[(a.model == "Hybrid (Kalman + GB residual)") & a.policy.str.startswith("threshold")].iloc[0]
    ax.plot(op.alerts_per_day, op.episode_recall, "o", color="#B3261E", ms=10, markeredgecolor=SURF, markeredgewidth=2)
    ax.annotate(f"Operating point: {op.alerts_per_day:.1f} alerts/day,\n{op.false_alerts_per_day:.1f} false; "
                f"warns {op.episode_recall:.0%} of excursions,\nmedian {op.median_lead_min:.0f} min ahead",
                (op.alerts_per_day, op.episode_recall), xytext=(op.alerts_per_day + 1.5, op.episode_recall - 0.22),
                fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Alerts per patient-day")
    ax.set_ylabel("Share of excursions warned in advance")
    ax.set_title("Every extra warning costs clinician attention", loc="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "Alert = risk above threshold, then 60 min silence. Cross-validated, 45 participants.",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "alert_tradeoff.png", dpi=160)


if __name__ == "__main__":
    fig_mae()
    fig_auprc()
    fig_noise()
    fig_alerts()
    print("figures in", FIG)
