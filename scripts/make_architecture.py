"""Architecture diagram -> docs/architecture.pdf and docs/architecture.png"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, TEAL, PAPER, LINE, SYN = "#1F2A30", "#66737B", "#0F6E74", "#FCFCFB", "#C9D2CE", "#7A5B00"
fig, ax = plt.subplots(figsize=(14, 8.2))
ax.set_xlim(0, 140); ax.set_ylim(0, 82); ax.axis("off"); fig.patch.set_facecolor(PAPER)


def box(x, y, w, h, title, body, edge=LINE, fill="white", tcol=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fill, ec=edge, lw=1.4))
    ax.text(x + 1.2, y + h - 2.2, title, fontsize=10.5, weight="bold", color=tcol, va="top")
    ax.text(x + 1.2, y + h - 5.4, body, fontsize=8.6, color=INK, va="top", linespacing=1.38)


def arrow(x1, y1, x2, y2, label=None, col=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=col, lw=1.3))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.0, label, fontsize=8, color=MUTED, ha="center")


ax.text(1, 80, "Glucose Excursion Twin: architecture", fontsize=15, weight="bold", color=INK, va="top")
ax.text(1, 76.2, "Offline training (top) is separate from the live twin engine and dashboard (bottom). "
        "Real data never ships to the public app; demo patients are synthetic or a small CC BY-NC-SA slice.",
        fontsize=9.2, color=MUTED, va="top")

# --- offline lane
ax.text(1, 70.5, "Offline: data and training", fontsize=10, color=TEAL, weight="bold")
box(1, 46, 24, 22, "Open data sources", "CGMacros (PhysioNet)\n45 adults, Dexcom G6\nCC BY-NC-SA 4.0\nShanghaiT2DM (Figshare)\n100 T2D, Libre H\nexternal test only")
box(29, 46, 25, 22, "Data engineering", "de-interpolate 1-min release\nto real 5-min readings\nrange / spike / gap checks\nregular grid, gaps kept\nas missing (never filled)\nprovenance per table")
box(58, 46, 25, 22, "Leakage-safe dataset", "features use data ≤ t only\nlabels: any reading ≥180\nin (t, t+120 min]\n≥80% future coverage\nscored only if G(t) < 180\nunit tests for leakage")
box(87, 46, 25, 22, "Models", "Kalman twin: identify p, τ,\nnoise by multi-step error\nHybrid: GB residual + event\nBaselines: persistence,\nridge, gradient boosting\nconformal + Platt on held-out")
box(116, 46, 23, 22, "Evaluation", "5-fold CV by participant\nbootstrap CIs, paired tests\nablations: inputs, personal-\nisation, missing, noise,\nintervals, alert workload\nexternal: ShanghaiT2DM")
for x in (25, 54, 83, 112):
    arrow(x + 0.4, 57, x + 3.6, 57)

# --- online lane
ax.text(1, 39.5, "Online: digital twin engine and clinician dashboard", fontsize=10, color=TEAL, weight="bold")
box(1, 8, 24, 29, "Virtual patient", "static record (EHR-like):\nage, sex, HbA1c, group,\ndiagnoses, medications\nCGM stream, replayed\n5 min at a time\nsynthetic patients S-01..06\nlabelled SYNTHETIC", edge=SYN)
box(29, 8, 25, 29, "Input validation", "duplicate timestamps\nimplausible values\nmissing reading\nstale data (>15 min)\ninsufficient history\nend of stream\npoor quality → no risk\nshown, explicit warning")
box(58, 8, 25, 29, "Twin state engine", "hidden state per patient:\nglucose, inflow, baseline\nKalman update each 5 min\nforecast 0–120 min\nMonte-Carlo P(any ≥180)\nhybrid residual correction\nconformal 90% interval\none engine per patient", edge=TEAL)
box(87, 8, 25, 29, "Alert logic", "alert if risk ≥ threshold\nthreshold chosen on\ncalibration patients\n(≤1 false alert/day)\n60-min refractory period\noutcome tracked: excursion\nfollowed / false alert")
box(116, 8, 23, 29, "Clinician dashboard", "review queue by risk\nforecast fan + 180 line\ndata-quality flags\ntwin states explained\nalerts and outcomes\nwhat-if (model scenario)\nworkload calculator")
for x in (25, 54, 83, 112):
    arrow(x + 0.4, 22.5, x + 3.6, 22.5)
arrow(99.5, 45.6, 70.5, 37.6, col=TEAL)
ax.text(101, 41.2, "trained artefacts:\ntwin_params.json, hybrid.joblib, serving.json", fontsize=8, color=TEAL, va="center")
ax.text(1, 3.5, "Research prototype: not a medical device, not clinically validated, not validated on Indian patients. "
        "Forecasts are scenarios for discussion, not treatment instructions.", fontsize=8.6, color=MUTED)
fig.savefig(ROOT / "docs" / "architecture.pdf", bbox_inches="tight", facecolor=PAPER)
fig.savefig(ROOT / "docs" / "architecture.png", dpi=150, bbox_inches="tight", facecolor=PAPER)
print("ok")
