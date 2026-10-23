"""Dataset loaders. Each returns a tidy long table: participant_id, timestamp, glucose (regular grid)
plus a per-participant profile table. Provenance is attached to every table via `.attrs`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gtwin.config import GML_CGMACROS, GML_SHANGHAI, PHYSIONET_CGMACROS
from gtwin.data.cgm import clean_values, deinterpolate_1min, to_regular_grid

GROUP_MAP = {"No diabetes": "Healthy", "Prediabetes": "Prediabetes", "T2D": "T2D"}


def _read_meta(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, encoding="utf-8-sig")
    m = m.loc[:, ~m.columns.str.startswith("Unnamed")]
    return m


def load_cgmacros(base: Path = GML_CGMACROS, step_min: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CGMacros Dexcom G6 stream (de-interpolated to real 5-minute readings).

    Returns (cgm, profile, qc).
    """
    meta = _read_meta(base / "CGMacros_Dexcom-metadata.csv")
    rows, qc = [], []
    for f in sorted((base / "CGMacros_Dexcom-extracted-glucose-files").glob("*.csv")):
        pid = f.stem
        raw = pd.read_csv(f, parse_dates=["timestamp"]).rename(columns={"glucose_value_mg_dl": "glucose"})
        real = deinterpolate_1min(raw)
        grid = to_regular_grid(real, step_min)
        grid["glucose"], flags = clean_values(grid["glucose"])
        grid["participant_id"] = pid
        rows.append(grid)
        qc.append({
            "participant_id": pid,
            "rows_1min": len(raw),
            "real_readings": int(real["observed"].sum()) if len(real) else 0,
            "filled_on_phase_set_missing": int((~real["observed"]).sum()) if len(real) else 0,
            "grid_slots": len(grid),
            "grid_missing_pct": round(100 * grid["glucose"].isna().mean(), 2),
            "days": round(len(grid) * step_min / 1440, 2),
            **flags,
        })
    cgm = pd.concat(rows, ignore_index=True)[["participant_id", "timestamp", "glucose"]]
    prof = meta.rename(columns={"person_id": "participant_id", "hba1c_%": "hba1c"})
    prof["group"] = prof["diabetes_type"].map(GROUP_MAP).fillna(prof["diabetes_type"])
    prof = prof[["participant_id", "group", "age", "gender", "race_ethnicity", "hba1c"]]
    prof = _attach_physionet_bio(prof)
    cgm.attrs["source"] = "CGMacros v1.0.0 (PhysioNet, CC BY-NC-SA 4.0) via Glucose-ML harmonised files"
    return cgm, prof, pd.DataFrame(qc)


def load_shanghai_t2dm(base: Path = GML_SHANGHAI, step_min: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ShanghaiT2DM (FreeStyle Libre H, 15-min). External test only - never redistributed."""
    meta = _read_meta(base / "ShanghaiT2DM-metadata.csv")
    rows = []
    for f in sorted((base / "ShanghaiT2DM-extracted-glucose-files").glob("*.csv")):
        raw = pd.read_csv(f, parse_dates=["timestamp"]).rename(columns={"glucose_value_mg_dl": "glucose"})
        raw["glucose"] = raw["glucose"].round(1)
        grid = to_regular_grid(raw, step_min)
        grid["glucose"], _ = clean_values(grid["glucose"])
        grid["participant_id"] = "SH-" + f.stem
        rows.append(grid)
    cgm = pd.concat(rows, ignore_index=True)[["participant_id", "timestamp", "glucose"]]
    prof = meta.rename(columns={"person_id": "participant_id", "hba1c_%": "hba1c"})
    prof["participant_id"] = "SH-" + prof["participant_id"].astype(str)
    prof["group"] = "T2D"
    prof = prof[["participant_id", "group", "age", "gender", "hba1c"]]
    cgm.attrs["source"] = "ShanghaiT2DM (Figshare 10.6084/m9.figshare.20444397; no licence attached)"
    return cgm, prof


# ---------------------------------------------------------------- optional PhysioNet extras
def _attach_physionet_bio(prof: pd.DataFrame) -> pd.DataFrame:
    """If the original bio.csv is present, add BMI, fasting glucose/insulin, HOMA-IR, lipids."""
    path = next(iter(PHYSIONET_CGMACROS.rglob("bio.csv")), None) if PHYSIONET_CGMACROS.exists() else None
    if path is None:
        prof.attrs["bio"] = "absent"
        return prof
    b = pd.read_csv(path)
    b.columns = [c.strip() for c in b.columns]
    sid = next(c for c in b.columns if c.lower() in ("subject", "participant", "id", "sub"))
    b["participant_id"] = "CGMacros-" + b[sid].astype(int).astype(str).str.zfill(3)

    def num(col):
        return pd.to_numeric(b[col].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce") if col in b else np.nan

    out = pd.DataFrame({"participant_id": b["participant_id"]})
    w_kg, h_m = num("Body weight") * 0.453592, num("Height") * 0.0254
    out["bmi"] = w_kg / h_m**2
    out["fasting_glucose"] = num("Fasting GLU - PDL (Lab)")
    out["fasting_insulin"] = num("Insulin")
    out["homa_ir"] = out["fasting_glucose"] * out["fasting_insulin"] / 405  # proxy, single time point
    out["triglycerides"] = num("Triglycerides")
    out["hdl"] = num("HDL")
    out["ldl"] = num("LDL (Cal)")
    prof = prof.merge(out, on="participant_id", how="left")
    prof.attrs["bio"] = str(path)
    return prof


def load_cgmacros_events(step_min: int = 5) -> pd.DataFrame | None:
    """Meals (carbs etc.) and Fitbit HR/METs from the original PhysioNet per-participant CSVs.

    Returns long table on the 5-minute grid: participant_id, timestamp, carbs, protein, fat, fiber,
    calories, meal (0/1), hr, mets. None if the PhysioNet release has not been downloaded.
    """
    if not PHYSIONET_CGMACROS.exists():
        return None
    files = sorted(PHYSIONET_CGMACROS.rglob("CGMacros-0*.csv"))
    if not files:
        return None
    rows = []
    for f in files:
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        d["timestamp"] = pd.to_datetime(d["Timestamp"])
        d["slot_ts"] = d["timestamp"].dt.floor(f"{step_min}min")
        g = pd.DataFrame({"timestamp": d["slot_ts"]})
        for src, dst in [("Carbs", "carbs"), ("Protein", "protein"), ("Fat", "fat"), ("Fiber", "fiber"), ("Calories", "calories")]:
            g[dst] = pd.to_numeric(d.get(src), errors="coerce")
        g["meal"] = d.get("Meal Type").notna().astype(int) if "Meal Type" in d else 0
        g["hr"] = pd.to_numeric(d.get("HR"), errors="coerce") if "HR" in d else np.nan
        g["mets"] = pd.to_numeric(d.get("METs"), errors="coerce") if "METs" in d else np.nan
        agg = g.groupby("timestamp").agg(
            carbs=("carbs", "sum"), protein=("protein", "sum"), fat=("fat", "sum"), fiber=("fiber", "sum"),
            calories=("calories", "sum"), meal=("meal", "max"), hr=("hr", "mean"), mets=("mets", "mean"),
        ).reset_index()
        agg["participant_id"] = f.stem
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)
