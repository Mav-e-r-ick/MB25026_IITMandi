"""Loading and cleaning CGM streams.

Key finding (documented in reports/data_quality.md): the CGMacros release stores Dexcom G6 values on a
1-minute grid that the dataset creators filled by **linear interpolation** between the real 5-minute
sensor readings. Using the 1-minute rows directly would leak up to 4 minutes of future information
into every feature (an interpolated value at minute t+1 is computed from the reading at t+5).

`deinterpolate_1min` recovers the real readings:
  * real Dexcom readings are integers in mg/dL and fall on a fixed 5-minute phase;
  * for every contiguous block we pick the 5-minute phase whose values are most often integers;
  * on that phase, integer values are kept as observed readings; non-integer values are linear fills
    across a missing reading and are set to NaN (treated as missing, never as data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GLU_MIN, GLU_MAX = 40.0, 400.0  # Dexcom G6 reportable range


def _is_int(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    return np.abs(x - np.round(x)) < tol


def deinterpolate_1min(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """Return real sensor readings from a linearly interpolated 1-minute series.

    Input columns: timestamp, glucose. Output: timestamp, glucose, observed (bool) on the recovered
    5-minute grid of each block (NaN where the original reading was missing).
    """
    d = df[["timestamp", "glucose"]].dropna().sort_values("timestamp").drop_duplicates("timestamp")
    if d.empty:
        return pd.DataFrame(columns=["timestamp", "glucose", "observed"])
    # contiguous 1-minute blocks
    new_block = d["timestamp"].diff().dt.total_seconds().ne(60).cumsum()
    out = []
    for _, b in d.groupby(new_block):
        b = b.reset_index(drop=True)
        if len(b) < period:
            continue
        vals = b["glucose"].to_numpy(float)
        scores = [_is_int(vals[p::period]).mean() for p in range(period)]
        p = int(np.argmax(scores))
        sub = b.iloc[p::period].copy()
        sub["observed"] = _is_int(sub["glucose"].to_numpy(float))
        sub.loc[~sub["observed"], "glucose"] = np.nan
        out.append(sub)
    if not out:
        return pd.DataFrame(columns=["timestamp", "glucose", "observed"])
    return pd.concat(out, ignore_index=True)


def to_regular_grid(df: pd.DataFrame, step_min: int) -> pd.DataFrame:
    """Snap readings to a regular grid (nearest slot, no interpolation). Missing slots stay NaN."""
    d = df.dropna(subset=["glucose"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["timestamp", "glucose"])
    t0 = d["timestamp"].min().floor(f"{step_min}min")
    slot = ((d["timestamp"] - t0).dt.total_seconds() / 60 / step_min).round().astype(int)
    d["slot"] = slot
    d = d.groupby("slot", as_index=False)["glucose"].mean()  # duplicates in a slot -> mean
    full = pd.DataFrame({"slot": np.arange(d["slot"].max() + 1)})
    full = full.merge(d, on="slot", how="left")
    full["timestamp"] = t0 + pd.to_timedelta(full["slot"] * step_min, unit="min")
    return full[["timestamp", "glucose"]]


def clean_values(g: pd.Series) -> tuple[pd.Series, dict]:
    """Flag physiologically implausible values and isolated spikes. Returns cleaned series + counts."""
    g = g.astype(float).copy()
    out_of_range = (g < GLU_MIN) | (g > GLU_MAX)
    # isolated spike: jump > 40 mg/dL per 5 min that immediately reverses (sensor artefact)
    d1 = g.diff()
    d2 = g.diff(-1)
    spike = (d1.abs() > 40) & (d2.abs() > 40) & (np.sign(d1) == np.sign(d2))
    flags = {"out_of_range": int(out_of_range.sum()), "isolated_spikes": int(spike.sum())}
    g[out_of_range | spike] = np.nan
    return g, flags
