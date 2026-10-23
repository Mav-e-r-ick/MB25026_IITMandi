"""Turn the Kalman twin into per-example forecast columns (used by the evaluation and the hybrid)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtwin.config import TASK
from gtwin.models.mechanistic import GlucoseKF, MechParams, run_filter

HORIZONS = TASK.forecast_points_min


def mech_columns(series_by_pid: dict[str, np.ndarray], examples: pd.DataFrame, params: MechParams | dict,
                 n_samples: int = 200, seed: int = 0) -> pd.DataFrame:
    """For every example row (participant_id, t_idx) compute the state-space forecast.

    `params` may be one MechParams for everyone or a dict pid -> MechParams (personalised).
    Returns a DataFrame aligned with `examples.index`.
    """
    rng = np.random.default_rng(seed)
    H = TASK.n_hor
    out = []
    for pid, ex in examples.groupby("participant_id", sort=False):
        prm = params[pid] if isinstance(params, dict) else params
        kf = GlucoseKF(prm, dt=TASK.step_min)
        means, covs = run_filter(kf, series_by_pid[pid])
        t = ex["t_idx"].to_numpy()
        mean, sd, pev = kf.forecast_batch(means[t], covs[t], H, n_samples=n_samples,
                                          threshold=TASK.threshold_mgdl, rng=rng)
        d = pd.DataFrame(index=ex.index)
        for h in HORIZONS:
            k = h // TASK.step_min - 1
            d[f"mech_{h}"] = mean[:, k]
            d[f"mech_sd_{h}"] = sd[:, k]
        d["mech_p_event"] = pev
        d["mech_R"] = means[t, 1]
        d["mech_B"] = means[t, 2]
        out.append(d)
    return pd.concat(out).loc[examples.index]
