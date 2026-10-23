"""Grey-box glucose state-space model with Kalman filtering.

Physiology (simplified, see docs/methodology.md):
    dG/dt = -p (G - B) + R(t)          G: sensor glucose (mg/dL), B: personal basal level
    dR/dt = -R/tau + noise              R: latent net glucose appearance (mg/dL/min), driven by
                                         unobserved meals, hepatic output, stress ... decays with tau
    dB/dt = noise (very slow)           personal basal drift
    y = G + v                           CGM reading with sensor noise v

Optionally, when meal records exist, carbohydrate grams enter through a two-compartment gut model
(Q1 -> Q2 -> R) with absorption rate k_abs and an effect gain c (mg/dL per g, absorbed into R).

`p` is an *effective* return-to-baseline rate. It bundles insulin action, insulin-independent uptake
and hepatic suppression; it is NOT a measured insulin sensitivity. With CGM alone, insulin action is
not identifiable, so we do not claim to estimate it.

The discretised model is LINEAR in the state, so the standard Kalman filter is the exact Bayesian
filter for Gaussian noise; an Unscented KF would add cost without benefit here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

LN2PI = np.log(2 * np.pi)


@dataclass
class MechParams:
    p: float = 0.02          # 1/min   effective clearance towards basal
    tau: float = 45.0        # min     decay time of latent appearance
    q_g: float = 0.5         # process noise on G   (mg/dL)^2 per min
    q_r: float = 0.01        # process noise on R   (mg/dL/min)^2 per min
    q_b: float = 0.05        # basal drift          (mg/dL)^2 per min
    r: float = 9.0           # sensor noise variance (mg/dL)^2
    k_abs: float = 0.03      # 1/min   gut absorption (meal mode only)
    c_meal: float = 1.0      # mg/dL per g carbohydrate (meal mode only)
    use_meals: bool = False

    # -------------------------------------------------- vector <-> params for optimisation
    def to_vec(self) -> np.ndarray:
        v = [self.p, self.tau, self.q_g, self.q_r, self.q_b, self.r]
        if self.use_meals:
            v += [self.k_abs, self.c_meal]
        return np.log(np.asarray(v))

    def from_vec(self, v: np.ndarray) -> "MechParams":
        e = np.exp(v)
        kw = dict(p=e[0], tau=e[1], q_g=e[2], q_r=e[3], q_b=e[4], r=e[5])
        if self.use_meals:
            kw.update(k_abs=e[6], c_meal=e[7])
        return replace(self, **kw)


class GlucoseKF:
    """Kalman filter for the model above on a regular grid with step `dt` minutes."""

    def __init__(self, params: MechParams, dt: float = 5.0):
        self.prm, self.dt = params, dt
        self._build()

    def _build(self):
        p, dt, prm = self.prm, self.dt, self.prm
        a = np.exp(-dt / p.tau)
        if not prm.use_meals:
            self.A = np.array([[1 - dt * p.p, dt, dt * p.p], [0, a, 0], [0, 0, 1.0]])
            self.Bu = np.zeros(3)
            self.Q = np.diag([p.q_g, p.q_r, p.q_b]) * dt
        else:  # state [G, R, B, Q1, Q2]
            k = p.k_abs
            ek = np.exp(-k * dt)
            self.A = np.array([
                [1 - dt * p.p, dt, dt * p.p, 0, 0],
                [0, a, 0, 0, dt * k * p.c_meal / 10.0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, ek, 0],
                [0, 0, 0, 1 - ek, ek],
            ])
            self.Bu = np.array([0, 0, 0, 1.0, 0])  # carbs (g) enter Q1
            self.Q = np.diag([p.q_g, p.q_r, p.q_b, 1e-6, 1e-6]) * dt
        self.nx = self.A.shape[0]
        self.H = np.zeros(self.nx)
        self.H[0] = 1.0

    def init_state(self, y0: float, basal: float | None = None):
        m = np.zeros(self.nx)
        m[0] = y0
        m[2] = basal if basal is not None else y0
        P = np.diag([self.prm.r, 0.05, 400.0] + [1.0] * (self.nx - 3))
        return m, P

    def step(self, m, P, y: float, u: float = 0.0):
        """One predict+update. Returns new (m, P, innovation, innovation variance) - y may be NaN."""
        m = self.A @ m + self.Bu * u
        P = self.A @ P @ self.A.T + self.Q
        if np.isnan(y):
            return m, P, np.nan, np.nan
        S = P[0, 0] + self.prm.r
        K = P[:, 0] / S
        e = y - m[0]
        m = m + K * e
        P = P - np.outer(K, P[0, :])
        return m, P, e, S

    def filter(self, y: np.ndarray, u: np.ndarray | None = None):
        """Run over a whole series. Returns filtered means (n, nx), covariances (n, nx, nx), loglik."""
        n = len(y)
        u = np.zeros(n) if u is None else np.nan_to_num(u)
        first = int(np.argmax(~np.isnan(y)))
        means = np.full((n, self.nx), np.nan)
        covs = np.full((n, self.nx, self.nx), np.nan)
        m, P = self.init_state(y[first])
        means[first], covs[first] = m, P
        ll, burn = 0.0, first + 12  # ignore first hour while basal is learned
        for t in range(first + 1, n):
            m, P, e, S = self.step(m, P, y[t], u[t])
            if not np.isnan(e) and t > burn:
                ll += -0.5 * (LN2PI + np.log(S) + e * e / S)
            means[t], covs[t] = m, P
        return means, covs, ll

    # ------------------------------------------------------------ forecasting
    def forecast_mean(self, m: np.ndarray, H: int, u_future: np.ndarray | None = None) -> np.ndarray:
        """Mean sensor glucose for steps 1..H (assumes no unannounced meals)."""
        out = np.empty(H)
        x = m.copy()
        for k in range(H):
            x = self.A @ x + (self.Bu * u_future[k] if u_future is not None else 0)
            out[k] = x[0]
        return out

    def forecast_batch(self, M: np.ndarray, Ps: np.ndarray, H: int, n_samples: int = 200,
                       threshold: float = 180.0, rng: np.random.Generator | None = None):
        """Vectorised forecast for many prediction points.

        M: (N, nx) filtered means, Ps: (N, nx, nx). Returns
          mean (N, H), sd (N, H) of future *readings*, and P(max reading over horizon >= threshold),
        the latter by Monte-Carlo over whole trajectories so that temporal dependence is respected.
        """
        rng = rng or np.random.default_rng(0)
        N = len(M)
        A, Q, r = self.A, self.Q, self.prm.r
        mean = np.empty((N, H))
        var = np.empty((N, H))
        x, P = M.copy(), Ps.copy()
        for k in range(H):
            x = x @ A.T
            P = A @ P @ A.T + Q
            mean[:, k] = x[:, 0]
            var[:, k] = P[:, 0, 0] + r
        # Monte Carlo trajectories
        jitter = 1e-9 * np.eye(self.nx)
        L = np.linalg.cholesky(Ps + jitter)
        z = rng.standard_normal((N, n_samples, self.nx))
        X = M[:, None, :] + np.einsum("nij,nsj->nsi", L, z)
        LQ = np.linalg.cholesky(Q + jitter)
        mx = np.full((N, n_samples), -np.inf)
        for k in range(H):
            X = X @ A.T + rng.standard_normal((N, n_samples, self.nx)) @ LQ.T
            Y = X[..., 0] + np.sqrt(r) * rng.standard_normal((N, n_samples))
            mx = np.maximum(mx, Y)
        p_event = (mx >= threshold).mean(axis=1)
        return mean, np.sqrt(var), p_event


# ----------------------------------------------------------------- fast filtering (optional numba)
try:  # numba is used only to speed up training; the app runs without it
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]


@njit(cache=True)
def _kf_run(y, u, A, Bu, Q, r, P0, first):
    n, nx = y.shape[0], A.shape[0]
    means = np.full((n, nx), np.nan)
    covs = np.full((n, nx, nx), np.nan)
    m = np.zeros(nx)
    m[0] = y[first]
    m[2] = y[first]
    P = P0.copy()
    means[first] = m
    covs[first] = P
    for t in range(first + 1, n):
        m = A @ m + Bu * u[t]
        P = A @ P @ A.T + Q
        if not np.isnan(y[t]):
            S = P[0, 0] + r
            K = P[:, 0] / S
            e = y[t] - m[0]
            m = m + K * e
            P = P - np.outer(K, P[0, :])
        means[t] = m
        covs[t] = P
    return means, covs


def run_filter(kf: GlucoseKF, y: np.ndarray, u: np.ndarray | None = None):
    """Fast filtered means/covariances for a whole series (NaN = missing reading)."""
    y = np.asarray(y, float)
    u = np.zeros(len(y)) if u is None else np.nan_to_num(np.asarray(u, float))
    first = int(np.argmax(~np.isnan(y)))
    _, P0 = kf.init_state(y[first])
    return _kf_run(y, u, kf.A, kf.Bu.astype(float), kf.Q, float(kf.prm.r), P0, first)


FIT_HORIZONS_MIN = (15, 30, 60, 90, 120)


def multistep_errors(kf: GlucoseKF, y: np.ndarray, u: np.ndarray | None = None, stride_min: int = 15,
                     horizons_min: tuple[int, ...] = FIT_HORIZONS_MIN, burn_min: int = 180):
    """Forecast errors (n_points, n_horizons) of the filter's mean forecast, plus predicted variances."""
    dt = kf.dt
    horizons = tuple(int(h // dt) for h in horizons_min)
    stride, burn = max(1, int(stride_min // dt)), int(burn_min // dt)
    means, covs = run_filter(kf, y, u)
    n = len(y)
    t_idx = np.arange(burn, n - max(horizons), stride)
    t_idx = t_idx[~np.isnan(y[t_idx])]
    if len(t_idx) == 0:
        return np.empty((0, len(horizons))), np.empty((0, len(horizons)))
    M, P = means[t_idx], covs[t_idx]
    errs, vars_ = [], []
    Ak = np.eye(kf.nx)
    Pk = P.copy()
    k_prev = 0
    for h in horizons:
        for _ in range(h - k_prev):
            Ak = kf.A @ Ak
            Pk = kf.A @ Pk @ kf.A.T + kf.Q
        k_prev = h
        pred = (M @ Ak.T)[:, 0]
        errs.append(y[t_idx + h] - pred)
        vars_.append(Pk[:, 0, 0] + kf.prm.r)
    return np.column_stack(errs), np.column_stack(vars_)


def fit_forecast_params(series: list[np.ndarray], init: MechParams | None = None, prior: MechParams | None = None,
                        prior_sd: float = 0.5, free: tuple[int, ...] = (0, 1, 2, 3, 4), maxiter: int = 400,
                        inputs: list[np.ndarray] | None = None, dt: float = 5.0) -> MechParams:
    """Identify parameters by minimising multi-step (15-120 min) forecast MSE - the quantity the twin is
    used for - instead of one-step likelihood, which favours trend-tracking and leaves the physiological
    decay (tau) unidentified. Sensor variance r is fixed (gain depends only on noise *ratios*) and the
    overall noise scale is then calibrated so predicted variances match observed squared errors.
    Optional `prior` gives MAP shrinkage (used for per-patient personalisation).
    """
    init = init or MechParams(r=1.0)
    v0 = init.to_vec()
    idx = np.asarray(free)
    pv = prior.to_vec() if prior is not None else None

    def loss(z):
        v = v0.copy()
        v[idx] = z
        prm = init.from_vec(v)
        if not (0.002 < prm.p < 0.2 and 5 < prm.tau < 600):
            return 1e12
        kf = GlucoseKF(prm, dt)
        e = [multistep_errors(kf, y, None if inputs is None else inputs[i])[0] for i, y in enumerate(series)]
        e = np.concatenate([x for x in e if len(x)])
        val = np.nanmean(e ** 2)
        if pv is not None:
            # Gaussian prior on log-parameters; overlapping 2-h windows are correlated, so the
            # effective number of independent errors is ~ points / 8
            n_eff = max(len(e) / 8.0, 1.0)
            val *= 1 + 0.5 * np.sum(((z - pv[idx]) / prior_sd) ** 2) / n_eff
        return val

    res = minimize(loss, v0[idx], method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3})
    v = v0.copy()
    v[idx] = res.x
    prm = init.from_vec(v)
    return calibrate_noise_scale(prm, series, inputs, dt)


def calibrate_noise_scale(prm: MechParams, series: list[np.ndarray], inputs=None, dt: float = 5.0) -> MechParams:
    """Scale all noise variances by s so that mean(err^2 / predicted var) = 1 across horizons.
    Kalman gains are unchanged (ratios fixed), so mean forecasts are unaffected."""
    kf = GlucoseKF(prm, dt)
    E, V = [], []
    for i, y in enumerate(series):
        e, v = multistep_errors(kf, y, None if inputs is None else inputs[i])
        E.append(e)
        V.append(v)
    E, V = np.concatenate(E), np.concatenate(V)
    s = float(np.nanmean(E ** 2 / V))
    return replace(prm, q_g=prm.q_g * s, q_r=prm.q_r * s, q_b=prm.q_b * s, r=prm.r * s)


# ----------------------------------------------------------------- system identification
def fit_params(series: list[np.ndarray], init: MechParams | None = None, inputs: list[np.ndarray] | None = None,
               prior: MechParams | None = None, prior_sd: float = 0.5, free: tuple[int, ...] | None = None,
               maxiter: int = 300) -> MechParams:
    """Maximum-likelihood (or MAP with a log-normal prior) estimate of model parameters from
    one or more glucose series, using the Kalman-filter innovation likelihood.

    `free` selects which log-parameters are estimated (others held at `init`). Personalisation uses
    free=(0, 1) (p, tau) with the population fit as prior - shrinkage towards the population.
    """
    init = init or MechParams()
    v0 = init.to_vec()
    idx = np.arange(len(v0)) if free is None else np.asarray(free)
    pv = prior.to_vec() if prior is not None else None

    def nll(z):
        v = v0.copy()
        v[idx] = z
        prm = init.from_vec(v)
        if not (0.001 < prm.p < 0.3 and 5 < prm.tau < 400):
            return 1e12
        kf = GlucoseKF(prm)
        tot = 0.0
        for i, y in enumerate(series):
            tot -= kf.filter(y, None if inputs is None else inputs[i])[2]
        if pv is not None:
            tot += 0.5 * np.sum(((z - pv[idx]) / prior_sd) ** 2)
        return tot

    res = minimize(nll, v0[idx], method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-2})
    v = v0.copy()
    v[idx] = res.x
    return init.from_vec(v)
