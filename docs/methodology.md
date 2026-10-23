# Methodology

## 1. The question

Can a physiology-guided, uncertainty-aware, personalisable Digital Twin forecast 2-hour glucose excursions
better than standard statistical and machine-learning baselines, especially when data are missing or noisy?
We treated this as a hypothesis to test, not a result to demonstrate.

## 2. Physiology the model has to respect

Glucose measured by a CGM is interstitial glucose. It follows blood glucose with a lag of roughly 5–10 minutes.
Over hours, three processes dominate:

1. **Appearance.** After a meal, carbohydrate is digested and absorbed over roughly 1–3 hours. The liver also
   releases glucose (hepatic output), which insulin normally suppresses.
2. **Disposal.** Insulin drives uptake into muscle and fat and suppresses hepatic output. Some uptake is
   insulin-independent (brain, and muscle during exercise).
3. **Return to baseline.** In people without diabetes, glucose returns to a stable fasting level within about
   2 hours of a meal. In type 2 diabetes, insulin resistance and impaired insulin secretion make excursions taller
   and slower to resolve, and the fasting level is higher.

The Bergman minimal model (Bergman et al., *Am J Physiol* 1979) describes this with glucose, remote insulin action
and plasma insulin. The Dalla Man meal model (Dalla Man et al., *IEEE TBME* 2007) adds gut absorption. Both need
insulin measurements, which we do not have.

**What a CGM alone cannot tell us.** A rise in glucose looks the same whether it comes from a meal, from hepatic
output or from stress hormones. Insulin sensitivity and insulin secretion cannot be separated from glucose data
alone, so we do not pretend to estimate them.

## 3. The twin's state-space model

Discretised every Δt = 5 minutes:

```
G[t+1] = G[t] + Δt·( −p·(G[t] − B[t]) + R[t] ) + w_G
R[t+1] = exp(−Δt/τ)·R[t] + w_R
B[t+1] = B[t] + w_B
y[t]   = G[t] + v                       (CGM reading)
```

| Symbol | Meaning | Observed? |
|---|---|---|
| `G` | sensor glucose | yes, noisily (`y`) |
| `R` | unexplained net glucose inflow: meals, hepatic output and stress combined | **no**, a modelling proxy |
| `B` | personal baseline the patient returns to | **no**, estimated |
| `p` | effective return-to-baseline rate, which bundles insulin action and non-insulin uptake | parameter |
| `τ` | how long an inflow persists | parameter |
| `w, v` | process and sensor noise | variances are parameters |

The model is **linear in the state**, so the ordinary Kalman filter is the exact Bayesian filter under Gaussian
noise. We considered extended and unscented Kalman filters. They only help with nonlinear dynamics, so they would
add cost without benefit here. Missing readings are handled naturally: the filter predicts but skips the update,
and its uncertainty grows.

When meal records are available (PhysioNet release), carbohydrate enters through two gut compartments
`Q1 → Q2 → R` (a simplified Dalla Man-style absorption). This code path exists and is tested, but meal data were
not available in this run (see Limitations).

### System identification

1. **One-step likelihood was the wrong objective.** Maximising the standard Kalman innovation likelihood drove τ to
   its upper bound (about 400 min), shrank sensor noise to almost zero and let the baseline drift freely. The model
   became a trend-follower that is good for 5 minutes and wrong for 2 hours. This is a known identifiability
   problem: one-step fit does not constrain slow dynamics.
2. **What we used instead.** Parameters are chosen to minimise **multi-step forecast error** at 15, 30, 60, 90 and
   120 minutes. That is what the twin is used for. Fitted values: `p ≈ 0.015 min⁻¹` (a return time constant of
   about 1 hour) and `τ ≈ 15–18 min`. Both are physiologically plausible.
3. **Noise scale.** The mean forecast depends only on noise *ratios*, so we fix the ratios first, then scale all
   variances so that predicted variances match observed squared errors on the training participants.

### Personalisation

For a new patient, `p` and `τ` are refitted on that patient's first 3 days. A Gaussian prior on the log-parameters
is centred on the population fit (MAP, shrinkage). The refitted model is evaluated only on later days.

## 4. Forecasts and event probability

From the filtered state, the twin propagates the mean and covariance 24 steps ahead. The event probability is
`P(max reading over the next 120 min ≥ 180 mg/dL)`. It is computed by **Monte-Carlo sampling of whole
trajectories**, not from the marginal probability at one time point, because readings 5 minutes apart are
strongly correlated.

## 5. Machine-learning models

All models share one feature set, computed only from data at or before the prediction time:

- Current reading and lags at 5–180 minutes; changes over those lags.
- Rolling mean, SD, minimum and maximum over 30, 60 and 180 minutes; 24-hour mean and time above 180.
- Time of day; history coverage.
- Static profile: age, sex, HbA1c. BMI, fasting glucose/insulin, HOMA-IR and lipids are added automatically when
  the PhysioNet `bio.csv` is present.

| Model | Forecast target | Event model |
|---|---|---|
| Persistence | last value | trend extrapolation, Platt-scaled |
| Ridge AR | change from last value | logistic regression |
| Gradient boosting (HistGradientBoosting) | change from last value | GB classifier, Platt-scaled |
| Mechanistic (Kalman) | twin mean | twin Monte-Carlo probability |
| **Hybrid** | twin mean + GB-learned residual | GB classifier using twin states and forecasts |

## 6. Uncertainty

Split-conformal prediction intervals at 90%, calibrated on held-out *participants*. The twin and hybrid use
normalised scores `|error| / twin SD`, so intervals widen when the twin is uncertain (after a rise, after a gap).
**Group-conditional** (Mondrian) calibration gives each diabetes group its own quantile. Guarantees hold only for
participants exchangeable with the calibration set. They hold neither for individuals nor for other populations.

## 7. Evaluation design

- **Unit of independence = participant.** Five-fold `StratifiedGroupKFold` by diabetes group. Within each training
  fold, 20% of participants are held out for calibration (conformal, Platt scaling, alert threshold). The test fold
  is never used for any choice.
- Predictions are made every 15 minutes, and only when the current reading exists and history coverage is at
  least 80%.
- **Event label:** any reading ≥ 180 mg/dL in the next 120 minutes. It is scored only when the current reading is
  below 180 (otherwise trivial) and at least 80% of future readings exist (otherwise excluded, not counted as
  negative). 180 mg/dL is the upper limit of the consensus time-in-range target (Battelino et al., *Diabetes Care*
  2019). Here it is a configurable research definition, not a treatment target.
- **Uncertainty around metrics** comes from a participant-level bootstrap, plus paired per-participant comparisons
  between models.
- **Ablations:** hybrid vs ML vs mechanistic; population vs personalised; sensor-only vs profile-only vs fused;
  30% random missing readings; added sensor noise (SD 10 mg/dL), including a noise-aware twin that only changes its
  sensor-noise parameter; conformal marginal vs group-conditional; alert-policy workload.
- **External test:** ShanghaiT2DM (100 T2D, FreeStyle Libre H, 15-min). Models are trained on CGMacros
  downsampled to 15 min. The two datasets are never pooled.

## 8. Data engineering notes

- **Interpolation leak.** The CGMacros release stores Dexcom readings on a 1-minute grid filled by linear
  interpolation. Each filled value at minute t+1 depends on the reading at t+5, so using them leaks the future.
  Real readings are integers on a fixed 5-minute phase. We recover them (125,819 real readings) and keep gaps as
  missing. This is covered by unit tests.
- Readings outside 40–400 mg/dL and isolated reversing spikes over 40 mg/dL are rejected (none were found).
- Implausible, duplicate and stale inputs are flagged at run time by the twin engine. The dashboard refuses to show
  a risk when the history is too short, too sparse or stale.

## References

- Bergman RN et al. Quantitative estimation of insulin sensitivity. *Am J Physiol* 236:E667 (1979).
- Dalla Man C, Rizza RA, Cobelli C. Meal simulation model of the glucose-insulin system. *IEEE TBME* 54:1740 (2007).
- Battelino T et al. Clinical targets for CGM data interpretation: International Consensus on Time in Range.
  *Diabetes Care* 42:1593 (2019).
- Nathan DM et al. Translating the A1C assay into estimated average glucose values. *Diabetes Care* 31:1473 (2008).
- Angelopoulos AN, Bates S. A gentle introduction to conformal prediction. arXiv:2107.07511 (2021).
- Anjana RM et al. ICMR-INDIAB-17. *Lancet Diabetes Endocrinol* 11:474 (2023).
- Das A et al. CGMacros. *Sci Data* 12:1557 (2025).
- Zhao Q et al. Chinese diabetes datasets for data-driven machine learning. *Sci Data* 10 (2023).
