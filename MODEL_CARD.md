# Model Card — BBO Capstone Gaussian Process / Bayesian Optimization System

Following the template proposed by Mitchell et al. (2019), *Model Cards for
Model Reporting*.

## Model Details

- **Developed by:** Author (postgraduate ML programme participant), as
  capstone coursework.
- **Model type:** Sequential decision-making system combining a Gaussian
  Process (GP) surrogate regressor with an acquisition-function optimiser
  (Expected Improvement or Upper Confidence Bound). Not a single trained
  model but a *procedure* re-fit at every round.
- **Model architecture:** `sklearn.gaussian_process.GaussianProcessRegressor`
  with a `ConstantKernel × Matern` kernel. Acquisition optimisation via
  `scipy.optimize.minimize` (L-BFGS-B) with multi-start random restarts.
- **Version:** Final (Module 24 / round 22 of 22).
- **Hyperparameters:** length scale, Matern ν, noise α — tuned per function
  via grid search + 3-fold cross-validation (Module 18); acquisition
  parameter ξ (EI) or κ (UCB) adapted per round based on recent performance
  trend. See `src/bbo_optimizer.py` for exact defaults and `notebooks/` for
  the full tuning history.
- **License:** MIT (code); underlying black-box functions not owned/licensed
  by the author (see `DATASHEET.md`).

## Intended Use

**Primary intended use:** Sample-efficient sequential optimisation of
expensive, gradient-free, black-box functions under a strict one-query-per-round
budget, specifically the 8 functions provided in this capstone project.

**Primary intended users:** The author, for coursework completion; secondarily,
peers/instructors reviewing methodology; secondarily still, anyone learning
Bayesian Optimization who wants a worked, documented example.

**Out-of-scope uses:**
- Not intended as a general-purpose optimisation library for production use
  without further validation (no unit tests beyond notebook demonstration).
- Not validated for functions with a much larger sample budget (>100 points)
  or dimensionality (>8D), where GP scaling (`O(n^3)`) becomes prohibitive.
- Not intended for safety-critical or high-stakes optimisation (e.g. clinical
  dosing, engineering safety margins) — this is an educational artefact.

## Factors

**Relevant factors:** Function dimensionality (2D–8D) and smoothness had the
largest observed effect on performance. Low-dimensional, apparently smooth
functions (5, 7) responded well to standard GP-BO; a uniformly near-zero
function (1) never yielded any exploitable signal regardless of strategy;
high-dimensional functions (8D, Function 8) suffered from sparse coverage
(curse of dimensionality).

**Evaluation factors:** Performance was tracked per-function as "best value
found so far," not against any external benchmark, since ground truth optima
are unknown.

## Metrics

- **Primary metric:** Cumulative gain (final best value − initial best value)
  per function, and summed across all 8 functions (**+8,418 total**).
- **Secondary/diagnostic metrics:** Cross-validated R² of the GP fit (used
  during hyperparameter tuning, Module 18) to detect overfitting; e.g.
  Function 1 showed catastrophic negative CV R² (~-10⁶⁰), correctly flagging
  that its training fit was pure noise memorisation, not genuine signal.

No held-out test set exists in the traditional sense — "evaluation" here is
the realised, one-shot outcome of each query, since black-box functions
cannot be repeatedly sampled for validation without consuming budget.

## Training / Evaluation Data

See `DATASHEET.md` for full details. In short: 10 seed points per function
were provided by the course, followed by 12 points chosen adaptively by this
system across Modules 12–23. There is no separate "training" and "evaluation"
split — the GP is refit on all available data at each round, in an online
fashion.

## Quantitative Analysis — Results by Function

| Function | Dims | Initial best | Final best | Gain | Notes |
|---|---|---|---|---|---|
| 1 | 2 | 0.00 | 0.00 | +0.00 | No exploitable signal found after 22 queries |
| 2 | 2 | 0.61 | 0.69 | +0.08 | Weak, noisy signal |
| 3 | 3 | -0.03 | -0.01 | +0.03 | Weak, noisy signal |
| 4 | 4 | -10.07 | 0.57 | +10.64 | Phase transition out of a negative regime; later regressed on off-manifold probes |
| 5 | 4 | 258.00 | 8,662.48 | **+8,404.48** | Dominant success; boundary/ridge structure discovered in dims 2–3 |
| 6 | 5 | -0.71 | -0.17 | +0.54 | Steady, modest improvement |
| 7 | 6 | 1.36 | 2.75 | +1.39 | Consistent improvement; near-zero-manifold structure in dim 0 |
| 8 | 8 | 8.61 | 9.99 | +1.38 | Sparse coverage in 8D limited gains |

## Ethical Considerations

This is a low-stakes educational optimisation task with no personal, biometric,
or otherwise sensitive data involved, and no direct human subjects. The main
ethical/responsible-AI consideration is **honest reporting of limitations**
(below) rather than overstating a method's generality — in particular, not
implying the GP-BO approach "solved" Function 1, which it demonstrably did
not.

## Limitations & Known Biases

1. **No exploitable signal for Function 1.** Twenty-two queries produced no
   improvement whatsoever. This may indicate the function is genuinely flat
   / adversarially designed, or that GP assumptions (smoothness, continuity)
   do not hold. The model cannot distinguish between these explanations from
   data alone.

2. **Non-stationarity / boundary-only structure (Function 5, Function 4).**
   Function 5's large gain is concentrated in a specific boundary region; a
   single probe away from that boundary (Module 19, all-zero query) dropped
   the result to 163 from a peak of 8,662 — an over 50x regression. This
   shows the "improvement" is highly local and should not be read as a
   smooth global trend. Function 4 showed a similar late-stage regression
   (peak 0.57, later probe -40.3) when queried off its discovered cluster.

3. **Sampling bias toward high-performing regions.** Because queries are
   chosen adaptively to maximise expected improvement, the accumulated
   dataset is heavily skewed toward regions the model already believes are
   good. Vast regions of the input space — particularly in higher dimensions
   (Function 8, 8D) — remain entirely unexplored. Any claim about "the
   optimum" is therefore a claim about the best point found within a narrow,
   self-reinforcing search trajectory, not a verified global optimum.

4. **Curse of dimensionality.** With only 22 points in an 8-dimensional unit
   hypercube (Function 8), coverage is astronomically sparse. GP uncertainty
   estimates in unexplored regions should be treated as unreliable extrapolation.

5. **Assumption of smoothness.** The Matern kernel assumes a degree of
   differentiability that some functions (e.g. Function 4's erratic
   trajectory) appear to violate. Where this assumption fails, GP predictions
   and the resulting acquisition-driven queries are not well-justified.

6. **Small-sample hyperparameter tuning.** Grid search + 3-fold CV on 15–20
   points per function (Module 18) is itself a small-sample estimate and may
   not have reliably identified the true optimal kernel hyperparameters for
   every function, particularly the noisier, low-signal ones (Functions 1–3).

7. **Single-query-per-round design.** The whole system is tuned for a
   sequential, one-point-at-a-time setting. It has not been validated for
   batch/parallel query settings, which require different acquisition
   functions (e.g. q-EI) to properly account for within-batch diversity.

## Recommendations

- Treat Function 5 and Function 4's peak results as **regime-specific**, not
  globally robust — any downstream use should re-verify near the discovered
  boundary rather than assuming smooth behaviour nearby.
- For functions showing no signal (Function 1), further budget is better
  spent elsewhere; continued uniform sampling is unlikely to help without a
  fundamentally different method (e.g. non-GP surrogate, active-subspace
  methods).
- Any reuse of this code on new black-box functions should re-run the
  hyperparameter tuning step (Module 18 methodology) rather than reusing this
  project's tuned values, which are specific to these 8 functions.
