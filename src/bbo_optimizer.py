"""
bbo_optimizer.py

Core Bayesian Optimization utilities used throughout the BBO capstone project.
Implements Gaussian Process surrogate modelling with Expected Improvement (EI)
and Upper Confidence Bound (UCB) acquisition functions.

References
----------
Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization
    of expensive black-box functions. Journal of Global Optimization, 13(4), 455-492.
Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization
    of machine learning algorithms. NeurIPS.
Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian process
    optimization in the bandit setting: No regret and experimental design. ICML.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def expected_improvement(x, gp, y_best, xi=0.01):
    """
    Compute the negative Expected Improvement at a point x (for minimisation
    by scipy.optimize, since we maximise EI by minimising -EI).

    Parameters
    ----------
    x : ndarray, shape (n_dims,)
        Candidate point.
    gp : GaussianProcessRegressor
        Fitted GP surrogate model.
    y_best : float
        Current best observed value.
    xi : float
        Exploration-exploitation trade-off parameter. Higher xi favours
        exploration; lower xi favours exploitation near y_best.

    Returns
    -------
    float
        Negative expected improvement (for use with a minimiser).
    """
    mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
    sigma = sigma.reshape(-1, 1)
    with np.errstate(divide="warn"):
        improvement = mu - y_best - xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma == 0.0] = 0.0
    return -ei[0, 0]


def upper_confidence_bound(x, gp, kappa=2.5):
    """
    Compute the negative GP-UCB acquisition value at point x.

    Parameters
    ----------
    x : ndarray, shape (n_dims,)
        Candidate point.
    gp : GaussianProcessRegressor
        Fitted GP surrogate model.
    kappa : float
        Controls the exploration bonus (Srinivas et al., 2010).

    Returns
    -------
    float
        Negative UCB value (for use with a minimiser).
    """
    mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
    return -(mu[0] + kappa * sigma[0])


def build_gp(length_scale=0.3, nu=2.5, alpha=1e-6, n_restarts=25, random_state=42):
    """
    Construct a Gaussian Process regressor with a Matern kernel.

    Parameters
    ----------
    length_scale : float
        Kernel length scale (controls the "receptive field" of the GP).
    nu : float
        Matern smoothness parameter (0.5, 1.5, or 2.5 are common choices).
    alpha : float
        Observation noise / regularisation term.
    n_restarts : int
        Number of restarts for the internal kernel hyperparameter optimiser.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    GaussianProcessRegressor
    """
    kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu)
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=random_state,
    )


def propose_next_query(X, y, bounds, acquisition="EI", xi=0.01, kappa=2.5,
                        length_scale=0.3, nu=2.5, alpha=1e-6,
                        n_restarts=100, random_state=42):
    """
    Fit a GP to observed data and propose the next query point by optimising
    an acquisition function over the bounded search space.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_dims)
        Observed input points.
    y : ndarray, shape (n_samples,)
        Observed function values.
    bounds : list of (float, float)
        Per-dimension bounds, e.g. [(0, 1)] * n_dims.
    acquisition : {"EI", "UCB"}
        Acquisition function to use.
    xi : float
        Exploration parameter for EI.
    kappa : float
        Exploration parameter for UCB.
    length_scale, nu, alpha : float
        GP kernel hyperparameters.
    n_restarts : int
        Number of random restarts for acquisition optimisation.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    best_x : ndarray, shape (n_dims,)
        Proposed next query point.
    gp : GaussianProcessRegressor
        The fitted GP model (useful for diagnostics/plotting).
    """
    rng = np.random.RandomState(random_state)
    gp = build_gp(length_scale, nu, alpha, random_state=random_state)
    gp.fit(X, y)

    y_best = y.max()
    best_x, best_val = None, np.inf

    for _ in range(n_restarts):
        x0 = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        if acquisition == "EI":
            objective = lambda x: expected_improvement(x, gp, y_best, xi)
        elif acquisition == "UCB":
            objective = lambda x: upper_confidence_bound(x, gp, kappa)
        else:
            raise ValueError("acquisition must be 'EI' or 'UCB'")

        result = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
        if result.fun < best_val:
            best_val, best_x = result.fun, result.x

    return best_x, gp


def format_query(x):
    """Format a query point as the six-decimal, hyphen-separated string
    required by the capstone submission portal, e.g. '0.123456-0.654321'."""
    return "-".join(f"{xi:.6f}" for xi in x)
