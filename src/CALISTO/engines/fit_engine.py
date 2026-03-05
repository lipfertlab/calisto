# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
from math import floor

import numpy as np
from scipy.optimize import minimize_scalar, minimize, lsq_linear, curve_fit


def fit_single_exp_multiplicative(t, y, n_grid=400, n_iter=2, eps=1e-12):
    """
    Fit y(t) = A * exp(-t/tau), C=0, assuming multiplicative noise:
        Var(y) ∝ y^2   (constant coefficient of variation)
    => weights w ~ 1 / yhat^2 (iteratively reweighted, no fitted nuisance params)
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    mask = (y > 0) & np.isfinite(y) & np.isfinite(t)
    t = t[mask]
    y = y[mask]

    # sort by time
    idx = np.argsort(t)
    t, y = t[idx], y[idx]

    # Data-driven tau bounds
    tspan = max(t.max() - t.min(), eps)
    difft = np.diff(t)
    dt = np.median(difft[difft > 1e-8]) if len(t) > 1 else tspan
    dt = max(dt, eps)

    tau_min = max(dt / 10.0, tspan * 1e-4)
    tau_max = tspan * 1e3
    loglo, loghi = np.log(tau_min), np.log(tau_max)

    def solve_A_given_tau(tau, w):
        phi = np.exp(-t / tau)
        denom = np.sum(w * phi * phi) + eps
        A = np.sum(w * y * phi) / denom
        return max(0.0, A), phi

    def weighted_sse_logtau(logtau, w):
        tau = np.exp(logtau)
        A, phi = solve_A_given_tau(tau, w)
        r = A * phi - y
        return np.sum(w * r * r)

    def minimize_tau(w):
        # grid in log(tau) to get a safe bracket, then bounded refine
        grid = np.linspace(loglo, loghi, n_grid)
        vals = np.array([weighted_sse_logtau(g, w) for g in grid])
        k = int(np.argmin(vals))
        lo = grid[max(0, k - 1)]
        hi = grid[min(n_grid - 1, k + 1)]
        if lo == hi:
            lo, hi = loglo, loghi

        res = minimize_scalar(
            lambda z: weighted_sse_logtau(z, w), bounds=(lo, hi), method="bounded"
        )
        tau = float(np.exp(res.x))
        A, phi = solve_A_given_tau(tau, w)
        yhat = A * phi
        return A, tau, yhat, res

    # Start unweighted
    w = np.ones_like(y)
    A, tau, yhat, res = minimize_tau(w)

    # Weight floor to avoid gigantic weights near the tail
    floor = 0.01 * max(np.max(yhat), eps)

    for _ in range(n_iter):
        w = 1.0 / (np.maximum(yhat, floor) ** 2)
        w /= np.mean(w) + eps  # normalize for conditioning
        A, tau, yhat, res = minimize_tau(w)

    return {
        "A": A,
        "tau": tau,
        "tau_bounds_used": (tau_min, tau_max),
        "opt_result": res,
    }


def _taus_from_uv(u, v):
    tau_f = np.exp(u)
    tau_s = tau_f + np.exp(v)  # guarantees tau_s > tau_f
    return tau_f, tau_s


def _weighted_nnls_AB(t, y, tau_f, tau_s, w, eps=1e-12):
    """
    Solve min_{A,B>=0} sum_i w_i (A*phi_f + B*phi_s - y)^2
    using a numerically stable bounded least-squares solver.

    Fixes NNLS maxiter issues by:
      - clipping extreme weights
      - scaling columns
      - using lsq_linear(bounds=(0,inf))
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    # Build design matrix
    X = np.column_stack([np.exp(-t / tau_f), np.exp(-t / tau_s)])

    # Guard against NaN/inf
    if not (
        np.all(np.isfinite(X)) and np.all(np.isfinite(y)) and np.all(np.isfinite(w))
    ):
        return 0.0, 0.0

    # Clip weights to avoid insane scaling (crucial!)
    # This does NOT change the optimum much, it just prevents numerical blow-ups.
    w = np.maximum(w, 0.0)
    wmax = np.percentile(w[w > 0], 99.5) if np.any(w > 0) else 1.0
    wmin = np.percentile(w[w > 0], 0.5) if np.any(w > 0) else 1.0
    w = np.clip(w, wmin, wmax)

    sw = np.sqrt(w + eps)
    Xw = X * sw[:, None]
    yw = y * sw

    # Column scaling for conditioning
    col_norm = np.linalg.norm(Xw, axis=0) + eps
    Xws = Xw / col_norm[None, :]

    # Bounded linear least squares (nonnegative)
    res = lsq_linear(Xws, yw, bounds=(0.0, np.inf), method="trf", lsmr_tol="auto")
    x = res.x / col_norm  # unscale

    A = float(max(0.0, x[0]))
    B = float(max(0.0, x[1]))
    return A, B


def fit_double_exp_multiplicative(
    t,
    y,
    n_iter=2,  # IRLS iterations for weights
    n_inits=200,  # random global search points in (u,v)
    seed=0,
    max_local_iter=500,
    eps=1e-12,
):
    """
    Fit y(t) = A exp(-t/tau_f) + B exp(-t/tau_s), with:
      - C=0
      - A,B >= 0 (NNLS)
      - tau_s > tau_f enforced via uv-param
      - multiplicative noise: weights w ~ 1 / yhat^2 (IRLS, no fitted nuisance params)

    Returns A, B, tau_fast, tau_slow.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    mask = (y > 0) & np.isfinite(y) & np.isfinite(t)
    t = t[mask]
    y = y[mask]

    # sort by time

    idx = np.argsort(t)
    t, y = t[idx], y[idx]

    # Data-driven tau bounds
    tspan = max(t.max() - t.min(), eps)
    difft = np.diff(t)
    dt = np.median(difft[difft > 1e-8]) if len(t) > 1 else tspan
    dt = max(dt, eps)

    tau_min = max(dt / 10.0, tspan * 1e-4)
    tau_max = tspan * 1e3
    loglo, loghi = np.log(tau_min), np.log(tau_max)

    # Bounds in uv space: u in [loglo, loghi], v in [log(eps), loghi]
    bounds = [(loglo, loghi), (np.log(eps), loghi)]

    rng = np.random.default_rng(seed)

    # Initialize weights: unweighted first
    w = np.ones_like(y)

    # weight floor to avoid exploding weights in the tail
    # (set from data scale; updated after first fit)
    floor = 0.01 * max(np.max(y), eps)

    def objective_uv(uv, w):
        u, v = uv
        tau_f, tau_s = _taus_from_uv(u, v)
        A, B = _weighted_nnls_AB(t, y, tau_f, tau_s, w, eps=eps)
        yhat = A * np.exp(-t / tau_f) + B * np.exp(-t / tau_s)
        r = yhat - y
        return float(np.sum(w * r * r))

    def solve_for_fixed_w(w):
        # --- global random search for a good start in 2D ---
        best_uv = None
        best_val = np.inf
        for _ in range(n_inits):
            u = rng.uniform(loglo, loghi)
            # sample delta=tau_s-tau_f in log-space via v
            v = rng.uniform(np.log(eps), loghi)
            uv = np.array([u, v], float)
            val = objective_uv(uv, w)
            if val < best_val:
                best_val = val
                best_uv = uv

        # --- local refinement (stable 2D) ---
        res = minimize(
            lambda z: objective_uv(z, w),
            best_uv,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_local_iter},
        )
        uv_opt = res.x
        tau_f, tau_s = _taus_from_uv(uv_opt[0], uv_opt[1])
        A, B = _weighted_nnls_AB(t, y, tau_f, tau_s, w, eps=eps)
        yhat = A * np.exp(-t / tau_f) + B * np.exp(-t / tau_s)
        return A, B, tau_f, tau_s, yhat, res

    # IRLS loop: update weights from current yhat for Var ~ yhat^2
    A, B, tau_f, tau_s, yhat, res = solve_for_fixed_w(w)
    floor = 0.01 * max(np.max(yhat), eps)

    for _ in range(n_iter):
        w = 1.0 / (np.maximum(yhat, floor) ** 2)
        w /= np.mean(w) + eps  # normalize (conditioning)
        A, B, tau_f, tau_s, yhat, res = solve_for_fixed_w(w)

    return {
        "A": float(A),
        "B": float(B),
        "Fmax": float(A + B),
        "c": float(A / (A + B)),
        "tau_fast": float(tau_f),
        "tau_slow": float(tau_s),
        "tau_bounds_used": (float(tau_min), float(tau_max)),
        "opt_result": res,
    }


def info_criteria(y, yhat, k, eps=1e-6):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)

    # only use positive points
    mask = (y > 0) & (yhat > 0) & np.isfinite(y) & np.isfinite(yhat)
    y = y[mask]
    yhat = yhat[mask]
    N = y.size
    if N <= k + 2:
        return {"N": N, "rss": np.nan, "AICc": np.nan, "BIC": np.nan}

    res = y - yhat
    s = yhat
    s[s < eps] = eps  # floor
    wrss = float(np.sum((res / s) ** 2))
    c = 2 * np.sum(np.log(s))

    # -2 log L up to constants
    crit = N * np.log(wrss / N) + c

    AIC = crit + 2 * k
    AICc = AIC + (2 * k * (k + 1)) / (N - k - 1)
    BIC = crit + k * np.log(N)

    return {"N": N, "rss": wrss, "AICc": AICc, "BIC": BIC}


def wlcfunc(ext, Lp, Lc, T):
    """
    % Given a vector of extensions and the parameter
    % Lp = persistence length  (in nano-m)
    % Lc = contour length      (in nano-m)
    % T  = absolute temperature (in Kelvin)
    % This function returns the forces computed from a
    % 7 parameter model of the WLC,
    % using the model by Bouchiat, et al. Biophys J 76:409 (1999)
    """
    kT = 1.3806503e-2 * T  # k_B T in units pN nano-m
    z_scaled = ext / Lc
    coef = np.array(
        [-0.25, 1, -0.5164228, -2.737418, 16.07497, -38.87607, 39.49944, -14.17718]
    )
    Fwlc = 1.0 / (4.0 * (1.0 - z_scaled) ** 2)
    for p, a in enumerate(coef):
        Fwlc += a * z_scaled**p
    return Fwlc * kT / Lp


def fit_wlc_multiplicative(
    xr,
    yr,
    temperatue,
    p0=None,
    wlcfit_cutoff_force=10,
    n_trial=100,
    eps=1e-10,
    n_iter=2,
):
    wlcfit = lambda x, Lp, Lc: wlcfunc(x, Lp, Lc, temperatue)
    nanmask = np.isfinite(xr) & np.isfinite(yr)
    x = xr[nanmask]
    y = yr[nanmask]
    mask = (y > 0) & (y < wlcfit_cutoff_force) & (x > 0)
    x = x[mask]
    y = y[mask]
    maxext = np.max(x)
    bounds = ([1e-8, maxext + 1e-8], [1e3, 100 * maxext])
    # seed rng for reproducibility
    rng = np.random.default_rng(seed=0x250796)

    def solve_for_fixed_w(w):
        # --- global random search for a good start in 2D ---
        try:
            p0 = [45, maxext * 1.2]
            bestpopt, _ = curve_fit(
                wlcfit,
                x,
                y,
                p0=p0,
                bounds=bounds,
                sigma=1 / np.sqrt(w),
                absolute_sigma=False,
            )
            residuals = y - wlcfit(x, *bestpopt)
            bestresidual = np.sum(w * residuals**2)
        except:
            bestresidual = np.inf
            bestpopt = None
        for _ in range(n_trial):
            # randomly sample p0 from bounds
            Lp0 = rng.uniform(bounds[0][0], bounds[1][0])
            Lc0 = rng.uniform(bounds[0][1], bounds[1][1])
            p0 = [Lp0, Lc0]
            try:
                popt, pcov = curve_fit(
                    wlcfit,
                    x,
                    y,
                    p0=p0,
                    bounds=bounds,
                    sigma=1 / np.sqrt(w),
                    absolute_sigma=False,
                )
                residuals = y - wlcfit(x, *popt)
                ss_res = np.sum(w * residuals**2)
                if ss_res < bestresidual:
                    bestresidual = ss_res
                    bestpopt = popt
            except RuntimeError:
                continue

        return bestpopt

    # Start unweighted
    w = np.ones_like(y)
    popt = solve_for_fixed_w(w)
    if popt is None:
        return None

    for _ in range(n_iter):
        yhat = wlcfit(x, *popt)
        floor = 0.01 * max(np.max(yhat), eps)
        w = 1.0 / (np.maximum(yhat, floor) ** 2)
        w /= np.mean(w) + eps  # normalize (conditioning)
        popt = solve_for_fixed_w(w)

    return popt
