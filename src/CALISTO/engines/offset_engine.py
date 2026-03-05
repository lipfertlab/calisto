# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
import numpy as np
from scipy.optimize import curve_fit, minimize
from .engine import BeadType
from scipy import stats


def process_raw_data(state_manager):
    rawdata = state_manager.get_state("offset_rawdata")
    bead_specs = state_manager.get_state("bead_specs")

    ztraces = rawdata[:, :, 2]

    refbead_mask = (bead_specs["Type"] == BeadType.REFERENCE).to_numpy()
    include_mask = bead_specs["Include"].to_numpy()

    mask = refbead_mask & include_mask
    reftrace = ztraces[:, mask].mean(axis=1)
    reftrace = reftrace.reshape([reftrace.shape[0], 1])

    ztraces = ztraces - reftrace

    state_manager.set_state("offset_traces", ztraces)


def get_minimum_offset(trace):
    return trace.min()


def step_fitfunc(x, widthl, widthr, mean):
    out = np.zeros_like(x)
    mask = np.logical_and(x > widthl, x < np.max(x) - widthr)
    out[mask] = mean

    return out


def line_passing_points(x, x0, y0, x1, y1):
    m = (y1 - y0) / (x1 - x0)
    return y0 + m * (x - x0)


def detailedstep_fitfunc(x, low, center, widthc, widthl, widthr):

    out = np.zeros_like(x)
    regC = np.abs(x - center) <= 0.5 * widthc
    out[regC] = low
    regL = np.abs(x - center + 0.5 * widthc + 0.5 * widthl) < 0.5 * widthl
    regR = np.abs(x - center - 0.5 * widthc - 0.5 * widthr) < 0.5 * widthr

    xL = x[regL]
    xR = x[regR]

    yL = line_passing_points(
        xL, center - 0.5 * widthc - widthl, 0.0, center - 0.5 * widthc, low
    )
    yR = line_passing_points(
        xR, center + 0.5 * widthc, low, center + 0.5 * widthc + widthr, 0.0
    )

    out[regL] = yL
    out[regR] = yR

    return out


def fit_window(x, y, p0):
    offset_lowerbound = y.min()

    low0, widthl0, widthr0 = p0
    full_width = np.max(x) - np.min(x)
    popt, _ = curve_fit(
        step_fitfunc,
        x,
        y,
        p0=(widthl0, widthr0, low0),
        bounds=(
            [0.0, 0.0, offset_lowerbound],
            [0.5 * full_width, 0.5 * full_width, 0.0],
        ),
    )

    widthl0, widthr0, low0 = popt
    widthc0 = full_width - widthl0 - widthr0
    widthlt0 = 0.5 * (widthl0 + widthc0)
    widthrt0 = 0.5 * (widthr0 + widthc0)
    center0 = widthl0 + 0.5 * widthc0
    dpopt, _ = curve_fit(
        detailedstep_fitfunc,
        x,
        y,
        p0=(low0, center0, widthc0, widthlt0, widthrt0),
        bounds=(
            [offset_lowerbound, 0.0, 0.0, 0.0, 0.0],
            [0.0, np.max(x), full_width, full_width, full_width],
        ),
    )

    return dpopt


def get_rect_offset(trace):
    faketime = np.arange(trace.shape[0])
    full_width = float(faketime[-1])
    p0 = (trace.min(), full_width / 3.0, full_width / 3.0)
    offset = fit_window(faketime, trace, p0)[0]
    return offset


from sklearn import mixture


def minGMM(trace):
    faketime = np.arange(trace.shape[0])
    X = np.vstack((faketime, trace)).T

    # Fit a Gaussian mixture with EM using 4 components
    gmm = mixture.GaussianMixture(
        n_components=4, covariance_type="diag", random_state=0x250796
    ).fit(trace.reshape(-1, 1))
    minidx = np.argmin(gmm.means_)
    mean = gmm.means_[minidx]
    std = np.sqrt(gmm.covariances_[minidx])

    # Fit a Dirichlet process Gaussian mixture using 4 components
    dpgmm = mixture.BayesianGaussianMixture(
        n_components=4, covariance_type="diag", random_state=0x250796
    ).fit(trace.reshape(-1, 1))

    minidx = np.argmin(dpgmm.means_)

    if dpgmm.means_[minidx] < mean:
        mean = dpgmm.means_[minidx]
        std = np.sqrt(dpgmm.covariances_[minidx])

    # Fit a Gaussian mixture with EM using 6 components
    gmm = mixture.GaussianMixture(
        n_components=6, covariance_type="full", random_state=0x250796
    ).fit(X)
    minidx = np.argmin(gmm.means_[:, 1])
    if gmm.means_[minidx, 1] < mean:
        mean = gmm.means_[minidx, 1]
        std = np.sqrt(gmm.covariances_[minidx, 1, 1])

    # Fit a Dirichlet process Gaussian mixture using 6 components
    dpgmm = mixture.BayesianGaussianMixture(
        n_components=6, covariance_type="full", random_state=0x250796
    ).fit(X)
    minidx = np.argmin(dpgmm.means_[:, 1])

    if dpgmm.means_[minidx, 1] < mean:
        mean = dpgmm.means_[minidx, 1]
        std = np.sqrt(dpgmm.covariances_[minidx, 1, 1])

    return mean, std


def get_gmm_offset(trace):
    mean, std = minGMM(trace)
    # if mean is ndarray, convert it to scalar
    if isinstance(mean, np.ndarray):
        mean = mean[0]
    if isinstance(std, np.ndarray):
        std = std[0]

    return mean - std


def MLE_Gumbel(data):

    def neg_LL_Gumbel(parameters):

        # extract parameters
        loc, scale = parameters

        # predict the output
        # Calculate the log-likelihood for normal distribution
        LL = np.sum(stats.gumbel_r.logpdf(data, loc, scale))

        # Calculate the negative log-likelihood
        neg_LL = -1.0 * LL

        return neg_LL

    return neg_LL_Gumbel


def get_gumbel_offset(trace):
    faketime = np.arange(trace.shape[0])
    full_width = float(faketime[-1])
    p0 = (trace.min(), full_width / 3.0, full_width / 3.0)
    popt = fit_window(faketime, trace, p0)

    _, center, widthc, _, _ = popt
    regC = np.abs(faketime - center) <= 0.5 * widthc

    mintrace = trace[regC]
    mean0 = np.mean(mintrace)
    var0 = np.var(mintrace)

    scale0 = np.sqrt(6.0 * var0) / np.pi
    loc0 = mean0 - np.euler_gamma * scale0

    neg_LL_func = MLE_Gumbel(mintrace)
    mle_model = minimize(neg_LL_func, np.array([loc0, scale0]), method="L-BFGS-B")

    return mle_model.x[0]
