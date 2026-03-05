# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.ensemble import IsolationForest
from tweezepy import HV, AV


def smoothed_trace_lag_1(traces, axis, verbose=False):
    """
    This function applies a simple exponential smoothing to the difference of traces along a specified axis.
    It iterates over the first dimension of the traces, calculates the difference along the specified axis,
    and applies the smoothing. If the maximum likelihood estimation is not successful or if 20 times the time
    constant is greater than the length of the difference trace, the function applies the smoothing with a heuristic
    initialization method. The function keeps track of the maximum time constant and the minimum smoothing level.
    If verbose is True, the function plots the difference trace and the smoothed trace for each iteration.

    Parameters:
    traces (numpy.ndarray): The input traces to be smoothed. The first dimension is iterated over.
    axis (int): The axis along which the difference of the traces is calculated.
    verbose (bool, optional): If True, plots the difference trace and the smoothed trace for each iteration. Defaults to False.

    Returns:
    numpy.ndarray: The smoothed difference traces.
    float: The maximum time constant encountered during the smoothing.
    """
    smooth_dtrc = []
    taumax = 0
    alphamin = np.inf
    for idx in range(0, traces.shape[0]):
        if verbose:
            plt.figure()
        dtrc = np.diff(traces[idx, :, axis])
        fit = SimpleExpSmoothing(dtrc, initialization_method="estimated").fit()
        tau = -1 / np.log(1 - fit.params["smoothing_level"])
        if not fit.mle_retvals.success or 20 * tau > len(dtrc):
            fit = SimpleExpSmoothing(dtrc, initialization_method="heuristic").fit(
                smoothing_level=alphamin, optimized=False
            )
        smooth_dtrc.append(fit.fittedvalues)
        tau = -1 / np.log(1 - fit.params["smoothing_level"])
        if tau > taumax:
            taumax = tau
        if fit.params["smoothing_level"] < alphamin:
            alphamin = fit.params["smoothing_level"]
        if verbose:
            plt.plot(dtrc, label="refbead trace")  # plotting y axis
            plt.plot(fit.fittedvalues, label="refbead trace")  # plotting y axis
            plt.axvline(x=10 * tau, color="r", linestyle="--", label=r"$\tau$")
            plt.title(r"$\alpha=%s$" % fit.model.params["smoothing_level"])
            plt.show()
            plt.close()
    return np.array(smooth_dtrc), taumax


def mask_outliers(smooth_dtrc, ntrees=50, verbose=False):
    """
    This function uses the Isolation Forest algorithm to detect and mask outliers in the input data.
    Isolation Forest is an unsupervised learning algorithm that identifies anomalies or outliers in the data.
    The function fits the model to the input data and makes predictions. The predictions are then used to create a mask
    where True indicates normal data and False indicates an outlier.

    Parameters:
    smooth_dtrc (numpy.ndarray): The input data to be processed. It should be a 1D or 2D array.
    ntrees (int, optional): The number of trees in the Isolation Forest. More trees provide a better detection rate but increase computation time. Defaults to 50.
    verbose (bool, optional): Controls the verbosity of the Isolation Forest algorithm. If True, the algorithm will output more information about its progress. Defaults to False.

    Returns:
    numpy.ndarray: A boolean mask where True indicates normal data and False indicates an outlier.
    """

    res = IsolationForest(n_estimators=ntrees, verbose=verbose).fit_predict(
        np.nan_to_num(smooth_dtrc, nan=0.0)
    )
    return res == 1


def get_smooth_mean_trace(traces):
    """
    This function calculates the mean of the input traces along the first axis, applies exponential smoothing to each
    channel (assuming the second dimension of the traces corresponds to different channels), and then subtracts the mean
    of the smoothed trace from each channel. The smoothing is done using the SimpleExpSmoothing function from the statsmodels
    library, with the initialization method set to "estimated". If the maximum likelihood estimation is not successful,
    the function replaces the original mean trace with the smoothed values.

    Parameters:
    traces (numpy.ndarray): The input traces to be processed. It should be a 2D array where the first dimension corresponds
    to different traces and the second dimension corresponds to different channels.

    Returns:
    numpy.ndarray: The smoothed mean traces with the mean of each channel subtracted.
    """
    mgrb = np.mean(traces, axis=0)
    smooth_mgrb = mgrb.copy()
    fit = SimpleExpSmoothing(mgrb[:, 0], initialization_method="estimated").fit()
    if fit.mle_retvals.success:
        smooth_mgrb[:, 0] = fit.fittedvalues
    fit = SimpleExpSmoothing(mgrb[:, 1], initialization_method="estimated").fit()
    if fit.mle_retvals.success:
        smooth_mgrb[:, 1] = fit.fittedvalues
    fit = SimpleExpSmoothing(mgrb[:, 2], initialization_method="estimated").fit()
    if fit.mle_retvals.success:
        smooth_mgrb[:, 2] = fit.fittedvalues
    smooth_mgrb[:, 0] -= np.mean(smooth_mgrb[:, 0])
    smooth_mgrb[:, 1] -= np.mean(smooth_mgrb[:, 1])
    return smooth_mgrb


def get_smooth_trace(trace):
    smooth_trace = trace.copy()
    fit = SimpleExpSmoothing(trace[:, 0], initialization_method="estimated").fit()
    if not fit.mle_retvals.success:
        smooth_trace[:, 0] = fit.fittedvalues
    fit = SimpleExpSmoothing(trace[:, 1], initialization_method="estimated").fit()
    if not fit.mle_retvals.success:
        smooth_trace[:, 1] = fit.fittedvalues
    fit = SimpleExpSmoothing(trace[:, 2], initialization_method="estimated").fit()
    if not fit.mle_retvals.success:
        smooth_trace[:, 2] = fit.fittedvalues
    smooth_trace[:, 0] -= np.mean(smooth_trace[:, 0])
    smooth_trace[:, 1] -= np.mean(smooth_trace[:, 1])
    return smooth_trace


def choose_beads(traces, axis, verbose=False):
    """
    This function applies a series of operations to the input traces to select and process the data.
    It first applies exponential smoothing to the difference of the traces along a specified axis,
    discards a portion of the smoothed traces based on the maximum time constant, and masks outliers
    using the Isolation Forest algorithm. The function then calculates the mean of the remaining traces,
    detrends the traces by subtracting the mean, and repeats the smoothing and outlier detection process.
    If verbose is True, the function plots the detrended traces.

    Parameters:
    traces (numpy.ndarray): The input traces to be processed. It should be a 3D array where the first dimension corresponds
    to different traces, the second dimension corresponds to time axis and the third dimension corresponds to spatial axes.
    axis (int): The axis along which the difference of the traces is calculated.
    verbose (bool, optional): If True, plots the detrended traces. Defaults to False.

    Returns:
    numpy.ndarray: A boolean mask where True indicates a selected trace and False indicates a discarded trace.
    """
    smooth_dtrc, taumax = smoothed_trace_lag_1(traces, axis, verbose=verbose)
    discart = np.min([int(10 * taumax), smooth_dtrc.shape[1] // 2])
    smooth_dtrc = smooth_dtrc[:, discart:]

    gbmask = mask_outliers(smooth_dtrc, ntrees=200, verbose=verbose)
    grb = traces[gbmask]
    smooth_mgrb = get_smooth_mean_trace(grb)

    grb_detrend = grb - smooth_mgrb

    smooth_dtrc, taumax = smoothed_trace_lag_1(grb_detrend, axis, verbose=verbose)
    discart = int(10 * taumax)
    smooth_dtrc = smooth_dtrc[:, discart:]
    gbmask[gbmask] = mask_outliers(smooth_dtrc, ntrees=200, verbose=verbose)

    return gbmask


def get_noise_profile(traces, fsample):
    """
    This function calculates the noise profile of the input traces. It iterates over the traces,
    applies the HV estimator to each trace, and performs a maximum likelihood estimation (MLE) fit
    using the 'noiseHV' function. If the support of the MLE fit is greater than 0.95, the results
    of the fit are stored. The function then calculates the mean of each parameter of the fit results
    across all stored results.

    Parameters:
    traces (numpy.ndarray): The input traces to be processed. It should be a 2D array where the first dimension corresponds
    to different traces and the second dimension corresponds to different channels.
    fsample (float): The sampling frequency of the traces.

    Returns:
    numpy.ndarray: The mean of each parameter of the fit results across all stored results.
    """
    fitres = []
    for reftrc in traces:
        est = HV(reftrc[:, 1], fsample)
        est.mlefit(fitfunc="noiseHV", guess=[1e-2, 1e-2, 1e-1, 0, 0])
        if est.results["support"] > 0.95:
            fitres.append(est.results)
    eps = np.zeros(5)
    for res in fitres:
        eps[0] += res["h_m2"]
        eps[1] += res["h_m1"]
        eps[2] += res["h0"]
        eps[3] += res["h1"]
        eps[4] += res["h2"]

    eps = eps / len(fitres)

    return eps
