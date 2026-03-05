"""
This module performs operations associated with calculating the Hadamard variance.

Adapted from allantools https://github.com/aewallin/allantools.
"""

import numpy as np
import scipy

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is an optional speedup

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from warnings import warn
from tweezepy.allanvar import m_generator, noise_id, edf_greenhall, edf_approx


@njit(cache=True)
def calc_hvar_phase(phase, rate, mj, stride):
    """main calculation fungtion for HDEV and OHDEV
    Parameters
    ----------
    phase: np.array
        Phase data in seconds.
    rate: float
        The sampling rate for phase or frequency, in Hz
    mj: int
        M index value for stride
    stride: int
        Size of stride
    Returns
    -------
    (dev, deverr, n): tuple
        Array of computed values.
    Notes
    -----
    http://www.leapsecond.com/tools/adev_lib.c
                         1        N-3
         s2y(t) = --------------- sum [x(i+3) - 3x(i+2) + 3x(i+1) - x(i) ]^2
                  6*tau^2 (N-3m)  i=1
        N=M+1 phase measurements
        m is averaging factor
    NIST [SP1065]_ eqn (18) and (20) pages 20 and 21
    """

    tau0 = 1.0 / rate
    mj_i = int(mj)
    stride_i = int(stride)

    n0 = (phase.shape[0] - 0 * mj_i + stride_i - 1) // stride_i
    n1 = (phase.shape[0] - 1 * mj_i + stride_i - 1) // stride_i
    n2 = (phase.shape[0] - 2 * mj_i + stride_i - 1) // stride_i
    n3 = (phase.shape[0] - 3 * mj_i + stride_i - 1) // stride_i

    if n0 < 0:
        n0 = 0
    if n1 < 0:
        n1 = 0
    if n2 < 0:
        n2 = 0
    if n3 < 0:
        n3 = 0

    n = n0
    if n1 < n:
        n = n1
    if n2 < n:
        n = n2
    if n3 < n:
        n = n3

    if n < 0:
        n = 0

    s = 0.0
    base0 = 0
    base1 = mj_i
    base2 = 2 * mj_i
    base3 = 3 * mj_i

    for idx in range(n):
        i0 = base0 + idx * stride_i
        i1 = base1 + idx * stride_i
        i2 = base2 + idx * stride_i
        i3 = base3 + idx * stride_i
        diff = phase[i3] - 3.0 * phase[i2] + 3.0 * phase[i1] - phase[i0]
        s += diff * diff

    if n == 0:
        n = 1

    tau_factor = tau0 * mj_i
    h = (s / (6.0 * n)) / (tau_factor * tau_factor)
    return h


@njit(cache=True)
def calc_hvar_phase_batch(phase, rate, m_values, stride_values):
    out = np.empty(m_values.shape[0], dtype=np.float64)
    for idx in range(m_values.shape[0]):
        out[idx] = calc_hvar_phase(phase, rate, m_values[idx], stride_values[idx])
    return out


def hvar(data, rate=1.0, taus="octave", overlapping=True, edf="real"):
    """
    Calculates the standard and overlapping Hadamard variance.
    Takes an array of bead positions. Returns the taus, edfs, and ohvs.
    Rejects frequency drift, and handles divergent noise.

    .. math::
        \\sigma^2_{HDEV}( \\tau ) = { 1 \\over 6 \\tau^2 (N-3) }
        \\sum_{i=1}^{N-3} ( {x}_{i+3} - 3x_{i+2} + 3x_{i+1} - x_{i} )^2
    where :math:`x_i` is the time-series of phase observations, spaced
    by the measurement interval :math:`\\tau`, and with length :math:`N`.
    NIST [SP1065]_ eqn (17) and (18), page 20

    Parameters
    ----------
    data : array-like
        1-D array of bead positions in nm.
    rate : float
        Frequency of acquisition in Hz.

    Returns
    -------
    (taus, edfs, oavs) : tuple
        Tuple of computed values.
    taus : array
        Observation times.
    edfs : array
        Equivalent degrees of freedom.
    ohvs : array
        Computed Hadamard variance for each tau value.
    """
    assert (
        type(overlapping) == bool
    ), "overlapping keyword argument should be a boolean."
    assert edf in ["approx", "real"], "edf keyword argument should be approx or real."
    rate = float(rate)
    data = scipy.signal.detrend(np.asarray(data))  # convert to numpy array

    N = len(data)
    m = m_generator(N, taus=taus)
    n = N - 3 * m + 1
    m = m[n >= 2]
    taus = m / rate  # tau = m*tau_c
    if edf == "real":
        edfs = np.empty(len(m))
        for i, mj in enumerate(m):
            if N // mj > 32:
                alpha_int = noise_id(data, mj)[0]
            if (alpha_int < 3) and (alpha_int > -5):
                edfs[i] = edf_greenhall(alpha_int, 3, mj, N, overlapping=overlapping)
            else:
                warn(
                    "Real edf failed to identify noise for %s. Falling back to approximate edf."
                    % mj
                )
                edfs[i] = edf_approx(N, mj)
    elif edf == "approx":
        edfs = edf_approx(N, m)
    else:
        warn("edf keyword argument %s not recognized." % edf)
        raise UserWarning
    # Calculate phasedata from Eq. 18b (in erratum)
    phase = (
        np.cumsum(data) / rate
    )  # integrate positions, converting frequency to phase data
    phase = np.insert(phase, 0, 0)  # phase data should start at 0
    phase = np.ascontiguousarray(phase, dtype=np.float64)
    assert len(phase) > 0, "Data array length is too small: %i" % len(phase)
    # Calculate ohv from TODO: cite
    m_int = np.ascontiguousarray(m, dtype=np.int64)
    if overlapping:
        strides = np.ones_like(m_int)
        ohvs = calc_hvar_phase_batch(phase, rate, m_int, strides)
    elif not overlapping:
        strides = m_int
        ohvs = calc_hvar_phase_batch(phase, rate, m_int, strides)
    return taus, edfs, ohvs
