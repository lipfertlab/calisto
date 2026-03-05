import numpy as np


def table_cell_changed(bid, pid, state, status_manager):
    # the cell in the table has been changed, update the bead_specs
    include = status_manager.get_state("inclusion")
    include[bid, pid] = state
    status_manager.set_state("inclusion", include)


def noise_id(x, af, dmin=0, dmax=2):

    # Split time series into average positions of nonoverlapping bins
    N = len(x)
    x = x[: N // af * af].reshape((N // af, af))  # cut to correct lengths
    # x = x[:N/af * af].reshape((N/af,af)) # cut to correct lengths
    x = np.average(x, axis=1)
    # require minimum length for time-series
    # if N < 32:
    #   warn(("noise_id() Can't determine noise-ID for time-series of length= %d") %len(x))
    #  return np.nan,np.nan
    # raise NotImplementedError
    d = 0  # number of differentiations
    while True:
        r1 = np.corrcoef(x[1:], x[:-1])[0][1]
        rho = r1 / (1.0 + r1)
        if d >= dmin and (rho < 0.25 or d >= dmax):
            alpha = -2.0 * (rho + d)
            alpha_int = int(-1.0 * np.round(2 * rho) - 2.0 * d)
            return alpha_int, alpha

        else:
            x = np.diff(x)
            d = d + 1


def m_generator(N, taus="octave"):
    assert taus in ["all", "octave", "decade"]
    if taus == "all":
        maxn = N // 2
        m = np.linspace(1.0, maxn, maxn, dtype="int")
    elif taus == "octave":
        # octave sampling break bin sizes, m, into powers of 2^n
        maxn = int(np.floor(np.log2(N / 2)))  # m =< N/2
        m = np.logspace(0, maxn - 1, maxn, base=2, dtype="int")  # bin sizes
    elif taus == "decade":
        maxn = int(np.floor(np.log10(N / 2)))
        m = [
            np.array([1, 2, 4]) * k
            for k in np.logspace(0, maxn, maxn + 1, base=10, dtype="int")
        ]
        m = np.ravel(m)
    return m


def is_trace_stable(trace, method):
    N = len(trace)
    m = m_generator(N, taus="octave")

    max_alpha = 2
    if method == "HV":
        n = N - 3 * m + 1
        min_alpha = -4
    elif method == "AV":
        n = N - 2 * m + 1
        min_alpha = -2
    else:
        raise ValueError("Invalid method: %s" % method)

    m = m[n >= 2]

    if len(m) == 0:
        return False, np.nan

    for mj in m:
        try:
            alpha_int, _ = noise_id(trace, mj)
        except:
            return False, np.nan

        if alpha_int < min_alpha or alpha_int > max_alpha:
            return False, alpha_int

    return True, alpha_int
