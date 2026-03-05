# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
import numpy as np
from scipy.stats import skewnorm, FitError


def skewnorm_mode(shape, loc, scale):

    delta = shape * np.sqrt(2 / np.pi) / np.sqrt(1 + shape**2)
    mo = (
        delta
        - (1 - np.pi / 4) * delta**3 / (1 - delta**2)
        - 0.5 * np.sign(shape) * np.exp(-2 * np.pi / np.abs(shape))
    )

    return loc + scale * mo


def skewnorm_mle_fit(data):
    success = True
    shape, loc, scale = None, None, None
    try:
        mu0 = np.mean(data[np.isfinite(data)])
        sig0 = np.std(data[np.isfinite(data)])
        a0 = 0.0
        shape, loc, scale = skewnorm.fit(
            data[np.isfinite(data)], a0, loc=mu0, scale=sig0
        )

    except FitError as e:
        print(f"Error fitting skewnorm: {e}")
        success = False

    res = {
        "shape": shape,
        "shape_error": 0.0,
        "location": loc,
        "location_error": 0.0,
        "scale": scale,
        "scale_error": 0.0,
        "success": success,
    }

    return res
