#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import tweezepy as tp
from enum import Enum
import pandas as pd
from PySide6.QtCore import Signal, QObject
import yaml
import numpy as np
from warnings import warn
from scipy.stats import sem, tvar
from collections.abc import MutableMapping
from scipy import integrate
from .mtstats import skewnorm_mle_fit


class FileType(Enum):
    PLAINTEXT = 0
    CSV = 1
    HDF5 = 2


class BeadType(Enum):
    MAGNETIC = 0
    REFERENCE = 1

    def __str__(self):
        mapping = {
            BeadType.MAGNETIC: "Magnetic",
            BeadType.REFERENCE: "Reference",
        }
        return mapping.get(self, self.name)


class TetherType(Enum):
    NICKED = 0
    SUPERCOILABLE = 1
    DOUBLETETHERED = 2
    # OFF_CENTER = 3
    NO_TETHER = 3

    def __str__(self):
        mapping = {
            TetherType.NICKED: "Nicked",
            TetherType.SUPERCOILABLE: "Supercoilable",
            TetherType.DOUBLETETHERED: "Double Tethered",
            # TetherType.OFF_CENTER: "Off-Center Attached",
            TetherType.NO_TETHER: "No Tether",
        }
        return mapping.get(self, self.name)


class AttachmentType(Enum):
    ONCENTER = 0
    OFFCENTER = 1

    def __str__(self):
        mapping = {
            AttachmentType.ONCENTER: "On-Center",
            AttachmentType.OFFCENTER: "Off-Center",
        }
        return mapping.get(self, self.name)


class TestResult(Enum):
    PASSED = 0
    FAILED = 1
    NOT_TESTED = 2
    INCLUDED_BY_USER = 3
    EXCLUDED_BY_USER = 4
    INCLUDED_BY_USER_TRACEPLOTTER = 5
    EXCLUDED_BY_USER_TRACEPLOTTER = 6
    INCLUDED_BY_USER_CALIBRATION = 7
    EXCLUDED_BY_USER_CALIBRATION = 8
    INCLUDED_BY_USER_REFPLOTTER = 9
    EXCLUDED_BY_USER_REFPLOTTER = 10
    INCLUDED_BY_ALGO_REFPLOTTER = 11
    EXCLUDED_BY_ALGO_REFPLOTTER = 12
    EXCLUDED_BY_OFFSET_FILTER = 13
    INCLUDED_BY_USER_ROTATION = 14
    EXCLUDED_BY_USER_ROTATION = 15

    def __str__(self):
        mapping = {
            TestResult.PASSED: "Passed",
            TestResult.FAILED: "Failed",
            TestResult.NOT_TESTED: "Included as Default",
            TestResult.INCLUDED_BY_USER: "Included by User",
            TestResult.EXCLUDED_BY_USER: "Excluded by User",
            TestResult.INCLUDED_BY_USER_TRACEPLOTTER: "Included by User (Choose & Plot Traces)",
            TestResult.EXCLUDED_BY_USER_TRACEPLOTTER: "Excluded by User (Choose & Plot Traces)",
            TestResult.INCLUDED_BY_USER_CALIBRATION: "Included by User (Forces per Bead)",
            TestResult.EXCLUDED_BY_USER_CALIBRATION: "Excluded by User (Forces per Bead)",
            TestResult.INCLUDED_BY_USER_REFPLOTTER: "Included by User (Suitable Reference Bead)",
            TestResult.EXCLUDED_BY_USER_REFPLOTTER: "Excluded by User (Suitable Reference Bead)",
            TestResult.INCLUDED_BY_ALGO_REFPLOTTER: "Included by Algorithm (Suitable Reference Bead)",
            TestResult.EXCLUDED_BY_ALGO_REFPLOTTER: "Excluded by Algorithm (Suitable Reference Bead)",
            TestResult.EXCLUDED_BY_OFFSET_FILTER: "Excluded by Offset Filter",
            TestResult.INCLUDED_BY_USER_ROTATION: "Included by User (Rotation Plot)",
            TestResult.EXCLUDED_BY_USER_ROTATION: "Excluded by User (Rotation Plot)",
        }
        return mapping.get(self, self.name)


class StateManager(QObject):
    stateChanged = Signal(dict)

    def __init__(self, initial_state=None):
        super().__init__()
        self._state = initial_state if initial_state is not None else {}

        if self._state.get("bead_specs") is None:
            self._state["bead_specs"] = pd.DataFrame(
                columns=["Type", "Tether Type", "Offset", "Test Result", "Include"]
            )

    def set_state(self, key, value):
        self._state[key] = value
        self._state["last_modified"] = key
        self.stateChanged.emit(self._state)

    def get_state(self, key, default=None):
        return self._state.get(key, default)

    def delete_state(self, key):
        if key in self._state:
            del self._state[key]
            self._state["last_modified"] = key
            self.stateChanged.emit(self._state)

    # @property
    def keys(self):
        return self._state.keys()


def load_config(fpath):
    # load yaml file
    with open(fpath, "r") as f:
        config = yaml.safe_load(f)
    return config


def process_motor_data_position_units(magpos, state_manager):
    const = state_manager.get_state("config")[state_manager.get_state("config_setup")][
        "magnet_position_to_millimeters"
    ]
    return magpos * const


def process_motor_data_time_units(magtime, state_manager):
    const = state_manager.get_state("config")[state_manager.get_state("config_setup")][
        "magnet_time_to_seconds"
    ]
    return magtime * const


def process_bead_data_position_units(pos, state_manager):
    constxy = state_manager.get_state("config")[
        state_manager.get_state("config_setup")
    ]["bead_xy_position_to_nanometers"]
    constz = state_manager.get_state("config")[state_manager.get_state("config_setup")][
        "bead_z_position_to_nanometers"
    ]
    rpos = pos.copy()
    rpos[:, :, 0] *= constxy
    rpos[:, :, 1] *= constxy
    rpos[:, :, 2] *= constz
    return rpos


def process_bead_data_time_units(time, state_manager):
    const = state_manager.get_state("config")[state_manager.get_state("config_setup")][
        "bead_time_to_seconds"
    ]
    return time * const


class SingleBeadMeasurement:

    def __init__(
        self,
        fs,
        ax,
        trace,
        radius=500.0,
        refbead=False,
        ref_subtracted=False,
        offset_subtracted=False,
    ):
        self.fs = float(fs)
        self.ax = int(ax)
        self.radius = float(radius)
        self.trace = trace
        self.refbead = bool(refbead)
        self.ref_subtracted = bool(ref_subtracted)
        self.offset_subtracted = bool(offset_subtracted)
        self.good = True

        if not isinstance(self.trace, np.ndarray):
            self.trace = np.array(self.trace)

        assert len(self.trace.shape) == 2
        assert self.trace.shape[0] == 3

        self.trace[:2, :] -= np.mean(self.trace[:2, :], axis=1).reshape([2, 1])

        self.outdated = {"eom": False, "ext": False, "force": False}

        self._extension = None
        self._EoMparams = {}
        self._force = {}

    def subtract_reference(self, reftrace):
        if self.refbead:
            warn("Subtracting reference trace from a reference bead.")
        if self.ref_subtracted:
            raise Exception("Refenrence trace was already subtracted.")

        self.trace -= reftrace
        self.trace[:2, :] -= np.mean(self.trace[:2, :], axis=1).reshape([2, 1])

        self.ref_subtracted = True
        self.outdate()

        return self

    def outdate(self):
        for k in self.outdated:
            self.outdated[k] = True

    def subtract_offset(self, offset):
        offset = float(offset)
        if self.offset_subtracted:
            raise Exception("Offset was already subtracted.")

        self.trace[2] -= offset
        self.outdate()

        return self

    def get_extension(self, method="skew"):
        if method not in ["skew", "gaussian"]:
            raise Exception("Invalid method: %s" % method)
        if (
            self._extension is None
            or self.outdated["ext"]
            or self._extension["method"] != method
        ):

            self._extension = {}
            self._extension["method"] = method
            if method == "skew":
                res = skewnorm_mle_fit(self.trace[2, :])
                self._extension["mean"] = res["location"]
                self._extension["stderr"] = res["location_error"]
                if not res["success"]:
                    method = "gaussian"
                    self._extension["method"] = method

            if method == "gaussian":
                self._extension["mean"] = np.mean(self.trace[2, :])
                self._extension["stderr"] = sem(self.trace[2, :])

            self.outdated["ext"] = False

        return self._extension

    def get_EoM_parameters(self, method, correct_tracking_error=False, kT=4.1):
        """Calculates the parameters kappa and gamma of the Langevin
        EoM given by

        .. math::
            \\kappa x(t) = \\gamma \\dot{x}(t) + F_L(t).

        Parameters
        ----------
        method : str,
            The method to calculate the parameters, can be one of
                PSD: Power spectral density
                AV: Allan Variance
                HV: Hadamard Variance

        Raises
        ------
        TODO: Fill
        """
        assert type(method) == str
        assert method in ["PSD", "AV", "HV", "auto", "naive", "fusion"]

        if self.refbead:
            raise Exception("Attempt to extract EoM parameters from reference bead")

        if not self.ref_subtracted:
            warn(
                "Reference trace is not subtracted. Mechanical drift might alter the results."
            )

        for v in self._EoMparams.values():
            if ("e" in v) != correct_tracking_error:
                self.outdated["eom"] = True
                break

        if self.outdated["eom"]:
            self._EoMparams = {}

        if method not in self._EoMparams.keys():

            if method == "fusion":
                self.outdated["eom"] = False

                _, _, psdres = self.get_EoM_parameters("PSD", correct_tracking_error)
                _, _, avres = self.get_EoM_parameters("AV", correct_tracking_error)
                _, _, hvres = self.get_EoM_parameters("HV", correct_tracking_error)

                ress = [psdres, avres, hvres]

                ks = np.array([r["k"] for r in ress])
                kerrs = np.array([r["k_error"] for r in ress])
                gs = np.array([r["g"] for r in ress])
                gerrs = np.array([r["g_error"] for r in ress])

                kvar = 1.0 / np.sum(kerrs ** (-2))
                k = kvar * np.sum(ks * kerrs ** (-2))

                gvar = 1.0 / np.sum(gerrs ** (-2))
                g = kvar * np.sum(gs * gerrs ** (-2))

                res = {
                    "k": k,
                    "k_error": np.sqrt(kvar),
                    "g": g,
                    "g_error": np.sqrt(gvar),
                }
                self._EoMparams[method] = res
            elif method == "auto":
                self.outdated["eom"] = False
                _, _, autores = self.get_EoM_parameters("AV", correct_tracking_error)
                _, _, hvres = self.get_EoM_parameters("HV", correct_tracking_error)

                if hvres["AICc"] < autores["AICc"]:
                    autores = hvres

                self._EoMparams[method] = autores
            elif method == "naive":
                variance = tvar(self.trace[self.ax])
                variance_err = variance / np.sqrt(0.5 * (len(self.trace[self.ax]) - 1))

                kest = kT / variance
                kerr = kest * np.sqrt(variance_err) / variance

                res = {"k": kest, "k_error": kerr, "g": np.nan, "g_error": np.nan}
                self._EoMparams[method] = res

            else:
                if method == "PSD":
                    estimator = tp.PSD
                if method == "AV":
                    estimator = tp.AV
                if method == "HV":
                    estimator = tp.HV

                est = estimator(self.trace[self.ax], fsample=self.fs)
                est.mlefit(
                    radius=self.radius, tracking_error=correct_tracking_error, kT=kT
                )
                if method == "PSD":
                    est.results["k"] = np.abs(
                        est.results["k"]
                    )  # PSD is quadratic in kappa, take the positive value

                self._EoMparams[method] = est.results
        self.outdated["eom"] = False

        kappa = [self._EoMparams[method]["k"], self._EoMparams[method]["k_error"]]
        gamma = [self._EoMparams[method]["g"], self._EoMparams[method]["g_error"]]

        return kappa, gamma, self._EoMparams[method]

    def get_force(self, method, correct_tracking_error=False):

        assert type(method) == str
        assert method in ["PSD", "AV", "HV", "auto", "naive", "fusion"]

        if method not in self._force.keys() or self.outdated["force"]:
            kappa, _, _ = self.get_EoM_parameters(method, correct_tracking_error)
            zpos = self.get_extension()
            self._force[method] = np.array([0.0, 0.0])

            self._force[method][0] = kappa[0] * zpos["mean"]
            self._force[method][1] = np.abs(self._force[method][0]) * np.sqrt(
                (zpos["stderr"] / zpos["mean"]) ** 2 + (kappa[1] / kappa[0]) ** 2
            )

            self.outdated["force"] = False

        return self._force[method]

    def get_power_ratio(self, method, correct_tracking_error=False, kT=4.1):
        k, g, _ = self.get_EoM_parameters(
            method, correct_tracking_error=correct_tracking_error
        )
        cA = kT / (2.0 * np.pi * np.pi * g[0])
        fc = k[0] / (2.0 * np.pi * g[0])
        fn = 0.5 * self.fs

        power_sig = 2.0 * (cA / fc) * np.arctan(fn / fc)
        power_trace = (
            integrate.simpson(self.trace[self.ax] ** 2, dx=1.0 / self.fs)
            * self.fs
            / self.trace[self.ax].size
        )

        return power_sig / power_trace


class MultiBeadMeasurement(MutableMapping):
    """
    A class used to represent a measurement.

    A measurement is defined by a fixed magnet position and
    the traces (time series) of the beads that have been tracked.

    ...

    Attributes
    ----------
    fs : float
        Sampling frequency in Hz

    mag_pos : float
        Magnet position(mm) of the magnetic tweezer

    mag_rot : float
        Magnet rotation(turns) of the magnetic tweezer

    ax : int
        Force measurement axis, the axis parallel to the B-field.

    traces : numpy.ndarray
        3D array of bead traces in nm. First index represents the bead,
        second index represents the axis and the third index
        represents the time.

    reference_traces : numpy.ndarray, optional
        3D array of reference bead traces in nm. First index represents
        the bead, second index represents the axis and the third index
        represents the time.

    bead_radius : float, optional
        The radius of the beads in nm. (default is 500)

    Methods
    -------
    TODO: Fill
    """

    def __init__(
        self,
        fs,
        mag_pos,
        mag_rot,
        ax,
        traces,
        bead_ids=None,
        refbead_ids=None,
        bead_radius=500.0,
    ):

        self.fs = float(fs)
        self.mag_pos = float(mag_pos)
        self.mag_rot = float(mag_rot)
        self.ax = int(ax)
        self.radius = float(bead_radius)

        assert type(traces) == np.ndarray
        assert len(traces.shape) == 3

        self.nbeads = traces.shape[0]

        if bead_ids is None:
            self.bead_ids = np.arange(self.nbeads)
        else:
            assert hasattr(bead_ids, "__iter__")
            assert len(bead_ids) == self.nbeads
            self.bead_ids = bead_ids

        self.traces = traces
        self.traces[:, :2, :] -= np.mean(self.traces[:, :2, :], axis=2).reshape(
            [-1, 2, 1]
        )

        self.bead = {}

        for id, trace in zip(self.bead_ids, self.traces):
            self.bead[id] = SingleBeadMeasurement(self.fs, self.ax, trace, self.radius)

        self.ref_subtracted = False
        self.refbead_ids = refbead_ids
        self.reference_traces = []
        if isinstance(self.refbead_ids, int):
            self.bead[self.refbead_ids].refbead = True
            self.reference_traces = self.bead[self.refbead_ids].trace
        elif hasattr(self.refbead_ids, "__iter__"):
            for rid in self.refbead_ids:
                self.bead[rid].refbead = True
                self.reference_traces.append(self.bead[rid].trace)

        self.reference_traces = np.array(self.reference_traces)

        self.extension = {}
        self.kappa = {}
        self.force = {}
        self.gamma = {}

    def __getitem__(self, id):
        return self.bead[id]

    def __setitem__(self, id, value):
        if not isinstance(value, SingleBeadMeasurement):
            raise TypeError("The value must be a SingleBeadMeasurement.")
        if id not in self.bead.keys():
            self.nbeads += 1
        self.bead[id] = value

    def __delitem__(self, id):
        del self.bead[id]
        self.nbeads -= 1

    def __iter__(self):
        return iter(self.bead.items())

    def __len__(self):
        return self.nbeads

    def set_ref_bead(self, bid):
        if not self.bead[bid].refbead:
            self.bead[bid].refbead = True
            self.refbead_ids.append(bid)
            self.reference_traces.append(self.bead[bid].trace)

    def get_extensions(self, recalculate=False, onlygoodbeads=True):
        if not self.extension or recalculate:
            exts = []
            exts_e = []
            for bead in self.bead.values():
                if (not bead.refbead) and ((not onlygoodbeads) or bead.good):
                    ext = bead.get_extension()
                    exts.append(ext["mean"])
                    exts_e.append(ext["stderr"])

            exts = np.array(exts)
            exts_e = np.array(exts_e)
            self.extension["mean"] = exts

            self.extension["stderr"] = exts_e
            self.extension["stdev"] = self.extension[
                "stderr"
            ]  # yes, this is wrong, but this is here for compatibility with older versions
        return self.extension

    def get_power_ratios(
        self, method, correct_tracking_error=False, onlygoodbeads=True, kT=4.1
    ):
        prs = []
        for bead in self.bead.values():
            if (not bead.refbead) and ((not onlygoodbeads) or bead.good):
                pr = bead.get_power_ratio(
                    method, correct_tracking_error=correct_tracking_error, kT=kT
                )
                prs.append(pr)

        prs = np.array(prs)
        pr_mean = np.mean(prs)
        pr_var = tvar(prs)
        pr_err = np.sqrt(pr_var / prs.size)

        return pr_mean, pr_err, prs

    def get_mean_extension(self, recalculate=False):
        ext = np.mean(self.extension["mean"])
        ext_e = (
            np.sqrt(np.sum(self.extension["stderr"] ** 2))
            / self.extension["stderr"].size
        )

        return ext, ext_e

    def get_EoM_parameters(
        self, method, correct_tracking_error=False, onlygoodbeads=True
    ):
        """Calculates the parameters kappa and gamma of the Langevin
        EoM given by

        .. math::
            \\kappa x(t) = \\gamma \\dot{x}(t) + F_L(t).

        Parameters
        ----------
        method : str,
            The method to calculate the parameters, can be one of
                PSD: Power spectral density
                AV: Allan Variance
                HV: Hadamard Variance

        Raises
        ------
        TODO: Fill
        """
        assert type(method) == str
        assert method in ["PSD", "AV", "HV", "auto", "naive", "fusion"]

        kappas = []  # kappa, kappa_error
        gammas = []  # gamma, gamma_error

        for bead in self.bead.values():
            if bead.refbead or not ((not onlygoodbeads) or bead.good):
                continue
            kappa, gamma, _ = bead.get_EoM_parameters(method, correct_tracking_error)
            kappas.append(kappa)
            gammas.append(gamma)

        self.kappa[method] = kappas
        self.gamma[method] = gammas

        return self.kappa[method], self.gamma[method]

    def get_forces(self, method, recalculate=False, onlygoodbeads=True):

        assert type(method) == str
        assert method in ["PSD", "AV", "HV", "naive", "fusion"]

        if not method in self.force.keys() or recalculate:
            kappas, _ = self.get_EoM_parameters(method, onlygoodbeads=onlygoodbeads)
            zpos = self.get_extensions(onlygoodbeads=onlygoodbeads)
            self.force[method] = np.empty_like(kappas)
            for kappa, force, z_m, z_e in zip(
                kappas, self.force[method], zpos["mean"], zpos["stderr"]
            ):
                force[0] = kappa[0] * z_m
                force[1] = np.abs(force[0]) * np.sqrt(
                    (z_e / z_m) ** 2 + (kappa[1] / kappa[0]) ** 2
                )

        return self.force[method]

    def _subtract_global_reftrace(self, idx):
        self.traces -= self.reference_traces[idx]

    def _subtract_mean_reftrace(self):
        meanreftrace = np.mean(self.reference_traces, axis=0)
        self.traces -= meanreftrace

    def subtract_reference(self, onlygoodbeads=False, onlygoodref=True):
        """Subtracts the reference trace from the bead traces"""
        assert self.ref_subtracted == False

        if len(self.reference_traces.shape) == 2:
            reftrace = self.reference_traces
        elif len(self.reference_traces.shape) == 3:
            reftraces = self.reference_traces
            if onlygoodref:
                goodrefmask = np.array(
                    [self.bead[rid].good for rid in self.refbead_ids]
                )
                reftraces = reftraces[goodrefmask]
            if len(self.reference_traces.shape) == 2:
                reftrace = reftraces
            else:
                reftrace = np.mean(reftraces, axis=0)
        else:
            raise Exception("Reference trace parsing error.")

        for bead in self.bead.values():
            if not bead.refbead and ((not onlygoodbeads) or bead.good):
                bead.subtract_reference(reftrace)

        self.ref_subtracted = True
