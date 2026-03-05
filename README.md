# CALISTO — Calibration of Forces in Single-Molecule Tweezers Operations

**CALISTO** is a desktop GUI application for force calibration of Magnetic Tweezers (MT) experiments. It provides an interactive, step-by-step workflow for loading bead tracking data, inspecting traces, selecting reference beads, determining offsets, and extracting calibrated force–magnet position curves 

---

## Features

- **Multi-format data import** — Load bead tracking data from plain-text (`.txt`) or HDF5 (`.h5`) files, together with motor/magnet position data.
- **Configurable MT setups** — Define unit-conversion factors (position → nm, time → s, magnet → mm) per setup in a YAML config file.
- **Interactive trace plotter** — Visualise X/Y/Z bead traces, toggle individual beads on/off, classify beads as magnetic or reference.
- **Magnet-position & plateau detection** — Automatically segment constant-magnet-position plateaus with adjustable tolerance and minimum length.
- **Reference-bead selection** — Automated outlier detection (Isolation Forest) and manual include/exclude of reference beads, with side-by-side included vs. excluded trace views.
- **Offset determination** — Interactive slider and Gaussian-mixture-model-based offset estimation with import/export of offset tables.
- **Rotation curve analysis** — Load rotation measurement data, visualise magnet rotation vs. bead extension, and classify tether types (nicked, supercoilable, double-tethered).
- **Per-bead force calibration** — Extract magnetic trap stiffness via Power Spectral Density (PSD), Allan Variance (AV), or Hadamard Variance (HV) methods.
- **Master force–magnet position curve** — Aggregate forces across beads, fit single- or double-exponential decay models.
- **Noise & stability diagnostics** — Per-plateau noise identification and stationarity checks using Allan/Hadamard variance scaling.
- **Built-in `tweezepy` library** — Bundled calibration engine implementing MLE fitting of PSD, AV, and HV.

---

## Requirements

- **Python 3.12** (exact version required)
- All dependencies are declared in `pyproject.toml` and installed automatically.

### Key dependencies

| Package | Purpose |
|---|---|
| PySide6 | Qt 6 GUI framework |
| pyqtgraph | Fast interactive plotting |
| NumPy / SciPy / pandas | Numerical computing & data handling |
| numba | JIT-accelerated variance calculations |
| emcee / corner | tweezepy dependency |
| scikit-learn | Outlier detection (Isolation Forest) |
| statsmodels | Exponential smoothing for reference-bead processing |
| h5py | HDF5 file support |
| autograd | Automatic differentiation for MLE Hessians |
| PyYAML | Configuration file parsing |
| Bottleneck | Fast moving-median filter |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/lipfertlab/calisto.git
cd calisto

# (Recommended) Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# Install
pip install .
```

---

## Usage

After installation two console entry-points are available:

```bash
calisto          # launch the GUI
mt-calibrate     # alias for the same command
```

### Typical workflow

1. **Select MT Setup** — Choose a predefined setup from the config (unit conversions are applied automatically).
2. **Load Data** — Import bead trace data and motor position data (plain-text or HDF5).
3. **Set Parameters** — Enter bead radius, and temperature.
4. **Inspect Traces** — Open the trace plotter to review X/Y/Z traces, toggle beads.
5. **Select Reference Beads** — Use automated or manual selection to pick suitable reference beads.
6. **Detect Plateaus** — View the magnet position plot, adjust tolerance and minimum plateau length.
7. **Determine Offsets** — Fit, load or manually set the surface offset for each bead.
8. **Noise Diagnostics** — Check per-plateau noise type and stationarity.
9. **(Optional) Rotation Curves** — Load rotation data to classify tether topology.
10. **Calibrate Forces** — Run PSD/AV/HV calibration per bead and review force–extension curves with WLC fits.
11. **Master Curve** — Generate the aggregated force–magnet distance curve with exponential model fits.

---

## Input File Formats

CALISTO supports two data formats: **plain-text** (`.txt`) and **HDF5** (`.h5`). Unit conversion factors defined in the config are applied automatically after loading.

### Plain-Text Files (`.txt`)

Whitespace-delimited (spaces or tabs), no header row. The token `-1.#IND000` is treated as NaN and linearly interpolated.

#### Bead trace data file

Each row is one time frame. Column layout:

| Column | Content |
|---|---|
| 0 | Frame index (ignored) |
| 1 | Timestamp |
| 2 | Bead 0 — X position |
| 3 | Bead 0 — Y position |
| 4 | Bead 0 — Z position |
| 5 | Bead 1 — X position |
| 6 | Bead 1 — Y position |
| 7 | Bead 1 — Z position |
| … | … (repeating X, Y, Z triplets for each additional bead) |

#### Motor data file (force calibration)

Each row is one time point of the magnet stage.

| Column | Content |
|---|---|
| 0 | Frame index (ignored) |
| 1 | Timestamp |
| 2 | Magnet Z position |

#### Motor data file (rotation measurement)

Same whitespace-delimited format with an additional rotation column.

| Column | Content |
|---|---|
| 0 | Frame index (ignored) |
| 1 | Timestamp |
| 2 | Magnet Z position |
| 3 | Magnet rotation (turns) |

### HDF5 Files (`.h5`)

A single HDF5 file contains both bead and motor data. The expected structure:

```
/
├── timestamp          # 1-D dataset: frame timestamps (float64)
├── stage/
│   ├── t_s            # 1-D dataset: motor timestamps (seconds)
│   ├── mag_pos_mm     # 1-D dataset: magnet Z position (mm)
│   └── mag_rot_turn   # 1-D dataset: magnet rotation (turns) — only for rotation files
├── M0/                # Magnetic bead 0 (prefix "M")
│   ├── x_nm           # 1-D dataset: X position (nm)
│   ├── y_nm           # 1-D dataset: Y position (nm)
│   └── z_nm           # 1-D dataset: Z position (nm)
├── M1/                # Magnetic bead 1
│   ├── x_nm
│   ├── y_nm
│   └── z_nm
├── R2/                # Reference bead 2 (prefix "R")
│   ├── x_nm
│   ├── y_nm
│   └── z_nm
└── ...                # Additional M<id> or R<id> groups
```

- Bead groups can be named `M<id>` for magnetic beads or `R<id>` for reference beads, where `<id>` is a zero-based integer index. The `M`/`R` prefix is **optional** — groups named with just an integer (e.g. `0`, `1`, `2`) are also accepted and default to magnetic beads. The bead type can then be changed in the GUI.
- The `stage/mag_rot_turn` dataset is only required for rotation measurement files.

---

## Configuration

On first launch, a default `config.yaml` is copied to your platform's user config directory (e.g. `~/Library/Application Support/CALISTO/` on macOS). Edit this file to add custom MT setups:

```yaml
Default:
  bead_time_to_seconds: 1.0
  bead_xy_position_to_nanometers: 1.0
  bead_z_position_to_nanometers: 1.0
  magnet_position_to_millimeters: 1.0
  magnet_time_to_seconds: 1.0
  variance_axis: x

MySetup:
  bead_time_to_seconds: 0.001
  bead_xy_position_to_nanometers: 1000.0
  bead_z_position_to_nanometers: 1000.0
  magnet_position_to_millimeters: 1.0
  magnet_time_to_seconds: 0.001
  variance_axis: x
```

---

## Project Structure

```
src/CALISTO/
├── main.py                  # Application entry point
├── config.yaml              # Default configuration
├── engines/                 # Back-end logic (data loading, calibration, fitting)
│   ├── engine.py            # Core classes: StateManager, SingleBeadMeasurement, MultiBeadMeasurement
│   ├── landing_engine.py    # Data file loading & validation
│   ├── fcalibration_engine.py  # Full calibration pipeline orchestration
│   ├── calibration_engine.py   # Multi-bead measurement preparation & WLC fitting
│   ├── fit_engine.py        # Exponential decay model fitting (single & double)
│   ├── magposplotter_engine.py # Plateau detection
│   ├── traceplotter_engine.py  # Trace filtering & bead table management
│   ├── refplotter_engine.py    # Reference-bead table management
│   ├── refbead_processor.py    # Automated reference-bead selection (smoothing + Isolation Forest)
│   ├── offset_engine.py     # Surface-offset fitting
│   ├── rotation_engine.py   # Rotation curve analysis
│   ├── noisestability_engine.py # Noise identification & stationarity tests
│   └── mtstats.py           # Skew-normal MLE fitting
├── gui/                     # PySide6 GUI windows
│   ├── landing.py           # Main window (setup, file loading, parameters)
│   ├── fcalibration.py      # Calibration orchestration window
│   ├── traceplotter.py      # X/Y/Z trace visualisation
│   ├── magposplotter.py     # Magnet-position & plateau visualisation
│   ├── refplotter.py        # Reference-bead review
│   ├── offset.py            # Offset determination with slider
│   ├── rotation.py          # Rotation curve viewer
│   ├── calibration.py       # Per-bead force calibration plots
│   ├── mastercurve.py       # Aggregated force–distance curve & fits
│   └── noisestability.py    # Noise & stability diagnostics
└── tweezepy/                # Bundled calibration library
    ├── smmcalibration.py    # PSD, AV, HV calibration classes
    ├── allanvar.py          # Allan variance computation (numba-accelerated)
    ├── hadamardvar.py       # Hadamard variance computation (numba-accelerated)
    ├── MLE.py               # Maximum-likelihood estimation & MCMC sampling
    ├── expressions.py       # Analytical PSD/AV/HV model expressions
    └── simulations.py       # Trace simulation utilities
```

---

## Disclaimer — `tweezepy`

CALISTO bundles a fork of [**tweezepy**](https://github.com/ianlmorgan/tweezepy), an open-source Python package for single-molecule force calibration originally developed by Ian L. Morgan. The bundled version comes from [this fork](https://github.com/alptug/tweezepy), which extends the original package with **Hadamard Variance (HV)** calibration support. Both the original `tweezepy` and the fork are licensed under the **GNU General Public License v3.0 (GPLv3)**. The copy included in this repository (under `src/CALISTO/tweezepy/`) may differ from either upstream. All modifications remain under the same GPLv3 license. For the original package, visit the [tweezepy GitHub repository](https://github.com/ianlmorgan/tweezepy); for the fork bundled here, see [alptug/tweezepy](https://github.com/alptug/tweezepy).

---

## Authors

- **Alptuğ Ulugöl**
- **Stefanie Pritzl**

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.