# Task2 — SPM 1C Discharge, concentration profiles, and OCV fit

Contents:

- `run_spm.py` — Run a PyBaMM SPM simulation for "Discharge at 1C for 1 hours", extract negative particle concentration profiles at five timestamps (default: 0, 900, 1800, 2700, 3600 s), save plots and `voltage_vs_soc.npz`, and fit a 5th-degree polynomial to Voltage vs SOC. Saves polynomial coefficients to `ocv_coeffs.npy`.
- `ocv_fit.py` — Loads `ocv_coeffs.npy` and exposes `soc_to_voltage(soc)`.
- `requirements.txt` — Python dependencies.

Quick start:

1. Create and activate a Python environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Run the simulation (this will solve the SPM discharge and write outputs in this folder):

```bash
python run_spm.py
```

3. Use the fit function in `ocv_fit.py`:

```python
from ocv_fit import soc_to_voltage
v = soc_to_voltage(0.8)
print(v)
```

Notes:

- The script uses the discharge solution's terminal voltage vs time to build a Voltage-vs-SOC dataset (SOC computed assuming a 1C linear relation: SOC = 1 - t/3600). If you need the true equilibrium OCV curve, the code can be extended to run zero-current equilibrations at selected SOC values.
- Timestamps for concentration extraction can be changed by passing `times_to_sample` to `run_and_save()`.