# Task2 — SPM 1C Discharge, concentration profiles, and OCV fit

This task runs a simple SPM (single-particle model) discharge at 1C using PyBaMM, extracts particle concentration profiles, and fits a 5th-degree polynomial to map SOC to terminal voltage.

What I did and why
- I run the SPM for the experiment "Discharge at 1C for 1 hours" to get the cell voltage and internal state over time.
- I extract the negative electrode particle concentration across the particle radius at a few times so we can see how concentration changes inside particles during discharge.
- I use the simulated terminal voltage and the assumed SOC (SOC = 1 - t/3600 for 1C) to build a Voltage vs SOC dataset, then fit a 5th-degree polynomial. The fit is saved so it can be used as a simple SOC→Voltage mapping.

Main files
- `run_spm.py`: runs the SPM discharge, extracts concentration profiles at default times [0, 900, 1800, 2700, 3600] s, saves plots to `Task2/plots`, saves `voltage_vs_soc.npz`, and writes polynomial coefficients to `Task2/ocv_coeffs.npy`.
- `ocv_fit.py`: loads the saved polynomial coefficients and provides `soc_to_voltage(soc)`.
- `requirements.txt`: Python packages needed for Task 2.

How to run
1. Create and activate a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the simulation:

```bash
python run_spm.py
```

3. Test the fitted mapping:

```bash
python ocv_fit.py
```

Files produced by the script
- `Task2/plots/concentration_profiles.png`: particle concentration vs radius for the sampled times.
- `Task2/plots/voltage_vs_soc_fit.png`: simulated voltage vs SOC and the polynomial fit.
- `Task2/voltage_vs_soc.npz`: raw arrays `t`, `voltage`, and `soc`.
- `Task2/ocv_coeffs.npy`: numpy array with the polynomial coefficients.

Theory and logic
- Single-particle model (SPM): each electrode is approximated by a representative spherical particle. The model captures solid-phase diffusion inside each particle and gives a simple, fast approximation of the cell behavior.
- Particle concentration profiles: the concentration inside a particle varies from center to surface during charge/discharge. Plotting concentration vs radius at different times shows how lithium moves inside particles.
- SOC→Voltage fit: I use the simulated terminal voltage during the 1C discharge as a proxy for a voltage vs SOC relation. I then fit a 5th-degree polynomial to that curve to get a compact mapping SOC→Voltage. This is not the true open-circuit voltage (OCV), but it is a useful approximation for tasks that need a simple voltage model.

Notes:
- SOC in the script is computed assuming a constant 1C current, so SOC = 1 - t/3600. If you want a more accurate SOC or the true OCV curve, run zero-current equilibrations at chosen SOC points and refit.
- You can change the timestamps used for concentration extraction by passing `times_to_sample` to `run_and_save()` in `run_spm.py`.
- If you prefer a smoother mapping than a polynomial, I can add a small neural network fit and save the model.
