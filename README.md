# BatteryHealth — Tasks 1 and 2

This repository contains the WIDS project I worked on related to battery modeling and a simple physics-informed neural network.

Task 1 — Exponential Decay PINN
- Purpose: Train a physics-informed neural network (PINN) to learn the exponential decay ODE dy/dt + λy = 0 with initial condition y(0)=y0.
- Key idea: The network learns from the physics (ODE residual and initial condition) without seeing labeled target data.
- How to run: `python Task1/exponential_decay_pinn.py` (install packages from `Task1/requirements.txt`).

What it does:
- Builds a small neural network that takes time `t` and outputs `y(t)`.
- Uses automatic differentiation to compute dy/dt and minimizes the ODE residual and initial condition error.
- Saves training plots and compares the learned solution to the analytic solution y(t)=y0*exp(-λt).

Task 2 — SPM 1C Discharge and OCV fit
- Purpose: Run a single-particle model (SPM) discharge at 1C for one hour using PyBaMM, extract particle concentration profiles, and fit a 5th-degree polynomial mapping SOC → Voltage.
- Main files:
  - `Task2/run_spm.py`: runs the SPM discharge (experiment "Discharge at 1C for 1 hours"), extracts negative particle concentration profiles at five timestamps (default: 0, 900, 1800, 2700, 3600 s), saves plots and `voltage_vs_soc.npz`, and fits a 5th-degree polynomial. Coefficients are saved to `Task2/ocv_coeffs.npy`.
  - `Task2/ocv_fit.py`: loads the saved coefficients and provides `soc_to_voltage(soc)` to evaluate the fitted curve.
  - `Task2/requirements.txt`: Python dependencies for Task 2.

