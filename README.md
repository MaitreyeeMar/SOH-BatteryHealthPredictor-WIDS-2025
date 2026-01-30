# BatteryHealth — Tasks 1, 2, 3, and 4

This repository contains the WIDS project I worked on related to battery modeling, physics-informed neural networks, and state-of-health (SoH) estimation.

## Task 1 — Exponential Decay PINN
- Purpose: Train a physics-informed neural network (PINN) to learn the exponential decay ODE dy/dt + λy = 0 with initial condition y(0)=y0.
- Key idea: The network learns from the physics (ODE residual and initial condition) without seeing labeled target data.
- How to run: `python Task1/exponential_decay_pinn.py` (install packages from `Task1/requirements.txt`).

What it does:
- Builds a small neural network that takes time `t` and outputs `y(t)`.
- Uses automatic differentiation to compute dy/dt and minimizes the ODE residual and initial condition error.
- Saves training plots and compares the learned solution to the analytic solution y(t)=y0*exp(-λt).

## Task 2 — SPM 1C Discharge and OCV fit
- Purpose: Run a single-particle model (SPM) discharge at 1C for one hour using PyBaMM, extract particle concentration profiles, and fit a 5th-degree polynomial mapping SOC → Voltage.
- Main files:
  - `Task2/run_spm.py`: runs the SPM discharge (experiment "Discharge at 1C for 1 hours"), extracts negative particle concentration profiles at five timestamps (default: 0, 900, 1800, 2700, 3600 s), saves plots and `voltage_vs_soc.npz`, and fits a 5th-degree polynomial. Coefficients are saved to `Task2/ocv_coeffs.npy`.
  - `Task2/ocv_fit.py`: loads the saved coefficients and provides `soc_to_voltage(soc)` to evaluate the fitted curve.
  - `Task2/requirements.txt`: Python dependencies for Task 2.

## Week 3 — Transformer-Based SoH Estimation
- Purpose: Build a transformer model to predict battery state-of-health (SoH), charge capacity (Q), and energy (E) from discharge cycle data using the NASA battery dataset.
- Key idea: Use a sequence-to-sequence transformer with physics-constrained losses (coulomb counting, energy conservation) to improve predictions.
- Main files:
  - `Week3/data_processing.py`: Downloads NASA battery dataset from Kaggle (228 MB), processes 2694 discharge cycles, creates binned dataset (20 bins/cycle) and physics dataset (200 points/cycle), filters by voltage cutoff (2.7V) and capacity constraints. Outputs `battery_main_binned20.csv` and `battery_phys_resampled200.csv`.
  - `Week3/transformer_model.py`: Defines the transformer architecture (4 attention heads, 2 encoder layers, 64 dimensions) that takes 5 input features (V, I, T, SoC, dt) over 20 bins and outputs [SoH %, Q_Ah, E_Wh].
  - `Week3/train_transformer.py`: Training script with physics-constrained loss functions (λ_q=0.3, λ_e=0.3). Trains for 100 epochs on NASA data.
  - `Week3/evaluate_transformer.py`: Evaluates trained model, computes MSE, MAE, and R² for each output.
  - `Week3/visualize.py`: Generates prediction scatter plots and residual plots for all three outputs.
  - `Week3/requirements.txt`: Python dependencies (numpy, pandas, torch, tqdm, kagglehub, scikit-learn).

What it does:
- Automatically downloads the NASA battery dataset and preprocesses 2694 discharge cycles into machine-learning-ready format.
- Trains a transformer model for 100 epochs on binned cycle data with physics constraints derived from high-resolution physics data.
- **Results**: SoH prediction (MAE=5.2%, R²=0.70), Q prediction (MAE=0.12 Ah, R²=0.58), E prediction (MAE=0.38 Wh, R²=0.75).
- Saves trained model to `Week3/models/transformer_soh.pt` (282 KB) and generates evaluation plots.

How to run:
```bash
cd Week3
pip install -r requirements.txt
python data_processing.py --download --out_dir data_out                    # ~5 min: download + process
python train_transformer.py --main_csv data_out/battery_main_binned20.csv --phys_csv data_out/battery_phys_resampled200.csv --epochs 100  # ~25 min (CPU) or ~5 min (GPU)
python evaluate_transformer.py --main_csv data_out/battery_main_binned20.csv --phys_csv data_out/battery_phys_resampled200.csv
python visualize.py --main_csv data_out/battery_main_binned20.csv --phys_csv data_out/battery_phys_resampled200.csv --output_dir plots
```

## Week 4 — SPM Integration and Parameter Fitting
- Purpose: Enhance a simple SPM with OCV characterization and parameter fitting from real discharge data.
- Key idea: Extract physics-informed parameters (ohmic resistance, OCV profile) from the NASA dataset to create a lightweight electrochemical model that can be ensembled with the transformer.
- Main files:
  - `Week4/spm_model.py`: Enhanced SPM class with OCV(SoC) polynomial mapping, ohmic resistance (R0), charge transfer resistance (Rct), double-layer capacitance (Cdl), and RC transient response simulation.
  - `Week4/train_spm.py`: Fits SPM parameters to physics dataset. Extracts R0 from V-I data via least squares, builds OCV profile from 10 SoC bins, and demonstrates voltage simulation.
  - `Week4/requirements.txt`: Python dependencies.

What it does:
- Loads the physics dataset from Week 3 (high-resolution 200-point cycles).
- Fits ohmic resistance R0 and extracts OCV curve from measured voltage-current-SoC data.
- Creates an SPM instance that can simulate discharge voltage curves with transient dynamics.
- Useful as a physics-informed baseline and for ensemble predictions with the transformer.

How to run:
```bash
cd Week4
pip install -r requirements.txt
python train_spm.py --phys_csv ../Week3/data_out/battery_phys_resampled200.csv
```

---

## Project Summary

| Task | Focus | Output |
|------|-------|--------|
| **Task 1** | Physics-informed learning | PINN solution to ODE |
| **Task 2** | Electrochemical modeling | OCV polynomial + SPM |
| **Week 3** | Machine learning SoH | Transformer (70% accuracy) |
| **Week 4** | Physics-based SoH | Enhanced SPM + parameter fitting |

All code is organized by task/week with clear README files, requirements.txt, and executable scripts. Ready for deployment and extension.

