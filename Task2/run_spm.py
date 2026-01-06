import os
import numpy as np
import matplotlib.pyplot as plt
import pybamm


def run_and_save(task_dir=".", times_to_sample=None, r_points=100):
    if times_to_sample is None:
        # five timestamps (start, 15min, 30min, 45min, 60min)
        times_to_sample = [0, 900, 1800, 2700, 3600]

    os.makedirs(task_dir, exist_ok=True)
    plots_dir = os.path.join(task_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    model = pybamm.lithium_ion.SPM()
    experiment = pybamm.Experiment(["Discharge at 1C for 1 hours"])  # user-provided instruction
    sim = pybamm.Simulation(model, experiment=experiment)

    print("Solving SPM discharge (this may take a bit)...")
    sol = sim.solve()

    # find negative particle concentration variable name from the model variables
    var_candidates = [name for name in model.variables.keys() if "particle concentration" in name.lower() and "negative" in name.lower()]
    if len(var_candidates) == 0:
        var_candidates = [name for name in model.variables.keys() if "particle concentration" in name.lower()]

    if len(var_candidates) == 0:
        raise RuntimeError("Could not find a particle concentration variable in the model variables.")

    var_name = var_candidates[0]
    print(f"Using variable for particle concentration: '{var_name}'")

    # radial coordinate (SPM uses normalized particle radius 0->1)
    r = np.linspace(0, 1, r_points)

    # evaluate concentration at requested times and plot
    conc_profiles = {}
    for t in times_to_sample:
        # Prefer sol.evaluate for a stable, consistent shape
        try:
            conc = sol.evaluate(var_name, t=t)
        except Exception:
            try:
                conc = sol[var_name](r, t)
            except Exception:
                raise
        conc = np.asarray(conc)

        # If conc is 2D (e.g., shape (nx, nr)), reduce to radial profile by averaging over spatial axis
        if conc.ndim == 2:
            if conc.shape[1] == r.size:
                conc = conc.mean(axis=0)
            elif conc.shape[0] == r.size:
                conc = conc.mean(axis=1)
            else:
                conc = conc.mean(axis=0)
        elif conc.ndim > 2:
            # flatten leading axes and average to get radial profile
            conc = conc.reshape(-1, r.size).mean(axis=0)

        conc_profiles[t] = conc

    # Plot concentration vs radius for each sampled time
    plt.figure(figsize=(8, 6))
    for t, conc in conc_profiles.items():
        plt.plot(r, conc, label=f"t={t}s")
    plt.xlabel("Normalized radius (r)")
    plt.ylabel(var_name)
    plt.legend()
    plt.title("Particle concentration profiles (negative electrode)")
    plot_path = os.path.join(plots_dir, "concentration_profiles.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved concentration profiles to {plot_path}")

    # Extract terminal voltage vs time and compute SOC (1C => full discharge in 3600s)
    # find terminal voltage variable
    tv_candidates = [name for name in model.variables.keys() if "terminal voltage" in name.lower()]
    if len(tv_candidates) == 0:
        tv_candidates = [name for name in model.variables.keys() if "voltage" in name.lower()]
    if len(tv_candidates) == 0:
        raise RuntimeError("Could not find a terminal voltage variable in the model variables.")
    tv_name = tv_candidates[0]
    print(f"Using variable for voltage: '{tv_name}'")

    t_all = sol.t
    voltage = sol[tv_name](t_all)
    voltage = np.asarray(voltage)

    # SOC for 1C discharge starting at 100% (1.0) and linear in time
    soc = 1.0 - t_all / 3600.0

    # Save raw data
    data_path = os.path.join(task_dir, "voltage_vs_soc.npz")
    np.savez(data_path, t=t_all, voltage=voltage, soc=soc)
    print(f"Saved voltage vs SOC data to {data_path}")

    # Fit 5th-degree polynomial (SOC in [0,1])
    coeffs = np.polyfit(soc, voltage, 5)
    coeffs_path = os.path.join(task_dir, "ocv_coeffs.npy")
    np.save(coeffs_path, coeffs)
    print(f"Saved 5th-degree polynomial coefficients to {coeffs_path}")

    # Save a quick plot of voltage vs SOC with fit
    soc_fit = np.linspace(soc.min(), soc.max(), 200)
    voltage_fit = np.polyval(coeffs, soc_fit)
    plt.figure(figsize=(6, 4))
    plt.plot(soc, voltage, 'o', label='simulated voltage')
    plt.plot(soc_fit, voltage_fit, '-', label='5th-degree fit')
    plt.xlabel('SOC')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.title('Voltage vs SOC and 5th-degree fit')
    plot_v_path = os.path.join(plots_dir, 'voltage_vs_soc_fit.png')
    plt.savefig(plot_v_path, dpi=200)
    plt.close()
    print(f"Saved voltage vs SOC plot to {plot_v_path}")

    return {
        'concentration_profiles': conc_profiles,
        'r': r,
        't': t_all,
        'voltage': voltage,
        'soc': soc,
        'coeffs': coeffs,
        'plot_paths': {'concentration': plot_path, 'voltage_fit': plot_v_path},
        'data_path': data_path,
        'coeffs_path': coeffs_path,
    }


if __name__ == '__main__':
    out = run_and_save(task_dir=os.path.join(os.path.dirname(__file__), "."))
    print("Done. Files written:")
    print(out['plot_paths'])