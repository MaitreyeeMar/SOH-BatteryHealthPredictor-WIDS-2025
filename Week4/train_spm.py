"""
Week 4: Train SPM parameters on physics dataset.
Estimates OCV curve and resistance parameters.
"""
import argparse
import pandas as pd
import numpy as np
from spm_model import EnhancedSPM


def train_spm(phys_csv, n_samples_per_cycle=10):
    """
    Train SPM by fitting to physics data.
    
    Args:
        phys_csv: Path to physics dataset (M=200 time points per cycle)
        n_samples_per_cycle: Number of points to sample per cycle for fitting
    """
    phys_df = pd.read_csv(phys_csv)
    
    # Extract unique cycles
    cycles = phys_df.groupby(["battery_id", "cycle_number"])
    
    print("Training SPM on physics data...")
    print(f"Processing {len(cycles)} discharge cycles...")
    
    # Collect all V-I-SoC points
    v_data = []
    i_data = []
    soc_data = []
    
    for k, cycle_df in cycles:
        soc = cycle_df["SoC"].to_numpy()
        v = cycle_df["Voltage_measured"].to_numpy()
        i = cycle_df["Current_measured"].to_numpy()
        
        if len(soc) < 5:
            continue
        
        # Sample evenly from this cycle
        indices = np.linspace(0, len(soc) - 1, min(n_samples_per_cycle, len(soc))).astype(int)
        v_data.extend(v[indices])
        i_data.extend(i[indices])
        soc_data.extend(soc[indices])
    
    v_data = np.array(v_data)
    i_data = np.array(i_data)
    soc_data = np.array(soc_data)
    
    print(f"Collected {len(v_data)} V-I-SoC points")
    
    # Fit R0
    R0 = EnhancedSPM.fit_ohmic_resistance(v_data, i_data, soc_data)
    print(f"Fitted R0: {R0:.6f} Ω")
    
    # Estimate OCV curve by removing ohmic drop
    v_ohmic_corrected = v_data + R0 * i_data
    
    # Create OCV profile: average voltage at each SoC bin
    soc_bins = np.linspace(0, 100, 11)  # 10 SoC bins
    ocv_at_soc = []
    for i in range(len(soc_bins) - 1):
        mask = (soc_data >= soc_bins[i]) & (soc_data < soc_bins[i + 1])
        if mask.sum() > 0:
            ocv_at_soc.append(np.mean(v_ohmic_corrected[mask]))
        else:
            ocv_at_soc.append(3.6)  # Default
    
    soc_bin_centers = (soc_bins[:-1] + soc_bins[1:]) / 2
    print(f"\nOCV Profile (SoC vs OCV):")
    for soc, ocv in zip(soc_bin_centers, ocv_at_soc):
        print(f"  SoC={soc:5.1f}% → OCV={ocv:.3f}V")
    
    # Create SPM
    spm = EnhancedSPM(
        soc_breakpoints=soc_bin_centers,
        ocv_values=np.array(ocv_at_soc),
        R0=R0,
        Rct=0.02,  # Default
        Cdl=5000   # Default
    )
    
    print(f"\n✓ SPM Created: {spm}")
    
    # Demo: simulate on first cycle
    first_cycle = next(iter(cycles))[1]
    soc = first_cycle["SoC"].to_numpy()
    i = first_cycle["Current_measured"].to_numpy()
    v_meas = first_cycle["Voltage_measured"].to_numpy()
    dt_hr = first_cycle["dt_hr"].to_numpy()
    
    v_sim = spm.simulate_voltage(soc, i, dt_hr)
    rmse = np.sqrt(np.mean((v_meas - v_sim) ** 2))
    
    print(f"\nDemo: Simulated first cycle")
    print(f"  Measured V (first 5): {v_meas[:5]}")
    print(f"  Simulated V (first 5): {v_sim[:5]}")
    print(f"  RMSE: {rmse:.4f} V")
    
    return spm


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phys_csv", required=True)
    p.add_argument("--samples_per_cycle", type=int, default=10)
    args = p.parse_args()
    
    spm = train_spm(args.phys_csv, n_samples_per_cycle=args.samples_per_cycle)
    print("\n✓ SPM training complete")
