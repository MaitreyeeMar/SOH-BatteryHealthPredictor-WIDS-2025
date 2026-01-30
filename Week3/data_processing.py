import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import kagglehub
    HAS_KAGGLEHUB = True
except ImportError:
    HAS_KAGGLEHUB = False


def process_dataset(cleaned_dataset_path, out_dir, V_CUTOFF=2.7, CAPACITY_MIN_AH=1.0, BINS=20, M=200, NOMINAL_AH=2.0):
    os.makedirs(out_dir, exist_ok=True)
    metadata_path = os.path.join(cleaned_dataset_path, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    metadata["battery_id"] = metadata["battery_id"].astype(str)

    excluded_batteries = ["B0049", "B0050", "B0051", "B0052"]

    discharge_metadata = metadata[(metadata["type"] == "discharge") & (~metadata["battery_id"].isin(excluded_batteries))].copy()
    discharge_metadata["cycle_number"] = discharge_metadata.groupby("battery_id").cumcount() + 1

    main_rows = []
    phys_rows = []

    data_dir = os.path.join(cleaned_dataset_path, "data")

    for _, row in tqdm(discharge_metadata.iterrows(), total=len(discharge_metadata)):
        file_path = os.path.join(data_dir, row["filename"])
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path).copy()

        df = df.sort_values("Time").drop_duplicates(subset=["Time"], keep="first")
        if len(df) < 5:
            continue

        cutoff_idx = df[df["Voltage_measured"] < V_CUTOFF].index.min()
        if not pd.isna(cutoff_idx):
            df = df.loc[:cutoff_idx].copy()

        if len(df) < 5:
            continue

        df["dt_hr_raw"] = df["Time"].diff().fillna(0) / 3600.0
        df["dQ"] = df["Current_measured"] * df["dt_hr_raw"]
        capacity = abs(df["dQ"].sum())

        if capacity <= CAPACITY_MIN_AH or capacity >= NOMINAL_AH:
            continue

        df["CumQ"] = df["dQ"].cumsum()
        df["SoC"] = 100.0 * (1.0 + df["CumQ"] / capacity)

        soh_value = 100.0 * capacity / NOMINAL_AH

        t = df["Time"].to_numpy()
        t0, t1 = float(t[0]), float(t[-1])
        if t1 <= t0:
            continue

        t_grid = np.linspace(t0, t1, M)
        dt_hr = np.diff(t_grid, prepend=t_grid[0]) / 3600.0

        def interp(col):
            return np.interp(t_grid, t, df[col].to_numpy())

        Vg = interp("Voltage_measured")
        Ig = interp("Current_measured")
        Tg = interp("Temperature_measured") if "Temperature_measured" in df.columns else np.zeros_like(Vg)
        SOCg = interp("SoC")

        for j in range(M):
            phys_rows.append({
                "battery_id": row["battery_id"],
                "cycle_number": row["cycle_number"],
                "t_idx": j,
                "Time_s": t_grid[j] - t0,
                "dt_hr": dt_hr[j],
                "Voltage_measured": Vg[j],
                "Current_measured": Ig[j],
                "Temperature_measured": Tg[j],
                "SoC": SOCg[j],
                "Capacity_Ah": capacity,
                "SoH": soh_value,
                "t_end_s": (t1 - t0),
            })

        rs = pd.DataFrame({
            "Time_s": t_grid - t0,
            "dt_hr": dt_hr,
            "Voltage_measured": Vg,
            "Current_measured": Ig,
            "Temperature_measured": Tg,
            "SoC": SOCg,
        })

        chunks = np.array_split(rs, BINS)
        if len(chunks) != BINS or any(c.empty for c in chunks):
            continue

        for b_idx, c in enumerate(chunks):
            main_rows.append({
                "battery_id": row["battery_id"],
                "cycle_number": row["cycle_number"],
                "bin_idx": b_idx,

                "Time_s": c["Time_s"].mean(),
                "dt_hr": c["dt_hr"].sum(),

                "Voltage_measured": c["Voltage_measured"].mean(),
                "Current_measured": c["Current_measured"].mean(),
                "Temperature_measured": c["Temperature_measured"].mean(),
                "SoC": c["SoC"].mean(),

                "Capacity_Ah": capacity,
                "SoH": soh_value,
                "t_end_s": (t1 - t0),
            })

    main_df = pd.DataFrame(main_rows)
    phys_df = pd.DataFrame(phys_rows)

    main_out = os.path.join(out_dir, "battery_main_binned20.csv")
    phys_out = os.path.join(out_dir, "battery_phys_resampled200.csv")
    main_df.to_csv(main_out, index=False)
    phys_df.to_csv(phys_out, index=False)

    print("Saved:", main_out, main_df.shape, phys_out, phys_df.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="Download dataset from Kaggle (requires kagglehub)")
    p.add_argument("--dataset_path", default=None, help="Path to cleaned_dataset folder (contains metadata.csv and data/)")
    p.add_argument("--out_dir", default="data_out")
    args = p.parse_args()

    # Download from Kaggle if requested
    if args.download:
        if not HAS_KAGGLEHUB:
            print("ERROR: kagglehub not installed. Install with: pip install kagglehub")
            exit(1)
        print("Downloading NASA battery dataset from Kaggle...")
        path = kagglehub.dataset_download("patrickfleith/nasa-battery-dataset")
        cleaned_dataset_path = os.path.join(path, "cleaned_dataset")
        print(f"Downloaded to: {path}")
    else:
        if args.dataset_path is None:
            print("ERROR: --dataset_path required (or use --download)")
            exit(1)
        cleaned_dataset_path = args.dataset_path

    process_dataset(cleaned_dataset_path, args.out_dir)
