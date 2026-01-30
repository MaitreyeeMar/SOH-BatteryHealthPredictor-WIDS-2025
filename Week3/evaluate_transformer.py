"""
Evaluate the trained transformer model on the NASA battery dataset.
Generates predictions and computes metrics.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.insert(0, ".")
from transformer_model import TransformerSOH


class CycleDataset(torch.utils.data.Dataset):
    def __init__(self, main_df, phys_df, bins=20):
        self.groups = []
        grp = main_df.groupby(["battery_id", "cycle_number"])
        phys_grp = phys_df.groupby(["battery_id", "cycle_number"])
        for k, g in grp:
            if len(g) != bins:
                continue
            g = g.sort_values("bin_idx")
            seq_feats = g[["Voltage_measured", "Current_measured", "Temperature_measured", "SoC", "dt_hr"]].to_numpy(dtype=np.float32)
            soh = g["SoH"].iloc[0]
            try:
                p = phys_grp.get_group(k)
            except KeyError:
                continue
            q = abs((p["Current_measured"] * p["dt_hr"]).sum())
            e = (p["Voltage_measured"] * p["Current_measured"] * p["dt_hr"]).sum()
            self.groups.append((seq_feats, np.array([soh, q, e], dtype=np.float32)))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        x, y = self.groups[idx]
        return x, y


def collate(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


def evaluate(model_path, main_csv, phys_csv, device="cpu", batch_size=32):
    """Load model and evaluate on dataset."""
    main_df = pd.read_csv(main_csv)
    phys_df = pd.read_csv(phys_csv)
    ds = CycleDataset(main_df, phys_df, bins=20)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Load model
    model = TransformerSOH(input_dim=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(yb.cpu().numpy())

    preds_arr = np.vstack(all_preds)
    trues_arr = np.vstack(all_trues)

    # Compute metrics for each output
    metrics = {}
    for i, name in enumerate(["SoH", "Q_Ah", "E_Wh"]):
        y_pred = preds_arr[:, i]
        y_true = trues_arr[:, i]
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics[name] = {"MSE": mse, "MAE": mae, "R2": r2}
        print(f"\n{name}:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.6f}")

    return preds_arr, trues_arr, metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/transformer_soh.pt")
    p.add_argument("--main_csv", required=True)
    p.add_argument("--phys_csv", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    print("Evaluating Transformer Model")
    print("=" * 60)
    preds, trues, metrics = evaluate(args.model, args.main_csv, args.phys_csv, device=args.device)
    print("=" * 60)
    print(f"Total cycles evaluated: {len(preds)}")
