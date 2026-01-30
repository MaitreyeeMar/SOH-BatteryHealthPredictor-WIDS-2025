import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformer_model import TransformerSOH, physics_losses


class CycleDataset(Dataset):
    def __init__(self, main_df, phys_df, bins=20):
        # prepare sequences grouped by battery_id + cycle_number
        self.groups = []
        grp = main_df.groupby(["battery_id", "cycle_number"])
        phys_grp = phys_df.groupby(["battery_id", "cycle_number"])
        for k, g in grp:
            if len(g) != bins:
                continue
            # sort by bin_idx
            g = g.sort_values("bin_idx")
            seq_feats = g[["Voltage_measured", "Current_measured", "Temperature_measured", "SoC", "dt_hr"]].to_numpy(dtype=np.float32)
            soh = g["SoH"].iloc[0]
            # compute true Q and E from phys group
            try:
                p = phys_grp.get_group(k)
            except KeyError:
                continue
            # Q: absolute coulomb (Ah) from phys
            q = abs((p["Current_measured"] * p["dt_hr"]).sum())
            e = (p["Voltage_measured"] * p["Current_measured"] * p["dt_hr"]).sum()  # Wh
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


def train(main_csv, phys_csv, epochs=5, batch_size=16, lr=1e-3, device="cpu"):
    main_df = pd.read_csv(main_csv)
    phys_df = pd.read_csv(phys_csv)
    ds = CycleDataset(main_df, phys_df, bins=20)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    input_dim = 5
    model = TransformerSOH(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            pred_soh = preds[:, 0]
            pred_q = preds[:, 1]
            pred_e = preds[:, 2]
            true_soh = yb[:, 0]
            true_q = yb[:, 1]
            true_e = yb[:, 2]

            loss_data = mse(pred_soh, true_soh)
            loss_phys = physics_losses(pred_q, true_q, pred_e, true_e, lambda_q=0.3, lambda_e=0.3)
            loss = loss_data + loss_phys

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        avg = total_loss / len(ds) if len(ds) > 0 else 0.0
        print(f"Epoch {ep+1}/{epochs} — loss: {avg:.6f}")

    # save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/transformer_soh.pt")
    print("Saved model to models/transformer_soh.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--main_csv", required=True)
    p.add_argument("--phys_csv", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    train(args.main_csv, args.phys_csv, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
