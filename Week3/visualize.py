"""
Visualize transformer training results and predictions.
Creates loss curves and prediction plots.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluate_transformer import evaluate


def plot_predictions(preds, trues, output_dir="plots"):
    """Plot predictions vs true values for each output."""
    os.makedirs(output_dir, exist_ok=True)
    names = ["SoH (%)", "Q (Ah)", "E (Wh)"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ax, name) in enumerate(zip(axes, names)):
        y_pred = preds[:, i]
        y_true = trues[:, i]
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name} Prediction")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions.png"), dpi=150)
    print(f"✓ Saved: {output_dir}/predictions.png")


def plot_residuals(preds, trues, output_dir="plots"):
    """Plot residuals."""
    os.makedirs(output_dir, exist_ok=True)
    names = ["SoH (%)", "Q (Ah)", "E (Wh)"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ax, name) in enumerate(zip(axes, names)):
        y_pred = preds[:, i]
        y_true = trues[:, i]
        residuals = y_true - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color="r", linestyle="--", lw=2)
        ax.set_xlabel(f"Predicted {name}")
        ax.set_ylabel(f"Residual {name}")
        ax.set_title(f"{name} Residuals")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals.png"), dpi=150)
    print(f"✓ Saved: {output_dir}/residuals.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/transformer_soh.pt")
    p.add_argument("--main_csv", required=True)
    p.add_argument("--phys_csv", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default="plots")
    args = p.parse_args()

    print("Generating Visualizations...")
    preds, trues, metrics = evaluate(args.model, args.main_csv, args.phys_csv, device=args.device)
    
    plot_predictions(preds, trues, args.output_dir)
    plot_residuals(preds, trues, args.output_dir)
    
    print(f"\nAll plots saved to {args.output_dir}/")
