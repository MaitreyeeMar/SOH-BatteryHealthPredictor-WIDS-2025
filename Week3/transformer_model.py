import torch
import torch.nn as nn


class TransformerSOH(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 3)  # predict [SoH, Q_Ah, E_Wh]
        )

    def forward(self, x):
        # x: (B, S, F)
        x = self.input_fc(x)  # (B, S, d_model)
        x = x.permute(1, 0, 2)  # (S, B, d)
        x = self.transformer(x)  # (S, B, d)
        x = x.permute(1, 2, 0)  # (B, d, S)
        x = self.pool(x).squeeze(-1)  # (B, d)
        out = self.head(x)  # (B, 3)
        return out


def physics_losses(pred_q, true_q, pred_e, true_e, lambda_q=0.3, lambda_e=0.3):
    mse = nn.MSELoss()
    loss_q = mse(pred_q, true_q)
    loss_e = mse(pred_e, true_e)
    return lambda_q * loss_q + lambda_e * loss_e
