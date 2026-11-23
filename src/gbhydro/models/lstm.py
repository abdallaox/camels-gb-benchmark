"""Regional LSTM, the data-driven benchmark. Same architecture as NeuralHydrology's CudaLSTM
as used by Kratzert et al. 2019 and Lees et al. 2021: static attributes concatenated to every
day's dynamic inputs, one LSTM layer, dropout before a linear head, forget-gate bias 3.
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    name = "lstm"

    def __init__(self, n_dyn: int, n_static: int, hidden_size: int = 64, dropout: float = 0.4,
                 initial_forget_bias: float = 3.0):
        super().__init__()
        self.lstm = nn.LSTM(n_dyn + n_static, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            h = hidden_size
            self.lstm.bias_hh_l0[h:2 * h] = initial_forget_bias     # forget gate slice of the gate biases

    def forward(self, batch: dict) -> dict:
        x_d, x_s = batch["x_d"], batch["x_s"]
        x = torch.cat([x_d, x_s[:, None, :].expand(-1, x_d.shape[1], -1)], dim=-1)
        out, _ = self.lstm(x)
        y_hat = self.head(self.dropout(out[:, -1])).squeeze(-1)        # prediction for the last day
        return {"y_hat": y_hat}
