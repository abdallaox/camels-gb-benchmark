"""NSE* loss (Kratzert et al. 2019): squared error scaled by each basin's train-period discharge variance."""
import torch

EPS = 0.1   # mm/d, as in Kratzert et al. 2019 / NeuralHydrology


def nse_star(y_hat: torch.Tensor, y: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_hat - y) ** 2 / (std + EPS) ** 2)
