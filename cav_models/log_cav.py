import torch
from torch import nn
import torch.nn.functional as F


class LogCAV(nn.Module):

    def __init__(self, n_concepts: int, n_features: int, device="cpu") -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_concepts, n_features, device=device))
        self.bias = nn.Parameter(torch.zeros(n_concepts, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T + self.bias

    def train_step(self, x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
        logits = self(x)
        cav_loss = F.binary_cross_entropy_with_logits(logits, y)

        C = self.weights @ self.weights.T
        identity = torch.eye(C.shape[0], device=self.weights.device)
        orthogonality_loss = torch.norm(W * (C - identity), p="fro") / C.numel()

        return cav_loss, orthogonality_loss

    @torch.no_grad()
    def val_step(self, x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
        logits = self(x)
        cav_loss = F.binary_cross_entropy_with_logits(logits, y)

        C = self.weights @ self.weights.T
        identity = torch.eye(C.shape[0], device=self.weights.device)
        orthogonality_loss = torch.norm(W * (C - identity), p="fro") / C.numel()

        return cav_loss, orthogonality_loss

    def get_direction(self, idx: int):
        return self.weights[idx, :], self.bias[idx]

    def get_params(self, normalize: bool = False):
        weights = self.weights.detach().cpu().clone()
        bias = self.bias.detach().cpu().clone()
        if normalize:
            norms = torch.norm(weights, p=2, dim=1, keepdim=True).clamp_min(1e-12)
            weights = weights / norms
            bias = bias / norms.squeeze(1)
        return weights, bias

    def set_params(self, weights: torch.Tensor, bias: torch.Tensor):
        self.weights.data = weights.to(self.weights.device)
        self.bias.data = bias.to(self.bias.device)
