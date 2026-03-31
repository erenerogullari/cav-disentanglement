import math
import torch
import torch.nn.functional as F
from torch import nn


class _TopKSigmoid(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = int(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = max(1, min(self.k, x.shape[-1]))
        topk = torch.topk(x, k=k, dim=-1)
        values = torch.sigmoid(topk.values)
        out = torch.zeros_like(x)
        out.scatter_(-1, topk.indices, values)
        return out


class G_SAE(nn.Module):
    """Guided sparse autoencoder with a CAV-compatible interface.

    The model is trained with:
      - reconstruction loss on latent activations
      - concept conditioning loss on the first `n_concepts` latent activations
      - optional orthogonality regularization over exported concept directions
    """

    def __init__(
        self,
        n_concepts: int,
        n_features: int,
        device: str = "cpu",
        latent_factor: float = 4.0,
        topk_ratio: float = 0.10,
        n_latents: int | None = None,
        topk: int | None = None,
        recon_weight: float = 1.0,
        cond_weight: float = 1.0,
        direction_source: str = "decoder",
    ) -> None:
        super().__init__()

        self.n_concepts = int(n_concepts)
        self.n_features = int(n_features)
        self.n_latents = (
            int(n_latents)
            if n_latents is not None
            else int(math.ceil(float(latent_factor) * self.n_features))
        )
        if self.n_latents < self.n_concepts:
            raise ValueError(
                f"n_latents ({self.n_latents}) must be >= n_concepts ({self.n_concepts})."
            )

        topk_default = max(
            self.n_concepts, int(round(float(topk_ratio) * self.n_latents))
        )
        self.topk = int(topk) if topk is not None else topk_default
        self.topk = max(1, min(self.topk, self.n_latents))

        self.recon_weight = float(recon_weight)
        self.cond_weight = float(cond_weight)
        self.direction_source = str(direction_source)

        self.encoder = nn.Linear(
            self.n_features, self.n_latents, bias=True, device=device
        )
        self.decoder = nn.Linear(
            self.n_latents, self.n_features, bias=True, device=device
        )
        self.activation = _TopKSigmoid(self.topk)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents_pre = self.encoder(x)
        latents = self.activation(latents_pre)
        recons = self.decoder(latents)
        return latents_pre, latents, recons

    @staticmethod
    def _normalized_recon_mse(
        recons: torch.Tensor, x: torch.Tensor, eps: float = 1e-12
    ) -> torch.Tensor:
        per_sample = ((recons - x) ** 2).mean(dim=1) / (x.pow(2).mean(dim=1) + eps)
        return per_sample.mean()

    def _concept_directions(self, source: str | None = None) -> torch.Tensor:
        src = self.direction_source if source is None else str(source)
        if src == "decoder":
            return self.decoder.weight[:, : self.n_concepts].T
        if src == "encoder":
            return self.encoder.weight[: self.n_concepts, :]
        raise ValueError(
            f"Unknown direction source '{src}'. Use 'decoder' or 'encoder'."
        )

    def _concept_bias(self, source: str | None = None) -> torch.Tensor:
        src = self.direction_source if source is None else str(source)
        if src == "decoder":
            return torch.zeros(
                self.n_concepts,
                device=self.decoder.weight.device,
                dtype=self.decoder.weight.dtype,
            )
        if src == "encoder":
            if self.encoder.bias is None:
                return torch.zeros(
                    self.n_concepts,
                    device=self.encoder.weight.device,
                    dtype=self.encoder.weight.dtype,
                )
            return self.encoder.bias[: self.n_concepts]
        raise ValueError(
            f"Unknown direction source '{src}'. Use 'decoder' or 'encoder'."
        )

    def train_step(self, x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
        _, latents, recons = self(x)
        recon_loss = self._normalized_recon_mse(recons, x)
        cond_features = latents[:, : self.n_concepts]
        cond_loss = F.binary_cross_entropy(cond_features, y)
        cav_loss = self.recon_weight * recon_loss + self.cond_weight * cond_loss

        directions = self._concept_directions()
        C = directions @ directions.T
        identity = torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
        orthogonality_loss = torch.norm(W * (C - identity), p="fro") / C.numel()
        return cav_loss, orthogonality_loss

    @torch.no_grad()
    def val_step(self, x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
        _, latents, recons = self(x)
        recon_loss = self._normalized_recon_mse(recons, x)
        cond_features = latents[:, : self.n_concepts]
        cond_loss = F.binary_cross_entropy(cond_features, y)
        cav_loss = self.recon_weight * recon_loss + self.cond_weight * cond_loss

        directions = self._concept_directions()
        C = directions @ directions.T
        identity = torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
        orthogonality_loss = torch.norm(W * (C - identity), p="fro") / C.numel()
        return cav_loss, orthogonality_loss

    def get_direction(self, idx: int):
        if idx < 0 or idx >= self.n_concepts:
            raise IndexError(
                f"Concept index {idx} out of range [0, {self.n_concepts})."
            )

        if self.direction_source == "decoder":
            direction = self.decoder.weight[:, idx]
            bias = torch.tensor(0.0, device=direction.device, dtype=direction.dtype)
            return direction, bias

        direction = self.encoder.weight[idx, :]
        if self.encoder.bias is None:
            bias = torch.tensor(0.0, device=direction.device, dtype=direction.dtype)
        else:
            bias = self.encoder.bias[idx]
        return direction, bias

    def get_params(self, normalize: bool = False):
        weights = self._concept_directions().detach().cpu().clone()
        bias = self._concept_bias().detach().cpu().clone()
        if normalize:
            norms = torch.norm(weights, p=2, dim=1, keepdim=True).clamp_min(1e-12)
            weights = weights / norms
            if bias.ndim == 1 and bias.shape[0] == weights.shape[0]:
                bias = bias / norms.squeeze(1)
            elif bias.ndim == 2 and bias.shape[0] == weights.shape[0]:
                bias = bias / norms
        return weights, bias

    def set_params(self, weights: torch.Tensor, bias: torch.Tensor):
        if weights.shape != (self.n_concepts, self.n_features):
            raise ValueError(
                f"Expected weights shape {(self.n_concepts, self.n_features)}, "
                f"got {tuple(weights.shape)}."
            )

        src = self.direction_source
        if src == "decoder":
            self.decoder.weight.data[:, : self.n_concepts] = weights.to(
                device=self.decoder.weight.device, dtype=self.decoder.weight.dtype
            ).T
        elif src == "encoder":
            self.encoder.weight.data[: self.n_concepts, :] = weights.to(
                device=self.encoder.weight.device, dtype=self.encoder.weight.dtype
            )
        else:
            raise ValueError(f"Unknown direction source '{src}'.")

        # Decoder-source directions do not use a per-concept bias term in this interface.
        if src == "decoder":
            return

        if self.encoder.bias is None:
            return

        if bias.ndim == 0:
            self.encoder.bias.data[: self.n_concepts] = bias.to(
                device=self.encoder.bias.device, dtype=self.encoder.bias.dtype
            )
            return

        flat_bias = bias.reshape(-1)
        if flat_bias.shape[0] != self.n_concepts:
            raise ValueError(
                f"Expected bias with {self.n_concepts} entries, "
                f"got {flat_bias.shape[0]}."
            )
        self.encoder.bias.data[: self.n_concepts] = flat_bias.to(
            device=self.encoder.bias.device, dtype=self.encoder.bias.dtype
        )
