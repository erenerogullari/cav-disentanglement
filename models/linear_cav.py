import copy
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
    

class LinearCAV(nn.Module):


    def __init__(self, n_concepts: int, n_features: int, device='cpu') -> None:
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(n_concepts, n_features, requires_grad=True, device=device))
        self.biases = nn.Parameter(torch.randn(1, n_features, requires_grad=True, device=device))
        

    def forward(self, labels):
        return labels @ self.weights + self.biases     # (batch_size, n_features)
    

    def train_step(self, x, y, W):
        predictions = self(y)
        cav_loss = F.mse_loss(predictions, x)

        C = self.weights @ self.weights.T 
        identity = torch.eye(C.shape[0], device=self.weights.device)
        orthogonality_loss = torch.norm(W * (C - identity), p='fro') / C.numel()

        return cav_loss, orthogonality_loss


    @torch.no_grad()
    def val_step(self, x, y, W):
        predictions = self(y)
        cav_loss = F.mse_loss(predictions, x)

        C = self.weights @ self.weights.T 
        identity = torch.eye(C.shape[0], device=self.weights.device)
        orthogonality_loss = torch.norm(W * (C - identity), p='fro') / C.numel()

        return cav_loss, orthogonality_loss
    

    def get_direction(self, idx):
        return self.weights[idx, :]
    

    def get_params(self):
        return self.weights, self.biases


    def save_state_dict(self, dir_name):
        state_dict = OrderedDict({
            'weights': self.weights.cpu(),
            'biases': self.biases.cpu()
        })
        path = f'{dir_name}/state_dict.pth'
        torch.save(state_dict, path)
