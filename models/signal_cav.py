import copy
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
    

class SignalCAV(nn.Module):


    def __init__(self, n_concepts: int, n_features: int, device='cpu') -> None:
        
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, n_concepts, n_features, device=device))       # (1, n_concepts, n_features)
        self.biases = nn.Parameter(torch.randn(1, n_concepts, n_features, device=device))        # (1, n_concepts, n_features)
        

    def forward(self, labels):
        labels_expanded = labels.unsqueeze(-1)                  # (batch_size, n_concepts, 1)
        return labels_expanded * self.weights + self.biases     # (batch_size, n_concepts, input_dim)
    

    def train_step(self, x, y, W):
        predictions = self(y)
        cav_loss = F.mse_loss(predictions, x.unsqueeze(1).repeat(1, predictions.shape[1], 1))

        C = self.weights.squeeze(0) @ self.weights.squeeze(0).T 
        identity = torch.eye(C.shape[0], device=self.weights.device)
        orthogonality_loss = torch.norm(W * (C - identity), p='fro')

        return cav_loss, orthogonality_loss


    @torch.no_grad()
    def val_step(self, x, y, W):
        predictions = self(y)
        cav_loss = F.mse_loss(predictions, x.unsqueeze(1).repeat(1, predictions.shape[1], 1))

        C = self.weights.squeeze(0) @ self.weights.squeeze(0).T 
        identity = torch.eye(C.shape[0], device=self.weights.device)
        orthogonality_loss = torch.norm(W * (C - identity), p='fro')

        return cav_loss, orthogonality_loss
    

    def get_direction(self, idx):
        return self.weights[0, idx, :], self.biases[0, idx, :]
    

    def get_params(self):
        return self.weights.detach().cpu().clone().squeeze(0), self.biases.detach().cpu().clone().squeeze(0)


    def save_state_dict(self, dir_name):
        state_dict = OrderedDict({
            'weights': self.weights.squeeze(0).cpu(),
            'biases': self.biases.squeeze(0).cpu()
        })
        path = f'{dir_name}/state_dict.pt'
        torch.save(state_dict, path)
