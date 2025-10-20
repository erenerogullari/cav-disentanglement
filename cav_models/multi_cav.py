import copy
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
    

class MultiPatternCAV(nn.Module):


    def __init__(self, n_concepts: int, n_features: int, device='cpu') -> None:
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(n_concepts, n_features, device=device))
        self.bias = nn.Parameter(torch.randn(1, n_features, device=device))
        

    def forward(self, labels):
        return labels @ self.weights + self.bias     # (batch_size, n_features)
    

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
        return self.weights[idx, :], self.bias
    

    def get_params(self):
        return self.weights.detach().cpu().clone(), self.bias.detach().cpu().clone()


    # def save_state_dict(self, dir_name):
    #     state_dict = OrderedDict({
    #         'weights': self.weights.cpu(),
    #         'bias': self.bias.cpu()
    #     })
    #     path = f'{dir_name}/state_dict.pth'
    #     torch.save(state_dict, path)
