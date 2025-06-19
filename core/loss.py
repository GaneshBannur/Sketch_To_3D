import torch
import torch.nn as nn

class MSCELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        if target.dim() == 4:
            target = target.unsqueeze(1)
            
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        occupied = (target_flat == 1)
        unoccupied = (target_flat == 0)
        
        P = occupied.sum(dim=1, keepdim=True).float() + self.eps
        N = unoccupied.sum(dim=1, keepdim=True).float() + self.eps
        
        log_pred = torch.log(pred_flat + self.eps)
        log_one_minus_pred = torch.log(1 - pred_flat + self.eps)
        
        fnce = - (occupied * log_pred).sum(dim=1, keepdim=True) / P
        fpce = - (unoccupied * log_one_minus_pred).sum(dim=1, keepdim=True) / N
        
        loss = (fnce + fpce).mean()
        return loss