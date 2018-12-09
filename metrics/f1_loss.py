import torch
import torch.nn as nn
import torch.nn.functional as F


class F1Loss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, logits, targets):
        epsilon = 1e-6
        beta = 1
        batch_size = logits.size()[0]

        p = F.sigmoid(logits)
        l = targets
        if self.weight is not None:
            weights = self.weight.expand(batch_size, -1)
            p = p * weights
            l = l * weights

        num_pos = torch.sum(p, 1) + epsilon
        num_pos_hat = torch.sum(l, 1) + epsilon
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + epsilon)
        loss = fs.sum() / batch_size
        return 1 - loss
