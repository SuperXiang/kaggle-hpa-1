import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, criterion1, criterion2, alpha):
        super().__init__()
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.alpha = alpha

    def forward(self, input, target):
        return self.alpha * self.criterion1(input, target) + (1 - self.alpha) * self.criterion2(input, target)
