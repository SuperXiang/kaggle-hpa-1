import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target):
        target = target.view(-1, 1).long()

        if self.weight is None:
            self.weight = torch.FloatTensor([1] * 28).cuda()

        prob = F.sigmoid(logit)
        prob = prob.view(-1, 1)
        prob = torch.cat((1 - prob, prob), 1)
        select = torch.FloatTensor(len(prob), 2).zero_().cuda()
        select.scatter_(1, target, 1.)

        self.weight = self.weight.view(-1, 1)
        self.weight = torch.gather(self.weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        # focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)

        batch_loss = - self.weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss
