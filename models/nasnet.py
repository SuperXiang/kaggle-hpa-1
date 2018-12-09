from pretrainedmodels.models.nasnet import nasnetalarge
from torch import nn

from .common import ExpandChannels2d


class NasNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.nasnet = nasnetalarge(num_classes=1000, pretrained="imagenet")

        self.expand_channels = ExpandChannels2d(3)
        self.bn = nn.BatchNorm2d(3)

        self.nasnet.last_linear = nn.Linear(4032, num_classes)

    def forward(self, x):
        x = self.expand_channels(x)
        x = self.bn(x)
        x = self.nasnet(x)
        return x
