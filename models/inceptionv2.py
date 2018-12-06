from pretrainedmodels.models import bninception
from torch import nn


class InceptionV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.inception = bninception(pretrained="imagenet")
        self.inception.global_pool = nn.AdaptiveAvgPool2d(1)
        self.inception.conv1_7x7_s2 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.inception.last_linear = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.inception(x)
