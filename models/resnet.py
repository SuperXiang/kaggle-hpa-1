import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50


class ResNet(nn.Module):
    def __init__(self, type, num_classes):
        super().__init__()

        if type == "resnet18":
            self.resnet = resnet18(pretrained=True)
            num_fc_in_channels = 512
        elif type == "resnet34":
            self.resnet = resnet34(pretrained=True)
            num_fc_in_channels = 512
        elif type == "resnet50":
            self.resnet = resnet50(pretrained=True)
            num_fc_in_channels = 2048
        else:
            raise Exception("Unsupported resnet model type: '{}".format(type))

        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight.data[:, 0:3, :, :] = self.resnet.conv1.weight.data
        self.resnet.conv1 = conv1

        self.resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(num_fc_in_channels),
            nn.Dropout(0.5),
            nn.Linear(num_fc_in_channels, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)
