from torch import nn

from models.drn import drn_d_38, drn_d_54, drn_d_105
from .common import ExpandChannels2d


class Drn(nn.Module):
    def __init__(self, type, num_classes):
        super().__init__()

        if type == "drn_d_38":
            self.drn = drn_d_38(pretrained=True)
        elif type == "drn_d_54":
            self.drn = drn_d_54(pretrained=True)
        elif type == "drn_d_105":
            self.drn = drn_d_105(pretrained=True)
        else:
            raise Exception("Unsupported drn model type: '{}".format(type))

        self.expand_channels = ExpandChannels2d(3)
        self.bn = nn.BatchNorm2d(3)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Conv2d(self.drn.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.expand_channels(x)
        x = self.bn(x)

        x = self.drn.layer0(x)
        x = self.drn.layer1(x)
        x = self.drn.layer2(x)
        x = self.drn.layer3(x)
        x = self.drn.layer4(x)
        x = self.drn.layer5(x)
        x = self.drn.layer6(x)
        x = self.drn.layer7(x)
        x = self.drn.layer8(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x
