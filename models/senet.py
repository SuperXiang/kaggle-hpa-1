from pretrainedmodels.models.senet import se_resnext50_32x4d, senet154
from torch import nn


class SeNet(nn.Module):
    def __init__(self, type, num_classes):
        super().__init__()

        if type == "seresnext50":
            self.senet = se_resnext50_32x4d(pretrained="imagenet")
        elif type == "senet154":
            self.senet = senet154(pretrained="imagenet")
        else:
            raise Exception("Unsupported senet model type: '{}".format(type))

        self.senet.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.senet.dropout = nn.Dropout(0.5)
        self.senet.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.senet(x)
