from pretrainedmodels.models.senet import se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, \
    se_resnext101_32x4d, senet154
from torch import nn


class SeNet(nn.Module):
    def __init__(self, type, num_classes):
        super().__init__()

        if type == "seresnext50":
            self.senet = se_resnext50_32x4d(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        elif type == "seresnext101":
            self.senet = se_resnext101_32x4d(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        elif type == "seresnet50":
            self.senet = se_resnet50(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        elif type == "seresnet101":
            self.senet = se_resnet101(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        elif type == "seresnet152":
            self.senet = se_resnet152(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        elif type == "senet154":
            self.senet = senet154(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        else:
            raise Exception("Unsupported senet model type: '{}".format(type))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(2048, num_classes)

    def features(self, x):
        x = self.layer0(x)
        x = self.senet.layer1(x)
        x = self.senet.layer2(x)
        x = self.senet.layer3(x)
        x = self.senet.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
