from pretrainedmodels.models.senet import se_resnext50_32x4d, senet154
from torch import nn


class SeNet(nn.Module):
    def __init__(self, type, num_classes):
        super().__init__()

        if type == "seresnext50":
            self.senet = se_resnext50_32x4d(pretrained="imagenet")
            conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif type == "senet154":
            self.senet = senet154(pretrained="imagenet")
            conv1 = nn.Conv2d(4, 64, 3, stride=2, padding=1, bias=False)
        else:
            raise Exception("Unsupported senet model type: '{}".format(type))

        senet_layer0_children = list(self.senet.layer0.children())
        conv1.weight.data[:, 0:3, :, :] = senet_layer0_children[0].weight.data
        self.senet.layer0 = nn.Sequential(*([conv1] + senet_layer0_children[1:]))

        self.senet.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.senet.dropout = nn.Dropout(0.5)
        self.senet.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.senet(x)
