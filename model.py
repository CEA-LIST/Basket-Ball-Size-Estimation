
import torch
from torch import nn
from torch.nn import functional as F

import timm


class BallSizeModel(nn.Module):

    def __init__(self):
        super(BallSizeModel, self).__init__()

        self.backbone = timm.create_model('resnet18', pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)


    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
