from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import variable
import Resnet
from config import class_num
import numpy as np


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.pretrined_model = Resnet.resnet50(pretrained=True)
        self.pretrined_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrined_model.fc = nn.Linear(512 * 4, class_num)

    def forward(self, x):
        x, feature1, feature2 = self.pretrined_model(x)
        return x


class GAPNet(nn.Module):
    def __init__(self):
        super(GAPNet, self).__init__()
        self.pretrined_model = Resnet.resnet50(pretrained=True)
        self.con_a1x1 = nn.Conv2d(2048, class_num, kernel_size=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pretrined_model(x)  # return final class, feature before pooling, features after pooling
        x = self.con_a1x1(x)
        x = self.avgpool(x)
        x = x.squeeze()
        return x
