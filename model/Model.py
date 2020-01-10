from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import variable
import Resnet
import numpy as np

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.pretrined_model = Resnet.resnet50(pretrained=True)
        self.pretrined_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrined_model.fc = nn.Linear(512*4,200)
    def forward(self,x):
        x,feature1,feature2=self.pretrined_model(x)
        return x



