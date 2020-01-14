from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import variable
import Resnet
from config import class_num



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
        self.conv_avg_1 = nn.Conv2d(2048, class_num, kernel_size=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_max_1 = nn.Conv2d(2048, class_num, kernel_size=1, bias=False)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.pretrined_model.avgpool = nn.AdaptiveMaxPool2d(1)
        # self.pretrined_model.fc = nn.Linear(512*4, class_num)
        self.cat_fc = nn.Linear(200 * 2, class_num)


    def forward(self, x):
        x = self.pretrined_model(x)  # return final class, feature before pooling, features after pooling
        avg1 = self.conv_avg_1(x)
        avg1 = self.avgpool(avg1)
        avg1 = avg1.squeeze()
        max1 = self.conv_max_1(x)
        max1 = self.maxpool(max1)
        max1 = max1.squeeze()
        c = torch.cat((avg1, max1), dim=1)
        predict = self.cat_fc(c)
        return avg1, max1, predict


def metric_loss(predict,target):
    batch = predict.size(0)
    predict = nn.functional.softmax(predict, 1)
    target = nn.functional.softmax(target, 1)
    loss = torch.norm((predict - target).abs(), p=2, dim=1).sum() / float(batch)
    return loss
