import os
import torch.utils.data
from datetime import datetime
import torchvision.transforms as transforms
from dataset import CUB200
import Model
from PIL import Image
from config import BATCH_SIZE, save_dir
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    load_ckpt = "/home/luyaowan/Data/pycharmprojects/Fine-Grained/ckpt/20200119_214728/069.ckpt"
    cuda_flag = torch.cuda.is_available()
    test_transform = transforms.Compose([
        transforms.Resize((600, 600), Image.BILINEAR),
        transforms.RandomCrop(size=448),
        transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # test_transform = transforms.Compose([
    #     transforms.Resize((448, 448), Image.BILINEAR),
    #     transforms.RandomCrop(size=448),
    #     transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    test_set = CUB200.CUB(root='./DATA/CUB_200_2011', is_train=False, transform=test_transform, data_len=None)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                              drop_last=False, pin_memory=True)

    net = Model.MyNet()
    if cuda_flag:
        net = net.cuda()
    ckpt = torch.load(load_ckpt)
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    with torch.no_grad():
        total = 0
        test_correct1 = 0
        test_correct2 = 0
        test_correct3 = 0
        test_correct4 = 0
        test_correct5 = 0
        test_bar = tqdm(test_loader)
        for data in test_bar:
            test_bar.set_description("Testing eval")

            img, label = data[0], data[1]
            if cuda_flag:
                img, label = img.cuda(), label.cuda()
            batch_size = img.size(0)
            target = net(img)
            # avg1, max1, target = net(img)
            # calculate accuracy
            # _, predict = torch.max(target, 1)
            top3_val, top3_pos = torch.topk(target, 5)
            total += batch_size

            batch_corrects1 = torch.sum((top3_pos[:, 0] == label.data)).data.item()
            test_correct1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == label.data)).data.item()
            test_correct2 += (batch_corrects1 + batch_corrects2)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == label.data)).data.item()
            test_correct3 += (batch_corrects1 + batch_corrects2 + batch_corrects3)
            batch_corrects4 = torch.sum((top3_pos[:, 3] == label.data)).data.item()
            test_correct4 += (batch_corrects1 + batch_corrects2 + batch_corrects3 + batch_corrects4)
            batch_corrects5 = torch.sum((top3_pos[:, 3] == label.data)).data.item()
            test_correct5 += (batch_corrects1 + batch_corrects2 + batch_corrects3 + batch_corrects4 + batch_corrects4)

        print(test_correct1)
        print(test_correct2)
        print(test_correct3)

        test_acc1 = float(test_correct1) / total
        test_acc2 = float(test_correct2) / total
        test_acc3 = float(test_correct3) / total
        test_acc4 = float(test_correct4) / total
        test_acc5 = float(test_correct5) / total

        print("test1 acc: {:.3f} total sample:{}".format(
            test_acc1, total
        ))
        print("test2 acc: {:.3f} total sample:{}".format(
            test_acc2, total
        ))
        print("test3 acc: {:.3f} total sample:{}".format(
            test_acc3, total
        ))
        print("test4 acc: {:.3f} total sample:{}".format(
            test_acc4, total
        ))
        print("test5 acc: {:.3f} total sample:{}".format(
            test_acc5, total
        ))
