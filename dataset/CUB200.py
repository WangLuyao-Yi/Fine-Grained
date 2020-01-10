from torch.utils.data import Dataset
from PIL import Image
import imageio
import numpy as np
import scipy.misc
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CUB(Dataset):
    def __init__(self,root,is_train=False, transform=None, data_len=None):
        self.root =root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root,"images.txt"))
        label_txt_file = open(os.path.join(self.root, "image_class_labels.txt"))
        train_val_file = open(os.path.join(self.root,"train_test_split.txt"))
        img_name_list = [] #all the image name
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = [] # all the image label
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = [] # train/test list
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i,x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i,x in zip(train_test_list, img_name_list) if not i]

        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root,'images', train_file))for train_file in train_file_list[:data_len]]
            self.train_label = [x for i,x in zip(train_test_list, label_list) if i ][:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
    def __getitem__(self, index):
        if self.is_train:
            img ,target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img =np.stack([img]*3,2)
            img =Image.fromarray(img, mode="RGB")
            if self.transform is not None:
                img =self.transform(img)
        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
        return img, target
    def __len__(self):
       if self.is_train:
           return len(self.train_label)
       else:
           return len(self.test_label)

if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.Resize((600,600),Image.BILINEAR),
        transforms.CenterCrop(size=448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),#turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((600,600),Image.BILINEAR),
        transforms.CenterCrop(size=448),
        transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_set = CUB(root='./../DATA/CUB_200_2011',is_train=True,transform=train_transform,data_len=8)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=8,shuffle=True,num_workers=8,drop_last=False)
    plt.figure(0)
    for (step, i)  in enumerate(train_set):
        image = i[0]
        label = i[1]
        image = transforms.ToPILImage()(image)
        ax=plt.subplot(2,4,step+1)
        plt.imshow(image)
    plt.show()







