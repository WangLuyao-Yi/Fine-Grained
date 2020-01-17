import os
import torch.utils.data
from datetime import datetime
import torchvision.transforms as transforms
from dataset import CUB200
import Model
from PIL import Image
from config import BATCH_SIZE, save_dir
from tqdm import tqdm

resume = "/home/luyaowan/Data/pycharmprojects/Fine-Grained-master/ckpt/metric_Net5/115.ckpt"

cuda_flag = torch.cuda.is_available()
start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)

test_transform = transforms.Compose([
    transforms.Resize((448, 448), Image.BILINEAR),
    transforms.RandomCrop(size=448),
    transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_set = CUB200.CUB(root='./DATA/CUB_200_2011', is_train=False, transform=test_transform, data_len=None)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)

# define model
net = Model.GAPNet()  # Model.MyNet()
if cuda_flag:
    net = net.cuda()

ckpt = torch.load(resume)
net.load_state_dict(ckpt['net_state_dict'])
total = 0
test_correct = 0
test_acc = 0

test_bar = tqdm(test_loader)
for data in test_bar:
    test_bar.set_description("Testing eval")
    with torch.no_grad():
        img, label = data[0], data[1]
        if cuda_flag:
            img, label = img.cuda(), label.cuda()
        batch_size = img.size(0)
        # target = net(img)
        avg1, max1, target = net(img)
        # calculate accuracy
        _, predict = torch.max(target, 1)
        total += batch_size
        test_correct += torch.sum(predict.data == label.data)

test_acc = float(test_correct) / total

print("test acc: {:.3f} total sample:{}".format(
    test_acc, total
))
