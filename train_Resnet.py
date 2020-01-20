import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
import torchvision.transforms as transforms
from dataset import CUB200
import Model
from PIL import Image
from config import BATCH_SIZE, LR, resume, save_dir, WD, end_epoch, save_freq, tesorboard_dir,net_name,INPUT_SIZE
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import trainer

#inital checkpoint and tensorboard save dir
cuda_flag = torch.cuda.is_available()
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d')+net_name)
tesorboard_dir = os.path.join(tesorboard_dir,  datetime.now().strftime('%Y%m%d')+net_name)
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)

#inital dataloader
dataloader = {}
train_transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.RandomCrop(size=INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.RandomCrop(size=INPUT_SIZE),
    transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_set = CUB200.CUB(root='./DATA/CUB_200_2011', is_train=True, transform=train_transform, data_len=None)
dataloader['train'] = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                           drop_last=False)

test_set = CUB200.CUB(root='./DATA/CUB_200_2011', is_train=False, transform=test_transform, data_len=None)
dataloader['test'] = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)
print("Dataload finished!")

#load model
net = Model.MyNet()
if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt['test_acc']
    best_epoch = ckpt['epoch']
else:
    best_acc = 0.0
    start_epoch = 1
    best_epoch = None


#  frozen feature extracte layer
for name, param in net.named_parameters():
    # print(name,':',param.requires_grad)
    param.requires_grad = True
    # if name == "pretrined_model.fc.weight" or name == "pretrined_model.fc.bias":
    #     param.requires_grad = False

parameters = list(net.pretrined_model.parameters())

# for i in parameters:
#     print(i.requires_grad)

optimizer={}
optimizer['raw'] = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters), lr=LR, momentum=0.9, weight_decay=WD)
schedulers = [MultiStepLR(optimizer['raw'], milestones=[60, 100], gamma=0.1)]  # LR changes at 60 and 100

if cuda_flag:
    net = net.cuda()
    # net = DataParallel(net)

# tensoboard writer
write = SummaryWriter(tesorboard_dir)
images = torch.zeros((BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE))
if cuda_flag:
    images = images.cuda()
write.add_graph(net, (images,))

#train entry
trainer.train(net=net,
          epoch_num=end_epoch,
          start_epoch=start_epoch,
          optimizer=optimizer,
          schedulers=schedulers,
          data_loader=dataloader,
          best_acc=best_acc,
          write=write,
          save_dir=save_dir,
          save_freq=save_freq)


