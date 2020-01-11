import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
import torchvision.transforms as transforms
from dataset import CUB200
import  Model
from PIL import Image
from config import BATCH_SIZE,LR,resume,save_dir,WD,end_epoch,SAVE_FREQ
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


cuda_flag = torch.cuda.is_available()
start_epoch = 1
save_dir = os.path.join(save_dir,datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
transforms.RandomHorizontalFlip(),

# read data
train_transform = transforms.Compose([
        transforms.Resize((600,600),Image.BILINEAR),
        transforms.RandomCrop(size=448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),#turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
test_transform = transforms.Compose([
    transforms.Resize((600,600),Image.BILINEAR),
    transforms.RandomCrop(size=448),
    transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_set = CUB200.CUB(root='./DATA/CUB_200_2011',is_train=True,transform=train_transform,data_len=None)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,num_workers=8,drop_last=False)

test_set = CUB200.CUB(root='./DATA/CUB_200_2011', is_train=False, transform=train_transform, data_len=None)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)

#define model
net = Model.MyNet()

if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1

criterion = torch.nn.CrossEntropyLoss()
parameters = list(net.pretrined_model.parameters())

optimizer = torch.optim.SGD(parameters, lr=LR, momentum=0.9, weight_decay=WD)
schedulers = MultiStepLR(optimizer, milestones=[60, 100, 160], gamma=0.1)  # LR changes at 60 and 100
if cuda_flag:
    net = net.cuda()
    # net = DataParallel(net)
#tensorboard add 2020-1-11
write = SummaryWriter("./tensorboard")


best_acc = 0.0
best_epoch = None
for epoch in range(start_epoch, end_epoch):
    schedulers.step()

    # begin training
    print("--" * 50)
    net.train()

    train_bar = tqdm(train_loader)

    for data in train_bar:
        train_bar.set_description("epoch %d :Training " % epoch)
        img, label =data[0], data[1]
        if cuda_flag:
            img, label = img.cuda(), label.cuda()
        batch_size = img.size(0)
        optimizer.zero_grad()
        target = net(img)
        loss = criterion(target,label)
        loss.backward()
        optimizer.step()

    if epoch % SAVE_FREQ == 0:
        train_loss = 0
        train_correct = 0
        total = 0
        net.eval()
        train_bar = tqdm(train_loader)
        for data in train_bar:
            train_bar.set_description("epoch %d: Traning eval" % epoch)
            with torch.no_grad():
                img, label = data[0], data[1]
                if cuda_flag:
                    img, label = img.cuda(), label.cuda()
                batch_size = img.size(0)
                target = net(img)
                #calculate loss
                loss = criterion(target, label)
                _,predict = torch.max(target,1)
                total += batch_size
                train_correct += torch.sum(predict.data == label.data)
                train_loss += loss.item()*batch_size
        train_acc = float(train_correct)/total
        train_loss = train_loss/total
        print("epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample:{}".format(
            epoch, train_loss, train_acc, total
        ))

        #evaluate on test
        test_loss = 0
        test_correct = 0
        total = 0
        test_bar = tqdm(test_loader)
        for data in test_bar:
            test_bar.set_description("epoch %d: Testing eval" % epoch)
            with torch.no_grad():
                img, label = data[0], data[1]
                if cuda_flag:
                    img, label = img.cuda(), label.cuda()
                batch_size = img.size(0)
                target = net(img)
                #calculate loss
                loss =criterion(target,label)
                #calculate accuracy
                _,predict = torch.max(target,1)
                total += batch_size
                test_correct += torch.sum(predict.data == label.data)
                test_loss += loss.item()*batch_size
        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        print("epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample:{}".format(
            epoch,test_loss,test_acc,total
        ))

        write.add_scalars("lOSS",{'train': train_loss, "test": test_loss },epoch)
        write.add_scalars("Accuracy",{"train": train_acc, "test": test_acc},epoch)
        write.flush()
        # save model
        if test_acc>best_acc:
            best_acc = test_acc
            best_epoch = epoch
            net_state_dict = net.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))
    write.close()

print("Finish training")





