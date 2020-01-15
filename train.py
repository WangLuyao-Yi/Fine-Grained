import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
import torchvision.transforms as transforms
from dataset import CUB200
import Model
from PIL import Image
from config import BATCH_SIZE, LR, resume, save_dir, WD, end_epoch, SAVE_FREQ, tesorboard_dir
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

cuda_flag = torch.cuda.is_available()
start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)

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
train_set = CUB200.CUB(root='./DATA/CUB_200_2011', is_train=True, transform=train_transform, data_len=None)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                           drop_last=False)

test_set = CUB200.CUB(root='./DATA/CUB_200_2011', is_train=False, transform=test_transform, data_len=None)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)

# define model
net = Model.GAPNet()  # Model.MyNet()

write = SummaryWriter(tesorboard_dir)
images = torch.zeros((BATCH_SIZE, 3, 448, 448))
write.add_graph(net, (images,))

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
        img, label = data[0], data[1]
        if cuda_flag:
            img, label = img.cuda(), label.cuda()

        batch_size = img.size(0)
        optimizer.zero_grad()
        avg1,max1,target = net(img)
        avg1_loss = criterion(avg1,label)
        max1_loss = criterion(max1,label)
        concat_loss = criterion(target,label)
        metric_loss = Model.metric_loss(avg1, max1)
        loss = avg1_loss + max1_loss + concat_loss + 5.0 * metric_loss
        loss.backward()
        optimizer.step()

    if epoch % SAVE_FREQ == 0:
        train_loss = 0
        train_correct = 0
        total = 0
        train_avg_loss = 0
        train_max_loss = 0
        train_concat_loss = 0
        train_metric_loss = 0
        net.eval()
        train_bar = tqdm(train_loader)
        for data in train_bar:
            train_bar.set_description("epoch %d: Traning eval" % epoch)
            with torch.no_grad():
                img, label = data[0], data[1]
                if cuda_flag:
                    img, label = img.cuda(), label.cuda()
                batch_size = img.size(0)
                avg1, max1, target = net(img)
                avg1_loss = criterion(avg1, label)
                max1_loss = criterion(max1, label)
                concat_loss = criterion(target, label)
                metric_loss = Model.metric_loss(avg1, max1)
                loss = avg1_loss + max1_loss + concat_loss + 5.0 * metric_loss


                # target = net(img)
                # #calculate loss
                # loss = criterion(target, label)
                _,predict = torch.max(target,1)
                total += batch_size
                train_correct += torch.sum(predict.data == label.data)
                train_loss += loss.item()*batch_size
                train_avg_loss += avg1_loss.item()*batch_size
                train_max_loss += max1_loss.item() * batch_size
                train_concat_loss += concat_loss.item()*batch_size
                train_metric_loss += metric_loss.item()*batch_size
        train_acc = float(train_correct)/total
        train_loss = train_loss/total
        train_avg_loss = train_avg_loss/total
        train_max_loss = train_max_loss/total
        train_concat_loss = train_concat_loss / total
        train_metric_loss = 5.0 * train_metric_loss / total
        print("epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample:{}".format(
            epoch, train_loss, train_acc, total
        ))

        #evaluate on test
        test_avg_loss = 0
        test_max_loss = 0
        test_concat_loss = 0
        test_metric_loss = 0
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
                # target = net(img)
                avg1, max1, target = net(img)
                #calculate loss
                avg1_loss = criterion(avg1, label)
                max1_loss = criterion(max1, label)
                concat_loss = criterion(target, label)
                metric_loss = Model.metric_loss(avg1, max1)
                loss = avg1_loss + max1_loss + concat_loss + 5.0 * metric_loss
                #calculate accuracy
                _,predict = torch.max(target,1)
                total += batch_size
                test_correct += torch.sum(predict.data == label.data)
                test_loss += loss.item()*batch_size
                test_avg_loss += avg1_loss.item() * batch_size
                test_max_loss += max1_loss.item() * batch_size
                test_concat_loss += concat_loss.item() * batch_size
                test_metric_loss += metric_loss.item() * batch_size
        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        test_avg_loss = test_avg_loss / total
        test_max_loss = test_max_loss / total
        test_concat_loss = test_concat_loss / total
        test_metric_loss = 5.0 * test_metric_loss / total
        print("epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample:{}".format(
            epoch,test_loss,test_acc,total
        ))

        write.add_scalars("lOSS",{'train': train_loss, "test": test_loss },epoch)
        write.add_scalars("AVG_loss", {'train': train_avg_loss, "test": test_avg_loss}, epoch)
        write.add_scalars("MAX_loss", {'train': train_max_loss, "test": test_max_loss}, epoch)
        write.add_scalars("Cat_loss", {'train': train_concat_loss, "test": test_concat_loss}, epoch)
        write.add_scalars("Metric_loss", {'train': train_metric_loss, "test": test_metric_loss}, epoch)
        write.add_scalars("Accuracy",{"train": train_acc, "test": test_acc},epoch)

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
