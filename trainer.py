import os
import tqdm
import torch
from config import th



def Variance_loss(predict,th):
    batch_size = predict.size(0)
    loss = 0.0
    for i in range(batch_size):
        mask = (predict[i] > th).nonzero()
        prob_i = predict[i].index_select(0, mask[:, 0])
        if prob_i.size(0)==1:
            variance = torch.tensor(0)
        else:
            variance = torch.var(prob_i, 0)
            variance = torch.rsqrt(variance)
            variance = torch.log(variance)
        print(variance)
        loss += variance
    return loss/batch_size


def train(net,
          epoch_num,
          start_epoch,
          optimizer,
          schedulers,
          data_loader,
          best_acc,
          write,
          save_dir,
          save_freq):

    criterion = torch.nn.CrossEntropyLoss()
    cuda_flag = torch.cuda.is_available()

    for epoch in range(start_epoch, epoch_num + 1):
        for scheduler in schedulers:
            scheduler.step()
            print(scheduler.get_lr())

        # begin training
        net.train()
        train_bar = tqdm(data_loader['train'])
        for data in train_bar:
            train_bar.set_description("epoch %d :Training " % epoch)
            img, label = data[0], data[1]
            if cuda_flag:
                img, label = img.cuda(), label.cuda()
            batch_size = img.size(0)
            optimizer.zero_grad()
            target = net(img)
            raw_loss = criterion(target, label)
            var_loss = Variance_loss(target, th)
            loss = raw_loss + var_loss
            loss.backward()
            optimizer['raw'].step()

        if epoch % save_freq == 0:
            train_loss = 0
            train_raw_loss = 0
            train_var_loss = 0
            train_correct = 0
            total = 0
            net.eval()
            train_bar = tqdm(data_loader['train'])
            for data in train_bar:
                train_bar.set_description("epoch %d: Traning eval" % epoch)
                with torch.no_grad():
                    img, label = data[0], data[1]
                    if cuda_flag:
                        img, label = img.cuda(), label.cuda()
                    batch_size = img.size(0)
                    target = net(img)
                    raw_loss = criterion(target, label)
                    var_loss = Variance_loss(target,th)
                    loss = raw_loss + var_loss

                    # target = net(img)
                    # #calculate loss
                    # loss = criterion(target, label)
                    _, predict = torch.max(target, 1)
                    total += batch_size
                    train_correct += torch.sum(predict.data == label.data)
                    train_loss += loss.item() * batch_size
                    train_raw_loss += raw_loss.item() * batch_size
                    train_var_loss += var_loss.item() * batch_size



            train_acc = float(train_correct) / total
            train_loss = train_loss / total
            train_raw_loss = train_raw_loss / total
            train_var_loss = train_var_loss / total

            print("epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample:{}".format(
                epoch, train_loss, train_acc, total
            ))

            # evaluate on test
            test_loss = 0
            test_correct = 0
            total = 0
            test_raw_loss = 0
            test_var_loss = 0

            test_bar = tqdm(data_loader['test'])
            for data in test_bar:
                test_bar.set_description("epoch %d: Testing eval" % epoch)
                with torch.no_grad():
                    img, label = data[0], data[1]
                    if cuda_flag:
                        img, label = img.cuda(), label.cuda()
                    batch_size = img.size(0)
                    # target = net(img)
                    target = net(img)
                    # calculate loss
                    raw_loss = criterion(target, label)
                    var_loss = Variance_loss(target, th)
                    loss = raw_loss + var_loss
                    # calculate accuracy
                    _, predict = torch.max(target, 1)
                    total += batch_size
                    test_correct += torch.sum(predict.data == label.data)
                    test_loss += loss.item() * batch_size
                    test_raw_loss += raw_loss.item() * batch_size
                    test_var_loss += var_loss.item() * batch_size

            test_acc = float(test_correct) / total
            test_loss = test_loss / total
            test_raw_loss = test_raw_loss / total
            test_var_loss = test_var_loss / total
            print("epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample:{}".format(
                epoch, test_loss, test_acc, total
            ))
            write.add_scalars("var_loss", {'train': train_var_loss, "test": test_var_loss}, epoch)
            write.add_scalars("lOSS", {'train': train_loss, "test": test_loss}, epoch)
            write.add_scalars("raw_loss",{'train':train_raw_loss, "test": train_raw_loss}, epoch)
            write.add_scalars("Accuracy", {"train": train_acc, "test": test_acc}, epoch)

            # save model
            if test_acc > best_acc:
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
                    'net_state_dict': net_state_dict,
                    'best_epoch': best_epoch},
                    os.path.join(save_dir, '%03d.ckpt' % epoch))
    write.close()
    print("Finish training")