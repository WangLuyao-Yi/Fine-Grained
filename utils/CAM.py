from torchvision import transforms
from PIL import Image
import CUB200
import torch
import Model
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

resume = "/home/luyaowan/Data/pycharmprojects/Fine-Grained-master/ckpt/metric_Net/005.ckpt"

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448), Image.BILINEAR),
    # transforms.RandomCrop(size=448),
    transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_set = CUB200.CUB(root='./../DATA/CUB_200_2011', is_train=False, transform=transforms.ToTensor(), data_len=100)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=8, drop_last=False)

net = Model.GAPNet()
net = net.cuda()
ckpt = torch.load(resume)
net.load_state_dict(ckpt['net_state_dict'])

net.eval()
data = next(iter(test_loader))
img, label = next(iter(test_loader))
ori_img = img.squeeze()
ori_img = transforms.ToPILImage()(ori_img)
# plt.imshow(ori_img)
# plt.show()

img = img.squeeze()
img = test_transform(img)
img, label = img.cuda(), label.cuda()
img = img.unsqueeze(0)

# hook the feature extractor

features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(input[0].data.cpu().numpy())


net._modules.get("conv_max_1").register_forward_hook(hook_feature)

avg1_logit, max1_logit, predict_logit = net(img)

parm = {}
for name, parameters in net.named_parameters():
    # print(name,':',parameters.size())
    parm[name] = parameters.detach().cpu().numpy()

weight_max = parm['conv_avg_1.weight'].squeeze()
weight_avg = parm['conv_max_1.weight'].squeeze()


def returnCAM(feature_conv, weight, idx):
    # generate the calss activate maps sample to 448x448
    size_upsample = (448, 448)
    bz, nc, h, w = feature_conv.shape
    cam = weight[idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, size_upsample)
    return cam_img


avg_x = F.softmax(avg1_logit, dim=1).data.squeeze()
probs, idx = avg_x.sort(0, True)
avg_probs = probs.cpu().numpy()
avg_idx = idx.cpu().numpy()

max_x = F.softmax(max1_logit, dim=1).data.squeeze()
max_probs, max_idx = avg_x.sort(0, True)
max_probs = probs.cpu().numpy()
max_idx = idx.cpu().numpy()

predict_logit = F.softmax(predict_logit, dim=1).data.squeeze()
probs, idx = predict_logit.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()

# output the prediction
for i in range(0, 5):
    print('{:.3f}->{}'.format(avg_probs[i], avg_idx[i]))
for i in range(0, 5):
    print('{:.3f}->{}'.format(max_probs[i], max_idx[i]))

for i in range(0, 5):
    print('{:.3f}->{}'.format(probs[i], idx[i]))

avgCAM = returnCAM(features_blobs[0], weight_avg, idx[0])

maxCAM = returnCAM(features_blobs[0], weight_max, idx[0])

width, height = ori_img.size
avgheatmap = cv2.applyColorMap(cv2.resize(avgCAM, (width, height)), 2)
maxheatmap = cv2.applyColorMap(cv2.resize(maxCAM, (width, height)), 2)
ori_img = np.array(ori_img)
avgresult = avgheatmap * 0.3 + ori_img * 0.5
maxresult = avgheatmap * 0.3 + ori_img * 0.5
cv2.imwrite('avgresult.jpg', avgresult)
cv2.imwrite('maxresult.jpg', maxresult)
