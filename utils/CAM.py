from torchvision import transforms
from PIL import Image
import CUB200
import torch
import Model
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

IMG_URL = "/home/luyaowan/Data/pycharmprojects/Fine-Grained-master/DATA/Test_image/188_Pileated_Woodpecker_0097_180392.jpg"
resume = "/home/luyaowan/Data/pycharmprojects/Fine-Grained-master/ckpt/metric_Net5/095.ckpt"

# load model
net = Model.GAPNet()
net = net.cuda()
ckpt = torch.load(resume)
net.load_state_dict(ckpt['net_state_dict'])
net.eval()

# hook the feature extractor: extracte the last layer feature of resnet
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(input[0].data.cpu().numpy())
net._modules.get("conv_max_1").register_forward_hook(hook_feature)

#get the 1x1 kernel of conv_max_1  and conv_avg_1
parm = {}
for name, parameters in net.named_parameters():
    # print(name,':',parameters.size())
    parm[name] = parameters.detach().cpu().numpy()
weight_max = parm['conv_avg_1.weight'].squeeze()#【200，2048，1，1】get rid of the last two dim
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

def get_test_sample_from_test():
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448), Image.BILINEAR),
        # transforms.RandomCrop(size=448),
        transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_set = CUB200.CUB(root='./../DATA/CUB_200_2011', is_train=False, transform=transforms.ToTensor(), data_len=100)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=8, drop_last=False)
    img, label = next(iter(test_loader))
    ori_img = img.squeeze()
    ori_img = transforms.ToPILImage()(ori_img)
    # plt.imshow(ori_img)
    # plt.show()
    ori_img.save("test.jpg")
    img = img.squeeze()
    img = test_transform(img)
    img, label = img.cuda(), label.cuda()
    img = img.unsqueeze(0)
    width, height = ori_img.size
    ori_img = np.array(ori_img)
    return img, height, width, ori_img


def get_test_image(img_url):
    process = transforms.Compose([
        transforms.Resize((448, 448), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ori_img = Image.open(img_url)
    ori_img.save('test.jpg')
    img = process(ori_img)
    width, height = ori_img.size
    ori_img = np.array(ori_img)
    img = img.unsqueeze(0)

    return img, height, width, ori_img


img, height, width, ori_img = get_test_image(IMG_URL)
# img, height, width, ori_img = get_test_sample_from_test()
print(img.shape)
print(ori_img.shape)

img = img.cuda()

avg1_logit, max1_logit, predict_logit = net(img)

avg_x = F.softmax(avg1_logit, dim=1).data.squeeze()
avg_probs, avg_idx = avg_x.sort(0, True)
avg_probs = avg_probs.cpu().numpy()
avg_idx = avg_idx.cpu().numpy()

max_x = F.softmax(max1_logit, dim=1).data.squeeze()
max_probs, max_idx = avg_x.sort(0, True)
max_probs = max_probs.cpu().numpy()
max_idx = max_idx.cpu().numpy()

predict_logit = F.softmax(predict_logit, dim=1).data.squeeze()
probs, idx = predict_logit.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()

# output the prediction
for i in range(0, 5):
    print('avg{:.3f}->{}'.format(avg_probs[i], avg_idx[i]))
for i in range(0, 5):
    print('max{:.3f}->{}'.format(max_probs[i], max_idx[i]))
for i in range(0, 5):
    print('pred{:.3f}->{}'.format(probs[i], idx[i]))
avgCAM = returnCAM(features_blobs[0], weight_avg, idx[0])
maxCAM = returnCAM(features_blobs[0], weight_max, idx[0])

avgheatmap = cv2.applyColorMap(cv2.resize(avgCAM, (width, height)), 2)
maxheatmap = cv2.applyColorMap(cv2.resize(maxCAM, (width, height)), 2)

avgresult = avgheatmap * 0.3 + ori_img * 0.5
maxresult = avgheatmap * 0.3 + ori_img * 0.5
cv2.imwrite('avgresult.jpg', avgresult)
cv2.imwrite('maxresult.jpg', maxresult)
