from torchvision import transforms
from PIL import Image
import CUB200
import torch
import Model
import torch.nn.functional as F
import numpy as np
import cv2

resume=""

test_transform = transforms.Compose([
    transforms.Resize((600,600),Image.BILINEAR),
    transforms.RandomCrop(size=448),
    transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_set = CUB200.CUB(root='./DATA/CUB_200_2011', is_train=False, transform=None, data_len=None)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=8, drop_last=False)

net = Model.GAPNet()
net = net.cuda()
ckpt= torch.load(resume)
net.load_state_dict(ckpt['net_state_dict'])


net.eval()
img, label = next(iter(test_loader))
ori_img = img

img = test_transform(img)
img, label = img.cuda(), label.cuda()
img = test_transform(img)
logit = net(img)


# parm={}
# for name,parameters in net.named_parameters():
#     print(name,':',parameters.size())
#     parm[name]=parameters.detach().numpy()
# parm['layer1.0.conv1.weight'].size()

#hook the feature extractor
features_blobs =[]
def hook_feature(module,input,output):
    features_blobs.append(output.data.cpu().numpy)
net._modules.get("conv_max_1").regiter_forward_hook(hook_feature)

weight_max = net._parameters.get('conv_max_1.weight')
weight_avg = net._parameters.get('conv_avg_1.weight')

def returnCAM(feature_conv, weight,idx):
    #generate the calss activate maps sample to 448x448
    size_upsample = (448,448)
    bz,nc,h,w = feature_conv.shape
    cam =weight[idx].dot(feature_conv.reshape((nc,h,w)))
    cam = cam.reshape(h,w)
    cam = cam-np.min(cam)
    cam_img = cam/np.max(cam)
    cam_img = np.uint8(255*cam_img)
    cam_img = cv2.resize(cam_img,size_upsample)
    return cam_img

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
for i in range(0,5):
    print('{:.3f}->{}'.format(probs[i],idx[i]))

avgCAM = returnCAM(features_blobs,weight_avg,idx)

maxCAM = returnCAM(features_blobs,weight_max,idx)

height, width, _= ori_img.shape
avgheatmap = cv2.applyColorMap(cv2.resize(avgCAM,(width, height)),2)
maxheatmap = cv2.applyColorMap(cv2.resize(maxCAM,(width, height)),2)
avgresult = avgheatmap*0.3 + img*0.5
maxresult = avgheatmap*0.3 + img*0.5
cv2.imwrite('avgresult.jpg',avgresult)
cv2.imwrite('maxresult.jpg',maxresult)














