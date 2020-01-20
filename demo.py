import numpy as np
import torch
import time
from tqdm import tqdm
import torch.tensor as  tensor
import torch.nn.functional as F

# list  = np.arange(1,1000,1000)
#
# for i,data in tqdm(range(1000)):
#     time.sleep(0.01)



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

a = tensor([[0.9, 0.5, 0, 0, 0], [0.5, 0.3, 0.2, 0, 0]])

print(Variance_loss(a,0.2))
# for i in range(a.size(0)):
#     print(a[i] > 0.4)
#     mask = (a[i] > 0.4).nonzero()
#     print(mask.size())
#     prob_i = a[i].index_select(0,mask[:,0])
#     print(prob_i)
#     variance = torch.norm(prob_i - torch.mean(prob_i), p=2)
#     # variance = torch.var(prob_i, 0)
#     # print(variance)
#     # variance = torch.rsqrt(variance)
#     variance = torch.log(variance)
#     print(variance)

# a1=torch.tensor([5.0])
# print(a1-torch.mean(a1))
# print(torch.norm(a1-torch.mean(a1),p=2))
# print(torch.log(1/torch.norm(a1-torch.mean(a1),p=2)))
# print(torch.var(a1,0))
# print(a[0].index_select(0,torch.tensor([0,2,4])))



# print(torch.index_select(a, mask))
# print(a.index_select(0,mask))








#方差
var = torch.var(a,1)

var2 = torch.var_mean(a,1)
#开根号倒数
var3 = torch.rsqrt(var)
b = torch.nn.functional.softmax(a,1)






# b = tensor([[4, 3, 2, 1, ], [1, 2, 3, 4]])
# a = a.cuda()
# b = b.cuda()
# print(a)
# a_soft = F.softmax(a, 1)
# print(a_soft)
