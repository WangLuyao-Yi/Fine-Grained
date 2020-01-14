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

a = tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
b = tensor([[4, 3, 2, 1, ], [1, 2, 3, 4]])
a = a.cuda()
b = b.cuda()
print(a)
a_soft = F.softmax(a, 1)
print(a_soft)
