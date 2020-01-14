import numpy as np
import torch
import time
from tqdm import tqdm
import torch.tensor as  tensor

# list  = np.arange(1,1000,1000)
#
# for i,data in tqdm(range(1000)):
#     time.sleep(0.01)

a = tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
b = tensor([[4, 3, 2, 1, ], [1, 2, 3, 4]])
a = a.cuda()
b = b.cuda()
print(a)
print(b)
c = torch.cat((a, b), 1)
print(c)
