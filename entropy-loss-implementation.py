import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


criterion = HLoss()

x = Variable(torch.randn(10, 10))
# w = Variable(torch.randn(10, 3), requires_grad=True)
w = nn.Parameter(torch.randn(10, 3), requires_grad=True)
# optimizer = Adam([nn.Parameter(w)], lr=1)
optimizer = Adam([w], lr=1)

for i in range(10):
    optimizer.zero_grad()
    output = torch.matmul(x, w)
    # print("===========")
    print(output)

    loss = criterion(output)
    print("===========")
    print(loss)
    # print("===========")

    loss.backward()
    # print(w.grad)
    optimizer.step()
