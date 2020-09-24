
import torch
import torch.nn as nn
import math
class GroupLinearLayer(nn.Module):
 def __init__(self, din, dout, num_blocks, bias=True, a = None):
     super(GroupLinearLayer, self).__init__()
     self.nb = num_blocks
     #din = din // num_blocks
     #dout = dout // num_blocks
     self.dout = dout
     if a is None:
         a = 1. / math.sqrt(dout)
     self.weight = nn.Parameter(torch.FloatTensor(num_blocks,din,dout).uniform_(-a,a))
     self.bias = bias
     if bias is True:
         self.bias = nn.Parameter(torch.FloatTensor(num_blocks,dout).uniform_(-a,a))
         #self.bias = nn.Parameter(torch.zeros(dout*num_blocks))
     else:
         self.bias = None
 def forward(self,x):
     ts,bs,m = x.shape
     #x = x.reshape((ts*bs, self.nb, m//self.nb))
     x = x.permute(1,0,2)
     x = torch.bmm(x,self.weight)
     x = x.permute(1,0,2)
     if not self.bias is None:
         x = x + self.bias
     #x = x.reshape((ts, bs, self.dout*self.nb))
     return x

