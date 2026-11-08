"""Synthetic arithmetic test only; no Stage 3 data or formal host training."""
import argparse
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args()
rng=np.random.default_rng(170)
x=torch.tensor(rng.standard_normal((4,3,5,5)).astype(np.float32),requires_grad=True)
dy=torch.tensor(rng.standard_normal((4,3,5,5)).astype(np.float32))
bn=torch.nn.BatchNorm2d(3,eps=1e-5,momentum=.1)
opt=torch.optim.SGD(bn.parameters(),lr=.01)
ref={'x':x.detach().numpy(),'dy':dy.numpy()}
for step in range(3):
 opt.zero_grad();x.grad=None;y=bn(x);(y*dy).sum().backward()
 if step==0:
  ref.update(y=y.detach().numpy(),dx=x.grad.numpy().copy(),dgamma=bn.weight.grad.numpy().copy(),dbeta=bn.bias.grad.numpy().copy())
 opt.step()
ref.update(gamma=bn.weight.detach().numpy(),beta=bn.bias.detach().numpy(),mean=bn.running_mean.numpy(),var=bn.running_var.numpy())
np.savez(a.output,**ref)
