"""BN graph writes raw batch statistics; the native bridge applies the EMA.

Keeping old-statistic multiplication outside the graph avoids Lite constant folding.
Forward values and parameter gradients remain the original BatchNorm equations.
"""
import mindspore as ms
from mindspore import nn,ops,Tensor,Parameter
import numpy as np

class SafeRsqrt(nn.Cell):
 def construct(self,x):return ops.rsqrt(x)
 def bprop(self,x,out,dout):return (-0.5*dout*out*out*out,)

class ExplicitBatchNorm(nn.Cell):
 def __init__(self,channels):
  super().__init__();self.gamma=Parameter(Tensor(np.ones(channels,np.float32)),name='gamma');self.beta=Parameter(Tensor(np.zeros(channels,np.float32)),name='beta')
  self.moving_mean=Parameter(Tensor(np.zeros(channels,np.float32)),name='moving_mean',requires_grad=False)
  self.moving_variance=Parameter(Tensor(np.ones(channels,np.float32)),name='moving_variance',requires_grad=False)
  self.mean=ops.ReduceMean(keep_dims=True);self.rsqrt=SafeRsqrt()
 def construct(self,x):
  if self.training:
   mean=self.mean(x,(0,2,3));diff=x-mean;var=self.mean(diff*diff,(0,2,3))
   n=x.shape[0]*x.shape[2]*x.shape[3]
   y=diff*self.rsqrt(var+1e-5)*self.gamma.reshape((1,-1,1,1))+self.beta.reshape((1,-1,1,1))
   y=ops.depend(y,ops.assign(self.moving_mean,mean.reshape((-1,))))
   y=ops.depend(y,ops.assign(self.moving_variance,var.reshape((-1,))*(n/(n-1))))
   return y
  return (x-self.moving_mean.reshape((1,-1,1,1)))*self.rsqrt(self.moving_variance.reshape((1,-1,1,1))+1e-5)*self.gamma.reshape((1,-1,1,1))+self.beta.reshape((1,-1,1,1))
