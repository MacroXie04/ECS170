"""Compare the decomposed graph and native EMA formula with PyTorch reference."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import mindspore as ms
from mindspore import Tensor,ops,nn
from decomposed_bn import ExplicitBatchNorm
p=argparse.ArgumentParser();p.add_argument('reference');a=p.parse_args()
r=np.load(a.reference);ms.set_context(mode=ms.GRAPH_MODE)
bn=ExplicitBatchNorm(3);bn.set_train();x=Tensor(r['x']);dy=Tensor(r['dy'])
def check(key,value):
 error=float(np.max(np.abs(value-r[key])));assert error<2e-5,(key,error);print(key,error,flush=True)
check('y',bn(x).asnumpy())
class Loss(nn.Cell):
 def __init__(self,network):super().__init__();self.network=network
 def construct(self,x,dy):return (self.network(x)*dy).sum()
inputs,weights=ops.GradOperation(get_all=True,get_by_list=True)(Loss(bn),bn.trainable_params())(x,dy)
check('dx',inputs[0].asnumpy());check('dgamma',weights[0].asnumpy());check('dbeta',weights[1].asnumpy())
mean=np.zeros(3,np.float32);var=np.ones(3,np.float32)
for _ in range(3):
 bn(x);mean=.9*mean+.1*bn.moving_mean.asnumpy();var=.9*var+.1*bn.moving_variance.asnumpy()
check('mean',mean);check('var',var)
print('BN_FORWARD_GRADIENT_AND_THREE_STEP_EMA_PASS')
