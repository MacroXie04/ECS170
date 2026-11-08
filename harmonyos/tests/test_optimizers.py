from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import mindspore as ms
from mindspore import Parameter,Tensor
from export_models import TorchAdam,InputRateSGD
ms.set_context(mode=ms.GRAPH_MODE)
for kind in ['adam','sgd']:
 p=Parameter(Tensor(np.array([1.,-2.,0.25],np.float32)),name='test_weight')
 opt=TorchAdam([p],0.001,1e-4) if kind=='adam' else InputRateSGD([{'params':[p],'weight_decay':5e-4}],learning_rate=0.05,momentum=0.9,nesterov=True)
 expected=p.asnumpy().astype(np.float64);m=np.zeros(3);v=np.zeros(3)
 for t in range(1,4):
  g=np.array([0.2*t,-0.3,0.4],np.float32);lr=(0.001 if kind=='adam' else 0.05)/t
  gg=g.astype(np.float64)+(1e-4 if kind=='adam' else 5e-4)*expected
  if kind=='adam':
   m=.9*m+.1*gg;v=.999*v+.001*gg*gg;expected-=lr*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
  else:
   m=gg if t==1 else .9*m+gg;expected-=lr*(gg+.9*m)
  opt((Tensor(g),),Tensor(np.array([lr],np.float32)))
  error=float(np.max(np.abs(p.asnumpy()-expected)));assert error<2e-6,(kind,t,error)
  print(kind,t,error,flush=True)
print('OPTIMIZER_AND_EXTERNAL_LR_PASS',flush=True)
