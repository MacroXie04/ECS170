"""Export small reproducible arithmetic diagnostics for the physical-device app."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from export_models import nn,ops,ms,np,Tensor,Parameter,InputRateSGD,TorchAdam,TrainWithLR
from decomposed_bn import ExplicitBatchNorm
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--reference',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
ms.set_context(mode=ms.GRAPH_MODE)
class DotLoss(nn.Cell):
 def construct(self,pred,target):return (pred*target).sum()
r=np.load(a.reference);net=ExplicitBatchNorm(3)
opt=InputRateSGD(net.trainable_params(),learning_rate=.01,momentum=0.)
train=TrainWithLR(nn.WithLossCell(net,DotLoss()),opt);train.set_train()
ms.export(train,Tensor(r['x']),Tensor(r['dy']),Tensor(np.array([.01],np.float32)),file_name=str(a.output/'bn_diagnostic'),file_format='MINDIR')
class Linear(nn.Cell):
 def __init__(self):
  super().__init__();self.weight=Parameter(Tensor(np.array([[1.],[-2.],[.25]],np.float32)),name='weight')
 def construct(self,x):return ops.matmul(x,self.weight)
net=Linear();opt=TorchAdam(net.trainable_params(),.001,1e-4)
train=TrainWithLR(nn.WithLossCell(net,nn.MSELoss()),opt);train.set_train()
ms.export(train,Tensor(np.array([[.2,-.3,.4]],np.float32)),Tensor(np.array([[.5]],np.float32)),Tensor(np.array([.001],np.float32)),file_name=str(a.output/'optimizer_diagnostic'),file_format='MINDIR')
net=Linear();opt=InputRateSGD([{'params':net.trainable_params(),'weight_decay':5e-4}],learning_rate=.05,momentum=.9,nesterov=True)
train=TrainWithLR(nn.WithLossCell(net,nn.MSELoss()),opt);train.set_train()
ms.export(train,Tensor(np.array([[.2,-.3,.4]],np.float32)),Tensor(np.array([[.5]],np.float32)),Tensor(np.array([.05],np.float32)),file_name=str(a.output/'optimizer_sgd_diagnostic'),file_format='MINDIR')
