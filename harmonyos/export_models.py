"""Export exact PR6 initialization and train/evaluation graphs; no host training."""
import argparse,json
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore import nn,ops,Tensor,Parameter
from export_pr6_probe import CNN,FixedRateSGD
from mindspore.nn.optim.sgd import _sgd_opt
from mindspore.ops import functional as F

class TorchAdam(nn.Optimizer):
 def __init__(self,params,lr,decay):
  super().__init__(lr,params,weight_decay=0.0)
  self.m=self.parameters.clone(prefix='adam_m',init='zeros');self.v=self.parameters.clone(prefix='adam_v',init='zeros')
  self.b1=Parameter(Tensor(np.ones(1,np.float32)),name='beta1_power',requires_grad=False)
  self.b2=Parameter(Tensor(np.ones(1,np.float32)),name='beta2_power',requires_grad=False)
  self.decay=decay;self.adam=ops.Adam();self.beta1=Tensor(0.9,ms.float32);self.beta2=Tensor(0.999,ms.float32)
 def construct(self,grads,lr):
  b1=self.b1*0.9;b2=self.b2*0.999;success=True
  eps=1e-8*ops.sqrt(1.0-b2)
  for p,m,v,g in zip(self.parameters,self.m,self.v,grads):
   g=g+self.decay*p
   success=ops.depend(success,self.adam(p,m,v,b1,b2,lr,self.beta1,self.beta2,eps,g))
  success=ops.depend(success,ops.assign(self.b1,b1));success=ops.depend(success,ops.assign(self.b2,b2))
  return success

class InputRateSGD(FixedRateSGD):
 def construct(self,grads,lr):
  grads=self.decay_weight(grads)
  return self.hyper_map_reverse(F.partial(_sgd_opt,self.momentum,lr),grads,self._parameters,self.accum,self.stat,self.opt)

class TrainWithLR(nn.Cell):
 def __init__(self,network,optimizer):
  super().__init__(auto_prefix=False);self.network=network;self.optimizer=optimizer
  self.network.set_grad();self.weights=optimizer.parameters;self.grad=ops.GradOperation(get_by_list=True)
 def construct(self,x,y,lr):
  loss=self.network(x,y);grads=self.grad(self.network,self.weights)(x,y)
  return ops.depend(loss,self.optimizer(grads,lr))

class Loss(nn.Cell):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.ce=nn.SoftmaxCrossEntropyWithLogits(sparse=False,reduction='mean')
 def construct(self,logits,y):
  if self.kind=='cross_entropy':return self.ce(logits,y)
  diff=ops.softmax(logits,axis=1)-y
  return ops.mean(ops.sum(diff*diff,dim=1))

def initialize(net,source):
 state=np.load(source)
 for name,param in net.parameters_and_names():
  key=name
  if name.endswith('.gamma'):key=name[:-6]+'.weight'
  elif name.endswith('.beta'):key=name[:-5]+'.bias'
  elif name.endswith('.moving_mean'):key=name[:-12]+'.running_mean'
  elif name.endswith('.moving_variance'):key=name[:-16]+'.running_var'
  assert key in state,(name,key)
  assert tuple(state[key].shape)==tuple(param.shape),(name,state[key].shape,param.shape)
  param.set_data(Tensor(state[key]))

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--run',required=True);a=p.parse_args();root=a.root
 run=next(r for r in json.loads((root/'protocol.json').read_text())['runs'] if r['id']==a.run)
 c=run['config'];meta=json.loads((root/'data/manifest.json').read_text())[c['dataset']]
 shape=meta['input_shape'];classes=40 if c['dataset']=='orl' else 10
 out=root/'models'/run['id'];out.mkdir(parents=True,exist_ok=True)
 bs=c['batch_size'];rem=meta['optimization_samples']%bs
 ms.set_context(mode=ms.GRAPH_MODE,device_target='CPU')
 variables=set();learned=set()
 for mode,batch in [('train',bs)]+([('tail',rem)] if rem else [])+[('eval',bs)]:
  ms.set_seed(run['seed']);net=CNN(c,shape,classes);initialize(net,root/'models/initial'/(run['id']+'.npz'))
  learned.update(p.name for p in net.trainable_params())
  if mode=='eval':
   for name,param in net.parameters_and_names():
    if name.endswith(('.moving_mean','.moving_variance')):param.requires_grad=True
  if c['optimizer']=='sgd':
   opt=InputRateSGD([{'params':net.trainable_params(),'weight_decay':c['weight_decay']}],learning_rate=c['learning_rate'],momentum=c['momentum'],nesterov=c['nesterov'])
  else:opt=TorchAdam(net.trainable_params(),c['learning_rate'],c['weight_decay'])
  train=TrainWithLR(nn.WithLossCell(net,Loss(c['loss'])),opt);train.set_train(mode!='eval')
  variables.update(p.name for p in train.get_parameters())
  ms.export(train,Tensor(np.zeros((batch,*shape),np.float32)),Tensor(np.eye(classes,dtype=np.float32)[np.zeros(batch,np.int32)]),Tensor(np.array([c['learning_rate']],np.float32)),file_name=str(out/mode),file_format='MINDIR')
  print('EXPORTED',run['id'],mode,batch,flush=True)
 (out/'variables.json').write_text(json.dumps({'variables':sorted(variables),'learned':sorted(learned)},indent=2))
if __name__=='__main__':main()
