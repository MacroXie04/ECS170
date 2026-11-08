"""Build random-initialized training graphs. Never execute an optimizer step here."""
from pathlib import Path
import json,math
import numpy as np
import mindspore as ms
from mindspore import nn,Tensor
from mindspore.ops import functional as F
from mindspore.nn.optim.sgd import _sgd_opt
ms.set_context(mode=ms.GRAPH_MODE,device_target='CPU')
class FixedRateSGD(nn.SGD):
 def construct(self, gradients):
  gradients=self.flatten_gradients(gradients)
  return self.hyper_map_reverse(F.partial(_sgd_opt,self.momentum,self.get_lr()),gradients,self._parameters,self.accum,self.stat,self.opt)
class CNN(nn.Cell):
 def __init__(self,c):
  super().__init__();layers=[];ch=c['inputChannels'];h=c['height'];w=c['width']
  for out in c['channels']:
   layers.extend([nn.Conv2d(ch,out,3,pad_mode='pad',padding=1,has_bias=True,weight_init='HeNormal'),nn.ReLU(),nn.MaxPool2d(2,2)])
   ch=out;h=h//2;w=w//2
  self.features=nn.SequentialCell(layers);self.flatten=nn.Flatten();self.fc1=nn.Dense(ch*h*w,c['hidden'],weight_init='HeNormal');self.relu=nn.ReLU();self.fc2=nn.Dense(c['hidden'],c['classes'],weight_init='HeNormal')
 def construct(self,x):return self.fc2(self.relu(self.fc1(self.flatten(self.features(x)))))
root=Path('/work/stage3_models');root.mkdir(exist_ok=True)
for c in json.loads(Path('/work/stage3_plan.json').read_text()):
 ms.set_seed(c['seed']);net=CNN(c)
 optimizer=FixedRateSGD(net.trainable_params(),learning_rate=c['learningRate'],momentum=0.9)
 train=nn.TrainOneStepCell(nn.WithLossCell(net,nn.SoftmaxCrossEntropyWithLogits(sparse=False,reduction='mean')),optimizer);train.set_train()
 x=Tensor(np.zeros((c['batchSize'],c['inputChannels'],c['height'],c['width']),np.float32));y=Tensor(np.eye(c['classes'],dtype=np.float32)[np.zeros(c['batchSize'],np.int32)])
 ms.export(train,x,y,file_name=str(root/c['id']),file_format='MINDIR')
 print('EXPORTED',c['id'],'parameters',sum(p.size for p in net.trainable_params()),flush=True)
