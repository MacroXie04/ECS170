"""Compatibility probe for the full PR6 CIFAR baseline; no host training.

A successful export/conversion is NOT evidence that the device supports the graph.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mindspore as ms
from decomposed_bn import ExplicitBatchNorm
from mindspore import nn, Tensor
from mindspore.ops import functional as F
from mindspore.nn.optim.sgd import _sgd_opt

class FixedRateSGD(nn.SGD):
    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        return self.hyper_map_reverse(F.partial(_sgd_opt, self.momentum, self.get_lr()),
                                      gradients, self._parameters, self.accum, self.stat, self.opt)

class CNN(nn.Cell):
    def __init__(self, c, shape, classes):
        super().__init__()
        ch,h,w = shape
        layers=[]
        for out in c['channels']:
            for i in range(c['convs_per_stage']):
                stride=2 if c['downsample']=='stride' and i==c['convs_per_stage']-1 else 1
                layers.append(nn.Conv2d(ch,out,c['kernel_size'],stride=stride,
                    pad_mode='pad',padding=c['padding'],has_bias=not c['batch_norm']))
                h=(h+2*c['padding']-c['kernel_size'])//stride+1
                w=(w+2*c['padding']-c['kernel_size'])//stride+1
                if c['batch_norm']:
                    layers.append(ExplicitBatchNorm(out))
                layers.append(nn.ReLU());ch=out
            if c['downsample']!='stride':
                pool=nn.MaxPool2d if c['downsample']=='maxpool' else nn.AvgPool2d
                layers.append(pool(2,2));h//=2;w//=2
        self.features=nn.SequentialCell(layers)
        layers=[nn.Flatten()];width=ch*h*w
        for hidden in c['hidden_dims']:
            layers.extend([nn.Dense(width,hidden),nn.ReLU()])
            if c['dropout']:layers.append(nn.Dropout(p=c['dropout']))
            width=hidden
        layers.append(nn.Dense(width,classes));self.classifier=nn.SequentialCell(layers)
    def construct(self,x):
        return self.classifier(self.features(x))

def main():
    p=argparse.ArgumentParser();p.add_argument('protocol');p.add_argument('output');a=p.parse_args()
    c=json.loads(Path(a.protocol).read_text())['runs'][0]['config']
    assert c['dataset']=='cifar' and c['name']=='baseline'
    ms.set_context(mode=ms.GRAPH_MODE,device_target='CPU');ms.set_seed(2026)
    net=CNN(c,(3,32,32),10)
    opt=FixedRateSGD(net.trainable_params(),learning_rate=c['learning_rate'],
         momentum=c['momentum'],weight_decay=c['weight_decay'],nesterov=c['nesterov'])
    train=nn.TrainOneStepCell(nn.WithLossCell(net,nn.SoftmaxCrossEntropyWithLogits(sparse=False,reduction='mean')),opt)
    train.set_train()
    ms.export(train,Tensor(np.zeros((128,3,32,32),np.float32)),
         Tensor(np.eye(10,dtype=np.float32)[np.zeros(128,dtype=np.int32)]),
         file_name=a.output,file_format='MINDIR')
    print('COMPATIBILITY_GRAPH_EXPORTED',sum(p.size for p in net.trainable_params()))

if __name__=='__main__':main()
