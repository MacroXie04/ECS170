"""MLP adapted from the instructor's Method_MLP/base method interface.
Uses logits with CrossEntropyLoss and fit(), preserving nn.Module.train().
"""
import time
import numpy as np
import torch
from torch import nn
from local_code.base_class.method import method

class Method_MLP(method, nn.Module):
    def __init__(self, config, device='cpu'):
        method.__init__(self, config['name'], 'Stage 2 multiclass MLP')
        nn.Module.__init__(self)
        self.config = config
        self.device = torch.device(device)
        widths = [784] + config['hidden'] + [10]
        layers = []
        for i, (a,b) in enumerate(zip(widths[:-1],widths[1:])):
            layer = nn.Linear(a,b)
            # Kaiming for ReLU hidden layers; Xavier for output logits.
            if i < len(widths)-2:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if i < len(widths)-2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self,x):
        return self.network(x)

    def fit(self,x,y):
        cfg=self.config
        opt=(torch.optim.Adam(self.parameters(),lr=cfg['lr']) if cfg['optimizer']=='Adam'
             else torch.optim.SGD(self.parameters(),lr=cfg['lr'],momentum=0.9))
        criterion=nn.CrossEntropyLoss() if cfg['loss']=='CE' else nn.MSELoss()
        x=torch.from_numpy(x).to(self.device); y=torch.from_numpy(y).to(self.device)
        generator=torch.Generator().manual_seed(cfg['seed'])
        history=[]; start=time.perf_counter()
        for epoch in range(1,cfg['epochs']+1):
            self.train()
            order=torch.randperm(len(y),generator=generator).to(self.device)
            total=0.0
            for ids in order.split(cfg['batch_size']):
                opt.zero_grad(set_to_none=True)
                logits=self(x[ids])
                loss=(criterion(logits,y[ids]) if cfg['loss']=='CE' else
                      criterion(logits.softmax(1),nn.functional.one_hot(y[ids],10).float()))
                loss.backward(); opt.step()
                total+=float(loss.detach().cpu())*len(ids)
            history.append({'epoch':epoch,'train_loss':total/len(y)})
            print(f"{cfg['name']} epoch {epoch:02}: loss={total/len(y):.6f}",flush=True)
        return history,time.perf_counter()-start

    @torch.no_grad()
    def predict(self,x):
        self.eval()
        return np.concatenate([self(torch.from_numpy(b).to(self.device)).argmax(1).cpu().numpy()
                               for b in np.array_split(x,max(1,(len(x)+1023)//1024))])

    def run(self):
        history,seconds=self.fit(self.data['train']['X'],self.data['train']['y'])
        return {'pred_y':self.predict(self.data['test']['X']),
                'true_y':self.data['test']['y'],'history':history,'seconds':seconds}
