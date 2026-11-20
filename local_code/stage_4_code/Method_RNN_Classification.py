import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from local_code.base_class.method import method


class Method_RNN_Classification(method,nn.Module):
    def __init__(self,mName,mDescription,vocab_size,mode='rnn',emb_dim=50,hidden_size=128,epochs=5,batch_size=64,learning_rate=0.001,device='cpu',embeddings=None):
        method.__init__(self,mName,mDescription)
        nn.Module.__init__(self)
        self.mode=mode.lower()
        self.epochs=epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.device=device
        self.history=[]
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
        if self.mode=='lstm':
            self.rnn=nn.LSTM(emb_dim,hidden_size,batch_first=True)
        elif self.mode=='gru':
            self.rnn=nn.GRU(emb_dim,hidden_size,batch_first=True)
        elif self.mode=='rnn':
            self.rnn=nn.RNN(emb_dim,hidden_size,batch_first=True)
        else:
            raise ValueError('Unknown mode: '+self.mode)
        self.dropout=nn.Dropout(0.2)
        self.fc=nn.Linear(hidden_size,2)
        self.to(device)

    def forward(self,x,lengths):
        emb=self.embedding(x)
        packed=nn.utils.rnn.pack_padded_sequence(emb,lengths.cpu(),batch_first=True,enforce_sorted=False)
        output,h=self.rnn(packed)
        if isinstance(h,tuple):
            h=h[0]
        return self.fc(self.dropout(h[-1]))

    def fit(self,data):
        x=torch.from_numpy(data['X'])
        y=torch.from_numpy(data['y'])
        n=torch.from_numpy(data['lengths'])
        loader=DataLoader(TensorDataset(x,y,n),batch_size=self.batch_size,shuffle=True)
        optimizer=torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        criterion=nn.CrossEntropyLoss()
        for epoch in range(1,self.epochs+1):
            self.train()
            loss_sum=0.0
            correct=0
            total=0
            for x,y,n in loader:
                x=x.to(self.device)
                y=y.to(self.device)
                logits=self.forward(x,n)
                loss=criterion(logits,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum+=loss.item()*len(y)
                correct+=(logits.argmax(1)==y).sum().item()
                total+=len(y)
            record={'epoch':epoch,'train_loss':loss_sum/total,'train_accuracy':correct/total}
            self.history.append(record)
            print(f'Epoch {epoch}/{self.epochs}: Loss={record["train_loss"]:.4f}, Accuracy={record["train_accuracy"]:.4f}',flush=True)

    @torch.no_grad()
    def test(self,data):
        self.eval()
        x=torch.from_numpy(data['X'])
        n=torch.from_numpy(data['lengths'])
        loader=DataLoader(TensorDataset(x,n),batch_size=self.batch_size)
        preds=[]
        for x,n in loader:
            logits=self.forward(x.to(self.device),n)
            preds.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(preds)

    def run(self):
        self.fit(self.data['train'])
        pred=self.test(self.data['test'])
        return {'pred_y':pred,'true_y':self.data['test']['y']}
