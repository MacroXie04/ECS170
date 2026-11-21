import re
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from local_code.base_class.method import method
from local_code.stage_4_code.xiaoyu.Dataset_Loader import tokenize


class Method_RNN_Generation(method,nn.Module):
    def __init__(self,mName,mDescription,vocab,mode='rnn',emb_dim=50,hidden_size=128,epochs=15,batch_size=32,learning_rate=0.001,device='cpu',embeddings=None):
        method.__init__(self,mName,mDescription)
        nn.Module.__init__(self)
        self.vocab=vocab
        self.rev={i:w for w,i in vocab.items()}
        self.mode=mode.lower()
        self.epochs=epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.device=device
        self.history=[]
        self.embedding=nn.Embedding(len(vocab),emb_dim,padding_idx=0)
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
        self.fc=nn.Linear(hidden_size,len(vocab))
        self.to(device)

    def forward(self,x,state=None):
        h,state=self.rnn(self.embedding(x),state)
        return self.fc(h),state

    def fit(self,data):
        sequences=[s for s in data['sequences'] if len(s)>1]
        width=max(len(s) for s in sequences)-1
        x=torch.zeros((len(sequences),width),dtype=torch.long)
        y=torch.zeros((len(sequences),width),dtype=torch.long)
        for i,s in enumerate(sequences):
            x[i,:len(s)-1]=torch.tensor(s[:-1])
            y[i,:len(s)-1]=torch.tensor(s[1:])
        loader=DataLoader(TensorDataset(x,y),batch_size=self.batch_size,shuffle=True)
        optimizer=torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        criterion=nn.CrossEntropyLoss(ignore_index=0)
        for epoch in range(1,self.epochs+1):
            self.train()
            loss_sum=0.0
            count=0
            for x,y in loader:
                x=x.to(self.device)
                y=y.to(self.device)
                logits,_=self.forward(x)
                loss=criterion(logits.reshape(-1,len(self.vocab)),y.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                words=(y!=0).sum().item()
                loss_sum+=loss.item()*words
                count+=words
            record={'epoch':epoch,'train_loss':loss_sum/count}
            self.history.append(record)
            print(f'Epoch {epoch}/{self.epochs}: Loss={record["train_loss"]:.4f}',flush=True)

    @torch.no_grad()
    def generate(self,words,max_words=40,temperature=0.8):
        self.eval()
        tokens=tokenize(words)
        if len(tokens)!=3:
            raise ValueError('Exactly three starting words are needed')
        x=torch.tensor([[self.vocab.get(w,1) for w in tokens]],device=self.device)
        logits,state=self.forward(x)
        output=list(tokens)
        for i in range(max_words):
            scores=logits[:,-1,:]/temperature
            scores[:,0]=-1e9
            scores[:,1]=-1e9
            nextid=torch.multinomial(torch.softmax(scores,dim=1),1).item()
            if nextid==2:
                break
            output.append(self.rev[nextid])
            logits,state=self.forward(torch.tensor([[nextid]],device=self.device),state)
        result=' '.join(output)
        return re.sub(r'\s+([.!?,;:])',r'\1',result)

    def run(self):
        self.fit(self.data)
        return {'train_loss':self.history[-1]['train_loss']}
