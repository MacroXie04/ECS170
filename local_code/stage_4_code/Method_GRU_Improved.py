import torch
from torch import nn
from torch.nn import functional as F
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification


class Method_GRU_Improved(Method_RNN_Classification):
    def __init__(self,mName,mDescription,vocab_size,hidden_size=96,embeddings=None,dropout=0.3):
        super().__init__(mName,mDescription,vocab_size,'gru',50,hidden_size,10,128,0.001,'cpu',embeddings)
        self.conv=nn.Conv1d(50,64,3,padding=1)
        self.att=nn.Linear(64,1)
        self.rnn=nn.GRU(64,hidden_size,batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Sequential(
            nn.Linear(hidden_size*6+64,128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,2)
        )

    def forward(self,x,lengths):
        emb=self.dropout(self.embedding(x))
        feat=F.relu(self.conv(emb.transpose(1,2))).transpose(1,2)
        mask=torch.arange(x.size(1),device=x.device)[None,:]<lengths.to(x.device)[:,None]
        scores=self.att(feat).squeeze(-1).masked_fill(~mask,-1e9)
        weights=torch.softmax(scores,dim=1)
        context=(feat*weights.unsqueeze(-1)).sum(dim=1)
        pooled=F.avg_pool1d(feat.transpose(1,2),kernel_size=5,stride=5,ceil_mode=True).transpose(1,2)
        n=(lengths+4)//5
        packed=nn.utils.rnn.pack_padded_sequence(pooled,n.cpu(),batch_first=True,enforce_sorted=False)
        output,h=self.rnn(packed)
        output,_=nn.utils.rnn.pad_packed_sequence(output,batch_first=True,total_length=pooled.size(1))
        compact_mask=torch.arange(pooled.size(1),device=x.device)[None,:]<n.to(x.device)[:,None]
        avg=(output*compact_mask.unsqueeze(-1)).sum(dim=1)/n.to(x.device).clamp(min=1).unsqueeze(1)
        top=output.masked_fill(~compact_mask.unsqueeze(-1),-1e9).max(dim=1).values
        last=torch.cat((h[-2],h[-1]),dim=1)
        return self.fc(self.dropout(torch.cat((avg,top,last,context),dim=1)))
