import csv
import html
import re
from collections import Counter
from pathlib import Path
import numpy as np
from local_code.base_class.dataset import dataset


def tokenize(s):
    s=html.unescape(re.sub(r'<br\s*/?>',' ',s,flags=re.I))
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+|[.!?,;:]",s.lower())


class Dataset_Loader(dataset):
    def __init__(self,dName=None,dDescription=None,folder=None,task='classification',max_len=180,vocab_size=20000,train_limit=None,test_limit=None):
        super().__init__(dName,dDescription)
        self.folder=Path(folder)
        self.task=task
        self.max_len=max_len
        self.vocab_size=vocab_size
        self.train_limit=train_limit
        self.test_limit=test_limit

    def _vocab(self,sentences):
        counts=Counter(w for s in sentences for w in s)
        vocab={'<pad>':0,'<unk>':1,'<eos>':2}
        for word,n in counts.most_common(self.vocab_size-len(vocab)):
            if word not in vocab:
                vocab[word]=len(vocab)
        return vocab

    def _reviews(self,split,limit):
        texts=[]
        labels=[]
        n=None if limit is None else max(1,limit//2)
        for label,name in enumerate(('neg','pos')):
            files=sorted((self.folder/'text_classification'/split/name).glob('*.txt'))
            if n is not None:
                files=files[:n]
            for filename in files:
                texts.append(tokenize(filename.read_text(encoding='utf-8',errors='replace')))
                labels.append(label)
        if not texts:
            raise FileNotFoundError(f'No reviews found for {split}')
        return texts,np.asarray(labels,dtype=np.int64)

    def _encode(self,texts,vocab):
        n=len(texts)
        x=np.zeros((n,self.max_len),dtype=np.int64)
        lengths=np.zeros(n,dtype=np.int64)
        for i,s in enumerate(texts):
            ids=[vocab.get(w,1) for w in s[:self.max_len]]
            lengths[i]=max(1,len(ids))
            if ids:
                x[i,:len(ids)]=ids
            else:
                x[i,0]=1
        return x,lengths

    def load(self):
        if self.task=='classification':
            train,y_train=self._reviews('train',self.train_limit)
            test,y_test=self._reviews('test',self.test_limit)
            vocab=self._vocab(train)
            x_train,len_train=self._encode(train,vocab)
            x_test,len_test=self._encode(test,vocab)
            print('IMDb:',len(train),'train,',len(test),'test,',len(vocab),'vocabulary')
            return {'train':{'X':x_train,'y':y_train,'lengths':len_train},'test':{'X':x_test,'y':y_test,'lengths':len_test},'vocab':vocab}
        if self.task=='generation':
            filename=self.folder/'text_generation'/'data'
            with filename.open('r',encoding='utf-8-sig',newline='') as f:
                jokes=[r['Joke'] for r in csv.DictReader(f) if r.get('Joke','').strip()]
            texts=[tokenize(s)[:self.max_len-1] for s in jokes]
            vocab=self._vocab(texts)
            sequences=[[vocab.get(w,1) for w in s]+[2] for s in texts]
            print('Jokes:',len(jokes),'vocabulary:',len(vocab))
            return {'jokes':jokes,'texts':texts,'sequences':sequences,'vocab':vocab}
        raise ValueError('Unknown task: '+self.task)
