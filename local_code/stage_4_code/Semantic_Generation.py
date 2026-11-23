"""Own GRU with an explicitly external GloVe word-vector prior.

Only individual word vectors are pretrained; recurrent and projection weights
are learned from the instructor training partition. No text is retrieved.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from local_code.stage_4_code.Dataset_Loader import Vocabulary


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def vocabulary_digest(vocab):
    return hashlib.sha256(b''.join(token.encode()+b'\0' for token in vocab.tokens)).hexdigest()


def load_word_prior(vocab, vector_file, manifest_file, dimension=100):
    """Read only training-vocabulary vectors after checking the archive member."""
    manifest = json.loads(Path(manifest_file).read_text())
    observed = file_digest(vector_file)
    if observed != manifest['member_sha256'] or Path(vector_file).name != manifest['member']:
        raise ValueError('Pretrained vector member does not match its manifest')
    if dimension < 1:
        raise ValueError('Word-vector dimension must be positive')
    vectors = torch.zeros((len(vocab),dimension),dtype=torch.float32)
    matched = torch.zeros(len(vocab),dtype=torch.bool)
    wanted = {word:index for index,word in enumerate(vocab.tokens) if index>=4}
    with Path(vector_file).open(encoding='utf-8') as stream:
        for line in stream:
            word, separator, values = line.partition(' ')
            if word not in wanted:
                continue
            fields = values.split()
            if not separator or len(fields) != dimension:
                raise ValueError('Malformed pretrained row for '+word)
            value = torch.tensor([float(field) for field in fields],dtype=torch.float32)
            if not torch.isfinite(value).all() or value.norm() == 0:
                raise ValueError('Invalid pretrained vector for '+word)
            index = wanted[word]
            if matched[index]:
                raise ValueError('Duplicate pretrained word: '+word)
            vectors[index] = value / value.norm()
            matched[index] = True
    if not matched.any():
        raise ValueError('No training vocabulary words match pretrained vectors')
    metadata = dict(manifest=manifest,manifest_sha256=file_digest(manifest_file),
        member_sha256=observed,dimension=dimension,matched_types=int(matched.sum()),
        eligible_types=len(wanted),matched_type_fraction=int(matched.sum())/len(wanted),
        vector_transform='Each matched GloVe vector divided by its L2 norm; norm1/RMS0.1 at100dimensions.',
        vector_tensor_sha256=tensor_digest(vectors),matched_mask_sha256=tensor_digest(matched),
        disclosure='External pretrained WORD embeddings; own GRU, no pretrained text generator or joke retrieval.')
    return vectors,matched,metadata


class SemanticGenerationGRU(nn.Module):
    """Causal GRU, projected to a semantic embedding space with tied output."""
    def __init__(self,vocab_size,embedding_dim=100,hidden_dim=256,dropout=.2,pad_id=0):
        super().__init__()
        if min(vocab_size,embedding_dim,hidden_dim)<1 or vocab_size<5 or pad_id!=0:
            raise ValueError('Invalid model dimensions or PAD index')
        if not math.isfinite(dropout) or not 0<=dropout<1:
            raise ValueError('Dropout must lie in [0,1)')
        self.task,self.cell='generation','semantic_prior_gru'
        self.vocab_size,self.pad_id=vocab_size,pad_id
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.recurrent=nn.GRU(embedding_dim,hidden_dim,batch_first=True)
        self.projection=nn.Linear(hidden_dim,embedding_dim,bias=False)
        self.output_bias=nn.Parameter(torch.zeros(vocab_size))
        self.dropout=nn.Dropout(dropout)
        self.register_buffer('prior',torch.zeros(vocab_size,embedding_dim))
        self.register_buffer('matched',torch.zeros(vocab_size,dtype=torch.bool))
        nn.init.normal_(self.embedding.weight,std=.1)
        with torch.no_grad():self.embedding.weight[0].zero_()
        self.embedding.weight.register_hook(self._zero_pad_gradient)

    def _zero_pad_gradient(self,gradient):
        gradient=gradient.clone();gradient[0].zero_();return gradient

    @torch.no_grad()
    def initialize_prior(self,vectors,matched):
        if vectors.shape!=self.prior.shape or matched.shape!=self.matched.shape or matched.dtype!=torch.bool:
            raise ValueError('Pretrained prior dimensions/mask differ from model')
        if not torch.isfinite(vectors).all() or bool(matched[:4].any()) or torch.count_nonzero(vectors[~matched]):
            raise ValueError('Invalid pretrained mask or special-token vectors')
        self.prior.copy_(vectors);self.matched.copy_(matched)
        self.embedding.weight[matched]=vectors[matched]
        self.embedding.weight[0].zero_()

    def anchor_penalty(self):
        if not bool(self.matched.any()):return self.embedding.weight.sum()*0
        return (self.embedding.weight[self.matched]-self.prior[self.matched]).square().sum(-1).mean()

    def step(self,tokens,hidden=None):
        if tokens.ndim!=2 or tokens.shape[1]==0:
            raise ValueError('tokens must have shape [batch,positive length]')
        states,hidden=self.recurrent(self.dropout(self.embedding(tokens)),hidden)
        projected=self.projection(self.dropout(states))
        return nn.functional.linear(projected,self.embedding.weight,self.output_bias),hidden

    def forward(self,tokens,lengths=None):return self.step(tokens)[0]
    def parameter_count(self):return sum(p.numel() for p in self.parameters())


def load_semantic_checkpoint(checkpoint,device='cpu'):
    checkpoint=Path(checkpoint).resolve();run=checkpoint.parent.parent
    config=json.loads((run/'config.json').read_text())
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    digest=canonical_digest(config)
    if saved.get('config_hash')!=digest or config.get('architecture')!='semantic_prior_gru_v1':
        raise ValueError('Mismatched configuration or architecture')
    repo=Path(__file__).resolve().parents[2]
    required={'local_code/stage_4_code/Semantic_Generation.py','local_code/stage_4_code/Dataset_Loader.py',
              'script/stage_4_script/train_semantic_generation.py','local_code/stage_4_code/Experiment_Runner.py'}
    if not required.issubset(config['source_sha256']):
        raise ValueError('Missing required training source bindings')
    for name,expected in config['source_sha256'].items():
        source=(repo/name).resolve()
        if not source.is_relative_to(repo) or file_digest(source)!=expected:
            raise ValueError('Training source changed: '+name)
    external=config['pretrained']
    for key in ('manifest_file','vector_file'):
        if not (repo/external[key]).resolve().is_relative_to(repo):
            raise ValueError('Pretrained source path escapes repository')
    manifest_path=repo/external['manifest_file']
    if file_digest(manifest_path)!=external['manifest_sha256']:
        raise ValueError('Pretrained manifest changed')
    manifest=json.loads(manifest_path.read_text())
    if manifest!=external['manifest'] or file_digest(repo/external['vector_file'])!=external['member_sha256']:
        raise ValueError('Pretrained vector source changed')
    vocab=Vocabulary.from_dict(saved['vocab'])
    if vocab.to_dict()!=saved['vocab'] or len(vocab)!=config['model']['vocab_size'] or vocabulary_digest(vocab)!=config['data_metadata']['vocabulary_sha256']:
        raise ValueError('Checkpoint vocabulary changed')
    state=saved['model']
    if tensor_digest(state['prior'])!=external['vector_tensor_sha256'] or tensor_digest(state['matched'])!=external['matched_mask_sha256']:
        raise ValueError('Checkpoint pretrained reference changed')
    if torch.count_nonzero(state['embedding.weight'][0]) or bool(state['matched'][:4].any()):
        raise ValueError('Checkpoint PAD/special-token invariant violated')
    with torch.random.fork_rng(devices=[]):model=SemanticGenerationGRU(**config['model'])
    model.load_state_dict(state,strict=True);model.to(device).eval()
    return model,vocab,dict(epoch=saved['epoch'],config=config,config_hash=digest,
        checkpoint_sha256=file_digest(checkpoint),external_word_embeddings=True)
