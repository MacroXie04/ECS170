"""Own GRU with GloVe only at the input and an unconstrained output matrix."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from local_code.stage_4_code.Dataset_Loader import Vocabulary
from local_code.stage_4_code.Semantic_Generation import (
    SemanticGenerationGRU,canonical_digest,file_digest,tensor_digest,vocabulary_digest,
    load_word_prior,
)


class SemanticInputGenerationGRU(SemanticGenerationGRU):
    """Retain input semantic anchors without constraining output geometry."""
    def __init__(self,vocab_size,embedding_dim=100,hidden_dim=256,dropout=.2,pad_id=0):
        super().__init__(vocab_size,embedding_dim,hidden_dim,dropout,pad_id)
        del self.projection
        del self.output_bias
        self.output=nn.Linear(hidden_dim,vocab_size)
        self.cell='semantic_input_gru'

    def step(self,tokens,hidden=None):
        if tokens.ndim!=2 or tokens.shape[1]==0:
            raise ValueError('tokens must have shape [batch,positive length]')
        states,hidden=self.recurrent(self.dropout(self.embedding(tokens)),hidden)
        return self.output(self.dropout(states)),hidden


def load_semantic_input_checkpoint(checkpoint,device='cpu'):
    checkpoint=Path(checkpoint).resolve();run=checkpoint.parent.parent
    config=json.loads((run/'config.json').read_text())
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    digest=canonical_digest(config)
    if saved.get('config_hash')!=digest or config.get('architecture')!='semantic_input_gru_v1':
        raise ValueError('Mismatched configuration or architecture')
    repo=Path(__file__).resolve().parents[2]
    required={'local_code/stage_4_code/Semantic_Generation.py','local_code/stage_4_code/Semantic_Input_Generation.py',
              'local_code/stage_4_code/Dataset_Loader.py','script/stage_4_script/train_semantic_input_generation.py',
              'local_code/stage_4_code/Experiment_Runner.py'}
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
    with torch.random.fork_rng(devices=[]):model=SemanticInputGenerationGRU(**config['model'])
    model.load_state_dict(state,strict=True);model.to(device).eval()
    return model,vocab,dict(epoch=saved['epoch'],config=config,config_hash=digest,
        checkpoint_sha256=file_digest(checkpoint),external_word_embeddings=True)
