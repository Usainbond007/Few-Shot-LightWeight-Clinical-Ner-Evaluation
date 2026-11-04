"""Frozen GLiNER backbone mean pooling, NOT GLiNER's learned span head.

Tokenize the complete note once, then center a bounded token window around each
known mention. Never substitute CLS or truncate away the target. No gold class
labels enter the encoder. Disk cache contains vectors, not raw note text.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from gliner import GLiNER


from src.data.context_window import centered_window

class GLiNEREncoder(nn.Module):
    def __init__(self, model_id='urchade/gliner_medium-v2.1', device='auto',
                 max_span_len=12, batch_size=8, context_tokens=64,
                 cache_dir='data/processed/embeddings', revision=None):
        super().__init__()
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self._device = torch.device(device)
        self.gliner = GLiNER.from_pretrained(model_id, **({'revision': revision} if revision else {})).to(device).eval()
        for p in self.gliner.parameters(): p.requires_grad_(False)
        self.tokenizer = self.gliner.data_processor.transformer_tokenizer
        if not self.tokenizer.is_fast:
            raise RuntimeError('A fast tokenizer with offsets is required.')
        self.backbone = self.gliner.model.token_rep_layer.bert_layer.model
        self.embedding_dim = self.backbone.config.hidden_size
        self.batch_size, self.context_tokens = batch_size, context_tokens
        self.max_span_len = max_span_len  # applies only to extraction candidate filtering
        self.max_tokens = min(512, getattr(self.backbone.config, 'max_position_embeddings', 512))
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Hash actual loaded weights: a changed remote checkpoint cannot reuse stale vectors.
        digest = hashlib.sha256()
        for name, value in self.backbone.state_dict().items():
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.fingerprint = hashlib.sha256(json.dumps([model_id, revision, digest.hexdigest(),
            self.tokenizer.backend_tokenizer.to_str(), context_tokens, self.max_tokens, 'pool-v2']).encode()).hexdigest()
        self._tokens = {}; self._cache = {}

    @property
    def device(self): return self._device

    def _prepare(self, text, start, end):
        if not 0 <= start < end <= len(text): raise ValueError('Invalid mention bounds')
        # Whitespace has no token representation; keep original gold offsets elsewhere.
        while start < end and text[start].isspace(): start += 1
        while end > start and text[end-1].isspace(): end -= 1
        if start == end: raise ValueError('Whitespace-only mention')
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self._tokens:
            self._tokens[key] = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        enc = self._tokens[key]
        budget = self.max_tokens - self.tokenizer.num_special_tokens_to_add(False)
        lo, hi, first, last = centered_window(enc['offset_mapping'], start, end, self.context_tokens, budget)
        left, right = enc['offset_mapping'][lo][0], enc['offset_mapping'][hi-1][1]
        prepared = self.tokenizer(text[left:right], add_special_tokens=True,
                                  return_offsets_mapping=True, truncation=False)
        local_offsets = prepared.pop('offset_mapping')
        positions = [i for i,(a,b) in enumerate(local_offsets)
                     if a < end-left and b > start-left and b > a]
        if len(prepared['input_ids']) > self.max_tokens:
            raise ValueError('Retokenized window exceeds budget; reduce context_tokens.')
        if not positions or local_offsets[positions[0]][0] > start-left or local_offsets[positions[-1]][1] < end-left:
            raise ValueError('Retokenized window lost part of the mention.')
        return prepared, positions

    @torch.no_grad()
    def encode_spans(self, sentences, spans, entity_types=None):
        if len(sentences) != len(spans): raise ValueError('Mismatched spans and sentences')
        keys = [hashlib.sha256((self.fingerprint + t + str(s)).encode()).hexdigest() for t,s in zip(sentences, spans)]
        missing = {}
        for key,text,(s,e) in zip(keys,sentences,spans):
            if key in self._cache: continue
            path = self.cache_dir / (key+'.npy')
            if path.exists(): self._cache[key] = np.load(path, allow_pickle=False)
            else: missing[key] = (text,s,e)
        items = list(missing.items())
        for i in range(0,len(items),self.batch_size):
            batch = items[i:i+self.batch_size]
            prepared = [self._prepare(*v) for _,v in batch]
            inputs = self.tokenizer.pad([p for p,_ in prepared], padding=True, return_tensors='pt').to(self._device)
            hidden = self.backbone(**inputs).last_hidden_state
            for j,((key,_),(_,positions)) in enumerate(zip(batch,prepared)):
                vector = hidden[j,positions].mean(0).float().cpu().numpy()
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                if not np.isfinite(vector).all(): raise ValueError('Nonfinite embedding')
                self._cache[key] = vector
                tmp = self.cache_dir / (key+'.tmp.npy'); np.save(tmp,vector); tmp.replace(self.cache_dir/(key+'.npy'))
        return np.stack([self._cache[k] for k in keys]) if keys else np.empty((0,self.embedding_dim))

    def encode_span_items(self, items):
        return self.encode_spans([x.sentence for x in items], [(x.span_start,x.span_end) for x in items])
