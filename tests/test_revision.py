import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data.context_window import centered_window
from src.data.validated import load_validated, split_notes
from src.data.synthetic_generator import ClinicalSample
from src.data.span_dataset import SpanDataset
from src.data.heldout_episodes import HeldoutEpisodeBuilder
from src.models.gaussian_prototype import GaussianPrototype
from scripts.run_extraction import scores, evaluate


def test_late_mention_preserved():
    offsets=[(i*5,i*5+4) for i in range(2000)]
    lo,hi,a,b=centered_window(offsets,9000,9004,64,510)
    assert lo+a==1800 and b-a==1 and hi-lo==129


def test_oversize_and_missing_mention_fail():
    with pytest.raises(ValueError):centered_window([(0,1)],3,4,64,510)
    with pytest.raises(ValueError):centered_window([(i,i+1) for i in range(600)],0,600,64,510)


def test_preprocessing_does_not_rewrite(tmp_path):
    pd.DataFrame([{'note_id':'001','text':'pain drug'}]).to_csv(tmp_path/'train_notes.csv',index=False)
    rows=[{'note_id':'001','start':0,'end':4,'span':'pain','entity_type':'problem'},
          {'note_id':'001','start':0,'end':4,'span':'pain','entity_type':'Problem'},
          {'note_id':'001','start':5,'end':9,'span':'wrong','entity_type':'Treatment'}]
    pd.DataFrame(rows).to_csv(tmp_path/'train_annotations.csv',index=False)
    before=(tmp_path/'train_annotations.csv').read_bytes()
    samples,audit=load_validated(tmp_path)
    assert len(samples[0].spans)==1 and samples[0].note_id=='001'
    assert audit['counts']['duplicate']==1 and audit['counts']['text_mismatch']==1
    assert (tmp_path/'train_annotations.csv').read_bytes()==before


def test_duplicate_text_stays_in_one_split():
    samples=[ClinicalSample(text=str(i//2),spans=[],note_id=str(i),source='csv') for i in range(20)]
    split=split_notes(samples)
    sets=[{s.text for s in group} for group in split.values()]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]


def test_paired_nested_episodes_and_insufficient_pool():
    def ds(prefix):
        return SpanDataset([ClinicalSample(text='pain',spans=[dict(text='pain',start=0,end=4,label='Problem')],note_id=f'{prefix}{i}',source='csv') for i in range(10)])
    builder=HeldoutEpisodeBuilder(ds('s'),ds('q'),['Problem'],3)
    a=builder.build_episodes(1,2);b=builder.build_episodes(5,2)
    assert a[0].query==b[0].query
    assert a[0].support['Problem']==b[0].support['Problem'][:1]
    with pytest.raises(ValueError):builder.build_episodes(11)
    with pytest.raises(ValueError):HeldoutEpisodeBuilder(ds('s'),ds('s'),['Problem'])


def test_shrunk_gaussian_one_shot_finite():
    x=np.array([[1.,0.],[0.,1.]])
    model=GaussianPrototype(covariance_type='diagonal',shrinkage=.8,regularisation=.001).fit(x,['A','B'])
    assert np.isfinite(model.predict_proba(x)).all()
    assert model.predict(x)==['A','B']


def test_extraction_counts_boundaries_and_false_positives():
    result=scores({('a',0,4,'Problem')},{('a',0,3,'Problem'),('a',5,8,'Test')})
    assert result['tp']==0 and result['fp']==2 and result['fn']==1
    records=[{'note_id':'a','gold':[(0,4,'Problem')], 'predictions':[
      dict(start=0,end=4,label='Problem',score=.6,words=1)]}]
    assert evaluate(records,.5,12)['f1']==1
    assert evaluate(records,.7,12)['f1']==0


def test_pipeline_evaluation_writes_real_metric_objects(tmp_path):
    pytest.importorskip('torch')
    from omegaconf import OmegaConf
    from scripts.run_pipeline import _run_shot_evaluation
    class Encoder:
        def encode_span_items(self,items):
            return np.asarray([[1.,0.] if x.note_id.startswith('a') else [0.,1.] for x in items])
    samples=[]
    for label,prefix in [('Problem','a'),('Test','b')]:
        for i in range(8):
            samples.append(ClinicalSample(text='word',spans=[dict(text='word',start=0,end=4,label=label)],note_id=f'{prefix}{i}',source='csv'))
    support=SpanDataset([s for s in samples if int(s.note_id[1:])<4])
    query=SpanDataset([s for s in samples if int(s.note_id[1:])>=4])
    builder=HeldoutEpisodeBuilder(support,query,['Problem','Test'],2)
    cfg=OmegaConf.load(Path(__file__).resolve().parents[1]/'configs/default.yaml')
    cfg.data.entity_types=['Problem','Test'];cfg.output.results_dir=str(tmp_path)
    result=_run_shot_evaluation(2,2,builder,Encoder(),cfg)
    assert result['point_prototype']['metrics']['f1']['mean']==1
    saved=json.loads((tmp_path/'metrics/episodes_2shot.json').read_text())
    assert isinstance(saved['point'][0],dict)
    assert len(list((tmp_path/'plots').glob('*.png')))==6


def test_encoder_real_tokenizer_and_tiny_backbone(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    pytest.importorskip('gliner')
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from src.models.gliner_encoder import GLiNEREncoder, GLiNER
    from types import SimpleNamespace
    vocab=tmp_path/'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nfiller\npain\ndrug\n')
    tokenizer=BertTokenizerFast(vocab_file=str(vocab),do_lower_case=True)
    backbone=BertModel(BertConfig(vocab_size=8,hidden_size=16,num_hidden_layers=1,
                                  num_attention_heads=2,intermediate_size=32))
    class FakeGLiNER(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=backbone
            self.model=SimpleNamespace(token_rep_layer=SimpleNamespace(bert_layer=SimpleNamespace(model=backbone)))
            self.data_processor=SimpleNamespace(transformer_tokenizer=tokenizer)
    monkeypatch.setattr(GLiNER,'from_pretrained',lambda *a,**k:FakeGLiNER())
    encoder=GLiNEREncoder(device='cpu',cache_dir=tmp_path/'cache',context_tokens=8,batch_size=2)
    text='filler '*1000+'pain drug'
    start=text.index('pain')
    result=encoder.encode_spans([text,text],[(start,start+4),(start+5,start+9)])
    assert result.shape==(2,16) and np.isfinite(result).all()
    assert not np.allclose(result[0],result[1])
    assert np.allclose(np.linalg.norm(result,axis=1),1)
    whitespace = encoder.encode_spans([text],[(start-1,start+5)])
    assert np.allclose(whitespace,result[:1])
    encoder._cache.clear()
    monkeypatch.setattr(backbone,'forward',lambda *a,**k:pytest.fail('Cache should avoid forward'))
    assert np.allclose(encoder.encode_spans([text],[(start,start+4)]),result[:1])
