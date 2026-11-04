"""Native GLiNER exact-span NER; tune labels, threshold and span cap on dev only."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data.validated import load_validated, split_notes

PRESETS = {
 'simple': {'problem':'Problem','treatment':'Treatment','test':'Test'},
 'clinical': {'condition':'Problem','symptom':'Problem','comorbidity':'Problem',
              'drug':'Treatment','treatment':'Treatment','operation':'Treatment',
              'diagnostic test':'Test','test':'Test'},
}


def windows(text, tokenizer, size=160, overlap=48):
    enc = tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
    offsets = enc['offset_mapping']
    for i in range(0,len(offsets),size-overlap):
        part = offsets[i:i+size]
        if not part: break
        a,b=part[0][0],part[-1][1]
        yield a,text[a:b]
        if i+size>=len(offsets): break


def scores(gold, predictions):
    tp=len(gold & predictions); fp=len(predictions-gold); fn=len(gold-predictions)
    p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
    return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'tp':tp,'fp':fp,'fn':fn}


def evaluate(records,threshold,cap):
    gold,pred=set(),set()
    for rec in records:
        nid=rec['note_id']
        gold.update((nid,*x) for x in rec['gold'])
        pred.update((nid,p['start'],p['end'],p['label']) for p in rec['predictions']
                    if p['score']>=threshold and p['words']<=cap)
    return scores(gold,pred)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='urchade/gliner_medium-v2.1')
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--selection',type=Path)
    p.add_argument('--output',type=Path,default=Path('results/extraction_dev'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if (args.output/'COMPLETE.json').exists(): raise FileExistsError('Use a fresh output directory')
    samples,audit=load_validated('data/raw'); split=split_notes(samples,args.seed)
    selected=None
    if args.split=='test':
        if not args.selection: p.error('--selection from dev is required for test')
        selected=json.loads(args.selection.read_text())
        if selected['model']!=args.model or selected['seed']!=args.seed or selected['data_sha256']!=audit['sha256']:
            raise ValueError('Selection model, seed or dataset does not match')
    from gliner import GLiNER
    model=GLiNER.from_pretrained(args.model).to(args.device).eval()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'data_audit.json').write_text(json.dumps(audit,indent=2))
    presets=[selected['preset']] if selected else list(PRESETS)
    rows=[]
    for preset in presets:
        mapping=PRESETS[preset];records=[]
        for idx,note in enumerate(split[args.split]):
            predictions={}
            for offset,text in windows(note.text,model.data_processor.transformer_tokenizer):
                for e in model.predict_entities(text,list(mapping),threshold=.1,flat_ner=False):
                    label=mapping[e['label']];a,b=offset+e['start'],offset+e['end']
                    key=(a,b,label)
                    rec={'start':a,'end':b,'label':label,'score':float(e['score']),
                         'words':len(list(model.data_processor.words_splitter(note.text[a:b])))}
                    if key not in predictions or predictions[key]['score']<rec['score']: predictions[key]=rec
            records.append({'note_id':note.note_id,
                'gold':[(s['start'],s['end'],s['label']) for s in note.spans],
                'predictions':list(predictions.values())})
            print(f'{preset}: {idx+1}/{len(split[args.split])}',flush=True)
        (args.output/f'predictions_{preset}.json').write_text(json.dumps(records))
        thresholds=[selected['threshold']] if selected else [.2,.3,.4,.5,.6]
        caps=[selected['span_cap']] if selected else [8,12,24]
        for threshold in thresholds:
            for cap in caps:
                rows.append({'preset':preset,'threshold':threshold,'span_cap':cap,**evaluate(records,threshold,cap)})
    best=max(rows,key=lambda r:r['f1'])
    payload={'model':args.model,'seed':args.seed,'data_sha256':audit['sha256'],
             'native_max_width':getattr(model.config,'max_width',None),**best}
    (args.output/'metrics.json').write_text(json.dumps(rows,indent=2))
    (args.output/('selection.json' if args.split=='dev' else 'test_result.json')).write_text(json.dumps(payload,indent=2))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(['Precision','Recall','F1'],[best[k] for k in ['precision','recall','f1']]);ax.set_ylim(0,1)
    ax.set_title(f'Exact-span NER ({args.split})');fig.tight_layout();fig.savefig(args.output/'extraction_metrics.png');plt.close(fig)
    (args.output/'COMPLETE.json').write_text(json.dumps({'split':args.split,'task':'exact_span_ner',**payload},indent=2))
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
