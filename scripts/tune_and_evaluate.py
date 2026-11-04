"""Small dev-only search; run the selected configuration on test exactly once."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/experiment'))
    parser.add_argument('--medical',action='store_true')
    parser.add_argument('--dev-episodes',type=int,default=10)
    parser.add_argument('--test-episodes',type=int,default=50)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    candidates=[{'context':32,'shrinkage':.8,'model':'urchade/gliner_medium-v2.1'},
                {'context':64,'shrinkage':.8,'model':'urchade/gliner_medium-v2.1'},
                {'context':64,'shrinkage':.5,'model':'urchade/gliner_medium-v2.1'}]
    if args.medical:candidates.append({'context':64,'shrinkage':.8,'model':'E3-JSI/gliner-multi-med-ner-synthetic-v1'})
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'selection.json').exists():raise FileExistsError('Choose a fresh experiment output path')
    def run(candidate,split,output,episodes):
        command=[sys.executable,'scripts/run_pipeline.py',f'model.device={args.device}',
            f'model.gliner_id={candidate["model"]}',f'model.context_tokens={candidate["context"]}',
            f'prototypes.gaussian.shrinkage={candidate["shrinkage"]}',f'evaluation.split={split}',
            f'support.n_episodes={episodes}',f'output.results_dir={output}']
        subprocess.run(command,check=True)
        return json.loads((output/'metrics/all_results.json').read_text())
    trials=[]
    for i,c in enumerate(candidates):
        result=run(c,'dev',args.output/f'dev_{i}',args.dev_episodes)
        score=sum(result[f'{k}_shot']['gaussian_prototype']['metrics']['f1']['mean'] for k in [5,10,20])/3
        trials.append({'config':c,'dev_gaussian_mean_f1':score})
    best=max(trials,key=lambda r:r['dev_gaussian_mean_f1'])
    (args.output/'selection.json').write_text(json.dumps({'objective':'Mean Gaussian dev micro F1 at 5,10,20 shots',
        'trials':trials,'selected':best},indent=2))
    run(best['config'],'test',args.output/'test',args.test_episodes)

if __name__=='__main__':main()
