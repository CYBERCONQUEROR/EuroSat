"""Review 2: train/select/freeze/test, with canonical jobs reused across stages.
No test metric is read by either selection function. Completed matching jobs
are reused; interrupted jobs are never silently mistaken for complete ones.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
from statistics import mean

from experiment_utils import fingerprint, read_json, save_json, validate_manifest
from partition import SEEDS, ALPHAS, RATIOS, freeze, prepare_all, partition_name
BASE=Path(__file__).resolve().parent
TUNE_SEEDS=SEEDS[:3]
GN_LRS=[.0003,.001,.003]
BETAS=[.99,.999,.9999]
GAMMAS=[1.,2.]
CORE=['train.py','evaluate.py','model.py','data.py','metrics.py','utils.py','losses.py','partition.py','experiment_utils.py','environment_capture.py','review_workflow.py']

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def make_job(kind, value, seed, loss='cross-entropy', beta=0., gamma=0., client_id=-1,
             initialization='imagenet', norm='batch', size=96, lr=None):
    return dict(kind=kind,value=float(value),seed=seed,loss=loss,beta=float(beta) if loss=='class-balanced' else 0.,
                gamma=float(gamma) if loss=='focal' else 0.,client_id=client_id,initialization=initialization,
                norm=norm,image_size=size,lr=lr if lr is not None else (.0001 if initialization=='imagenet' else .001))

def gn_jobs():
    return [make_job('pooled',0,seed,initialization='scratch',norm='group',lr=lr) for lr in GN_LRS for seed in TUNE_SEEDS]

def cases(seeds):
    for seed in seeds:
        for alpha in ALPHAS:
            for cid in range(10):yield 'dirichlet',alpha,seed,cid
        for ratio in RATIOS:yield 'longtail',ratio,seed,0

def candidates():
    return [('cross-entropy',0.,0.)]+[('class-balanced',b,0.) for b in BETAS]+[('focal',0.,g) for g in GAMMAS]

def tuning_jobs():
    return [make_job(kind,v,seed,loss,b,g,cid) for kind,v,seed,cid in cases(TUNE_SEEDS) for loss,b,g in candidates()]

def final_jobs(gn_lr, selections):
    jobs=[]
    for seed in SEEDS:
        jobs.extend(make_job('pooled',0,seed,size=size) for size in [64,96,224])
        jobs.append(make_job('pooled',0,seed,initialization='scratch'))
        jobs.append(make_job('pooled',0,seed,initialization='scratch',norm='group',lr=gn_lr))
    for kind,v,seed,cid in cases(SEEDS):
        pick=selections[kind]
        jobs.extend([make_job(kind,v,seed,client_id=cid),
                     make_job(kind,v,seed,'class-balanced',pick['beta'],client_id=cid),
                     make_job(kind,v,seed,'focal',gamma=pick['gamma'],client_id=cid)])
    best=selections['gn_comparison']
    for seed in SEEDS:
        jobs.extend([make_job('longtail',100,seed,client_id=0,initialization='scratch',norm='group',lr=gn_lr),
                     make_job('longtail',100,seed,best['loss'],best['beta'],best['gamma'],0,'scratch','group',96,gn_lr)])
    return jobs

class Workflow:
    def __init__(self,args):
        self.args=args;self.root=Path(args.output).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.manifest=read_json(args.split);validate_manifest(self.manifest)
        self.protocol=dict(schema=2,split_hash=fingerprint(self.manifest),image_size=96,epochs=args.epochs,patience=10,
            batch_size=32,weight_decay=.0001,seeds=SEEDS,tuning_seeds=TUNE_SEEDS,num_clients=10,
            alphas=ALPHAS,ratios=RATIOS,gn_lrs=GN_LRS,betas=BETAS,gammas=GAMMAS,
            class_order_seed=42,validation_loss='unweighted_cross_entropy',
            selection_metric='validation_macro_f1',hyperparameter_scope='one beta and gamma per regime',
            aggregation='unweighted client mean within seed; sample SD across five seed means',
            evaluation_split='original stratified 4050-image subsets, unchanged, not exactly class-balanced',
            original_commit='2658cf6da9522e5cb86ea3a5e48eec4eb21e3ef5',
            code_sha256={name:sha256(BASE/name) for name in CORE})
        freeze(self.root/'protocol.json',self.protocol)
        self.protocol_hash=fingerprint(self.protocol)
        self.count=0

    def prepare(self):
        prepare_all(self.args.split,self.root/'partitions')
        freeze(self.root/'splits.json',self.manifest)

    def identifier(self,job):return fingerprint(dict(protocol_hash=self.protocol_hash,job=job))[:24]
    def directory(self,job):return self.root/'runs'/self.identifier(job)
    def partition_path(self,job):
        if job['kind']=='pooled':return None
        return self.root/'partitions'/partition_name(job['kind'],job['value'],job['seed'])

    def runtime(self):
        packages={name:md.version(name) for name in ['torch','torchvision','numpy','scikit-learn','Pillow','matplotlib','pandas','tqdm']}
        freeze(self.root/'runtime_versions.json',dict(python=sys.version.split()[0],packages=packages))
        if not (self.root/'requirements.txt').exists():
            from environment_capture import capture
            capture(self.root/'requirements.txt')

    def value(self,job):
        directory=self.directory(job)
        status=read_json(directory/'status.json')
        if status['status']!='completed':raise ValueError(f'Incomplete run: {directory}')
        cfg=read_json(directory/'config.json')
        if cfg.get('recipe_id')!=self.identifier(job):raise ValueError('Recipe mismatch')
        receipt=read_json(directory/'completion.json')
        for f in ['config.json','best_validation_metrics.json','history.csv']:
            if sha256(directory/f)!=receipt['sha256'][f]:raise ValueError(f'Changed run evidence: {directory/f}')
        if not (directory/'best_model.pt').exists():raise ValueError(f'Missing checkpoint: {directory}')
        return read_json(directory/'best_validation_metrics.json')

    def train(self,jobs):
        self.runtime()
        for job in jobs:
            rid=self.identifier(job);directory=self.directory(job)
            freeze(self.root/'recipes'/f'{rid}.json',dict(protocol_hash=self.protocol_hash,job=job))
            if (directory/'completion.json').exists():self.value(job);continue
            if (self.root/'final_plan.json').exists():raise ValueError('Final plan is sealed; cannot add training')
            if self.args.limit is not None and self.count>=self.args.limit:
                print('Chunk complete. Re-run the same stage to continue.');return False
            if directory.exists() and any(directory.iterdir()):
                if not self.args.restart_incomplete:
                    raise ValueError(f'Incomplete run at {directory}. Re-run with --restart-incomplete to archive and restart it.')
                archive=self.root/'interrupted'/f'{rid}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}'
                archive.parent.mkdir(exist_ok=True);directory.rename(archive)
            cmd=[sys.executable,str(BASE/'train.py'),'--data-dir',str(Path(self.args.data_dir).resolve()),
                 '--split',str(self.root/'splits.json'),'--output',str(directory),'--epochs',str(self.args.epochs),
                 '--patience','10','--batch-size','32','--workers',str(self.args.workers),'--device',self.args.device,
                 '--image-size',str(job['image_size']),'--seed',str(job['seed']),'--norm',job['norm'],
                 '--lr',str(job['lr']),'--loss',job['loss'],'--beta',str(job['beta']),'--gamma',str(job['gamma']),
                 '--recipe-id',rid]
            if job['initialization']=='scratch':cmd.append('--from-scratch')
            part=self.partition_path(job)
            if part is not None:cmd.extend(['--partition',str(part),'--client-id',str(job['client_id'])])
            subprocess.run(cmd,check=True)
            receipt=dict(sha256={f:sha256(directory/f) for f in ['best_model.pt','config.json','best_validation_metrics.json','history.csv']})
            freeze(directory/'completion.json',receipt);self.count+=1
            self.export_results()
        return True

    def assert_no_tests(self):
        if any((self.root/'runs').glob('*/test_metrics.json')):raise ValueError('Test evaluation started; selection cannot be changed')

    def select_gn(self):
        self.assert_no_tests()
        scores={str(lr):mean(self.value(j)['macro_f1'] for j in gn_jobs() if j['lr']==lr) for lr in GN_LRS}
        lr=max(GN_LRS,key=lambda x:(scores[str(x)],-x))
        evidence={self.identifier(j):sha256(self.directory(j)/'best_validation_metrics.json') for j in gn_jobs()}
        selection=dict(lr=lr,scores=scores,metric='mean validation macro-F1 across three seeds',evidence=evidence)
        freeze(self.root/'gn_selection.json',selection);return selection

    def select_losses(self):
        self.assert_no_tests();jobs=tuning_jobs();selection={};all_scores={}
        # Equal client means -> equal severity means -> equal seed means.
        for kind in ['dirichlet','longtail']:
            scores={}
            for loss,b,g in candidates():
                per_seed=[]
                for seed in TUNE_SEEDS:
                    severity=[]
                    for val in (ALPHAS if kind=='dirichlet' else RATIOS):
                        matched=[j for j in jobs if j['kind']==kind and j['value']==val and j['seed']==seed and
                                 j['loss']==loss and j['beta']==b and j['gamma']==g]
                        severity.append(mean(self.value(j)['macro_f1'] for j in matched))
                    per_seed.append(mean(severity))
                scores[f'{loss}:{b:g}:{g:g}']=mean(per_seed)
            beta=max(BETAS,key=lambda b:(scores[f'class-balanced:{b:g}:0'],-b))
            gamma=max(GAMMAS,key=lambda g:(scores[f'focal:0:{g:g}'],-g))
            selection[kind]=dict(beta=beta,gamma=gamma);all_scores[kind]=scores
        # Choose the non-CE comparison on ImageNet BN ratio-100 validation only.
        pick=selection['longtail'];choice=[]
        for loss,b,g in [('class-balanced',pick['beta'],0.),('focal',0.,pick['gamma'])]:
            score=mean(self.value(make_job('longtail',100,seed,loss,b,g,0))['macro_f1'] for seed in TUNE_SEEDS)
            choice.append(dict(loss=loss,beta=b,gamma=g,score=score))
        selection['gn_comparison']=max(choice,key=lambda x:x['score'])
        selection['scores']=all_scores
        selection['evidence']={self.identifier(j):sha256(self.directory(j)/'best_validation_metrics.json') for j in jobs}
        freeze(self.root/'loss_selection.json',selection);return selection

    def selections(self):
        gn=read_json(self.root/'gn_selection.json');loss=read_json(self.root/'loss_selection.json')
        for selection in [gn,loss]:
            for rid,digest in selection['evidence'].items():
                if sha256(self.root/'runs'/rid/'best_validation_metrics.json')!=digest:raise ValueError('Changed selection evidence')
        return gn,loss

    def finals(self):
        gn,loss=self.selections();return final_jobs(gn['lr'],loss)

    def seal(self):
        jobs=self.finals()
        for job in jobs:self.value(job)
        plan=dict(protocol_hash=self.protocol_hash,gn_selection_hash=fingerprint(read_json(self.root/'gn_selection.json')),
                  loss_selection_hash=fingerprint(read_json(self.root/'loss_selection.json')),
                  jobs=jobs,run_ids=[self.identifier(j) for j in jobs],n=len(jobs))
        freeze(self.root/'final_plan.json',plan)
        print(f'Sealed {len(jobs)} final jobs before test access')

    def evaluate(self):
        self.runtime();plan=read_json(self.root/'final_plan.json')
        if plan['protocol_hash']!=self.protocol_hash:raise ValueError('Protocol mismatch')
        gn,loss=self.selections()
        if fingerprint(gn)!=plan['gn_selection_hash'] or fingerprint(loss)!=plan['loss_selection_hash']:raise ValueError('Selection changed after seal')
        if plan['jobs']!=self.finals():raise ValueError('Final job list changed')
        count=0
        for job in plan['jobs']:
            d=self.directory(job);self.value(job)
            if (d/'test_metrics.json').exists():continue
            if self.args.limit is not None and count>=self.args.limit:return False
            if sha256(d/'best_model.pt')!=read_json(d/'completion.json')['sha256']['best_model.pt']:raise ValueError('Checkpoint checksum mismatch')
            subprocess.run([sys.executable,str(BASE/'evaluate.py'),'--data-dir',str(Path(self.args.data_dir).resolve()),
                            '--checkpoint',str(d/'best_model.pt'),'--device',self.args.device],check=True)
            count+=1;self.export_results()
        return True

    def export_results(self):
        from results_review import export_results
        export_results(self.root)

@contextmanager
def exclusive(root):
    Path(root).mkdir(parents=True,exist_ok=True)
    with (Path(root)/'.workflow.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another workflow is using this output directory')
        yield

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','plan','gn-search','select-gn','loss-search','select-losses','final-train','seal','evaluate','report','all'])
    p.add_argument('--data-dir',default='EuroSAT_RGB');p.add_argument('--split',default=str(BASE/'splits.json'))
    p.add_argument('--output',default='outputs/review2');p.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    p.add_argument('--workers',type=int,default=0);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--limit',type=int);p.add_argument('--restart-incomplete',action='store_true');args=p.parse_args()
    if args.epochs<1 or args.workers<0 or (args.limit is not None and args.limit<1):p.error('Invalid numeric settings')
    with exclusive(args.output):
        w=Workflow(args)
        if args.stage in ['prepare','all']:w.prepare()
        if args.stage=='plan':
            dummy={k:dict(beta=.999,gamma=2.) for k in ['dirichlet','longtail']}
            dummy['gn_comparison']=dict(loss='class-balanced',beta=.999,gamma=0.)
            final=final_jobs(.001,dummy)
            unique={fingerprint(j) for j in gn_jobs()+tuning_jobs()+final}
            print(json.dumps(dict(gn_search=9,loss_search=len(tuning_jobs()),final_configurations=len(final),
                 total_unique_trainings=len(unique),final_pooled=25,final_dirichlet=600,final_longtail=45,
                 final_gn_check=10,note='Counts only; no hyperparameter selection has been made'),indent=2));return
        stage=args.stage
        if stage in ['gn-search','all']:
            if not w.train(gn_jobs()):return
        if stage in ['select-gn','all'] and not (w.root/'gn_selection.json').exists():w.select_gn()
        if stage in ['loss-search','all']:
            if not (w.root/'gn_selection.json').exists():raise ValueError('Finish select-gn first')
            if not w.train(tuning_jobs()):return
        if stage in ['select-losses','all'] and not (w.root/'loss_selection.json').exists():w.select_losses()
        if stage in ['final-train','all']:
            if not w.train(w.finals()):return
        if stage in ['seal','all']:w.seal()
        if stage in ['evaluate','all']:
            if not w.evaluate():return
        if stage in ['report','all']:
            w.export_results()
            from results_review import publish
            publish(w.root)
        w.export_results()

if __name__=='__main__':main()
