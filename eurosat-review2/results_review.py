"""One CSV row per completed run; publication artifacts read only this CSV.
Dirichlet summaries first average ten clients within each seed, then calculate
mean and sample SD over five seed means. Clients are not independent seeds.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_utils import fingerprint, read_json

LOSSES=['cross-entropy','class-balanced','focal']
LABELS={'cross-entropy':'Cross-entropy','class-balanced':'Class-balanced','focal':'Focal'}
COLORS={'cross-entropy':'#366A9F','class-balanced':'#DF7B25','focal':'#259373'}
SEEDS={42,123,2024,7,99}

def export_results(root):
    from review_workflow import final_jobs,gn_jobs,tuning_jobs
    root=Path(root);protocol=read_json(root/'protocol.json');ph=fingerprint(protocol)
    final_ids=set()
    if (root/'gn_selection.json').exists() and (root/'loss_selection.json').exists():
        jobs=final_jobs(read_json(root/'gn_selection.json')['lr'],read_json(root/'loss_selection.json'))
        final_ids={fingerprint(dict(protocol_hash=ph,job=j))[:24] for j in jobs}
    gn_ids={fingerprint(j) for j in gn_jobs()};tune_ids={fingerprint(j) for j in tuning_jobs()}
    rows=[]
    for path in sorted((root/'recipes').glob('*.json')):
        record=read_json(path);job=record['job'];rid=path.stem;d=root/'runs'/rid
        if not (d/'completion.json').exists():continue
        status=read_json(d/'status.json')
        if status['status']!='completed':continue
        cfg=read_json(d/'config.json');val=read_json(d/'best_validation_metrics.json')
        test=read_json(d/'test_metrics.json') if (d/'test_metrics.json').exists() else None
        with (d/'history.csv').open() as f:history=list(csv.DictReader(f))
        roles=[]
        if fingerprint(job) in gn_ids:roles.append('gn-search')
        if fingerprint(job) in tune_ids:roles.append('loss-search')
        if rid in final_ids:roles.append('final')
        row=dict(run_id=rid,roles=';'.join(roles),is_final=int(rid in final_ids),**job,
                 split_hash=cfg['split_hash'],partition_hash=cfg.get('partition_hash'),
                 checkpoint_sha256=read_json(d/'completion.json')['sha256']['best_model.pt'],
                 best_epoch=status['best_epoch'],epochs_completed=status['epochs_completed'],
                 training_loop_seconds=status['training_loop_seconds'],val_loss=val['loss'],
                 train_size=cfg['train_size'],train_counts_json=json.dumps(cfg['train_counts']),
                 history_json=json.dumps(history),test_evaluated=int(test is not None),
                 confusion_matrix_json=json.dumps(test['confusion_matrix']) if test else '',
                 classes_json=json.dumps(read_json(d/'splits.json')['classes']))
        for split,metrics in [('val',val),('test',test)]:
            for name in ['accuracy','macro_f1','balanced_accuracy']:row[f'{split}_{name}']=metrics[name] if metrics else None
            row[f'{split}_report_json']=json.dumps(metrics['classification_report']) if metrics else ''
            for cls in val['per_class_recall']:row[f'{split}_recall_{cls}']=metrics['per_class_recall'][cls] if metrics else None
        rows.append(row)
    if rows:
        dest=root/'results.csv';tmp=root/'results.csv.tmp';pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(dest)
    return rows

def seed_means(frame, metric):
    return frame.groupby('seed')[metric].mean().sort_index()

def mean_sd(frame,metric):
    vals=seed_means(frame,metric)
    if len(vals)!=5 or set(vals.index)!=SEEDS:raise ValueError('Five final seeds required')
    return float(vals.mean()),float(vals.std(ddof=1))

def cell(frame,metric='test_macro_f1',scale=100):
    m,s=mean_sd(frame,metric);return f'{m*scale:.2f} \\pm {s*scale:.2f}'

def latex_table(headers,rows,path,caption):
    lines=['\\begin{table}[t]','\\centering','\\caption{'+caption+'}',
           '\\resizebox{\\columnwidth}{!}{%', '\\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}', '\\toprule',
           ' & '.join(headers)+' \\\\', '\\midrule']
    lines += [' & '.join(str(x) for x in row)+' \\\\' for row in rows]
    lines += ['\\bottomrule','\\end{tabular}}','\\end{table}']
    Path(path).write_text('\n'.join(lines)+'\n')

def validate_final(root,df):
    plan=read_json(Path(root)/'final_plan.json')
    final=df[df.is_final==1].copy()
    if set(final.run_id)!=set(plan['run_ids']) or len(final)!=plan['n']:raise ValueError('Final jobs incomplete or duplicated')
    if len(final)!=680:raise ValueError('Expected 680 final model evaluations')
    if not (final.test_evaluated==1).all() or final.test_macro_f1.isna().any():raise ValueError('Test evaluation is incomplete')
    for _,g in final.groupby(['kind','value','initialization','norm','image_size','loss']):
        if set(g.seed)!=SEEDS:raise ValueError('Missing final seeds')
        if g.kind.iloc[0]=='dirichlet':
            for _,seed_group in g.groupby('seed'):
                if set(seed_group.client_id)!=set(range(10)) or len(seed_group)!=10:raise ValueError('Missing or duplicate clients')
        elif len(g)!=5:raise ValueError('Expected one run per seed')
    return final

def publish(root):
    root=Path(root);df=pd.read_csv(root/'results.csv',dtype={'run_id':str});final=validate_final(root,df)
    out=root/'generated';out.mkdir(exist_ok=True)
    metric_cols=['test_accuracy','test_macro_f1','test_balanced_accuracy']+[c for c in df if c.startswith('test_recall_')]
    keys=['kind','value','initialization','norm','image_size','loss']
    seedwise=final.groupby(keys+['seed'],dropna=False)[metric_cols+['training_loop_seconds']].mean().reset_index()
    seedwise.to_csv(out/'seed_level_results.csv',index=False)
    summary=seedwise.groupby(keys,dropna=False)[metric_cols+['training_loop_seconds']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'summary.csv',index=False)
    final[final.kind=='dirichlet'].drop(columns=['history_json','confusion_matrix_json','val_report_json','test_report_json']).to_csv(out/'per_client_results.csv',index=False)
    pooled=final[final.kind=='pooled'];rows=[]
    for init,norm,size in [('imagenet','batch',64),('imagenet','batch',96),('imagenet','batch',224),('scratch','batch',96),('scratch','group',96)]:
        g=pooled[(pooled.initialization==init)&(pooled.norm==norm)&(pooled.image_size==size)]
        name=('ImageNet' if init=='imagenet' else 'Scratch')+', '+('BN' if norm=='batch' else 'GN')
        rows.append([name,size,'$'+cell(g,'test_accuracy')+'$','$'+cell(g)+'$','$'+cell(g,'training_loop_seconds',1/60)+'$'])
    latex_table(['Model','Input','Accuracy (\\%)','Macro-F1 (\\%)','Time (min)'],rows,out/'table1.tex','Pooled baseline results over five seeds, mean $\\pm$ sample standard deviation. GN uses its validation-selected learning rate.')
    primary=final[(final.initialization=='imagenet')&(final.norm=='batch')&(final.image_size==96)]
    for idx,kind,values in [(2,'longtail',[10,50,100]),(3,'dirichlet',[.1,.5,1.,10.])]:
        rows=[[LABELS[loss]]+['$'+cell(primary[(primary.kind==kind)&(primary.value==v)&(primary.loss==loss)])+'$' for v in values] for loss in LOSSES]
        caption=('Long-tailed EuroSAT test macro-F1 (\\%), five-seed mean $\\pm$ sample SD.' if kind=='longtail' else
                 'Dirichlet local-only models on the global test set: client macro-F1 averaged within each seed, then five-seed mean $\\pm$ sample SD (\\%).')
        latex_table(['Loss']+[('Ratio ' if kind=='longtail' else '$\\alpha=')+f'{v:g}'+('' if kind=='longtail' else '$') for v in values],rows,out/f'table{idx}.tex',caption)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    mats=[]
    for v in [.1,10.]:
        g=primary[(primary.kind=='dirichlet')&(primary.value==v)&(primary.seed==42)&(primary.loss=='cross-entropy')].sort_values('client_id')
        mats.append(np.array([json.loads(x) for x in g.train_counts_json]))
    maximum=max(m.max() for m in mats)
    classes=json.loads(final.iloc[0].classes_json)
    for ax,m,v in zip(axes,mats,[.1,10.]):
        im=ax.imshow(m,cmap='Blues',vmin=0,vmax=maximum,aspect='auto');ax.set(title=f'alpha = {v:g}',xlabel='Class',ylabel='Client')
        ax.set_xticks(range(10),classes,rotation=90);ax.set_yticks(range(10))
    fig.colorbar(im,ax=axes,label='Training image count');save(fig,'figure1')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for ax,kind,values,xlabel in zip(axes,['longtail','dirichlet'],[[10,50,100],[.1,.5,1,10]],['Imbalance ratio','Dirichlet alpha']):
        for loss in LOSSES:
            stats=[mean_sd(primary[(primary.kind==kind)&(primary.value==v)&(primary.loss==loss)],'test_macro_f1') for v in values]
            ax.errorbar(values,[x[0]*100 for x in stats],yerr=[x[1]*100 for x in stats],label=LABELS[loss],color=COLORS[loss],marker='o',capsize=3)
        ax.set(xlabel=xlabel,ylabel='Test macro-F1 (%)',xscale='log');ax.set_xticks(values,[f'{v:g}' for v in values]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);save(fig,'figure2')
    gn=final[(final.kind=='longtail')&(final.norm=='group')]
    best=gn[gn.loss!='cross-entropy'].loss.unique()
    if len(best)!=1:raise ValueError('Ambiguous validation-selected comparison loss')
    best=best[0]
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained');x=np.arange(10)
    for offset,loss in [(-.2,'cross-entropy'),(.2,best)]:
        group=primary[(primary.kind=='longtail')&(primary.value==100)&(primary.loss==loss)]
        stats=[mean_sd(group,'test_recall_'+c) for c in classes]
        ax.bar(x+offset,[v[0]*100 for v in stats],.4,yerr=[v[1]*100 for v in stats],color=COLORS[loss],label=LABELS[loss],capsize=2)
    ax.set_xticks(x,classes,rotation=45,ha='right');ax.set_ylabel('Recall (%)');ax.legend();save(fig,'figure3')
    chosen=pooled[(pooled.initialization=='imagenet')&(pooled.image_size==96)&(pooled.seed==42)].iloc[0]
    cm=np.array(json.loads(chosen.confusion_matrix_json));fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    ax.imshow(cm,cmap='Blues');ax.set_xticks(range(10),classes,rotation=90);ax.set_yticks(range(10),classes)
    for i in range(10):
        for j in range(10):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=8)
    ax.set(xlabel='Predicted class',ylabel='Actual class',title='Selected pooled ImageNet BN 96, seed 42');save(fig,'figure4')
    # Supplementary learning histories are embedded in results.csv as well.
    for norm,init in [('batch','imagenet'),('batch','scratch'),('group','scratch')]:
        group=pooled[(pooled.norm==norm)&(pooled.initialization==init)&(pooled.image_size==96)]
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        for row in group.itertuples():
            history=pd.DataFrame(json.loads(row.history_json)).astype(float)
            axes[0].plot(history.epoch,history.val_accuracy*100,label=f'seed {row.seed}')
            axes[1].plot(history.epoch,history.val_loss,label=f'seed {row.seed}')
        axes[0].set(xlabel='Epoch',ylabel='Validation accuracy (%)');axes[1].set(xlabel='Epoch',ylabel='Validation CE');axes[0].legend(fontsize=7)
        save(fig,f'supplement_{init}_{norm}_curves')
    gn_summary=gn.groupby(['loss','seed'])[metric_cols].mean().groupby('loss').agg(['mean','std']);gn_summary.to_csv(out/'scratch_gn_ratio100.csv')
    # Trend diagnostics are descriptive warnings, never a demand to alter measurements.
    trends=[]
    for kind,values in [('longtail',[10,50,100]),('dirichlet',[.1,.5,1,10])]:
        vals=[mean_sd(primary[(primary.kind==kind)&(primary.value==v)&(primary.loss=='cross-entropy')],'test_macro_f1')[0] for v in values]
        monotonic=all((b<=a if kind=='longtail' else b>=a) for a,b in zip(vals,vals[1:]))
        trends.append(dict(kind=kind,values=values,mean_macro_f1=vals,expected_monotonic=monotonic,
                           action='Inspect if nonmonotonic; retain and report real values.'))
    (out/'trend_diagnostics.json').write_text(json.dumps(trends,indent=2))
    interpretations=['# Generated result interpretations','All uncertainty below is sample SD over five seeds.']
    interpretations.append('Table 1 compares complete pooled training recipes. ImageNet and scratch models have different learning rates, so their difference is not an isolated pretraining effect.')
    for kind in ['longtail','dirichlet']:
        interpretations.append(f'## {kind}')
        for value in sorted(primary[primary.kind==kind].value.unique()):
            ce=mean_sd(primary[(primary.kind==kind)&(primary.value==value)&(primary.loss=='cross-entropy')],'test_macro_f1')[0]
            for loss in LOSSES[1:]:
                score=mean_sd(primary[(primary.kind==kind)&(primary.value==value)&(primary.loss==loss)],'test_macro_f1')[0]
                interpretations.append(f'{value:g}, {loss}: mean macro-F1 difference versus CE = {(score-ce)*100:+.3f} percentage points. This descriptive difference is not a significance test.')
    (out/'interpretations.md').write_text('\n\n'.join(interpretations)+'\n')
    discussion=[]
    for line in interpretations[2:]:
        if not line.startswith('#'):
            discussion.append(line.replace('cross-entropy','CE').replace('class-balanced','class-balanced CE'))
    (out/'results_discussion.tex').write_text('\n\n'.join(discussion)+'\n')
    df[df.roles.str.contains('gn-search|loss-search', na=False)].drop(columns=['history_json','confusion_matrix_json','val_report_json','test_report_json']).to_csv(out/'tuning_scores.csv',index=False)
    import hashlib
    (out/'provenance.json').write_text(json.dumps(dict(results_sha256=hashlib.sha256((root/'results.csv').read_bytes()).hexdigest(),
        final_plan_hash=fingerprint(read_json(root/'final_plan.json')),final_runs=len(final),seeds=sorted(SEEDS)),indent=2))
    print(f'Publication tables, four figures and supplementary results written to {out}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/review2');p.add_argument('--export-only',action='store_true')
    args=p.parse_args();export_results(args.root)
    if not args.export_only:publish(args.root)
