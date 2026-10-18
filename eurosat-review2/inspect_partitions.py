"""Pre-training diagnostics; model performance is never simulated here."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def inspect(root='partitions', output='partition_diagnostics'):
    output=Path(output);output.mkdir(exist_ok=True);rows=[]
    for path in sorted(Path(root).glob('*_seed*.json')):
        p=json.loads(path.read_text())
        for client,counts in p['counts'].items():
            for name,count in zip(p['classes'],counts):
                rows.append(dict(kind=p['kind'],value=p.get('alpha',p.get('ratio')),seed=p['seed'],client_id=int(client),class_name=name,count=count))
    pd.DataFrame(rows).to_csv(output/'partition_counts.csv',index=False)
    df=pd.read_csv(output/'partition_counts.csv');d=df[(df.seed==42)&(df.kind=='dirichlet')]
    classes=p['classes'];fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    vmax=d[d.value.isin([.1,10])]['count'].max()
    for ax,value in zip(axes,[.1,10]):
        m=d[d.value==value].pivot(index='client_id',columns='class_name',values='count')[classes]
        im=ax.imshow(m,aspect='auto',vmin=0,vmax=vmax,cmap='Blues');ax.set(title=f'alpha = {value:g}',ylabel='Client',xlabel='Class');ax.set_xticks(range(10),classes,rotation=90);ax.set_yticks(range(10))
    fig.colorbar(im,ax=axes,label='Training images');fig.savefig(output/'client_counts.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(10)
    for i,value in enumerate([10,50,100]):
        g=df[(df.seed==42)&(df.kind=='longtail')&(df.value==value)].set_index('class_name').loc[classes]
        ax.bar(x+(i-1)*.25,g['count'],.25,label=f'Ratio {value}')
    ax.set_xticks(x,classes,rotation=45,ha='right');ax.set_ylabel('Training images');ax.legend();fig.savefig(output/'longtail_counts.png',dpi=170);plt.close(fig)
    print(f'Frozen partition diagnostics written to {output}')
if __name__=='__main__':inspect()
