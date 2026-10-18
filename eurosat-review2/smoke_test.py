"""End-to-end CPU check on synthetic images; not a EuroSAT result."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image
from experiment_utils import fingerprint

BASE=Path(__file__).resolve().parent

def main():
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    checks=[]
    with tempfile.TemporaryDirectory(prefix='eurosat_smoke_') as temp:
        root=Path(temp);data=root/'data';rng=np.random.RandomState(42)
        classes=[f'class_{i}' for i in range(10)];manifest=dict(seed=42,classes=classes,splits={})
        for split,n in [('train',3),('val',1),('test',1)]:
            rows=[]
            for label,cls in enumerate(classes):
                (data/cls).mkdir(parents=True,exist_ok=True)
                for i in range(n):
                    rel=f'{cls}/{split}_{i}.jpg';Image.fromarray(rng.randint(0,256,(64,64,3),dtype=np.uint8)).save(data/rel)
                    rows.append(dict(path=rel,label=label))
            manifest['splits'][split]=rows
        split=root/'split.json';split.write_text(json.dumps(manifest))
        selected=[x for x in manifest['splits']['train'] if x['label']<4]
        partition=dict(kind='longtail',seed=42,num_clients=1,parent_split_hash=fingerprint(manifest),classes=classes,
                       clients={'0':[x['path'] for x in selected]},counts={'0':[3]*4+[0]*6})
        part=root/'subset.json';part.write_text(json.dumps(partition))
        for loss,norm in [('cross-entropy','batch'),('class-balanced','group'),('focal','group')]:
            output=root/loss
            cmd=[sys.executable,str(BASE/'train.py'),'--data-dir',str(data),'--split',str(split),'--output',str(output),
                 '--epochs','1','--patience','10','--batch-size','8','--image-size','96','--from-scratch','--norm',norm,
                 '--loss',loss,'--device','cpu','--workers','0']
            if loss!='cross-entropy':cmd.extend(['--partition',str(part),'--client-id','0'])
            proc=subprocess.run(cmd,env=env,capture_output=True,text=True)
            if proc.returncode:raise RuntimeError(proc.stdout+'\n'+proc.stderr)
            status=json.loads((output/'status.json').read_text());assert status['status']=='completed'
            assert (output/'environment_requirements.txt').stat().st_size>0
            config=json.loads((output/'config.json').read_text())
            assert config['train_size']==(30 if loss=='cross-entropy' else 12)
            proc=subprocess.run([sys.executable,str(BASE/'evaluate.py'),'--data-dir',str(data),
                                 '--checkpoint',str(output/'best_model.pt'),'--device','cpu'],env=env,capture_output=True,text=True)
            if proc.returncode:raise RuntimeError(proc.stdout+'\n'+proc.stderr)
            metrics=json.loads((output/'test_metrics.json').read_text())
            assert sum(map(sum,metrics['confusion_matrix']))==10
            assert metrics['checkpoint_epoch']==1
            checks.append(dict(loss=loss,norm=norm,passed=True))
    print(json.dumps(dict(synthetic_images_only=True,checks=checks),indent=2))

if __name__=='__main__':main()
