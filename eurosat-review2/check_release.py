"""Read-only release gate. Tag only after this succeeds on real complete data."""
import argparse
import json
from pathlib import Path
import subprocess
import pandas as pd
from results_review import validate_final
from review_workflow import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/review2');p.add_argument('--requirements',default='requirements.txt')
    args=p.parse_args();root=Path(args.root)
    df=pd.read_csv(root/'results.csv');final=validate_final(root,df)
    requirements=Path(args.requirements)
    if not requirements.exists() or not requirements.read_text().strip():raise ValueError('Missing pip-freeze environment')
    if requirements.read_bytes()!=(root/'requirements.txt').read_bytes():raise ValueError('Repository requirements differ from training snapshot')
    for rid in final.run_id:
        d=root/'runs'/rid
        if sha256(d/'best_model.pt')!=json.loads((d/'completion.json').read_text())['sha256']['best_model.pt']:
            raise ValueError('Checkpoint checksum mismatch: '+rid)
    proc=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True)
    if proc.returncode:raise RuntimeError('Run inside your Git repository after committing the reviewed outputs')
    dirty=subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,check=True).stdout
    if dirty.strip():raise RuntimeError('Working tree is dirty. Commit the final code, locks, manifests and results first.')
    print('Numerical/environment/checkpoint gate passed at commit '+proc.stdout.strip())
    print('Publish checkpoint archives and record their download URL and SHA-256 before tagging phase1-final.')
    print('This script does not upload files or create a misleading final tag.')
if __name__=='__main__':main()
