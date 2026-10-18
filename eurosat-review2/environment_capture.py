"""Capture the ACTIVE training environment; never fabricate historical packages."""
import argparse
from datetime import datetime, timezone
import importlib.metadata as md
import json
from pathlib import Path
import platform
import subprocess
import sys

def capture(output, require_torch=True):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    proc=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
    if proc.returncode or not proc.stdout.strip():
        raise RuntimeError('pip freeze failed or returned no packages. Install pip in this environment. '+proc.stderr)
    if require_torch:
        for package in ['torch','torchvision','numpy','scikit-learn','Pillow','matplotlib','pandas','tqdm']:md.version(package)
    output.write_text(proc.stdout,encoding='utf-8')
    return dict(python=sys.version,platform=platform.platform(),captured_utc=datetime.now(timezone.utc).isoformat(),
                packages={d.metadata['Name']:d.version for d in md.distributions() if d.metadata['Name']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='requirements.txt');args=p.parse_args()
    data=capture(args.output)
    Path(args.output+'.environment.json').write_text(json.dumps(data,indent=2))
    print(f'Exact active-environment snapshot written to {args.output}')
