"""Archive actual checkpoints and SHA-256 manifests, never placeholder weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from datetime import datetime, timezone

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def package(root,output,final_only=False):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if final_only:
        plan=json.loads((root/'final_plan.json').read_text())
        paths=[root/'runs'/rid/'best_model.pt' for rid in plan['run_ids']]
    else:paths=sorted(root.rglob('best_model.pt'))
    if not paths or any(not p.is_file() for p in paths):raise ValueError('Actual checkpoint files are required')
    rows=[dict(file=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in paths]
    manifest=output.with_suffix('.manifest.json')
    if manifest.exists():raise FileExistsError(manifest)
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),models=rows),indent=2))
    # Uncompressed tar avoids a lengthy CPU compression pass over tensor weights.
    with tarfile.open(output,'w') as tar:
        for p in paths:tar.add(p,arcname=p.relative_to(root).as_posix())
        tar.add(manifest,arcname='CHECKPOINT_SHA256.json')
    print(json.dumps(dict(archive=str(output),archive_sha256=digest(output),models=len(rows),manifest=str(manifest)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--final-only',action='store_true')
    a=p.parse_args();package(a.root,a.output,a.final_only)
