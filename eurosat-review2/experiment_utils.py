"""Pure-Python provenance and selection helpers; usable without PyTorch."""
import hashlib
import json
import os
from pathlib import Path
from statistics import mean


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_manifest(manifest):
    classes = manifest['classes']
    if len(classes) != 10 or len(set(classes)) != 10:
        raise ValueError('Expected ten unique classes.')
    seen = set()
    for split in ['train', 'val', 'test']:
        rows = manifest['splits'][split]
        if not rows:
            raise ValueError(f'Empty {split} split')
        present = set()
        for row in rows:
            path, label = row['path'], row['label']
            if path in seen:
                raise ValueError(f'Duplicate path or split overlap: {path}')
            if not isinstance(label, int) or not 0 <= label < len(classes):
                raise ValueError(f'Invalid label: {label}')
            if Path(path).is_absolute() or '..' in Path(path).parts:
                raise ValueError('Manifest paths must be relative and remain inside data-dir.')
            present.add(label)
            seen.add(path)
        if present != set(range(len(classes))):
            raise ValueError(f'All ten classes must be represented in Phase 1 {split}.')


def select_size(scores, threshold=0.005):
    """Smallest input whose mean validation macro-F1 drops by <0.5 pp."""
    baseline = mean(scores[224])
    for size in [64, 96]:
        if baseline - mean(scores[size]) < threshold:
            return size
    return 224
