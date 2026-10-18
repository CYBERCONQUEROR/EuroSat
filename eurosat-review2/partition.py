"""Immutable training-only partitions. Label-dir adapted from NIID-Bench.
Upstream: Xtra-Computing/NIID-Bench, partition.py, noniid-labeldir branch,
commit 61b901ab645e62391772e9898384c2c3485b7a6c (MIT; see third_party).
The original balancing mask and legacy NumPy RNG are retained. Retry draws
advance the same seeded RNG; resetting it inside the loop would never recover.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiment_utils import fingerprint, read_json, validate_manifest

SEEDS = [42, 123, 2024, 7, 99]
ALPHAS = [0.1, 0.5, 1.0, 10.0]
RATIOS = [10, 50, 100]
UPSTREAM = '61b901ab645e62391772e9898384c2c3485b7a6c'

def freeze(path, value):
    path = Path(path)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Frozen file differs: {path}. Use a new output folder.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f: f.write(text)
    return path

def label_dirichlet(labels, alpha, num_clients, seed, min_size=10, max_attempts=10000):
    labels = np.asarray(labels)
    if not np.isfinite(alpha) or alpha <= 0 or num_clients < 1 or min_size < 1:
        raise ValueError('Invalid partition parameters')
    if len(labels) < num_clients * min_size:
        raise ValueError('Insufficient images for the minimum client size')
    rng = np.random.RandomState(seed)
    for attempt in range(1, max_attempts + 1):
        batches = [[] for _ in range(num_clients)]
        for k in sorted(np.unique(labels)):
            indices = np.flatnonzero(labels == k)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            mask = np.array([len(x) < len(labels) / num_clients for x in batches])
            proportions *= mask
            if proportions.sum() <= 0:  # floating-point underflow safeguard
                proportions = mask.astype(float)
            if proportions.sum() <= 0: raise RuntimeError('Invalid balancing mask')
            cuts = (np.cumsum(proportions / proportions.sum()) * len(indices)).astype(int)[:-1]
            batches = [old + new.tolist() for old, new in zip(batches, np.split(indices, cuts))]
        if min(map(len, batches)) >= min_size:
            for batch in batches: rng.shuffle(batch)
            return batches, attempt
    raise RuntimeError('Could not satisfy minimum client size; no partial partition saved')

def longtail_order(records, classes, order_seed=42, min_ratio=10):
    """Seeded fixed permutation feasible without replacement at every ratio.
    Head must be a largest available class. Rejection avoids clipping n_k when
    a naturally smaller EuroSAT class happens to be assigned an early rank.
    """
    counts = np.bincount([r['label'] for r in records], minlength=classes)
    nmax = int(counts.max())
    rng = np.random.RandomState(order_seed)
    for _ in range(10000):
        order = rng.permutation(classes)
        targets = np.array([int(nmax * min_ratio ** (-k / (classes - 1))) for k in range(classes)])
        if counts[order[0]] == nmax and np.all(counts[order] >= targets):
            return order.tolist()
    raise RuntimeError('No feasible class order found')

def generate(manifest, kind, seed, alpha=None, ratio=None, num_clients=10, order_seed=42):
    validate_manifest(manifest)
    records = manifest['splits']['train']; C = len(manifest['classes'])
    result = dict(schema=1, kind=kind, seed=seed, parent_split_hash=fingerprint(manifest),
                  classes=manifest['classes'], num_clients=num_clients if kind == 'dirichlet' else 1)
    if kind == 'dirichlet':
        batches, attempts = label_dirichlet([r['label'] for r in records], alpha, num_clients, seed)
        result.update(alpha=float(alpha), attempts=attempts, min_client_size=10, upstream_commit=UPSTREAM)
    elif kind == 'longtail':
        if ratio not in RATIOS: raise ValueError('Review ratios are 10, 50, 100')
        order = longtail_order(records, C, order_seed)
        nmax = max(sum(r['label'] == c for r in records) for c in range(C))
        rng = np.random.RandomState(seed); selected = []; desired = []
        for rank, c in enumerate(order):
            indices = np.array([i for i, row in enumerate(records) if row['label'] == c])
            rng.shuffle(indices)
            count = max(1, int(nmax * ratio ** (-rank / (C - 1))))
            if count > len(indices): raise ValueError('Infeasible long-tail target; cannot oversample')
            selected.extend(indices[:count].tolist()); desired.append(count)
        rng.shuffle(selected); batches = [selected]
        result.update(ratio=int(ratio), class_order_seed=order_seed, class_order=order,
                      rank_counts=desired, realized_ratio=max(desired)/min(desired))
    else: raise ValueError(kind)
    result['clients'] = {str(i): [records[j]['path'] for j in batch] for i, batch in enumerate(batches)}
    result['counts'] = {str(i): np.bincount([records[j]['label'] for j in batch], minlength=C).tolist()
                        for i, batch in enumerate(batches)}
    validate_partition(manifest, result)
    return result

def validate_partition(manifest, partition):
    if partition['parent_split_hash'] != fingerprint(manifest): raise ValueError('Partition belongs to another split')
    if partition['classes'] != manifest['classes']: raise ValueError('Class order mismatch')
    allowed = {r['path']: r['label'] for r in manifest['splits']['train']}; seen = set()
    for cid, paths in partition['clients'].items():
        if not paths or len(paths) != len(set(paths)): raise ValueError('Empty or duplicate client paths')
        if not set(paths).issubset(allowed): raise ValueError('Partition includes paths outside training')
        if seen.intersection(paths): raise ValueError('Clients overlap')
        seen.update(paths)
        counts = np.bincount([allowed[p] for p in paths], minlength=len(manifest['classes'])).tolist()
        if counts != partition['counts'][cid]: raise ValueError('Class counts do not match paths')
        if partition['kind'] == 'dirichlet' and len(paths) < 10: raise ValueError('Client has fewer than 10 images')
    if len(partition['clients']) != partition['num_clients']: raise ValueError('Client count mismatch')
    if partition['kind'] == 'dirichlet' and seen != set(allowed): raise ValueError('Dirichlet must cover the training set')

def partition_name(kind, value, seed):
    value = f'{value:g}'
    return f'{kind}_{value}_seed{seed}.json'

def prepare_all(split, destination):
    manifest = read_json(split); destination = Path(destination)
    for seed in SEEDS:
        for alpha in ALPHAS:
            freeze(destination/partition_name('dirichlet',alpha,seed), generate(manifest,'dirichlet',seed,alpha=alpha))
        for ratio in RATIOS:
            freeze(destination/partition_name('longtail',ratio,seed), generate(manifest,'longtail',seed,ratio=ratio))
    freeze(destination/'manifest.json', dict(split_hash=fingerprint(manifest),seeds=SEEDS,alphas=ALPHAS,
                                            ratios=RATIOS,num_clients=10,class_order_seed=42))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='splits.json');p.add_argument('--output',default='partitions')
    args=p.parse_args();prepare_all(args.split,args.output);print('35 immutable training partitions ready')
