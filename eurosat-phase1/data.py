"""Fixed stratified splits; augmentation is applied only to training images."""
import hashlib
import json
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = sorted([
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake',
])
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_transform(size=224, train=False):
    steps = [transforms.Resize((size, size), antialias=True)]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def prepare_split(root, destination, seed=42):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise FileExistsError(f'{destination} already exists. Reuse it, or choose a new filename.')
    if not all((root / name).is_dir() for name in CLASSES):
        raise ValueError('data-dir must directly contain all 10 EuroSAT class folders.')
    records, labels = [], []
    for label, name in enumerate(CLASSES):
        paths = sorted(p for p in (root / name).rglob('*')
                       if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if len(paths) < 10:
            raise ValueError(f'Too few images in {name}: {len(paths)}')
        for path in paths:
            records.append({'path': path.relative_to(root).as_posix(), 'label': label})
            labels.append(label)
    train, remaining = train_test_split(list(range(len(records))), test_size=0.30,
                                       stratify=labels, random_state=seed)
    val, test = train_test_split(remaining, test_size=0.50,
                               stratify=[labels[i] for i in remaining], random_state=seed)
    manifest = {'seed': seed, 'classes': CLASSES,
                'splits': {key: [records[i] for i in indices]
                           for key, indices in [('train', train), ('val', val), ('test', test)]}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print({key: len(rows) for key, rows in manifest['splits'].items()})
    if len(records) != 27000:
        print(f'NOTE: Found {len(records)} images; full EuroSAT RGB contains 27000.')


class EuroSATDataset(Dataset):
    def __init__(self, root, records, transform, client_id=None, partition=None):
        if client_id is not None:
            if partition is None:
                raise ValueError("client_id requires a partition mapping of client IDs to relative paths")
            requested = partition[str(client_id)]
            if len(requested) != len(set(requested)):
                raise ValueError("Duplicate client paths")
            allowed = set(requested)
            if not allowed or not allowed.issubset({r["path"] for r in records}):
                raise ValueError("Client paths must be a nonempty subset of the supplied records")
            records = [r for r in records if r["path"] in allowed]
        self.root, self.records, self.transform = Path(root), records, transform
        missing = [r['path'] for r in records if not (self.root / r['path']).is_file()]
        if missing:
            raise FileNotFoundError(f'Missing {len(missing)} images, e.g. {missing[0]}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, record['label']
