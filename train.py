"""Fine-tune every ResNet-18 layer, selecting checkpoints on validation loss."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import EuroSATDataset, load_manifest, make_transform, manifest_hash
from model import build_model
from utils import get_device, seed_everything, seed_worker


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc='Train' if training else 'Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / count, correct / count

    
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--split', default='splits.json')
    p.add_argument('--output', default='outputs/run1')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--from-scratch', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.image_size < 32 or args.workers < 0:
        p.error('epochs, batch-size, patience must be positive; image-size >=32; workers >=0.')
    if args.batch_size < 2:
        p.error('Use batch-size >=2 for BatchNorm training.')
    seed_everything(args.seed)
    device = get_device(args.device)
    manifest = load_manifest(args.split)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Output directory is not empty. Choose a new --output for each run.')
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    (output / 'splits.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    loaders = {}
    for split in ['train', 'val']:
        dataset = EuroSATDataset(args.data_dir, manifest['splits'][split],
                                 make_transform(args.image_size, train=(split == 'train')))
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=(split == 'train'), num_workers=args.workers,
                                    pin_memory=(device.type == 'cuda'), worker_init_fn=seed_worker,
                                    generator=torch.Generator().manual_seed(args.seed))
    model = build_model(len(manifest['classes']), pretrained=not args.from_scratch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    history, best_loss, stale = [], float('inf'), 0
    print(f'Device: {device}; classes: {manifest["classes"]}')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders['train'], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders['val'], criterion, device)
        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                   train_accuracy=train_acc, val_accuracy=val_acc, lr=optimizer.param_groups[0]['lr'])
        history.append(row)
        print(f'Epoch {epoch:02d} | train loss {train_loss:.4f},train accuracy {train_acc:.2%} '
              f'| val loss {val_loss:.4f}, val accuracy {val_acc:.2%}')
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save({'model_state': model.state_dict(), 'classes': manifest['classes'],
                        'image_size': args.image_size, 'split_hash': manifest_hash(manifest),
                        'epoch': epoch, 'val_loss': val_loss, 'val_accuracy': val_acc,
                        'config': vars(args)}, output / 'best_model.pt')
        else:
            stale += 1
        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if stale >= args.patience:
            print('Early stopping: validation loss stopped improving.')
            break
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        for split in ['train', 'val']:
            ax.plot([r['epoch'] for r in history], [r[f'{split}_{metric}'] for r in history], label=split)
        ax.set(xlabel='Epoch', ylabel=metric.title())
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png', dpi=160)
    plt.close(fig)
    print(f'Best checkpoint: {output / "best_model.pt"}')


if __name__ == '__main__':
    main()
