"""Phase 1: validation-selected training. Test data are never loaded here."""
import argparse
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import EuroSATDataset, make_transform
from experiment_utils import fingerprint, read_json, save_json, validate_manifest
from metrics import classification_metrics
from model import build_model
from utils import get_device, seed_everything, seed_worker


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def run_epoch(model, loader, device, classes, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, count, targets, predictions = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc='Train' if training else 'Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; inspect inputs and learning rate.')
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            count += labels.size(0)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
    result = classification_metrics(targets, predictions, classes)
    result['loss'] = loss_sum / count
    return result


def save_curves(history, output):
    for metric in ['accuracy', 'loss']:
        fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
        for split in ['train', 'val']:
            scale = 100 if metric == 'accuracy' else 1
            ax.plot([r['epoch'] for r in history], [r[f'{split}_{metric}'] * scale for r in history],
                    label='Training' if split == 'train' else 'Validation')
        ax.set(xlabel='Epoch', ylabel='Accuracy (%)' if metric == 'accuracy' else 'Cross-entropy loss',
               title=f'Training vs validation {metric}')
        ax.legend()
        ax.grid(alpha=.2)
        for ext in ['png', 'pdf']:
            fig.savefig(output / f'{metric}_curve.{ext}', dpi=180)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--split', default='splits.json')
    p.add_argument('--output', default='outputs/phase1_single')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--image-size', type=int, choices=[64, 96, 224], default=224)
    p.add_argument('--norm', choices=['batch', 'group'], default='batch')
    p.add_argument('--loss', choices=['cross-entropy'], default='cross-entropy')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--from-scratch', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.patience) < 1 or args.batch_size < 2 or args.workers < 0 or args.lr <= 0 or args.weight_decay < 0:
        p.error('Invalid training settings.')
    if args.norm == 'group' and not args.from_scratch:
        p.error('GroupNorm requires --from-scratch in Phase 1.')
    seed_everything(args.seed)
    device = get_device(args.device)
    manifest = read_json(args.split)
    validate_manifest(manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a new empty output directory; previous runs are never overwritten.')
    config = vars(args).copy()
    config.update(split_hash=fingerprint(manifest), checkpoint_rule='minimum_validation_cross_entropy',
                  initialization='scratch' if args.from_scratch else 'imagenet',
                  groupnorm_groups=32 if args.norm == 'group' else None)
    save_json(output / 'config.json', config)
    save_json(output / 'splits.json', manifest)
    save_json(output / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
              'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
              'cuda_runtime': torch.version.cuda, 'device': str(device),
              'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
              'command': sys.argv})
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (output / 'environment_requirements.txt').write_text(freeze.stdout, encoding='utf-8')
    started = time.perf_counter()
    loop_start = None
    status = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'best_epoch': None, 'epochs_completed': 0, 'stopping_reason': None}
    save_json(output / 'status.json', status)
    history = []
    try:
        loaders = {}
        for split in ['train', 'val']:
            dataset = EuroSATDataset(args.data_dir, manifest['splits'][split],
                                     make_transform(args.image_size, train=split == 'train'))
            loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == 'train',
                                        num_workers=args.workers, pin_memory=device.type == 'cuda',
                                        worker_init_fn=seed_worker,
                                        generator=torch.Generator().manual_seed(args.seed))
        model = build_model(len(manifest['classes']), not args.from_scratch, args.norm).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=2)
        best, stale = math.inf, 0
        synchronize(device)
        loop_start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train = run_epoch(model, loaders['train'], device, manifest['classes'], optimizer)
            val = run_epoch(model, loaders['val'], device, manifest['classes'])
            synchronize(device)
            row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                   'epoch_seconds': time.perf_counter() - epoch_start}
            for split, values in [('train', train), ('val', val)]:
                for key in ['loss', 'accuracy', 'macro_f1', 'balanced_accuracy']:
                    row[f'{split}_{key}'] = values[key]
            history.append(row)
            scheduler.step(val['loss'])
            if val['loss'] < best:
                best, stale = val['loss'], 0
                checkpoint = {'model_state': model.state_dict(), 'classes': manifest['classes'],
                              'image_size': args.image_size, 'split_hash': config['split_hash'],
                              'epoch': epoch, 'val_loss': best, 'val_accuracy': val['accuracy'], 'config': config}
                torch.save(checkpoint, output / 'best_model.tmp')
                os.replace(output / 'best_model.tmp', output / 'best_model.pt')
                save_json(output / 'best_validation_metrics.json', dict(val, epoch=epoch))
                status['best_epoch'] = epoch
            else:
                stale += 1
            with (output / 'history.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            status['epochs_completed'] = epoch
            status['elapsed_loop_seconds'] = time.perf_counter() - loop_start
            save_json(output / 'status.json', status)
            print(f"Epoch {epoch}: train acc={train['accuracy']:.4f}, val acc={val['accuracy']:.4f}, "
                  f"val F1={val['macro_f1']:.4f}, val loss={val['loss']:.5f}", flush=True)
            if stale >= args.patience:
                status['stopping_reason'] = 'early_stopping'
                break
        status.update(status='completed', stopping_reason=status['stopping_reason'] or 'max_epochs')
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      stopping_reason=type(error).__name__, error=str(error))
        raise
    finally:
        synchronize(device)
        finished = time.perf_counter()
        status.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                      total_seconds=finished - started,
                      training_loop_seconds=None if loop_start is None else finished - loop_start)
        save_json(output / 'status.json', status)
        (output / 'training_time.txt').write_text(
            f"Training + validation loop seconds: {status['training_loop_seconds']}\n"
            f"Total seconds including model/data setup: {status['total_seconds']:.3f}\n", encoding='utf-8')
        if history:
            save_curves(history, output)
    print(f'Completed: {output}; best epoch {status["best_epoch"]}')


if __name__ == '__main__':
    main()
