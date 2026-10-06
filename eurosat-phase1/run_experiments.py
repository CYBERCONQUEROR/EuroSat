"""Run resolution experiments, select using validation, then compare norms."""
import argparse
from pathlib import Path
import subprocess
import sys

from experiment_utils import fingerprint, read_json, save_json, select_size, validate_manifest

BASE = Path(__file__).resolve().parent


def call(script, args):
    subprocess.run([sys.executable, str(BASE / script), *map(str, args)], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--split', required=True, help='Your EXISTING splits.json; never regenerated.')
    p.add_argument('--output', default='outputs/phase1')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2024])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--pretrained-lr', type=float, default=1e-4)
    p.add_argument('--scratch-lr', type=float, default=1e-3)
    p.add_argument('--stage', choices=['train', 'evaluate'], default='train')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        p.error('Supply at least three unique seeds.')
    if min(args.epochs, args.patience) < 1 or args.batch_size < 2 or args.workers < 0:
        p.error('Invalid run settings')
    manifest = read_json(args.split)
    validate_manifest(manifest)
    root = Path(args.output).resolve()
    protocol = {k: v for k, v in vars(args).items() if k not in ['stage', 'dry_run', 'output']}
    protocol.update(data_dir=str(Path(args.data_dir).resolve()), split=str(Path(args.split).resolve()),
                    split_hash=fingerprint(manifest), selection_threshold=0.005)
    if args.dry_run:
        print('Resolution: pretrained BN sizes 224, 96, 64 x', args.seeds)
        print('Then: scratch BN and GN at the validation-selected size x', args.seeds)
        print('Total model trainings:', 5 * len(args.seeds), '; test evaluation is a separate stage.')
        return
    root.mkdir(parents=True, exist_ok=True)
    protocol_file = root / 'protocol.json'
    if protocol_file.exists():
        if read_json(protocol_file) != protocol:
            raise ValueError('Protocol changed. Use a new output root for a different experiment.')
    else:
        if args.stage == 'evaluate':
            raise ValueError('Train and select the configurations before test evaluation.')
        save_json(protocol_file, protocol)

    def run(size, norm, scratch, seed):
        name = f'{"scratch" if scratch else "imagenet"}_{norm}_{size}_seed{seed}'
        directory = root / name
        if (directory / 'status.json').exists() and read_json(directory / 'status.json')['status'] == 'completed':
            if not (directory / 'best_model.pt').is_file() or not (directory / 'best_validation_metrics.json').is_file():
                raise ValueError(f'Incomplete artifacts: {directory}')
            print('Reuse completed run:', name)
            return directory
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f'Incomplete run: {directory}. Archive/rename that folder, then rerun. No silent overwrite.')
        command = ['--data-dir', protocol['data_dir'], '--split', protocol['split'], '--output', directory,
                   '--image-size', size, '--norm', norm, '--seed', seed, '--epochs', args.epochs,
                   '--patience', args.patience, '--batch-size', args.batch_size, '--workers', args.workers,
                   '--device', args.device, '--lr', args.scratch_lr if scratch else args.pretrained_lr]
        if scratch:
            command.append('--from-scratch')
        call('train.py', command)
        return directory

    if args.stage == 'train':
        scores, runs = {}, []
        for size in [224, 96, 64]:
            scores[size] = []
            for seed in args.seeds:
                directory = run(size, 'batch', False, seed)
                runs.append(directory.name)
                scores[size].append(read_json(directory / 'best_validation_metrics.json')['macro_f1'])
        size = select_size(scores)
        save_json(root / 'selection.json', {'selected_size': size, 'scores': scores,
                  'metric': 'mean validation macro-F1 at lowest-validation-loss checkpoint',
                  'rule': 'smallest size with drop strictly below 0.005 from 224; otherwise 224',
                  'split_hash': protocol['split_hash']})
        for norm in ['batch', 'group']:
            for seed in args.seeds:
                runs.append(run(size, norm, True, seed).name)
        save_json(root / 'locked_runs.json', {'runs': runs, 'protocol_hash': fingerprint(protocol)})
        call('summarize_results.py', ['--root', root])
        print('Training finished. Review validation results before running --stage evaluate.')
    else:
        lock = read_json(root / 'locked_runs.json')
        if lock['protocol_hash'] != fingerprint(protocol):
            raise ValueError('Locked protocol mismatch')
        for name in lock['runs']:
            directory = root / name
            if not (directory / 'test_metrics.json').exists():
                call('evaluate.py', ['--data-dir', protocol['data_dir'], '--checkpoint', directory / 'best_model.pt',
                                    '--device', args.device, '--batch-size', args.batch_size])
        call('summarize_results.py', ['--root', root])


if __name__ == '__main__':
    main()
