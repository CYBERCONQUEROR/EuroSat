"""Rebuild aggregate tables from artifacts; never hand-enter measurements."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from experiment_utils import read_json


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    rows = []
    for config_path in sorted(root.glob('*/config.json')):
        folder = config_path.parent
        if not (folder / 'status.json').exists():
            continue
        status = read_json(folder / 'status.json')
        if status['status'] != 'completed':
            continue
        config = read_json(config_path)
        val = read_json(folder / 'best_validation_metrics.json')
        row = {key: config[key] for key in ['seed', 'image_size', 'norm', 'initialization', 'split_hash']}
        row.update(run=folder.name, best_epoch=status['best_epoch'], epochs_completed=status['epochs_completed'],
                   stopping_reason=status['stopping_reason'], training_loop_seconds=status['training_loop_seconds'])
        for scope, metrics in [('val', val), ('test', read_json(folder / 'test_metrics.json')
                                               if (folder / 'test_metrics.json').exists() else None)]:
            if metrics is not None:
                for key in ['accuracy', 'macro_f1', 'balanced_accuracy']:
                    row[f'{scope}_{key}'] = metrics[key]
                for cls, value in metrics['per_class_recall'].items():
                    row[f'{scope}_recall_{cls}'] = value
        rows.append(row)
    write_csv(root / 'results.csv', rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['initialization'], row['norm'], row['image_size'], row['split_hash'])].append(row)
    summary, table = [], []
    for (init, norm, size, split_hash), group in sorted(groups.items()):
        if len({r['seed'] for r in group}) != len(group):
            raise ValueError('Duplicate seed in configuration group.')
        result = dict(initialization=init, norm=norm, image_size=size, split_hash=split_hash,
                      seeds=';'.join(str(r['seed']) for r in group), n=len(group))
        for key in ['training_loop_seconds', 'best_epoch'] + [k for k in group[0] if k.startswith(('val_', 'test_'))]:
            values = [r[key] for r in group if key in r]
            if len(values) != len(group):
                continue  # Never report partial-seed test summaries as complete.
            result[key + '_mean'] = mean(values)
            result[key + '_std'] = stdev(values) if len(values) > 1 else None
        summary.append(result)
        def cell(key, scale=100):
            if key + '_mean' not in result:
                return 'pending'
            sd = result[key + '_std']
            return f"{result[key + '_mean'] * scale:.2f} ± {sd * scale:.2f}" if sd is not None else 'insufficient seeds'
        table.append([init, norm, str(size), str(len(group)), cell('val_macro_f1'), cell('test_accuracy'),
                      cell('test_macro_f1'), cell('test_balanced_accuracy'), cell('training_loop_seconds', 1)])
    write_csv(root / 'summary.csv', summary)
    headers = ['Initialisation', 'Norm', 'Size', 'Seeds', 'Val macro-F1 (%)', 'Test accuracy (%)',
               'Test macro-F1 (%)', 'Test balanced acc. (%)', 'Loop seconds']
    markdown = '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n'
    markdown += '\n'.join('| ' + ' | '.join(row) + ' |' for row in table)
    markdown += '\n\nMean ± sample standard deviation (ddof=1). Validation-selected checkpoints. Pending means not evaluated.\n'
    (root / 'phase1_results.md').write_text(markdown, encoding='utf-8')
    latex = ['% Requires \\usepackage{booktabs}', '\\begin{table*}[t]', '\\centering', '\\scriptsize',
             '\\caption{Phase 1 results: mean $\\pm$ sample standard deviation across seeds. Scores are percentages.}',
             '\\begin{tabular}{lllrrrrrr}', '\\toprule',
             'Init. & Norm & Size & Seeds & Val F1 & Test acc. & Test F1 & Test bal. acc. & Seconds \\\\', '\\midrule']
    latex += [' & '.join(row).replace('±', '$\\pm$') + ' \\\\' for row in table]
    latex += ['\\bottomrule', '\\end{tabular}', '\\label{tab:phase1}', '\\end{table*}']
    (root / 'phase1_results.tex').write_text('\n'.join(latex), encoding='utf-8')
    print(f'Summarized {len(rows)} completed runs into {root}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs/phase1')
    summarize(Path(p.parse_args().root))
