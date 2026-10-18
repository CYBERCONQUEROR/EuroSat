"""Read-only audit of saved Phase 2 experiments; exports never invoke training/inference.

Run from this directory with ../.venv/bin/python -B -u audit_phase2_closure.py.
Creates a new directory and refuses to replace an existing audit.
"""
import csv
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from experiment_utils import fingerprint, read_json, validate_manifest
from metrics import classification_metrics
from partition import generate, partition_name, validate_partition
from results_review import validate_final, LOSSES, LABELS, COLORS
from review_workflow import CORE, final_jobs, gn_jobs, tuning_jobs, candidates, sha256

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'outputs/review2'
OUT = ROOT / 'phase2_closure_20261018'
SEEDS = [42, 123, 2024, 7, 99]
HASHES = {}
STATS = {}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    path = Path(path).resolve()
    if str(path) not in HASHES:
        st = path.stat()
        HASHES[str(path)] = sha256(path)
        STATS[str(path)] = (st.st_size, st.st_mtime_ns)
    return HASHES[str(path)]


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def close(a, b, label):
    if isinstance(a, dict):
        check(a.keys() == b.keys(), f'{label}: keys differ')
        for key in a:
            close(a[key], b[key], f'{label}/{key}')
    elif isinstance(a, (list, tuple)):
        check(len(a) == len(b), f'{label}: lengths differ')
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f'{label}/{i}')
    elif isinstance(a, (int, float)):
        check(math.isfinite(a) and math.isfinite(float(b)) and
              math.isclose(a, float(b), rel_tol=1e-11, abs_tol=1e-12), f'{label}: {a} != {b}')
    else:
        check(a == b, f'{label}: values differ')


def copy(src, relative):
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)
    check(digest(src) == sha256(target), f'Copy mismatch: {src}')


def audit():
    torch.set_num_threads(1)
    protocol = read_json(ROOT / 'protocol.json')
    manifest = read_json(ROOT / 'splits.json')
    gn = read_json(ROOT / 'gn_selection.json')
    loss = read_json(ROOT / 'loss_selection.json')
    plan = read_json(ROOT / 'final_plan.json')
    ph = fingerprint(protocol)
    validate_manifest(manifest)
    check(protocol['split_hash'] == fingerprint(manifest), 'Global split mismatch')
    for name in CORE:
        check(digest(BASE / name) == protocol['code_sha256'][name], f'Frozen source changed: {name}')
    expected_jobs = final_jobs(gn['lr'], loss)
    ids = [fingerprint(dict(protocol_hash=ph, job=j))[:24] for j in expected_jobs]
    check(plan == dict(protocol_hash=ph, gn_selection_hash=fingerprint(gn),
                      loss_selection_hash=fingerprint(loss), jobs=expected_jobs, run_ids=ids, n=680),
          'Sealed manifest differs from frozen protocol/selections/canonical jobs')
    check(len(set(ids)) == 680, 'Duplicate final IDs')
    df = pd.read_csv(ROOT / 'results.csv', dtype={'run_id': str})
    final = validate_final(ROOT, df).set_index('run_id', drop=False)
    provenance = read_json(ROOT / 'generated/provenance.json')
    check(provenance['results_sha256'] == digest(ROOT / 'results.csv'), 'Generated CSV provenance mismatch')
    check(provenance['final_plan_hash'] == fingerprint(plan), 'Generated seal provenance mismatch')
    evidence_counts = {}
    for name, selection, jobs in [('gn', gn, gn_jobs()), ('loss', loss, tuning_jobs())]:
        expected = {fingerprint(dict(protocol_hash=ph, job=j))[:24] for j in jobs}
        check(set(selection['evidence']) == expected, f'{name} selection evidence incomplete')
        for rid, sha in selection['evidence'].items():
            directory = ROOT / 'runs' / rid
            check(digest(directory / 'best_validation_metrics.json') == sha, f'Changed selection evidence: {rid}')
            receipt = read_json(directory / 'completion.json')
            check(read_json(directory / 'status.json')['status'] == 'completed', f'Incomplete selection job: {rid}')
            for filename in ['config.json', 'best_validation_metrics.json', 'history.csv']:
                check(digest(directory / filename) == receipt['sha256'][filename], f'Selection receipt mismatch: {rid}/{filename}')
        evidence_counts[name] = len(expected)
    # Recompute selections using validation metrics only, preserving the original tie rules.
    def val(job):
        rid = fingerprint(dict(protocol_hash=ph, job=job))[:24]
        return read_json(ROOT / 'runs' / rid / 'best_validation_metrics.json')['macro_f1']
    gn_scores = {str(lr): sum(val(j) for j in gn_jobs() if j['lr'] == lr) / 3 for lr in protocol['gn_lrs']}
    close(gn['scores'], gn_scores, 'GN validation scores')
    check(gn['lr'] == max(protocol['gn_lrs'], key=lambda lr: (gn_scores[str(lr)], -lr)), 'GN selection mismatch')
    for kind, values in [('dirichlet', protocol['alphas']), ('longtail', protocol['ratios'])]:
        scores = {}
        for l, b, g in candidates():
            seed_scores = []
            for seed in protocol['tuning_seeds']:
                severity_scores = []
                for value in values:
                    matched = [j for j in tuning_jobs() if (j['kind'], j['value'], j['seed'], j['loss'], j['beta'], j['gamma']) == (kind, value, seed, l, b, g)]
                    severity_scores.append(sum(val(j) for j in matched) / len(matched))
                seed_scores.append(sum(severity_scores) / len(severity_scores))
            scores[f'{l}:{b:g}:{g:g}'] = sum(seed_scores) / len(seed_scores)
        close(loss['scores'][kind], scores, f'{kind} validation scores')
        check(loss[kind]['beta'] == max(protocol['betas'], key=lambda b: (scores[f'class-balanced:{b:g}:0'], -b)), 'Beta selection mismatch')
        check(loss[kind]['gamma'] == max(protocol['gammas'], key=lambda g: (scores[f'focal:0:{g:g}'], -g)), 'Gamma selection mismatch')
    partitions = {}
    count_rows = []
    for path in sorted((ROOT / 'partitions').glob('*.json')):
        if path.name == 'manifest.json':
            continue
        part = read_json(path)
        validate_partition(manifest, part)
        kwargs = dict(alpha=part['alpha']) if part['kind'] == 'dirichlet' else dict(ratio=part['ratio'])
        check(part == generate(manifest, part['kind'], part['seed'], **kwargs), f'Partition regeneration mismatch: {path.name}')
        partitions[path.name] = part
        for cid, counts in part['counts'].items():
            for label, count in zip(manifest['classes'], counts):
                count_rows.append(dict(kind=part['kind'], value=part.get('alpha', part.get('ratio')), partition_seed=part['seed'], num_clients=part['num_clients'], client_id=int(cid), class_name=label, train_count=count, partition_file=path.name, partition_hash=fingerprint(part)))
    check(len(partitions) == 35, 'Expected 35 frozen partitions')
    expected_paths = [r['path'] for r in manifest['splits']['test']]
    expected_targets = [r['label'] for r in manifest['splits']['test']]
    classes = manifest['classes']
    names = {name: i for i, name in enumerate(classes)}
    val_support = np.bincount([r['label'] for r in manifest['splits']['val']], minlength=10)
    pooled_counts = np.bincount([r['label'] for r in manifest['splits']['train']], minlength=10).tolist()
    pipeline_dir = ROOT / 'operations/final-after-reboot-20261017T103449Z'
    pipeline = read_json(pipeline_dir / 'pipeline.json')
    previous_end = None
    for stage in ['final-train', 'seal', 'evaluate', 'report']:
        record = pipeline['stage_results'][stage]
        check(record['state'] == 'completed' and record['exit_code'] == 0, f'Pipeline stage failed: {stage}')
        if previous_end:
            check(record['started_utc'] >= previous_end, f'Pipeline stage order: {stage}')
        previous_end = record['finished_utc']
    check(pipeline['exit_code'] == 0 and (pipeline_dir / 'pipeline.exitcode').read_text().strip() == '0', 'Pipeline exit failure')
    seal_end = datetime.fromisoformat(pipeline['stage_results']['seal']['finished_utc']).timestamp()
    records, class_rows = [], []
    instrumented = 0
    for i, (rid, job) in enumerate(zip(ids, expected_jobs), 1):
        d = ROOT / 'runs' / rid
        recipe = read_json(ROOT / 'recipes' / f'{rid}.json')
        check(recipe == dict(protocol_hash=ph, job=job), f'Recipe mismatch: {rid}')
        receipt = read_json(d / 'completion.json')
        for filename, expected in receipt['sha256'].items():
            check(digest(d / filename) == expected, f'Completion hash mismatch: {rid}/{filename}')
        check(set(receipt['sha256']) == {'best_model.pt', 'config.json', 'best_validation_metrics.json', 'history.csv'}, 'Unexpected receipt scope')
        cfg, status, validation, test = [read_json(d / f) for f in ['config.json', 'status.json', 'best_validation_metrics.json', 'test_metrics.json']]
        check(status['status'] == 'completed', f'Incomplete final run: {rid}')
        for key in ['seed', 'loss', 'beta', 'gamma', 'initialization', 'norm', 'image_size', 'lr']:
            check(cfg[key] == job[key], f'Configuration mismatch: {rid}/{key}')
        check(cfg['client_id'] == (None if job['client_id'] == -1 else job['client_id']), f'Client mismatch: {rid}')
        for key in ['epochs', 'patience', 'batch_size', 'weight_decay', 'validation_loss']:
            check(cfg[key] == protocol[key], f'Protocol setting mismatch: {rid}/{key}')
        check(cfg['checkpoint_rule'] == 'minimum_validation_cross_entropy', f'Checkpoint rule mismatch: {rid}')
        check(cfg['recipe_id'] == rid and cfg['split_hash'] == fingerprint(manifest), f'Config identity mismatch: {rid}')
        check(read_json(d / 'splits.json') == manifest, f'Global validation/test split changed: {rid}')
        if job['kind'] == 'pooled':
            counts = pooled_counts
            check(cfg['partition_hash'] is None, f'Unexpected pooled partition: {rid}')
        else:
            part = partitions[partition_name(job['kind'], job['value'], job['seed'])]
            check(read_json(d / 'partition.json') == part and cfg['partition_hash'] == fingerprint(part), f'Run partition changed: {rid}')
            counts = part['counts'][str(job['client_id'])]
        check(cfg['train_counts'] == counts and cfg['train_size'] == sum(counts), f'Training subset/count mismatch: {rid}')
        h = pd.read_csv(d / 'history.csv')
        check(np.isfinite(h.to_numpy(dtype=float)).all(), f'Non-finite history: {rid}')
        check(h.epoch.tolist() == list(range(1, len(h) + 1)) and len(h) == status['epochs_completed'], f'History length mismatch: {rid}')
        best = h.loc[h.val_loss.idxmin()]
        check(int(best.epoch) == status['best_epoch'] == validation['epoch'] == test['checkpoint_epoch'], f'Best checkpoint epoch mismatch: {rid}')
        for key in ['loss', 'accuracy', 'macro_f1', 'balanced_accuracy']:
            close(validation[key], best['val_' + key], f'Best validation {rid}/{key}')
        for name, support in zip(classes, val_support):
            close(validation['classification_report'][name]['support'], support, f'Global validation support: {rid}/{name}')
        check((status['stopping_reason'] == 'early_stopping' and len(h) - status['best_epoch'] == protocol['patience']) or
              (status['stopping_reason'] == 'max_epochs' and len(h) == protocol['epochs']), f'Stopping rule mismatch: {rid}')
        checkpoint = torch.load(d / 'best_model.pt', map_location='cpu', weights_only=True)
        for key, expected in [('config', cfg), ('classes', classes), ('split_hash', fingerprint(manifest)), ('epoch', status['best_epoch']), ('image_size', job['image_size'])]:
            check(checkpoint[key] == expected, f'Checkpoint metadata mismatch: {rid}/{key}')
        close(checkpoint['val_loss'], validation['loss'], f'Checkpoint CE: {rid}')
        for key, tensor in checkpoint['model_state'].items():
            check(torch.isfinite(tensor).all().item(), f'Non-finite saved state: {rid}/{key}')
            if key.endswith('running_var'):
                check((tensor >= 0).all().item(), f'Negative BN variance: {rid}/{key}')
        check('optimizer' not in checkpoint and 'optimizer_state' not in checkpoint, 'Unexpected resume state')
        del checkpoint
        predictions = pd.read_csv(d / 'test_predictions.csv', keep_default_na=False)
        check(predictions.path.tolist() == expected_paths, f'Test image identities/order changed: {rid}')
        targets = [names[x] for x in predictions.actual]
        preds = [names[x] for x in predictions.predicted]
        check(targets == expected_targets, f'Test labels changed: {rid}')
        recomputed = classification_metrics(targets, preds, classes)
        for key, value in recomputed.items():
            close(value, test[key], f'Saved test metrics {rid}/{key}')
        check(confusion_matrix(targets, preds, labels=list(range(10))).tolist() == test['confusion_matrix'], f'Confusion matrix mismatch: {rid}')
        check((d / 'classification_report.txt').read_text() == classification_report(targets, preds, labels=list(range(10)), target_names=classes, zero_division=0), f'Text report mismatch: {rid}')
        check(test['split_hash'] == fingerprint(manifest), f'Test split hash: {rid}')
        check((d / 'test_metrics.json').stat().st_mtime >= seal_end, f'Test artifact predates seal: {rid}')
        csvrow = final.loc[rid]
        for key in job:
            check(csvrow[key] == job[key], f'CSV job mismatch: {rid}/{key}')
        check(csvrow.checkpoint_sha256 == receipt['sha256']['best_model.pt'], f'CSV checkpoint mismatch: {rid}')
        check(json.loads(csvrow.train_counts_json) == counts and int(csvrow.train_size) == sum(counts), f'CSV counts mismatch: {rid}')
        close(json.loads(csvrow.test_report_json), test['classification_report'], f'CSV classification report: {rid}')
        for key in ['accuracy', 'macro_f1', 'balanced_accuracy']:
            close(float(csvrow['test_' + key]), test[key], f'CSV metric: {rid}/{key}')
        has_diagnostics = (d / '_diagnostics/instrumentation.json').exists()
        instrumented += has_diagnostics
        records.append(dict(run_id=rid, **job, train_size=sum(counts), absent_train_classes=sum(x == 0 for x in counts), best_epoch=status['best_epoch'], epochs_completed=len(h), validation_loss=validation['loss'], validation_macro_f1=validation['macro_f1'], test_accuracy=test['accuracy'], test_macro_f1=test['macro_f1'], test_balanced_accuracy=test['balanced_accuracy'], training_loop_seconds=status['training_loop_seconds'], checkpoint_sha256=receipt['sha256']['best_model.pt'], completion_sha256=digest(d / 'completion.json'), test_metrics_sha256=digest(d / 'test_metrics.json'), predictions_sha256=digest(d / 'test_predictions.csv'), partition_hash=cfg['partition_hash'], instrumented=has_diagnostics, verified=True))
        for split, metrics in [('validation', validation), ('test', test)]:
            for label, values in metrics['classification_report'].items():
                if isinstance(values, dict):
                    class_rows.append(dict(run_id=rid, **job, split=split, class_name=label, **values))
        if i % 25 == 0 or i == 680:
            print(f'Audited {i}/680: hashes, CPU model state, subsets, histories, saved predictions and reports', flush=True)
    recovery = read_json(ROOT / 'diagnostics/recovery-20261017T100521Z/recovery_report.json')
    recovered = ROOT / 'runs/8953d19d54840571685e0d64'
    original = Path(recovery['original_archive'])
    check(read_json(recovered / 'config.json') == read_json(original / 'config.json'), 'Recovered configuration changed')
    old_h, new_h = pd.read_csv(original / 'history.csv'), pd.read_csv(recovered / 'history.csv')
    cols = [c for c in old_h if c != 'epoch_seconds']
    check(len(old_h) == 12 and old_h[cols].equals(new_h.iloc[:12][cols]), 'Original first 12 epochs differ')
    check(recovery['completion_sha256'] == read_json(recovered / 'completion.json')['sha256'], 'Recovery receipt changed')
    for name, sha in recovery['diagnostic_files'].items():
        check(digest(BASE / name) == sha, f'Diagnostic source changed: {name}')
    verification = dict(audited_utc=datetime.now(timezone.utc).isoformat(), scope='saved-data audit; no training, inference or hyperparameter changes', final_required=680, final_verified=len(records), test_prediction_rows_verified=len(records) * len(expected_paths), checkpoints_hash_and_tensor_checks=len(records), partitions_regenerated_and_validated=len(partitions), selection_evidence_verified=evidence_counts, instrumented_final_runs=instrumented, protocol_fingerprint=ph, split_fingerprint=fingerprint(manifest), final_plan_fingerprint=fingerprint(plan), final_plan_sha256=digest(ROOT / 'final_plan.json'), results_sha256=digest(ROOT / 'results.csv'), pipeline_stage_order_verified=True, original_recovery_config_identical=True, original_first_12_epoch_metrics_identical=True, scientific_fix_applied=False, failure_cause=recovery['cause'], recovery=recovery, limitations=['Original reviewer documents not located: N=5 supersession unresolved.', 'Original seal binds jobs/protocol/selections; test artifacts have no historical signed receipt. Current audit binds saved predictions and reports by SHA-256; inference was not rerun.', 'Saved configuration and checkpoint metadata establish intended subsets; no per-image training access trace or historical image-byte hash manifest exists.', 'File timestamps corroborate seal-before-evaluation but are not cryptographic timestamps.', 'Checkpoint weights and image bytes are not included in the audit bundle.'])
    save(OUT / 'verification.json', verification)
    data = pd.DataFrame(records)
    data.to_csv(OUT / 'tables/final_run_verification.csv', index=False)
    pd.DataFrame(class_rows).to_csv(OUT / 'tables/classification_reports.csv', index=False)
    pd.DataFrame(count_rows).to_csv(OUT / 'tables/partition_class_counts.csv', index=False)
    data[data.kind == 'dirichlet'].to_csv(OUT / 'tables/per_client_metrics.csv', index=False)
    keys = ['kind', 'value', 'initialization', 'norm', 'image_size', 'loss', 'beta', 'gamma', 'lr']
    metrics = ['test_accuracy', 'test_macro_f1', 'test_balanced_accuracy']
    seedwise = data.groupby(keys + ['seed'], dropna=False)[metrics].mean().reset_index()
    seedwise['models_in_seed'] = data.groupby(keys + ['seed'], dropna=False).size().values
    check(set(seedwise.loc[seedwise.kind == 'dirichlet', 'models_in_seed']) == {10}, 'Client aggregation mismatch')
    seedwise.to_csv(OUT / 'tables/seed_level_metrics.csv', index=False)
    grouped = seedwise.groupby(keys, dropna=False)
    check((grouped.size() == 5).all(), 'Five seed aggregation mismatch')
    summary = grouped[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(x) for x in summary.columns]
    summary = summary.reset_index()
    summary['seed_count'] = 5
    summary.to_csv(OUT / 'tables/five_seed_summary.csv', index=False)
    for key in ['test_accuracy', 'test_macro_f1']:
        fig, ax = plt.subplots(figsize=(7.8, 4.7), layout='constrained')
        for l in LOSSES:
            g = summary[(summary.kind == 'dirichlet') & (summary.loss == l)].sort_values('value')
            ax.errorbar(g.value, g[key + '_mean'] * 100, yerr=g[key + '_std'] * 100, label=LABELS[l], color=COLORS[l], marker='o', capsize=4)
        ax.set(xscale='log', xlabel='Dirichlet alpha (N = 10 clients)', ylabel=('Test accuracy (%)' if key == 'test_accuracy' else 'Test macro-F1 (%)'), title='Mean of 10 clients per seed; five-seed mean ± sample SD')
        ax.set_xticks([.1, .5, 1, 10], ['0.1', '0.5', '1', '10'])
        ax.grid(alpha=.25)
        ax.legend()
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / 'figures' / f'{key}_versus_alpha.{ext}', dpi=200)
        plt.close(fig)
    counts_df = pd.DataFrame(count_rows)
    maximum = counts_df[counts_df.kind == 'dirichlet'].train_count.max()
    for seed in SEEDS:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout='constrained')
        for alpha, ax in zip([.1, .5, 1, 10], axes.flat):
            part = partitions[partition_name('dirichlet', alpha, seed)]
            m = np.array([part['counts'][str(cid)] for cid in range(10)])
            im = ax.imshow(m, cmap='Blues', vmin=0, vmax=maximum, aspect='auto')
            ax.set(title=f'alpha = {alpha:g}; partition seed = {seed}', ylabel='Client ID (zero-based)')
            ax.set_xticks(range(10), classes, rotation=65, ha='right', fontsize=8)
            ax.set_yticks(range(10))
            for row in range(10):
                for col in range(10):
                    ax.text(col, row, str(m[row, col]), ha='center', va='center', fontsize=6, color='white' if m[row, col] > maximum * .55 else 'black')
        fig.suptitle(f'Frozen training class counts; N = 10; shared color scale; seed {seed}')
        fig.colorbar(im, ax=axes, label='Training images')
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'figures/client_class_distributions_seed{seed}.{ext}', dpi=180)
        plt.close(fig)
    print('Verification and saved-data exports passed. Copying immutable evidence.', flush=True)
    for name in ['protocol.json', 'splits.json', 'gn_selection.json', 'loss_selection.json', 'final_plan.json', 'runtime_versions.json', 'requirements.txt', 'results.csv']:
        copy(ROOT / name, Path('frozen') / name)
    for name in CORE + ['results_review.py', 'numerical_diagnostics.py', 'diagnostic_workflow.py', 'replay_numerical_capture.py', 'test_numerical_diagnostics.py', 'audit_phase2_closure.py']:
        copy(BASE / name, Path('source') / name)
    for path in (ROOT / 'partitions').glob('*.json'):
        copy(path, Path('frozen/partitions') / path.name)
    for path in (ROOT / 'generated').iterdir():
        if path.is_file():
            copy(path, Path('existing_generated') / path.name)
    copy(BASE / 'docs/REVIEW_RESPONSE.md', Path('review_evidence/implementation_response.md'))
    for rid in ids:
        d = ROOT / 'runs' / rid
        filenames = ['completion.json', 'status.json', 'config.json', 'history.csv', 'best_validation_metrics.json', 'test_metrics.json', 'test_predictions.csv', 'classification_report.txt', 'environment.json', 'environment_requirements.txt', 'training_time.txt']
        for name in filenames:
            copy(d / name, Path('run_evidence') / rid / name)
        if (d / '_diagnostics/instrumentation.json').exists():
            copy(d / '_diagnostics/instrumentation.json', Path('run_evidence') / rid / '_diagnostics/instrumentation.json')
        copy(ROOT / 'recipes' / f'{rid}.json', Path('recipes') / f'{rid}.json')
    # All validation selection metrics and receipts make the two selection decisions auditable.
    for selection in [gn, loss]:
        for rid in selection['evidence']:
            for name in ['best_validation_metrics.json', 'completion.json', 'config.json', 'status.json']:
                copy(ROOT / 'runs' / rid / name, Path('selection_evidence') / rid / name)
    for path in pipeline_dir.iterdir():
        if path.is_file():
            copy(path, Path('operations/final_pipeline') / path.name)
    recovery_dir = ROOT / 'diagnostics/recovery-20261017T100521Z'
    for path in recovery_dir.iterdir():
        if path.is_file():
            copy(path, Path('recovery') / path.name)
    for path in original.iterdir():
        if path.is_file() and path.suffix not in ['.pt', '.png', '.pdf']:
            copy(path, Path('recovery/original_failed_run') / path.name)
    checkpoint_rows = [dict(run_id=rid, relative_path=f'runs/{rid}/best_model.pt', bytes=(ROOT / 'runs' / rid / 'best_model.pt').stat().st_size, sha256=digest(ROOT / 'runs' / rid / 'best_model.pt'), packaged=False) for rid in ids]
    pd.DataFrame(checkpoint_rows).to_csv(OUT / 'checkpoint_manifest.csv', index=False)
    for path, before in STATS.items():
        st = Path(path).stat()
        check((st.st_size, st.st_mtime_ns) == before, f'Source changed during audit: {path}')
    save(OUT / 'source_hashes.json', HASHES)
    print(f'PASS: all 680 final runs. Output: {OUT}', flush=True)
    return verification


if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    for name in ['tables', 'figures']:
        (OUT / name).mkdir()
    try:
        audit()
    except BaseException as error:
        save(OUT / 'AUDIT_FAILED.json', dict(error=repr(error), utc=datetime.now(timezone.utc).isoformat()))
        raise
