"""Package the completed saved-data audit. Does not train, evaluate or edit experiments."""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'outputs/review2'
OUT = ROOT / 'phase2_closure_20261018'


def write(name, text):
    (OUT / name).write_text(text.strip() + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    assert not (OUT / 'AUDIT_FAILED.json').exists()
    assert (OUT / 'source_hashes.json').exists(), 'Wait for the complete audit and evidence copies'
    verification = json.loads((OUT / 'verification.json').read_text())
    assert verification['final_verified'] == 680
    decision = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), source='User instructions in this conversation, 2026-10-18', initial_instruction='go with n=10', superseding_instruction='yse n=5 also', interpretation='Both N=5 and N=10 client configurations are required; n=5 independent seeds is a separate quantity.', current_completed_scope='Frozen N=10 plan: 680 final models, including pooled and long-tail controls.', N5_experiments_present=False, reviewer_supersession_verified=False, reviewer_documents_available=False, closure_state='N=10 experimental audit passed; full Phase 2 closure incomplete because N=5 is required and absent.', additional_training_launched=False)
    write('scope_decision.json', json.dumps(decision, indent=2))
    verification['scope_decision'] = decision
    verification['closure_state'] = decision['closure_state']
    verification['limitations'][0] = 'Both N=5 and N=10 are required by the latest user instruction. The frozen plan contains N=10 only; original reviewer documents were not located, so historical reviewer wording/supersession cannot be quoted or verified.'
    write('verification.json', json.dumps(verification, indent=2, allow_nan=False))
    summary = pd.read_csv(OUT / 'tables/five_seed_summary.csv')
    runs = pd.read_csv(OUT / 'tables/final_run_verification.csv')
    clients = pd.read_csv(OUT / 'tables/per_client_metrics.csv')
    counts = pd.read_csv(OUT / 'tables/partition_class_counts.csv')
    reports = pd.read_csv(OUT / 'tables/classification_reports.csv')
    assert len(runs) == 680 and len(clients) == 600
    assert len(reports) == 680 * 12 * 2  # ten classes, macro and weighted averages, validation + test
    # Independent export consistency check against the original publication tables.
    original = pd.read_csv(ROOT / 'generated/summary.csv')
    keys = ['kind', 'value', 'initialization', 'norm', 'image_size', 'loss']
    merged = summary.merge(original, on=keys, validate='one_to_one', suffixes=('_new', '_old'))
    assert len(merged) == len(summary) == 28
    for metric in ['test_accuracy', 'test_macro_f1', 'test_balanced_accuracy']:
        for stat in ['mean', 'std']:
            assert np.allclose(merged[f'{metric}_{stat}_new'], merged[f'{metric}_{stat}_old'], rtol=1e-11, atol=1e-12)
    for rid in runs.run_id:
        cfg = json.loads((ROOT / 'runs' / rid / 'config.json').read_text())
        assert cfg['device'] == 'cuda' and cfg['workers'] == 0
        assert cfg['from_scratch'] == (cfg['initialization'] == 'scratch')
    # Verify the validation-selected GN comparison independently from saved validation evidence.
    loss = json.loads((OUT / 'frozen/loss_selection.json').read_text())
    candidates = []
    for objective, beta, gamma in [('class-balanced', loss['longtail']['beta'], 0.), ('focal', 0., loss['longtail']['gamma'])]:
        vals = []
        for seed in [42, 123, 2024]:
            matching = runs[(runs.kind == 'longtail') & (runs.value == 100) & (runs.initialization == 'imagenet') & (runs.loss == objective) & (runs.seed == seed)]
            assert len(matching) == 1
            vals.append(float(matching.iloc[0].validation_macro_f1))
        candidates.append(dict(loss=objective, beta=beta, gamma=gamma, score=sum(vals) / 3))
    best = max(candidates, key=lambda x: x['score'])
    assert best['loss'] == loss['gn_comparison']['loss']
    assert abs(best['score'] - loss['gn_comparison']['score']) < 1e-12
    verification['gn_non_ce_comparator_recomputed'] = best
    verification['existing_generated_summary_matches_independent_export'] = True
    write('verification.json', json.dumps(verification, indent=2, allow_nan=False))
    labels = {'cross-entropy': 'CE', 'class-balanced': 'Class-balanced', 'focal': 'Focal'}
    def cell(row, metric):
        return f"{row[metric + '_mean'] * 100:.2f} ± {row[metric + '_std'] * 100:.2f}"
    dirichlet = summary[summary.kind == 'dirichlet']
    rows = []
    for alpha in [.1, .5, 1., 10.]:
        r = [f'{alpha:g}']
        for loss_name in labels:
            r.append(cell(dirichlet[(dirichlet.value == alpha) & (dirichlet.loss == loss_name)].iloc[0], 'test_macro_f1'))
        rows.append(r)
    f1_table = table(['Alpha', 'CE macro-F1 (%)', 'Class-balanced (%)', 'Focal (%)'], rows)
    rows = []
    for alpha in [.1, .5, 1., 10.]:
        rows.append([f'{alpha:g}'] + [cell(dirichlet[(dirichlet.value == alpha) & (dirichlet.loss == l)].iloc[0], 'test_accuracy') for l in labels])
    accuracy_table = table(['Alpha', 'CE accuracy (%)', 'Class-balanced (%)', 'Focal (%)'], rows)
    pooled_rows = [[f'{r.initialization}, {r.norm}, {int(r.image_size)}', cell(r, 'test_accuracy'), cell(r, 'test_macro_f1')] for _, r in summary[summary.kind == 'pooled'].iterrows()]
    longtail = summary[(summary.kind == 'longtail') & (summary.initialization == 'imagenet')]
    tail_rows = [[str(ratio)] + [cell(longtail[(longtail.value == ratio) & (longtail.loss == l)].iloc[0], 'test_macro_f1') for l in labels] for ratio in [10, 50, 100]]
    absences = clients[clients.loss == 'cross-entropy'].groupby('value').absent_train_classes.agg(['min', 'max', 'mean'])
    write('REVIEW_REQUIREMENTS.md', '''# Reviewer requirements and closure scope

The latest user instruction requires **both N=5 and N=10 clients**. The preceding instruction “go with n=10” was followed by “yse n=5 also”; this package records the latter as controlling. Five experimental seeds is a different quantity from N clients. No additional training was launched.

Neither the original review nor `Reviews_2.pdf` was located in the workspace, supplied implementation archives, home directories or mounted media searched. Consequently this audit cannot quote the reviewers, verify the original N=5 wording, or establish that a later reviewer explicitly waived it. The unrelated `Guide_review_memo.pdf` found on mounted media concerns WiRay and is not EuroSAT review evidence. **There is no claim of reviewer-approved supersession.**

The available [implementation response](review_evidence/implementation_response.md), lines 18–19, says:

> All 35 training partitions have been generated from the original manifest: four Dirichlet alpha values and three long-tail ratios, each for five seeds.

Lines 44–45 state:

> average ten clients inside each seed, then compute mean and sample SD across five seeds. Do not treat 50 clients as 50 independent seeds.

These quotations are the implementer's interpretation, not the original reviewer text. The controlling executed evidence is [protocol.json](frozen/protocol.json), `num_clients: 10`, [partition manifest](frozen/partitions/manifest.json), and the 680-job [sealed plan](frozen/final_plan.json). All 20 Dirichlet partitions contain ten clients, never five. The other 15 partitions are long-tail subsets with one training model per subset.

| Requirement | Evidence/status | Experiment or artifact gap |
| --- | --- | --- |
| N=10 completed experiment | 680/680 final models and test evaluations verified | Complete |
| N=5 client experiment, now also required | No N=5 partitions, selections or final records in the frozen plan | **Actual missing experiments**; not recoverable by exporting N=10 data |
| All four alpha heatmaps, all partition seeds | Newly exported 20 panels in five PDF/PNG pairs | Export gap closed |
| Accuracy and macro-F1 versus alpha with correct uncertainty | Newly exported PDF/PNG plots and seed-level tables | Export gap closed |
| Per-client metrics and classification reports | 600 client rows; all 680 validation/test reports and predictions | Export gap closed |
| Reviewer-to-plan comparison with exact review quotations | Original review and Reviews_2.pdf unavailable | Source-document gap; cannot certify all reviewer requirements |
| Manuscript cleanup and external model release/tag | Existing draft retains pending/historical prose; no publication/tag evidence supplied | Editorial/release gap; does not require retraining N=10 |

Full Phase 2 closure is **not certified**. N=5 needs its own explicit frozen extension, including partition/selection scope and run manifest; it must preserve this N=10 experiment. N=5 results must not be simulated by choosing five of the ten N=10 clients or relabeling five seeds as five clients. This package neither changes the existing scientific protocol nor starts that extension.
''')
    write('RECOVERY.md', '''# Epoch-13 non-finite-loss recovery

Run `8953d19d54840571685e0d64`: partition seed/training seed 7, Dirichlet alpha 10, client ID 5 (zero-based), ImageNet ResNet-18/BatchNorm, input 96, class-balanced training CE beta 0.999, LR 0.0001. Its 1,833-image frozen subset has class counts `[129,160,261,167,243,173,184,167,158,191]`.

The original attempt raised `FloatingPointError` during epoch-13 validation (progress 33/127 batches); the first 12 epochs had finite metrics. The best saved checkpoint was epoch 12, validation CE 0.15135302535545678. Its tensors, class weights (approximately 0.704–1.336), frozen memberships and validation image integrity passed the recorded investigation. These checks do not identify the missing epoch-13 state or prove a hardware/numerical cause.

Validation uses `model.train(False)`, equivalent to `model.eval()`, and unweighted `torch.nn.functional.cross_entropy` regardless of the training objective ([frozen source](source/train.py), run_epoch lines 37–49). Gradient recording is disabled for validation. CUDA/CPU autocast is disabled, tensors/default dtype are float32, matmul precision is highest, CUDA matmul TF32 is false, cuDNN TF32 is true, cuDNN deterministic true and benchmark false. There is no evidence justifying changes to LR, class weights or precision.

The original failed directory was preserved under `interrupted/8953d19d54840571685e0d64_diagnostic_original_20261017T100645092169`. Metadata/history copies are in [recovery/original_failed_run](recovery/original_failed_run); original weights remain at the original location. Diagnostic wrappers were added separately from the hash-frozen core sources. They capture failing image IDs, labels, input statistics, logits, unreduced loss/configuration, model/BatchNorm state, precision and autocast flags before the original exception is raised. They do not skip batches, replace non-finite values or suppress the guard. A replay utility and diagnostic verification tests were retained.

The checkpoint only held best-model weights and metadata, without optimizer/scheduler/RNG state needed for exact continuation. Recovery was a **fresh restart from the original seed, initialization and configuration**, not a resume from epoch 13 or from best weights. This audit compared the archived and recovered configs for exact equality and compared all numeric history columns except elapsed time for the first 12 epochs: they matched exactly.

The one reproduction finished 40 epochs, best epoch 30, validation CE 0.14090992494681734 and macro-F1 0.9560252843458066. The current receipt matches the recovery report and all saved model state and history metrics are finite. No failure recurred, so no failing-batch capture exists and no captured-batch float32 replay can be claimed. **The underlying cause remains unknown. No scientific fix, hyperparameter change, precision change or protocol change was applied.**

Diagnostics were used for 162 of the 680 final models. Bookkeeping adds elapsed-time overhead; comparing instrumented and original loop times as a controlled speed benchmark is inappropriate. Completed prior records were retained. A later laptop reboot interrupted a different run before its first completed epoch; its preserved incomplete directory was restarted rather than treated as resumable. Historical native SIGSEGVs are separate incidents and do not establish the cause of this NaN/Inf failure.

Evidence: [recovery report](recovery/recovery_report.json), [preflight](recovery/preflight.json), [reproduction record](recovery/reproduction.json), [reproduction log](recovery/reproduction.log), and the recovered run's [receipt](run_evidence/8953d19d54840571685e0d64/completion.json).
''')
    write('VERIFICATION_REPORT.md', f'''# Phase 2 saved-results verification

**PASS for all 680 final models in the frozen N=10 plan. Full Phase 2 closure remains incomplete because N=5 is also required.** Audit UTC: {verification['audited_utc']}. No training or model inference was run during this audit.

- Exact reconstruction of all 680 canonical jobs/IDs from frozen protocol and validation-selected settings; exact equality with the sealed final plan. Coverage: 25 pooled baselines, 600 Dirichlet client models, 45 pretrained long-tail models and 10 scratch-GN ratio-100 models.
- All 680 completion receipts verified against actual SHA-256 of checkpoints, configs, histories and best validation metrics. CPU loading checked every checkpoint's configuration, class order, split, selected epoch, finite tensors/BatchNorm statistics and nonnegative running variance. All 680 best-model files lack the training state for exact continuation.
- All 35 frozen partitions validated and deterministically regenerated with exact equality. Every model's copied split/partition, training counts and subset size agree with its canonical job. Dirichlet subsets are disjoint and cover only the frozen training split; held-out sets do not overlap training. Every validation report has the unchanged global support vector.
- Histories contain finite numeric values, contiguous epoch numbers, correct best-CE epochs and valid early-stop/max-epoch records. Test checkpoint epochs agree with selected best validation checkpoints.
- All 680 saved test prediction files have exactly the frozen 4,050 test paths in order with their correct labels. Recomputed confusion matrices, accuracy, macro-F1, balanced accuracy, per-class recall, full classification-report JSON and rounded text reports agree with saved metrics. **2,754,000 prediction rows checked.** No test-based tuning or seed selection was performed.
- CSV final membership, canonical settings, checkpoint hashes, training counts, test metrics and report values agree with per-run records. The independently aggregated 28 configuration summaries agree with the existing generated accuracy/F1/balanced-accuracy means and sample SDs.
- Validation selection evidence verified for all 9 GN and 774 loss-search jobs. GN learning-rate and regime beta/gamma choices recomputed using saved validation macro-F1; the non-CE GN comparison choice also recomputed. No test metric entered a selection.
- All 11 frozen core source hashes match the protocol. Recovery source hashes match the recovery evidence; original/restarted failed-run configurations and first 12 numeric epochs match.
- Pipeline records show successful sequential final-train → seal → evaluate → report, exit 0 for each. Test-metric file timestamps follow the recorded seal completion. Original generated provenance binds results.csv and the sealed plan.

Protocol fingerprint: `{verification['protocol_fingerprint']}`. Split fingerprint: `{verification['split_fingerprint']}`. Plan fingerprint: `{verification['final_plan_fingerprint']}`. Fingerprints are the implementation's canonical sorted-JSON hashes; raw-file SHA-256 values are listed separately.

Limits: the original seal binds protocol/selections/jobs and rechecks training receipts, but it has no historical signed test-results receipt. This audit newly hashes saved tests/predictions/reports; consistency with saved predictions does not independently prove the historic forward pass used the claimed weights. File timestamps are corroboration, not signed timestamps. Frozen configs, copied memberships and source/checkpoint metadata establish intended subsets, but no historical per-image training access trace or image-byte hash manifest exists. No new image-integrity or model-inference experiment is claimed. User review PDFs were not located. See [scope audit](REVIEW_REQUIREMENTS.md), [recovery](RECOVERY.md), [machine-readable verification](verification.json), [run table](tables/final_run_verification.csv), [checkpoint hashes](checkpoint_manifest.csv), and [source hashes](source_hashes.json).
''')
    write('PHASE2_RESULTS_SUMMARY.md', f'''# Phase 2 results: completed N=10 experiment

The frozen N=10 experiment has **680/680 verified final training runs and 680/680 saved test evaluations**. No completed model was restarted for this closure audit. This report packages measured results and closes export gaps; full reviewer closure remains incomplete because the latest user instruction also requires N=5 client experiments, which are absent.

## Evaluation setup and aggregation

The original EuroSAT RGB split contains 18,900 training, 4,050 validation and 4,050 test images across ten classes. Validation and test memberships stay fixed across every experiment, with per-class supports `[450,450,450,375,375,300,375,450,375,450]`. They are stratified, not exactly balanced. Synthetic label-Dirichlet partitions use alpha 0.1, 0.5, 1 and 10; long-tail ratios are 10, 50 and 100. All partitions are frozen before training.

Each client model trains independently on its own local subset and is evaluated on the same **global** test set, including classes absent from that client's training data. There is no server, model exchange, FedAvg, ensemble, geographic partition or federated optimization. Lower global-test performance measures local-model generalization under limited/skewed training evidence; it is not evidence about a federated aggregation algorithm.

For every loss/alpha, let m[s,c] be one client's saved metric: first take the unweighted mean of the ten clients within seed s, then the mean and **sample SD (ddof=1)** of those five seed means. Seeds are 42, 123, 2024, 7 and 99. Error bars are SD over five joint partition/training seeds, not confidence intervals and not SD over 50 independent clients. Macro-F1 is averaged as a per-model metric; it is not recomputed from pooled client predictions. Long-tail and pooled models have one model per seed.

Hyperparameters were chosen using validation only on seeds 42/123/2024. Scratch GN LR is 0.0003. Class-balanced beta is 0.999 for Dirichlet and 0.9999 for long-tail; focal gamma is 1 for both. Checkpoints and early stopping use minimum **unweighted validation CE**, patience 10 and maximum 100 epochs. The GN non-CE comparator is class-balanced beta 0.9999, chosen on pretrained ratio-100 validation. No setting was adjusted after observing test results.

## Global-test results of local client models

All entries below are five-seed mean ± sample SD, in percent.

{f1_table}

{accuracy_table}

For CE, decreasing alpha from 10 to 0.1 reduces mean global-test macro-F1 from 93.01% to 26.33% (66.68 percentage points) and accuracy from 93.75% to 37.67% (56.09 points). At alpha 0.1, class-balanced CE increases macro-F1 by 5.71 points relative to CE, to 32.03%; it does not recover the pooled baseline. Gains shrink to approximately 0.01 point at alpha 10. Focal has no consistent improvement: its mean macro-F1 differences versus CE across alpha 0.1/0.5/1/10 are +0.31/−0.64/−0.06/−0.15 points. These are descriptive paired-configuration differences, without a significance claim.

At alpha 0.1, clients lack an average of {absences.loc[.1, 'mean']:.2f} of ten training classes (range {int(absences.loc[.1, 'min'])}–{int(absences.loc[.1, 'max'])}); at alpha 10 the average is {absences.loc[10., 'mean']:.2f} (range {int(absences.loc[10., 'min'])}–{int(absences.loc[10., 'max'])}). A reweighted objective cannot supply training examples for a completely absent class. The heatmaps and per-client count/metric tables expose this allocation. Both class skew and variation in client sample size contribute, so the experiment does not isolate class imbalance as the sole cause.

## Pooled and long-tail context

{table(['Pooled recipe / input', 'Accuracy (%)', 'Macro-F1 (%)'], pooled_rows)}

The pooled ImageNet BN96 reference reaches 98.56% mean macro-F1, compared with 93.01% for CE clients even at alpha 10. The local/pooled gap includes the smaller per-client training set and independent training. Resolution and normalization comparisons remain conditional on their recipes; ImageNet and scratch use different learning rates. The BN224/BN96 test difference does not prove equivalence or override the predeclared validation-based resolution choice.

{table(['Long-tail ratio', 'CE macro-F1 (%)', 'Class-balanced (%)', 'Focal (%)'], tail_rows)}

Long-tail CE macro-F1 falls from 97.18% at ratio 10 to 93.25% at ratio 100. Class-balanced gains over CE are +0.21/+0.64/+0.69 points at ratios 10/50/100. Focal differences are −0.05/−0.09/−0.48 points. The scratch-GN ratio-100 comparison does not establish a general class-balanced benefit: CE macro-F1 is 71.57 ± 4.38%, versus 71.51 ± 1.84% for class-balanced (difference −0.05 points); accuracy is 74.95 ± 3.03% versus 73.08 ± 1.40%. The chosen comparator was retained despite its negative mean differences.

## Recovery, limitations and closure gaps

The epoch-13 non-finite validation loss did not recur during one fresh restart with the original seed/configuration. The first 12 epochs matched, and the recovered run finished 40 epochs with best epoch 30 and finite state/metrics. No experimental setting changed. **The underlying cause remains unknown.** See [RECOVERY.md](RECOVERY.md). Diagnostic bookkeeping affected 162 final runs' timing, so mixed timing records are not a controlled runtime comparison.

Only five joint partition/training seeds are available; these do not separate allocation uncertainty from optimizer/init randomness. Partitions are synthetic label-skew allocations, not real organizations or geographic shifts. Long-tail order is fixed across seeds. This is one RGB dataset, one architecture family and a specific augmentation/optimizer regime. The global test set was examined in historical Phase 1; the workflow separates new selection/test stages but cannot restore an unseen historical holdout. Unequal test-class supports and local missing classes make accuracy and macro-F1 complementary. Monotonic sample means do not prove statistical significance, equivalence or universal rankings.

**Actual missing experiments:** N=5 clients, now required in addition to N=10, have no frozen partitions/selected-settings record/final manifest or measurements here. N=10 data cannot establish N=5 outcomes. Define a separate frozen extension before running that condition; preserve the completed N=10 artifacts. No new training was launched during this task.

**Artifact/document gaps:** the two original reviewer documents are unavailable, so exact review quotations and historical N=5 supersession cannot be verified. The existing manuscript retains historical/pending prose and needs an editorial update; this summary is the current measured-results account. External checkpoint archive delivery/download URLs, a real committed release and Phase 2 tag are not evidenced here. These are publication/release tasks, not grounds to rerun N=10. Checkpoint weights remain intact locally with verified hashes, and are excluded from this portable evidence bundle along with dataset image bytes.

The requested missing heatmaps, accuracy curve, seed-level result tables, per-client metrics and classification-report exports have been generated from saved data. Their absence before this audit was an export gap, not an unfinished N=10 experiment.
''')
    write('README.md', '''# Phase 2 closure evidence package

**N=10: audited complete (680/680 final models and tests). Full Phase 2 closure: incomplete; latest user instruction also requires N=5 clients.** No new training or inference was run to create this package.

Start with [results summary](PHASE2_RESULTS_SUMMARY.md), [verification](VERIFICATION_REPORT.md), [review/scope audit](REVIEW_REQUIREMENTS.md), and [recovery evidence](RECOVERY.md). `verification.json` and `scope_decision.json` are machine-readable records.

- `frozen/`: exact split, 35 partitions and manifest, protocol, selections, sealed final manifest, training environment lock and full original results.csv (1,073 unique training rows; use is_final=1 for the 680 final models).
- `tables/`: final-run verification (680), Dirichlet per-client metrics (600), five-seed configuration summaries (28), within-seed means (140), partition/class counts, and validation/test classification-report rows (16,320). Classification CSV includes ten classes plus macro/weighted averages; scalar accuracy is in the metric tables.
- `figures/`: five four-panel heatmaps covering all four alphas and all five identified partition seeds (20 panels), plus separate accuracy-versus-alpha and macro-F1-versus-alpha curves, each in PDF and PNG. All error bars use five-seed sample SD after averaging clients within seeds. Heatmaps use the same count color scale across all 20 partitions.
- `run_evidence/<run_id>/`: final-run receipts, configs, finite histories, environment/timing, best validation metrics, saved test metrics/predictions, full text classification reports and any instrumentation sidecars. Common split/partition copies are deduplicated into frozen/; run configs and partition hashes map them back.
- `recipes/` and `selection_evidence/`: all final canonical recipes and validation metric/config/status/receipt evidence for 9 GN and 774 loss candidates. Some runs serve multiple stages.
- `existing_generated/`: original publication tables, figures and discussion preserved without changes.
- `recovery/`, `operations/`: archived original failure metadata, diagnostic reproduction evidence and successful final pipeline records/logs. Absolute paths are retained as original provenance, not assumed portable commands.
- `source/`: exact frozen core sources, diagnostics, saved-data audit and packaging scripts. Review implementation response is in review_evidence/ and must not be mistaken for the missing original reviews.
- `checkpoint_manifest.csv`: SHA-256, size and original relative path of each of 680 best-model checkpoints. **Weights are not bundled**; they remain in the original outputs/review2/runs directories. Images are also excluded. This is an audit/results bundle, not a self-contained inference/model-delivery archive.
- `source_hashes.json`: original local files read/bound during the audit. `SHA256SUMS`: checksums for every packaged file except itself. Verify from this directory with `sha256sum -c SHA256SUMS`.

Original experiment root: eurosat-review2/outputs/review2. No files in runs/, partitions/, original generated/, protocol, selections, final plan or results.csv were overwritten. PROJECT_STATUS.md records this closure audit. Rebuilding the audit intentionally requires a fresh output directory; never point a training command at the closure package.
''')
    shutil.copyfile(__file__, OUT / 'source/package_phase2_closure.py')
    shutil.copyfile(ROOT / 'phase2_closure_audit_20261018.log', OUT / 'audit_console.log')
    # Snapshot status before hashing/archiving. Status itself is updated by the caller first.
    shutil.copyfile(BASE.parent / 'PROJECT_STATUS.md', OUT / 'PROJECT_STATUS_snapshot.md')
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    write('SHA256SUMS', '\n'.join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}' for p in files))
    archive = OUT.with_suffix('.zip')
    assert not archive.exists(), 'Refusing to overwrite an existing archive'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for path in files + [OUT / 'SHA256SUMS']:
            z.write(path, Path(OUT.name) / path.relative_to(OUT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'Archive CRC check failed'
        assert len(z.namelist()) == len(files) + 1
    archive_hash = sha(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{archive_hash}  {archive.name}\n')
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=archive_hash, packaged_files=len(files) + 1, final_verified=680, N5_complete=False), indent=2))


if __name__ == '__main__':
    main()
