"""Correct closure interpretation from the supplied primary reviews; never launch runs.

The previous ZIP remains an immutable historical snapshot. A new revision keeps
the same verified N=10 evidence and adds quotations and conditional N=5 plans.
"""
import hashlib
import itertools
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent
ROOT = BASE / 'outputs/review2'
OLD = ROOT / 'phase2_closure_20261018'
NEW = ROOT / 'phase2_closure_20261018_review_correction'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    path = NEW / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + '\n', encoding='utf-8')


def save(name, obj):
    write(name, json.dumps(obj, indent=2, allow_nan=False))


def main():
    assert not NEW.exists(), 'Refuse to replace an existing revision'
    old_archive = OLD.with_suffix('.zip')
    old_archive_hash = sha(old_archive)
    assert old_archive_hash == old_archive.with_suffix('.zip.sha256').read_text().split()[0]
    # Confirm all previously packaged evidence before copying; do not touch training outputs.
    for line in (OLD / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert sha(OLD / relative) == expected, relative
    shutil.copytree(OLD, NEW)
    excerpts = [
        dict(file='review.pdf', page=2, section='3.A, Imbalanced data', quote='Dirichlet split into 5 and 10 clients with alpha values 0.1, 0.5, 1 and 10.'),
        dict(file='review.pdf', page=2, section='3.B, Imbalance-aware training', quote='Compare plain cross-entropy with weighted cross-entropy, class-balanced loss and focal loss. Choose beta and gamma on validation only.'),
        dict(file='Reviews_2.pdf', page=1, section='2, Part A, Dirichlet split', quote='Split the training set among 10 clients with alpha = 0.1, 0.5, 1 and 10.'),
        dict(file='Reviews_2.pdf', page=2, section='Part B, Train with normal cross-entropy', quote="Dirichlet: train each client's model on its own data and test it on the global test set. Also give the pooled-data model from Phase 1 as the reference."),
        dict(file='Reviews_2.pdf', page=2, section='Part C, Try better losses', quote='Choose beta and gamma on validation macro-F1, fix them, and then run the final five seeds. Run this for both the long-tailed and the Dirichlet setting.'),
        dict(file='Reviews_2.pdf', page=3, section='Section 3, Table 3 title', quote='Dirichlet split with 10 clients, global test macro-F1'),
    ]
    source_info = []
    for name in ['review.pdf', 'Reviews_2.pdf']:
        shutil.copyfile(WORKSPACE / name, NEW / 'review_evidence' / name)
        source_info.append(dict(file=name, sha256=sha(WORKSPACE / name), pages=3))
    save('review_evidence/excerpts.json', dict(method='Both PDFs are image-only. All pages rendered with pdftoppm and read visually; quotations manually transcribed from page images. Original review page 3 is blank.', sources=source_info, excerpts=excerpts))
    state = 'N=10 experiment verified complete; overall closure pending reviewer clarification of N=5 scope and remaining editorial/release requirements.'
    assessment = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), evidence='Attached original review.pdf and Reviews_2.pdf, visually read in full', correction='Earlier messages were incorrectly treated as introducing an unconditional N=5 requirement. That interpretation is withdrawn.', original_N5_and_N10_explicit=True, later_N10_explicit=True, later_N5_removal_explicit=False, reviewer_supersession_verified=False, N5_current_requirement='ambiguous: original includes N=5; later detailed next-phase plan specifies N=10 without explicitly waiving N=5', N5_loss_scope='ambiguous if N=5 is retained: baseline only versus extending Part C is not stated explicitly', reviewer_documents_available=True, N5_experiments_present=False, N10_final_verified=680, additional_training_launched=False, closure_state=state, prior_archive_sha256=old_archive_hash)
    save('scope_decision.json', assessment)
    verification = json.loads((NEW / 'verification.json').read_text())
    verification['scope_decision'] = assessment
    verification['closure_state'] = state
    verification['review_scope_corrected_utc'] = assessment['recorded_utc']
    verification['limitations'][0] = assessment['N5_current_requirement']
    save('verification.json', verification)
    write('REVIEW_REQUIREMENTS.md', '''# Primary-review comparison and corrected closure scope

**The later review specifies N=10 but does not explicitly remove N=5. Whether it replaces the original N=5-and-N=10 scope remains ambiguous.** This is a comparison of the attached primary reviews, not an implementation inference. The previous assertion that the user's short replies introduced a new mandatory N=5 experiment is withdrawn.

Both PDFs are image-only, so all pages were rendered and read visually. Quotations below are manual transcriptions, with PDF page numbers. The original review's third page is blank. Source PDF hashes and structured excerpts are in [excerpts.json](review_evidence/excerpts.json).

## Relevant passages

Original [review.pdf](review_evidence/review.pdf), page 2, section 3.A:

> Dirichlet split into 5 and 10 clients with alpha values 0.1, 0.5, 1 and 10.

Original review, page 2, section 3.B:

> Compare plain cross-entropy with weighted cross-entropy, class-balanced loss and focal loss. Choose beta and gamma on validation only.

Later [Reviews_2.pdf](review_evidence/Reviews_2.pdf), page 1, section 2, Part A:

> Split the training set among 10 clients with alpha = 0.1, 0.5, 1 and 10.

Later review, page 2, Part B:

> Dirichlet: train each client's model on its own data and test it on the global test set. Also give the pooled-data model from Phase 1 as the reference.

Later review, page 2, Part C:

> Choose beta and gamma on validation macro-F1, fix them, and then run the final five seeds. Run this for both the long-tailed and the Dirichlet setting.

Part C explicitly enumerates three methods: cross-entropy, class-balanced loss (beta 0.99, 0.999, 0.9999), and focal loss (gamma 1, 2). Later review page 3 reinforces the executed scope in its Table 3 title:

> Dirichlet split with 10 clients, global test macro-F1

The dummy numbers in that table are formatting examples, not experimental evidence; they are not used here.

## Determination and exact ambiguity

The original document unequivocally requests both client counts. The later document gives a detailed next-phase plan that mentions only ten clients, and its proposed table also uses ten. This supports interpreting the later plan as narrowed scope. However, it never says “replace the earlier 5-and-10-client requirement,” “N=10 only,” or “N=5 is no longer required.” Therefore **explicit removal is not established**. Equally, omission alone does not establish that the reviewer still wants N=5. Neither interpretation should be presented as confirmed.

Question for the reviewer:

> Does Reviews_2.pdf Part A (“Split the training set among 10 clients”) replace the original section 3.A request for “5 and 10 clients,” so that N=10 alone completes the client-count requirement? If N=5 is still wanted, should it cover only the normal-cross-entropy baseline in Part B, or also the class-balanced and focal comparisons in Part C? If the latter, should the already locked Dirichlet beta/gamma be reused, or should N=5 have a separate validation-only selection?

“Phase 2 baseline” and “Phase 3 loss comparisons” in the user's request map here to later-review Parts B and C; the review itself does not assign those phase numbers. If the original N=5 request is retained and interpreted as applying throughout the original section 3, its following loss-comparison paragraph suggests including losses too. But the later Part C refers back to a Dirichlet setting specified with ten clients, so **N=5 baseline-only versus loss-comparison scope is not explicit**. See the conditional [N=5 plan](N5_EXPERIMENT_PLAN.md); no extension is launched.

## Closure state and other requirements

All **680 final N=10-plan models and tests remain verified complete**. Twenty N=10 Dirichlet partitions, all four alpha values, five final seeds, fixed global evaluation sets, loss selections and recovery evidence are unchanged. The scope ambiguity is not a failed or missing N=10 experiment. N=5 would be an actual experimental gap only if confirmed within the required scope.

Even if the reviewer confirms N=10-only scope, this audit does not mark the entire submission complete automatically. The existing manuscript still contains pending/historical prose, and the later review's page-1 repository checklist requires downloadable model archives with filename/SHA-256 lists and a tagged final commit. A verified local checkpoint manifest is present, but an uploaded model release and final committed/tagged submission are not evidenced. Those are editorial/release gaps rather than reasons to retrain N=10. The repository/tag request on that page explicitly names `phase1-final`; it should not be misrepresented as a quoted demand for a `phase2-final` tag.

The reviews also ask for balanced validation/test sets while retaining the same established split. The preserved 4,050-image sets are stratified with unequal class supports. This discrepancy is disclosed in the results summary; the audit does not silently rebalance or claim exact balance. Reviewer acceptance of this interpretation is not documented. The image-level random split's possible spatial overlap, one-dataset/RGB-only setting, synthetic label skew, historical test access and five-seed limitations remain disclosed.
''')
    write('N5_EXPERIMENT_PLAN.md', '''# Conditional N=5 extension — plan only, no training authorized or launched

N=5 is not newly imposed by this audit. The original review requested it; the later review's N=10-only wording leaves replacement ambiguous. The scenarios below make the possible extension concrete for review. No scenario is silently chosen.

## Settings inherited without test-based changes

- Reuse the exact original `splits.json`: 18,900 train, 4,050 validation, 4,050 test; split fingerprint `314a000080506dac74ddf07a196d1d76311f847418149073f17c9628e901e7bf`. Do not rebalance or redraw held-out sets. Use the existing dataset and `.venv` on CUDA if training is subsequently approved.
- Four alphas: 0.1, 0.5, 1, 10. Five joint partition/training seeds: 42, 123, 2024, 7, 99. Create 20 new N=5 partitions with the unchanged NIID-Bench-derived label-Dirichlet algorithm, minimum ten images per client, deterministic retries advancing the same seeded RNG. Use all training images exactly once across the five clients. N=5 is not formed by keeping five of the N=10 clients.
- ImageNet ResNet-18, BatchNorm, 96×96; LR 0.0001, AdamW, weight decay 0.0001, batch size 32, workers 0, existing train augmentations and float32 precision. Maximum 100 epochs; patience 10; ReduceLROnPlateau factor 0.5/patience 2. Keep all model parameters trainable. Preserve numerical diagnostic capture and fail rather than skip non-finite batches.
- Train independent local models; validate/evaluate each against the unchanged global sets, including absent local classes. Checkpoint and early stop on minimum unweighted validation CE, including when the training loss is weighted/focal. Save histories, counts, environment, completion receipts, checkpoint SHA-256, predictions, confusion matrices and reports.
- Average the five client metrics within each seed, then report five-seed mean ± sample SD (ddof=1). Do not treat 25 client models as 25 independent seeds or pool their predictions as an ensemble.
- Prospective output root: `outputs/review2_n5`, with its own N=5 protocol hash, partition/recipe/selection records and sealed final manifest, referencing the existing N=10 protocol as parent. Existing `review_workflow.py` hard-codes ten clients; do not run it unchanged and relabel the outputs as N=5. A separate runner/generalized copy with a distinct source snapshot/protocol must be reviewed before use. Do not edit or unseal `outputs/review2`.

## Exact additional unique training counts

| Confirmed reviewer scope | Calculation | New training jobs | Final test evaluations |
| --- | --- | --- | --- |
| Later review replaces N=5; N=10 only | Reuse all completed runs | **0** | **0** |
| N=5 Part B / Phase 2 CE baseline only | 4 alphas × 5 clients × 5 seeds × 1 loss | **100** | **100** |
| N=5 Parts B+C, reusing locked Dirichlet settings | 4 × 5 × 5 × 3 losses | **300** | **300** |
| N=5 Parts B+C with separately authorized N=5 validation tuning | 360 search jobs + 300 finals − 180 identical jobs reused | **480** | **300** |

The 300-job scenario contains the same 100 CE baselines plus 200 non-CE finals. If the 100 CE jobs were already completed under an identical future N=5 protocol, only 200 more trainings would be needed for the fixed-setting loss extension; do not restart CE. In the separately tuned scenario, 380 more unique jobs would remain after those same 100 CE jobs. At present no N=5 jobs exist, so the full additional totals are 100, 300, or 480 as shown.

No extra pooled baseline, long-tail, scratch-GN ratio-100 or GN learning-rate training is needed: those conditions have no client-count axis and their matching completed evidence can be referenced. This plan adds only N=5 Dirichlet jobs.

## Tuning required versus an optional protocol amendment

**CE baseline:** no beta/gamma or learning-rate tuning is required. Use the established ImageNet BN96 recipe.

**Loss comparison using the existing selection scope:** the executed protocol selected one beta and gamma per regime, not one pair per alpha/client/seed. Its locked Dirichlet values are beta **0.999** and focal gamma **1**. Reusing those values prospectively for N=5 requires **zero additional search jobs** and preserves the established one-pair-per-regime rule. This is parameter transfer from validation-selected N=10 evidence, not a claim that N=5 was tuned separately. Obtain scope agreement before recording this extension. Do not reselect shared parameters from combined N=5/N=10 evidence or alter completed N=10 settings.

**Separate N=5 selection, only if requested:** this changes the selection scope to one pair per client-count-specific Dirichlet regime; it is not required by the existing frozen protocol and must be declared in advance. Retain the established three tuning seeds (42,123,2024), alpha grid and candidate grid: CE, CB beta 0.99/0.999/0.9999, focal gamma 1/2. The review does not specify the tuning-seed budget; three is the implementation's inherited choice. Search count = 4 alphas × 5 clients × 3 seeds × 6 objectives = **360**. Average validation macro-F1 at each best-CE checkpoint equally over clients within alpha, alphas within seed, then three seeds; apply the existing tie rules. Lock beta/gamma before any N=5 test access. Of 300 final models, 4×5×3×3 = **180** match already completed search jobs and are reused; the other two seeds need 4×5×2×3 = **120** new trainings. Total **360+120=480**, not 660. Only the 300 selected final models receive test evaluation.

N=10 test results have already been observed and cannot become unseen. They must not inform N=5 scope, candidate grids, learning rates, beta/gamma choices or stopping rules. Predeclare the extension and any N=5 validation-only tuning before its test stage. The count and selection rules above use design arithmetic and frozen settings, not observed test outcomes.

## Execution gates after plan/scope agreement

1. Resolve whether N=5 is required and whether it applies to baseline, all losses with fixed parameters, or an explicitly amended N=5 tuning scope.
2. Review a separate runner/source snapshot and frozen extension manifest. Check no training process is already active. Freeze and validate 20 N=5 partitions, including deterministic reproduction, full/disjoint training coverage, minimum sizes and exact unchanged held-out membership.
3. Run jobs sequentially on CUDA with persistent session/logs. Reuse matching valid completions; archive only incomplete attempts when an explicit restart is appropriate. Stop on real errors; preserve diagnostics. Do not resume from best weights without full training state.
4. Verify all required completion receipts, finite histories and checkpoint hashes. If losses may be separately tuned, do not test even the N=5 CE baselines before the N=5 selections and final plan are locked. Seal before all N=5 test access.
5. Evaluate selected final models once, generate separate N=5 result tables/figures/reports and compare descriptively with the retained N=10 evidence. Preserve negative results. Keep test-derived decisions out of training.

Proposed job lists are included for transparent arithmetic. They are design manifests, not sealed protocols, completion claims or launch requests. N=5 self-tuned final non-CE job settings cannot be instantiated until its validation selection is complete.
''')
    protocol = json.loads((NEW / 'frozen/protocol.json').read_text())
    base_job = dict(num_clients=5, kind='dirichlet', initialization='imagenet', norm='batch', image_size=96, lr=.0001)
    def jobs(seeds, objectives):
        return [dict(base_job, value=float(alpha), seed=seed, client_id=cid, loss=loss, beta=beta, gamma=gamma)
                for alpha, seed, cid, (loss, beta, gamma) in itertools.product([.1,.5,1.,10.], seeds, range(5), objectives)]
    ce = [('cross-entropy', 0., 0.)]
    fixed_losses = ce + [('class-balanced', .999, 0.), ('focal', 0., 1.)]
    candidates = ce + [('class-balanced', b, 0.) for b in [.99,.999,.9999]] + [('focal', 0., g) for g in [1.,2.]]
    baseline = jobs([42,123,2024,7,99], ce)
    fixed_final = jobs([42,123,2024,7,99], fixed_losses)
    search = jobs([42,123,2024], candidates)
    canonical = lambda job: json.dumps(job, sort_keys=True)
    a,b,c = [set(map(canonical, x)) for x in [baseline,fixed_final,search]]
    assert len(a) == 100 and len(b) == 300 and len(c) == 360
    assert len(b & c) == 180 and len(b | c) == 480 and a.issubset(b)
    for name, job_list in [('N5_CE_baseline_proposed.json', baseline), ('N5_fixed_losses_proposed.json', fixed_final), ('N5_separate_tuning_candidates_proposed.json', search)]:
        save('planning/' + name, dict(status='PROPOSED_ONLY_NOT_SEALED_NOT_LAUNCHED', parent_protocol_fingerprint=verification['protocol_fingerprint'], inherited_split_fingerprint=protocol['split_hash'], jobs=job_list, n=len(job_list)))
    save('planning/count_verification.json', dict(CE_baseline=100, fixed_loss_finals=300, separate_tuning_search=360, selected_finals_overlap_search=180, new_final_seed_trainings=120, separate_tuning_total_unique=480, extra_after_completed_CE_fixed=200, extra_after_completed_CE_separate_tuning=380, additional_training_launched=False))
    # Retain measured results verbatim; correct only the scope/provenance interpretation.
    summary = (NEW / 'PHASE2_RESULTS_SUMMARY.md').read_text()
    summary = summary.replace('This report packages measured results and closes export gaps; full reviewer closure remains incomplete because the latest user instruction also requires N=5 client experiments, which are absent.', 'This report packages measured results and closes export gaps. The supplied original review explicitly requests N=5 and N=10; Reviews_2.pdf specifies N=10 without explicitly removing N=5. Overall closure is pending clarification of that scope and the remaining editorial/release requirements. The previous interpretation that the user introduced a new N=5 requirement is withdrawn.')
    summary = summary.replace('**Actual missing experiments:** N=5 clients, now required in addition to N=10, have no frozen partitions/selected-settings record/final manifest or measurements here. N=10 data cannot establish N=5 outcomes. Define a separate frozen extension before running that condition; preserve the completed N=10 artifacts. No new training was launched during this task.', '**Conditional experiment gap:** no N=5 measurements are present. The original review requests both N=5 and N=10, while the later detailed plan specifies N=10 and does not explicitly address removal of N=5. Whether N=5 remains required, and whether it covers CE only or Part C losses too, is unresolved. Do not label this a confirmed new user requirement or silently launch an extension. See [primary-review quotations](REVIEW_REQUIREMENTS.md) and [conditional N=5 plan](N5_EXPERIMENT_PLAN.md): 100 CE jobs, 300 fixed-setting loss-comparison jobs, or 480 unique jobs if a separately approved N=5 tuning scope is required. Completed N=10 evidence is unchanged.')
    summary = summary.replace('the two original reviewer documents are unavailable, so exact review quotations and historical N=5 supersession cannot be verified.', 'both original reviewer PDFs are now supplied, included and quoted in REVIEW_REQUIREMENTS.md. The later document does not explicitly resolve the earlier N=5 requirement.')
    write('PHASE2_RESULTS_SUMMARY.md', summary)
    report = (NEW / 'VERIFICATION_REPORT.md').read_text().replace('Full Phase 2 closure remains incomplete because N=5 is also required.', 'Overall closure is pending clarification: the original review requests N=5 and N=10; the later plan specifies N=10 without explicitly waiving N=5.').replace('User review PDFs were not located.', 'Both user review PDFs have now been read visually; exact passages and the unresolved scope are recorded in REVIEW_REQUIREMENTS.md.')
    write('VERIFICATION_REPORT.md', report)
    readme = (NEW / 'README.md').read_text().replace('Full Phase 2 closure: incomplete; latest user instruction also requires N=5 clients.', 'Overall closure: pending reviewer clarification of original N=5 scope; the later review specifies N=10 without explicitly removing N=5.').replace('Review implementation response is in review_evidence/ and must not be mistaken for the missing original reviews.', 'Both primary review PDFs and exact excerpts are in review_evidence/. The implementation response is secondary evidence.').replace('Original experiment root:', 'This corrected revision supersedes the earlier package\'s scope interpretation only. The original ZIP is retained as a historical snapshot. The current review comparison and conditional N=5 plans supersede historical assertions about scope in audit logs, snapshots and source-generation scripts. No training command was launched. Read N5_EXPERIMENT_PLAN.md before considering an extension.\n\nOriginal experiment root:')
    write('README.md', readme)
    shutil.copyfile(__file__, NEW / 'source/correct_phase2_review_scope.py')
    # Scientific evidence must remain byte-identical to the previously verified package.
    unchanged = ['frozen', 'tables', 'figures', 'run_evidence', 'recipes', 'selection_evidence', 'recovery', 'operations', 'existing_generated']
    protected_count = 0
    for name in unchanged:
        for path in (OLD / name).rglob('*'):
            if path.is_file():
                assert sha(path) == sha(NEW / path.relative_to(OLD)), path
                protected_count += 1
    save('review_evidence/correction_provenance.json', dict(corrected_utc=assessment['recorded_utc'], previous_archive_sha256=old_archive_hash, unchanged_scientific_evidence_files=protected_count, original_N10_experiment_modified=False, training_launched=False, scope='Primary-review interpretation, quotations and conditional plans only'))
    shutil.copyfile(WORKSPACE / 'PROJECT_STATUS.md', NEW / 'PROJECT_STATUS_snapshot.md')
    files = sorted(p for p in NEW.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    write('SHA256SUMS', '\n'.join(f'{sha(path)}  {path.relative_to(NEW).as_posix()}' for path in files))
    archive = NEW.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files + [NEW / 'SHA256SUMS']:
            z.write(path, Path(NEW.name) / path.relative_to(NEW))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    archive_hash = sha(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{archive_hash}  {archive.name}\n')
    assert sha(old_archive) == old_archive_hash
    print(json.dumps(dict(archive=str(archive), sha256=archive_hash, bytes=archive.stat().st_size, unchanged_scientific_evidence_files=protected_count, final_verified=680, N5_requirement='ambiguous; reviewer clarification needed', training_launched=False), indent=2))


if __name__ == '__main__':
    main()
