# Conditional N=5 extension — plan only, no training authorized or launched

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
