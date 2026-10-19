# Phase 2 closure evidence package

**N=10: audited complete (680/680 final models and tests). Overall closure: pending reviewer clarification of original N=5 scope; the later review specifies N=10 without explicitly removing N=5.** No new training or inference was run to create this package.

Start with [results summary](PHASE2_RESULTS_SUMMARY.md), [verification](VERIFICATION_REPORT.md), [review/scope audit](REVIEW_REQUIREMENTS.md), and [recovery evidence](RECOVERY.md). `verification.json` and `scope_decision.json` are machine-readable records.

- `frozen/`: exact split, 35 partitions and manifest, protocol, selections, sealed final manifest, training environment lock and full original results.csv (1,073 unique training rows; use is_final=1 for the 680 final models).
- `tables/`: final-run verification (680), Dirichlet per-client metrics (600), five-seed configuration summaries (28), within-seed means (140), partition/class counts, and validation/test classification-report rows (16,320). Classification CSV includes ten classes plus macro/weighted averages; scalar accuracy is in the metric tables.
- `figures/`: five four-panel heatmaps covering all four alphas and all five identified partition seeds (20 panels), plus separate accuracy-versus-alpha and macro-F1-versus-alpha curves, each in PDF and PNG. All error bars use five-seed sample SD after averaging clients within seeds. Heatmaps use the same count color scale across all 20 partitions.
- `run_evidence/<run_id>/`: final-run receipts, configs, finite histories, environment/timing, best validation metrics, saved test metrics/predictions, full text classification reports and any instrumentation sidecars. Common split/partition copies are deduplicated into frozen/; run configs and partition hashes map them back.
- `recipes/` and `selection_evidence/`: all final canonical recipes and validation metric/config/status/receipt evidence for 9 GN and 774 loss candidates. Some runs serve multiple stages.
- `existing_generated/`: original publication tables, figures and discussion preserved without changes.
- `recovery/`, `operations/`: archived original failure metadata, diagnostic reproduction evidence and successful final pipeline records/logs. Absolute paths are retained as original provenance, not assumed portable commands.
- `source/`: exact frozen core sources, diagnostics, saved-data audit and packaging scripts. Both primary review PDFs and exact excerpts are in review_evidence/. The implementation response is secondary evidence.
- `checkpoint_manifest.csv`: SHA-256, size and original relative path of each of 680 best-model checkpoints. **Weights are not bundled**; they remain in the original outputs/review2/runs directories. Images are also excluded. This is an audit/results bundle, not a self-contained inference/model-delivery archive.
- `source_hashes.json`: original local files read/bound during the audit. `SHA256SUMS`: checksums for every packaged file except itself. Verify from this directory with `sha256sum -c SHA256SUMS`.

This corrected revision supersedes the earlier package's scope interpretation only. The original ZIP is retained as a historical snapshot. The current review comparison and conditional N=5 plans supersede historical assertions about scope in audit logs, snapshots and source-generation scripts. No training command was launched. Read N5_EXPERIMENT_PLAN.md before considering an extension.

Original experiment root: eurosat-review2/outputs/review2. No files in runs/, partitions/, original generated/, protocol, selections, final plan or results.csv were overwritten. PROJECT_STATUS.md records this closure audit. Rebuilding the audit intentionally requires a fresh output directory; never point a training command at the closure package.
