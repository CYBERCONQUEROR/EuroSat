# Verification and limits

- 12 unit tests passed: original Phase 1 checks plus deterministic partitions, unchanged held-out membership, long-tail counts, absent-class loss stability, loss identities, frozen artifacts, experiment counts, seed aggregation and test-access gating.
- Three end-to-end synthetic-image smoke tests passed: CE/BatchNorm, class-balanced/GroupNorm and focal/GroupNorm. These exercised training, checkpoint saving/loading, global test evaluation, classification reports and confusion matrices. Synthetic accuracy is not a scientific result and is not included in the paper.
- The publication pipeline was exercised with disposable fixtures representing all 680 final jobs. Fixtures are not distributed as results.
- All 35 real-manifest partitions were generated. Pre-training class-count plots and their source CSV are in partition_diagnostics. Those plots describe data allocation only.
- The active CPU verification environment was captured with pip freeze in requirements.txt and requirements.txt.environment.json. It is not the unavailable original GPU environment. Colab must capture its own training environment after setup.
- LaTeX compilation was attempted but the local TeX installation lacks IEEEtran.cls. The six-section source is supplied; compile in Overleaf with IEEEtran or a complete TeX installation. No compiled-paper verification is claimed.

Not performed: new EuroSAT GPU experiments, recovery of unavailable historical checkpoints, upload of real model archives, changes to the remote repository, or Git release tags. Run the staged notebook to produce the new measurements before submitting final claims. The paper clearly marks unfinished results.
