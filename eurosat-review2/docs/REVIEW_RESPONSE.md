# Response to Reviews 2

## What is implemented

1. Phase 1 wording is revised in the paper draft: 96 and 224 are comparable
   under the predeclared validation tolerance; 96 uses about one-third the
   recorded loop time. No statistical equivalence or universal superiority is
   claimed. BN versus GN is described as **under our training setup**.
2. Scratch GN learning-rate search: 0.0003, 0.001, 0.003, each with seeds
   42, 123 and 2024. Selection uses mean validation macro-F1 at each run's
   minimum-validation-CE checkpoint. No test metric enters the choice.
3. Different pretrained/scratch learning rates are disclosed. No claim assigns
   the full difference to pretraining alone.
4. Exact original folder and commit are recorded in phase1_reference/provenance.json.
   Environment capture now fails on empty or unsuccessful pip freeze. Model
   archive generation includes real filenames and SHA-256 checksums. The README
   provides reproducible commands and a final release checklist.
5. All 35 training partitions have been generated from the original manifest:
   four Dirichlet alpha values and three long-tail ratios, each for five seeds.
   NIID-Bench adaptation, commit, MIT license and paper citation are included.
6. CE, effective-number class-balanced CE and multiclass focal loss are available.
   Validation always uses unweighted CE for checkpointing/early stopping.
7. A staged runner tunes on validation, locks selections, trains final jobs,
   seals the final plan, then allows test evaluation. It reuses identical
   completed jobs and refuses changed protocols.
8. results.csv contains one row per unique completed run, with histories,
   class counts and confusion counts embedded for CSV-only figure generation.
   Publication generation requires all 680 final evaluations with five seeds.
9. Three tables, four main figures, per-class/client results, GN sensitivity
   evidence, selected hyperparameters, and supplementary histories are supported.
10. Six requested LaTeX sections and cited references are drafted. Historical
    three-seed numbers remain clearly labeled; no dummy results were copied.

## Important clarifications and implementation choices

- **Evaluation balance:** the original fixed validation and test subsets each
  have 4,050 images, with 300, 375 or 450 images per class. They are stratified,
  not exactly balanced. We preserve them because the review also requires them
  to stay unchanged. Macro metrics are reported. Exact balancing would require
  a new evaluation protocol and rerunning baselines; it is not done silently.
- **Not federated training:** clients train independently. There is no server,
  FedAvg or parameter exchange. The global-test score means the average of
  client-model scores on one global test set, not an ensemble prediction.
- **Uncertainty:** average ten clients inside each seed, then compute mean and
  sample SD across five seeds. Do not treat 50 clients as 50 independent seeds.
- **Long-tail class order:** a single seeded permutation (order seed 42) is fixed
  across all ratios and training seeds. It is deterministically rejected until
  its head is a largest class and every ratio-10 target is available. This
  avoids oversampling or silently clipping the required formula. Smaller
  ratios then retain the exact formula at all ranks. The largest class retains
  all 2,100 training images; tail counts are 210, 42 and 21.
- **Training versus partition seed:** each final seed controls both partition
  sampling and model randomness. Loss comparisons use exactly the same frozen
  subset within a seed. Class order itself is not rerandomized between seeds.
- **Loss selection scope:** three tuning seeds are used, a conservative choice
  where the review did not specify the tuning budget. One beta and one gamma
  are selected per regime, averaging validation macro-F1 equally over clients,
  severity settings and seeds. They remain fixed across severity and five final
  seeds. The GN ratio-100 comparison uses the better non-CE method selected on
  pretrained ratio-100 validation; GN gets no separate loss search.
- **Absent client classes:** effective-number weights are zero for absent
  classes and normalized across present classes. No division by zero occurs.
  A loss cannot manufacture training evidence for a completely absent class.
- **Monotonicity:** the code flags unexpected CE trends for inspection. It never
  retries partitions or changes results merely to obtain a desired trend.
- **Original results already tested:** the original Phase 1 test set has already
  been examined. The new workflow separates new tuning and test access, but
  cannot retroactively make the historical test set unseen. Disclose this.

## Work that requires the training machine

- Run the GN search and the subsequent experiments on the actual RGB images.
- Capture the exact original GPU environment if it still exists. The bundled
  CPU test environment cannot replace the original environment record.
- Supply the original best_model.pt files: they were not available in GitHub.
- Produce real new metrics; update the paper's result interpretation after them.
- Upload checkpoint archives to a release/shared archive, record the download
  URL and SHA-256, commit the exact report sources, and create phase1-final only
  after completing the Phase 1 correction/reproducibility gate. Use a separate
  phase2-final tag for the completed next-phase package. Never label this
  untrained implementation as the final experiment result.

No new EuroSAT training accuracy is claimed in this delivery. Unit tests and
synthetic-image smoke tests validate implementation, not scientific performance.
