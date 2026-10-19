# Phase 2 results: completed N=10 experiment

The frozen N=10 experiment has **680/680 verified final training runs and 680/680 saved test evaluations**. No completed model was restarted for this closure audit. This report packages measured results and closes export gaps. The supplied original review explicitly requests N=5 and N=10; Reviews_2.pdf specifies N=10 without explicitly removing N=5. Overall closure is pending clarification of that scope and the remaining editorial/release requirements. The previous interpretation that the user introduced a new N=5 requirement is withdrawn.

## Evaluation setup and aggregation

The original EuroSAT RGB split contains 18,900 training, 4,050 validation and 4,050 test images across ten classes. Validation and test memberships stay fixed across every experiment, with per-class supports `[450,450,450,375,375,300,375,450,375,450]`. They are stratified, not exactly balanced. Synthetic label-Dirichlet partitions use alpha 0.1, 0.5, 1 and 10; long-tail ratios are 10, 50 and 100. All partitions are frozen before training.

Each client model trains independently on its own local subset and is evaluated on the same **global** test set, including classes absent from that client's training data. There is no server, model exchange, FedAvg, ensemble, geographic partition or federated optimization. Lower global-test performance measures local-model generalization under limited/skewed training evidence; it is not evidence about a federated aggregation algorithm.

For every loss/alpha, let m[s,c] be one client's saved metric: first take the unweighted mean of the ten clients within seed s, then the mean and **sample SD (ddof=1)** of those five seed means. Seeds are 42, 123, 2024, 7 and 99. Error bars are SD over five joint partition/training seeds, not confidence intervals and not SD over 50 independent clients. Macro-F1 is averaged as a per-model metric; it is not recomputed from pooled client predictions. Long-tail and pooled models have one model per seed.

Hyperparameters were chosen using validation only on seeds 42/123/2024. Scratch GN LR is 0.0003. Class-balanced beta is 0.999 for Dirichlet and 0.9999 for long-tail; focal gamma is 1 for both. Checkpoints and early stopping use minimum **unweighted validation CE**, patience 10 and maximum 100 epochs. The GN non-CE comparator is class-balanced beta 0.9999, chosen on pretrained ratio-100 validation. No setting was adjusted after observing test results.

## Global-test results of local client models

All entries below are five-seed mean ± sample SD, in percent.

| Alpha | CE macro-F1 (%) | Class-balanced (%) | Focal (%) |
| --- | --- | --- | --- |
| 0.1 | 26.33 ± 6.13 | 32.03 ± 7.96 | 26.63 ± 6.54 |
| 0.5 | 71.39 ± 3.87 | 75.58 ± 4.37 | 70.75 ± 3.58 |
| 1 | 76.88 ± 1.73 | 79.50 ± 1.83 | 76.82 ± 1.71 |
| 10 | 93.01 ± 2.34 | 93.02 ± 2.35 | 92.86 ± 2.30 |

| Alpha | CE accuracy (%) | Class-balanced (%) | Focal (%) |
| --- | --- | --- | --- |
| 0.1 | 37.67 ± 6.30 | 42.83 ± 7.76 | 37.64 ± 6.63 |
| 0.5 | 76.85 ± 2.76 | 79.74 ± 3.13 | 76.25 ± 2.54 |
| 1 | 81.05 ± 1.28 | 83.08 ± 1.39 | 80.90 ± 1.27 |
| 10 | 93.75 ± 1.85 | 93.75 ± 1.86 | 93.63 ± 1.80 |

For CE, decreasing alpha from 10 to 0.1 reduces mean global-test macro-F1 from 93.01% to 26.33% (66.68 percentage points) and accuracy from 93.75% to 37.67% (56.09 points). At alpha 0.1, class-balanced CE increases macro-F1 by 5.71 points relative to CE, to 32.03%; it does not recover the pooled baseline. Gains shrink to approximately 0.01 point at alpha 10. Focal has no consistent improvement: its mean macro-F1 differences versus CE across alpha 0.1/0.5/1/10 are +0.31/−0.64/−0.06/−0.15 points. These are descriptive paired-configuration differences, without a significance claim.

At alpha 0.1, clients lack an average of 4.60 of ten training classes (range 0–8); at alpha 10 the average is 0.20 (range 0–2). A reweighted objective cannot supply training examples for a completely absent class. The heatmaps and per-client count/metric tables expose this allocation. Both class skew and variation in client sample size contribute, so the experiment does not isolate class imbalance as the sole cause.

## Pooled and long-tail context

| Pooled recipe / input | Accuracy (%) | Macro-F1 (%) |
| --- | --- | --- |
| imagenet, batch, 64 | 98.28 ± 0.18 | 98.22 ± 0.18 |
| imagenet, batch, 96 | 98.62 ± 0.07 | 98.56 ± 0.07 |
| imagenet, batch, 224 | 98.96 ± 0.17 | 98.92 ± 0.18 |
| scratch, batch, 96 | 96.86 ± 0.24 | 96.76 ± 0.26 |
| scratch, group, 96 | 95.58 ± 0.39 | 95.46 ± 0.39 |

The pooled ImageNet BN96 reference reaches 98.56% mean macro-F1, compared with 93.01% for CE clients even at alpha 10. The local/pooled gap includes the smaller per-client training set and independent training. Resolution and normalization comparisons remain conditional on their recipes; ImageNet and scratch use different learning rates. The BN224/BN96 test difference does not prove equivalence or override the predeclared validation-based resolution choice.

| Long-tail ratio | CE macro-F1 (%) | Class-balanced (%) | Focal (%) |
| --- | --- | --- | --- |
| 10 | 97.18 ± 0.25 | 97.39 ± 0.46 | 97.13 ± 0.38 |
| 50 | 94.80 ± 0.61 | 95.44 ± 0.40 | 94.71 ± 0.42 |
| 100 | 93.25 ± 0.32 | 93.94 ± 0.75 | 92.76 ± 0.49 |

Long-tail CE macro-F1 falls from 97.18% at ratio 10 to 93.25% at ratio 100. Class-balanced gains over CE are +0.21/+0.64/+0.69 points at ratios 10/50/100. Focal differences are −0.05/−0.09/−0.48 points. The scratch-GN ratio-100 comparison does not establish a general class-balanced benefit: CE macro-F1 is 71.57 ± 4.38%, versus 71.51 ± 1.84% for class-balanced (difference −0.05 points); accuracy is 74.95 ± 3.03% versus 73.08 ± 1.40%. The chosen comparator was retained despite its negative mean differences.

## Recovery, limitations and closure gaps

The epoch-13 non-finite validation loss did not recur during one fresh restart with the original seed/configuration. The first 12 epochs matched, and the recovered run finished 40 epochs with best epoch 30 and finite state/metrics. No experimental setting changed. **The underlying cause remains unknown.** See [RECOVERY.md](RECOVERY.md). Diagnostic bookkeeping affected 162 final runs' timing, so mixed timing records are not a controlled runtime comparison.

Only five joint partition/training seeds are available; these do not separate allocation uncertainty from optimizer/init randomness. Partitions are synthetic label-skew allocations, not real organizations or geographic shifts. Long-tail order is fixed across seeds. This is one RGB dataset, one architecture family and a specific augmentation/optimizer regime. The global test set was examined in historical Phase 1; the workflow separates new selection/test stages but cannot restore an unseen historical holdout. Unequal test-class supports and local missing classes make accuracy and macro-F1 complementary. Monotonic sample means do not prove statistical significance, equivalence or universal rankings.

**Conditional experiment gap:** no N=5 measurements are present. The original review requests both N=5 and N=10, while the later detailed plan specifies N=10 and does not explicitly address removal of N=5. Whether N=5 remains required, and whether it covers CE only or Part C losses too, is unresolved. Do not label this a confirmed new user requirement or silently launch an extension. See [primary-review quotations](REVIEW_REQUIREMENTS.md) and [conditional N=5 plan](N5_EXPERIMENT_PLAN.md): 100 CE jobs, 300 fixed-setting loss-comparison jobs, or 480 unique jobs if a separately approved N=5 tuning scope is required. Completed N=10 evidence is unchanged.

**Artifact/document gaps:** both original reviewer PDFs are now supplied, included and quoted in REVIEW_REQUIREMENTS.md. The later document does not explicitly resolve the earlier N=5 requirement. The existing manuscript retains historical/pending prose and needs an editorial update; this summary is the current measured-results account. External checkpoint archive delivery/download URLs, a real committed release and Phase 2 tag are not evidenced here. These are publication/release tasks, not grounds to rerun N=10. Checkpoint weights remain intact locally with verified hashes, and are excluded from this portable evidence bundle along with dataset image bytes.

The requested missing heatmaps, accuracy curve, seed-level result tables, per-client metrics and classification-report exports have been generated from saved data. Their absence before this audit was an export gap, not an unfinished N=10 experiment.
