# Primary-review comparison and corrected closure scope

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
