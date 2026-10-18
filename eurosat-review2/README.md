# EuroSAT Review 2 implementation

This package extends the Phase 1 code from commit
`2658cf6da9522e5cb86ea3a5e48eec4eb21e3ef5` of
https://github.com/CYBERCONQUEROR/EuroSat . The original project folder is
`eurosat-phase1/eurosat-phase1`; results are in that folder's `outputs/phase1`.
The original split SHA-256 is
`314a000080506dac74ddf07a196d1d76311f847418149073f17c9628e901e7bf`.

**This is runnable implementation plus a paper draft, not completed new GPU
experiments.** Original images and checkpoint weights were not supplied. Keep
all original Phase 1 outputs. `phase1_reference` is an immutable historical
three-seed reference, not a fabricated five-seed result.

## Start here

Read `docs/REVIEW_RESPONSE.md` for the reviewer-to-code mapping and important
experimental choices. `paper/main.tex` contains the requested six sections.
Use `review_workflow.py`, not the legacy `run_experiments.py`, for the new work.
The legacy runner is retained solely for reproducing the old Phase 1 protocol.

Open `Review2_Colab.ipynb` in Colab for the same staged workflow.

### Colab GPU

1. Extract this folder to Google Drive and mount Drive in Colab. Select a GPU
   runtime. Keep run outputs on Drive so completed runs survive disconnections.
2. Extract EuroSAT_RGB to the Colab local disk for faster reads. `--data-dir`
   must directly contain the ten class directories, not an extra nested folder.
3. Change directory to this package. Keep Colab's CUDA-enabled torch/torchvision.
   Do not install the CPU lock into the GPU notebook. Verify the pair first:

```python
import torch, torchvision
print(torch.__version__, torchvision.__version__)
assert torch.cuda.is_available(), 'Select a GPU runtime before starting'
```

```bash
python -m pip install -r requirements-bootstrap.txt
python environment_capture.py --output requirements.txt
python -m unittest test_phase1 test_review2
```

Restart the Colab session after package installation if requested, then rerun
imports and the environment capture. `requirements.txt` must be the successful
**pip freeze from the environment that actually trains your models**. The
bundled exact CPU snapshot is identified as such; it cannot reproduce the
historical RTX 4050 environment. The bundled `requirements.txt` is the full CPU verification environment snapshot, not a portable GPU installation recipe. For Colab, use its installed GPU pair and capture the resulting lock.

## Stage commands

Run these from the extracted package directory. Replace `/content/EuroSAT_RGB`
with the actual image root. Keep the same `--output`, `--split`, epoch budget
and source code across stages. Partition preparation does not need the images.

```bash
python review_workflow.py prepare --output outputs/review2
python review_workflow.py plan --output outputs/review2
python review_workflow.py gn-search --data-dir /content/EuroSAT_RGB --output outputs/review2
python review_workflow.py select-gn --output outputs/review2
python review_workflow.py loss-search --data-dir /content/EuroSAT_RGB --output outputs/review2
python review_workflow.py select-losses --output outputs/review2
python review_workflow.py final-train --data-dir /content/EuroSAT_RGB --output outputs/review2
python review_workflow.py seal --output outputs/review2
python review_workflow.py evaluate --data-dir /content/EuroSAT_RGB --output outputs/review2
python review_workflow.py report --output outputs/review2
```

Or run the stages in order without further approval:

```bash
python review_workflow.py all --data-dir /content/EuroSAT_RGB --output outputs/review2
```

Use `--limit 5` to train/evaluate at most five **new** jobs in one invocation;
rerun the same stage to continue. Matching completed jobs are reused. `all`
stops after reaching the chunk limit and continues on its next invocation.
An interrupted individual training is restarted from scratch, not resumed
mid-epoch. Pass `--restart-incomplete` to archive that incomplete run before
restarting. Completed metrics are never overwritten. One process at a time is
allowed per output root. A changed source/seed/split/protocol requires a new root.

## How many runs

| Work | Model trainings/evaluations |
|---|---:|
| GN learning-rate search: 3 rates × 3 seeds | 9 |
| Loss tuning: 43 cases × 6 candidates × 3 seeds | 774 |
| Final pooled baseline: 5 configurations × 5 seeds | 25 |
| Final Dirichlet: 4 alpha × 10 clients × 3 losses × 5 seeds | 600 |
| Final long-tail: 3 ratios × 3 losses × 5 seeds | 45 |
| Final scratch GN ratio 100: CE/best loss × 5 seeds | 10 |

There are **680 final model evaluations**. Reusing identical selected tuning
runs leaves **1,073 unique trainings** for the complete default workflow,
not the sum of every row above. This is a substantial experiment budget.
Each client uses only its own training images but all 4,050 validation images.
Do not estimate its cost from training-set size alone. Checkpoints can require
roughly 45–50 GB for the full search; check available Drive space first.
This budget uses three loss-tuning seeds as an explicit design choice. Do not
quietly cut the five final seeds or choose hyperparameters using test results.

## Scientific protocol

- Models: ImageNet BN at 96 for all skew experiments; scratch GN ratio-100
  comparison after GN learning-rate selection. All parameters are trained.
- Pooled final table also reruns ImageNet BN 64/96/224 and scratch BN/GN 96 at
  five seeds. The historical 15 runs remain archived. This avoids mixing new
  environments/recipes with historical three-seed statistics.
- Seeds: 42, 123, 2024, 7, 99. Tuning uses the first three. Early stopping:
  minimum unweighted validation CE, patience 10, maximum 100 epochs.
- AdamW, batch 32, weight decay 0.0001, ReduceLROnPlateau (factor 0.5, patience 2).
  ImageNet LR 0.0001; scratch BN LR 0.001; GN LR selected from its three rates.
- Training-only augmentation: horizontal and vertical flips. Resize then use
  ImageNet RGB normalization. Validation/test never use random augmentation.
- Dirichlet: NIID-Bench `noniid-labeldir`, ten clients, alpha 0.1/0.5/1/10,
  minimum ten images each, deterministic retries, no dropped images.
- Long tail: `floor(2100 * ratio**(-rank/9))`, ratios 10/50/100. One feasible
  seeded class permutation is fixed across all five seeds and all ratios.
- Frozen partitions are shared across compared losses. Training counts alone
  determine class weights. No test metric is read during tuning.
- The fixed validation/test sets are stratified, **not exactly balanced**;
  supports are 300/375/450. They are preserved rather than silently replaced.
- Dirichlet headline metric: equal mean of ten client-model global-test scores
  within each seed, then mean ± sample SD across five seeds. No FedAvg/ensemble.
- One beta and gamma per regime are locked using mean validation macro-F1,
  equally weighting clients, severity settings and tuning seeds. The selected
  non-CE method for the GN check comes from ratio-100 pretrained validation.
- Class-balanced CE uses `(1-beta)/(1-beta**n)` for present classes, with mean
  class weight one. Missing classes receive zero weight, not infinities.
  Focal loss is multiclass softmax focal with gamma 1 or 2 and no extra alpha.
- Validation CE remains unweighted even when the training objective changes.
  Training-objective and validation-CE magnitudes need not be directly comparable.

## Outputs

`outputs/review2/` contains frozen protocol and runtime records, 35 partition
JSONs, GN/loss selection files with evidence hashes, canonical run directories,
`results.csv`, and the final plan sealed before test evaluation.

Each run contains config, copied split/partition, training counts, history,
minimum-validation-loss checkpoint, best validation metrics, status, timing,
nonempty dependency snapshot, and (only in the final stage) test predictions,
classification report and confusion matrix. Checksums bind checkpoints to runs.

`generated/` contains the three LaTeX tables, four PDF/PNG figures, per-client
results, five-seed summaries, GN ratio-100 comparison, trend diagnostics and
interpretation notes. All publication numbers/figures come from `results.csv`.
Do not edit that CSV or manually type new measurements into the paper.

The paper compiles in draft mode using the verified historical table and
explicit pending placeholders until final outputs exist. From `paper/`, run
`latexmk -pdf main.tex` where IEEEtran and a LaTeX distribution are installed.
After completing experiments, the default output path is discovered automatically.
Read generated interpretations and replace pending discussion; do not copy the
reviewer's dummy values. Three-seed historical results are never labeled n=5.

## Checkpoints and release

The package has no original model weights. On the machine holding them:

```bash
python environment_capture.py --output requirements.txt
python package_models.py --root /path/to/original/outputs/phase1 --output phase1_models.tar
```

This creates an archive and a filename/SHA-256 manifest. Upload them to a GitHub
release or a shared archive; record the URL and archive checksum. Large files
may need multiple archives within the hosting service's actual size limits.
Do not put tens of GB of tensor checkpoints into ordinary Git commits.

After corrected Phase 1 GN tuning, commit the exact code, actual environment
lock, report sources, selection records and outputs; then create the requested
`phase1-final` tag on that real commit. The historical table explicitly cites
its original commit until a new tagged result snapshot exists. New Phase 2
results should use their own final tag, e.g. `phase2-final`.

For the complete five-seed next-phase package:

```bash
python package_models.py --root outputs/review2 --final-only --output review2_models.tar
cp outputs/review2/requirements.txt requirements.txt
# Commit code + protocol + results + paper + manifests in your repository.
python check_release.py --root outputs/review2 --requirements requirements.txt
```

Publishing archives and tagging are not claimed as done by this downloaded
package. Do not mark a release final while training, environment capture or
checkpoint delivery is still pending.
