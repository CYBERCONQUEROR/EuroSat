# EuroSAT ResNet-18: Phase 1 research upgrade

This package upgrades the earlier baseline. It implements the Phase 1 experiments from your review. No dataset, existing split, trained weights, or fabricated results are included. Keep your previous code and output folders as an archive.

## Start here

1. Extract this ZIP into its own directory and open a terminal there.
2. Activate your Python environment and run `python -m pip install -r requirements.txt`.
3. COPY your ORIGINAL `splits.json` into this directory. Do not regenerate it with prepare_data.py. If the root copy is missing, copy it from your original run's output directory. This must be the same split used previously.
4. Set --data-dir to the folder directly containing AnnualCrop, Forest, etc.
5. Preview the plan, train, review validation results, then evaluate.

Windows example (replace this dataset path with your actual one):

```powershell
python run_experiments.py --data-dir "D:\datasets\EuroSAT_RGB" --split splits.json --dry-run
python run_experiments.py --data-dir "D:\datasets\EuroSAT_RGB" --split splits.json
```

Linux/dev-container example:

```bash
python run_experiments.py --data-dir /path/to/EuroSAT_RGB --split splits.json --dry-run
python run_experiments.py --data-dir /path/to/EuroSAT_RGB --split splits.json
```

All following commands use `data/EuroSAT_RGB` as an example path. Use your real path consistently.

## What the automated training stage does

- Uses seeds 42, 123 and 2024 with the SAME immutable train/validation/test membership.
- Trains ImageNet-pretrained BatchNorm ResNet-18 at 224, 96 and 64 pixels: 9 runs.
- Uses validation macro-F1 at each run's LOWEST-VALIDATION-LOSS checkpoint to select the smallest size losing strictly less than 0.005 macro-F1 (0.5 percentage points) against mean score at 224. If neither smaller size qualifies, uses 224.
- Trains scratch BatchNorm and scratch GroupNorm at that selected size: 6 more runs.
- Maximum 100 epochs, early-stopping patience 10, AdamW, ReduceLROnPlateau.
- Pretrained LR 1e-4; scratch LR 1e-3. These are initial experimental settings, not claimed optimal settings. BN and GN scratch runs share their hyperparameters. Document any subsequent validation-only tuning equally for both methods.
- Uses standard ResNet-18 stem at ALL sizes: no architectural change is confounded with the image-size comparison.
- Does NOT evaluate the test set during training or resolution selection.
- Writes `selection.json`, `protocol.json`, and `locked_runs.json` to outputs/phase1.

For limited GPU memory add `--batch-size 16`. Use the same batch size across compared settings. Specify `--device cuda` if you want failure instead of a silent CPU fallback. All non-default options must also be supplied identically when rerunning or evaluating: the protocol guard rejects changed options.

The runner skips completed runs in an unchanged protocol. It deliberately stops at a partial run rather than overwrite or silently resume it. Rename/archive that partial run folder and rerun the command to repeat it from the beginning. Exact optimizer/RNG-state resume is not implemented. An interrupted best checkpoint is not labelled a completed experiment.

## Test evaluation: a separate final stage

After reviewing validation results and locking settings:

```bash
python run_experiments.py --data-dir data/EuroSAT_RGB --split splits.json --stage evaluate
```

This evaluates every locked configuration/seed once and skips runs with existing test_metrics.json. It rebuilds aggregate tables. Do not choose architectures, seeds or hyperparameters based on these test results.

The automatic experiment set has 15 runs. For five-seed final experiments use a NEW output root and explicitly choose five unique seeds, e.g. `--seeds 42 123 2024 31415 27182 --output outputs/phase1_five`. This repeats the selection protocol and creates 25 runs. Alternatively, predeclare the final configuration and run five individual runs, without retuning after seeing test scores. Specify which protocol you used in the paper.

## Individual run

```bash
python train.py --data-dir data/EuroSAT_RGB --split splits.json --output outputs/manual_bn64 --image-size 64 --norm batch --seed 42
python train.py --data-dir data/EuroSAT_RGB --split splits.json --output outputs/manual_gn64 --image-size 64 --norm group --from-scratch --lr 0.001 --seed 42
python evaluate.py --data-dir data/EuroSAT_RGB --checkpoint outputs/manual_gn64/best_model.pt
python predict.py --image path/to/patch.jpg --checkpoint outputs/manual_gn64/best_model.pt
```

GroupNorm uses 32 groups and RANDOM initialisation. The script rejects ImageNet+GroupNorm because silently loading BN weights into a GN network would change the experimental interpretation. Prediction supports both new normalisation settings and older BN checkpoints.

## Files changed or added

| File | Role |
|---|---|
| model.py | BN/GN options, random/pretrained guard, checkpoint-compatible loading |
| train.py | Validation metrics, timing, stop reason, atomic best checkpoints, PNG/PDF curves |
| data.py | Same transforms plus optional client subset API |
| evaluate.py | Accuracy, macro-F1, balanced accuracy, per-class recall, PDF confusion matrix |
| metrics.py | Shared metric definitions |
| experiment_utils.py | Manifest validation, fingerprints, atomic JSON and size selection |
| run_experiments.py | Staged 15-run automation and fixed protocol checks |
| summarize_results.py | Per-run CSV and mean/sample-SD CSV, Markdown and LaTeX tables |
| test_phase1.py | Tests for splitting guards, size selection, metrics and sample SD |

The optional dataset API is `EuroSATDataset(root, records, transform, client_id=None, partition=None)`. A future partition mapping must map string client IDs to lists of relative image paths drawn from the supplied records. It checks duplicates, missing paths and empty clients. Phase 1 passes no client ID. Partition generation, FedAvg, long-tail sampling and imbalance losses are intentionally deferred until Phase 1 review; --loss currently accepts cross-entropy only.

## Outputs per run

- config.json, original splits.json copy, environment.json, environment_requirements.txt
- best_model.pt, best_validation_metrics.json
- history.csv: train/validation loss, accuracy, macro-F1, balanced accuracy, LR and epoch seconds
- status.json: completion status, stop reason, epochs completed, selected epoch, timestamps and timings
- training_time.txt
- accuracy_curve.png/pdf and loss_curve.png/pdf (two separate figures)
- After evaluation: test_metrics.json, classification_report.txt, test_predictions.csv, confusion_matrix.png/pdf

Timing definitions: epoch_seconds covers train+validation computation and metric calculation. training_loop_seconds also includes scheduler/checkpoint/history writing. total_seconds includes data/model setup and possible pretrained-weight download, but not environment capture or final curve rendering. Compare loop time on the same hardware. Graceful interrupts record status; power loss or a forced process kill can leave status as running, which is never treated as complete.

## Aggregate paper table

```bash
python summarize_results.py --root outputs/phase1
```

Produces:
- results.csv: one row per completed run, raw fractional scores, per-class recalls.
- summary.csv: mean and SAMPLE standard deviation (ddof=1) per configuration, with seed count and split fingerprint.
- phase1_results.md: readable table; scores shown as percentages.
- phase1_results.tex: LaTeX table for the paper (`booktabs` required). Use `\input{phase1_results.tex}`.

Before evaluation, test cells say `pending`; there are no invented values. Partial-seed test summaries are not reported. Use a dedicated root per protocol; do not mix different learning rates/hardware into a summary directory.

## Research interpretation and limits

- The earlier CSV ends at epoch 16. It cannot establish why the run stopped. Inspect the old config and terminal logs; this package cannot retrospectively recover that event or elapsed time.
- Lowest validation loss and highest validation accuracy can select different epochs. The declared rule remains lowest loss; exact ties keep the earlier checkpoint.
- Fixed stratification preserves original class proportions, not an exactly balanced evaluation set. No evaluation images are resampled.
- Image-level splitting can mix related geographic patches across sets. Do not claim geographic generalisation or direct superiority over papers with different splits.
- Upsampling adds no new image detail; the actual runtime/accuracy tradeoff is measured, not assumed to be 12x.
- BatchNorm vs GroupNorm is an empirical comparison. Three seeds give a preliminary variability estimate, not statistical proof or guaranteed publication.
- Training accuracy is measured online with augmentation while weights change. Validation is measured after the epoch without random augmentation.
- Seeds reduce randomness but do not guarantee bit-identical results across hardware/software. Keep environment metadata.
- No Phase 2/3 result or monotonic degradation is assumed or manufactured.

## Verification

Run `python -m unittest test_phase1.py` for protocol tests. Python source compilation and these tests were checked during authoring. PyTorch/torchvision were not installed in the authoring environment, and your dataset and original splits.json were not provided. Full image loading, training, checkpoint reloading and CUDA execution therefore still require an end-to-end run on your machine. Start with an individual one-epoch run in a separate smoke-test directory if needed; do not include that run in the research table.

Sources: https://github.com/phelber/EuroSAT and https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
