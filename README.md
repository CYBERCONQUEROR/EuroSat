# EuroSAT RGB classification with ResNet-18

An end-to-end PyTorch project: fixed stratified splits, fine-tuning, validation-based early stopping, held-out test evaluation, and single-image prediction.

## 1. What you are training

Input: one RGB satellite patch. Output: one of ten land-use/land-cover classes.
The default uses ImageNet-pretrained ResNet-18 and replaces its final 512-to-1000 classifier with a 512-to-10 linear layer. Every layer is fine-tuned. Use --from-scratch for randomly initialized weights instead.

RGB JPEGs contain 3 channels, not all 13 Sentinel-2 bands. No change to ResNet-18's three-channel input convolution is needed. This is image classification, not pixel-level segmentation. The model gives one class per patch; it does not draw land-cover boundaries.

Original patches are 64 x 64. This baseline resizes them to 224 x 224 without cropping, uses ImageNet normalization, and applies horizontal/vertical flips only during training. Upsampling does not create additional spatial detail. The same deterministic resizing and normalization are used for validation, testing, and prediction.

## 2. Project files

| File | Purpose |
| --- | --- |
| requirements.txt | Dependencies |
| prepare_data.py | CLI to create the fixed 70/15/15 split |
| data.py | Class names, split creation, dataset and transforms |
| model.py | ResNet-18 construction and checkpoint loading |
| utils.py | Random seeds and device selection |
| train.py | Training, validation, best checkpoint, early stopping and curves |
| evaluate.py | Test accuracy, macro F1, class report and confusion matrix |
| predict.py | Top-k predictions for a new RGB patch |

Run all commands from this project directory. No dataset or trained weights are bundled.

## 3. Set up Python

Python 3.11 or 3.12 is a practical choice. Create an environment:

```bash
python -m venv .venv
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS / VS Code Linux dev container activation:

```bash
source .venv/bin/activate
```

Then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, use https://pytorch.org/get-started/locally/ to select a compatible PyTorch/torchvision CUDA installation command for your system first, then install requirements. CUDA must also be exposed to your container if using Docker. The default requirements do not guarantee GPU support on every OS. CPU execution is supported but slower.

Verify:

```bash
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

If PowerShell blocks activation, you can run the environment interpreter directly instead, for example `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, and use that interpreter for later commands.

## 4. Extract your downloaded ZIP

Extract EuroSAT_RGB.zip using your normal archive tool. Point --data-dir at the folder directly containing these ten folders:

```text
AnnualCrop/
Forest/
HerbaceousVegetation/
Highway/
Industrial/
Pasture/
PermanentCrop/
Residential/
River/
SeaLake/
```

For the commands below, place them under `data/EuroSAT_RGB/`. A directory named `2750` or an extra nested archive directory is also fine: use its actual path instead. Each class directory should contain that class's JPEGs. You do not need to copy files into train/validation/test folders.

Windows absolute-path example:

```powershell
python prepare_data.py --data-dir "D:\datasets\EuroSAT_RGB" --output splits.json
```

The remaining examples use `data/EuroSAT_RGB`.

## 5. Prepare a fixed split

```bash
python prepare_data.py --data-dir data/EuroSAT_RGB --output splits.json --seed 42
```

For all 27,000 images, this creates 18,900 training, 4,050 validation and 4,050 test records. Stratification preserves approximate class proportions. The JSON contains relative image paths and labels, not image copies. It refuses to overwrite an existing manifest. Keep it unchanged while comparing models, and do not modify the dataset after splitting.

This is a random image-level baseline split, not an official geographic benchmark. It does not verify geographic independence or remove duplicate content. Nearby/overlapping satellite patches can inflate apparent generalization. To assess performance on new areas, obtain geographic group information and split by nonoverlapping regions or scenes before training. Do not directly compare this result with published scores using different split protocols.

## 6. Train

```bash
python train.py --data-dir data/EuroSAT_RGB --split splits.json --output outputs/run1 --epochs 20 --batch-size 32
```

Defaults:

| Setting | Value |
| --- | --- |
| Initialization | ImageNet weights |
| Trainable layers | All |
| Image size | 224 x 224 |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| LR scheduler | Halve LR after validation-loss plateau |
| Checkpoint selection | Lowest validation loss |
| Early stopping | 5 epochs without validation-loss improvement |
| Device | CUDA if available; otherwise CPU |
| DataLoader workers | 0, suitable for Windows |

The first pretrained run downloads ImageNet weights if not cached and requires internet. Evaluation and prediction load your checkpoint without downloading pretrained weights.

For limited GPU memory, reduce --batch-size to 16 or 8. A laptop's system RAM size alone does not determine GPU training capacity. On a Linux GPU system you may increase --workers to 2 or 4. CPU is supported with --device cpu. Use --epochs 1 for a pipeline smoke run; that is not a performance evaluation.

To train from scratch:

```bash
python train.py --data-dir data/EuroSAT_RGB --split splits.json --output outputs/scratch1 --from-scratch --epochs 50 --lr 0.001 --patience 10
```

These are starting hyperparameters, not a claimed optimum. Tune using validation only. Random initialization typically requires more training and tuning. Each run requires a new empty output directory so earlier results are not accidentally overwritten. The checkpoint is for inference; exact interrupted-training resumption is not implemented. Seeds improve repeatability but do not guarantee bit-identical results across hardware or library versions.

## 7. Evaluate the selected model

After choosing your final settings using validation, run:

```bash
python evaluate.py --data-dir data/EuroSAT_RGB --checkpoint outputs/run1/best_model.pt
```

It uses the saved test split from the run directory and checks the split fingerprint against the checkpoint. It writes:

| Output | Meaning |
| --- | --- |
| best_model.pt | Best model weights plus class mapping, image size and configuration |
| config.json | Run parameters |
| splits.json | Exact split used for this run |
| history.csv | Per-epoch train/validation loss and accuracy |
| training_curves.png | Learning curves |
| test_metrics.json | Accuracy, macro F1 and per-class statistics |
| classification_report.txt | Human-readable precision, recall and F1 |
| test_predictions.csv | Each test image's actual/predicted class |
| confusion_matrix.png | True classes on rows; predicted classes on columns |

CrossEntropyLoss receives raw logits. Softmax is applied only for displayed prediction scores. Do not tune settings after inspecting test results and still describe that same test set as untouched.

## 8. Predict a new image

```bash
python predict.py --image "path/to/new_satellite_patch.jpg" --checkpoint outputs/run1/best_model.pt --top-k 3
```

Use a satellite RGB patch with a scale and appearance similar to EuroSAT. An arbitrary street photograph, a large map screenshot, or a full Sentinel-2 scene is outside this input setup. Full scenes require a separate geospatial tiling workflow. Softmax scores are not calibrated confidence and do not reliably detect unfamiliar inputs.

## 9. Troubleshooting

- Class-folder error: --data-dir must be immediately above AnnualCrop, Forest, etc.
- CUDA unavailable: check your GPU-capable PyTorch installation and container GPU access, or use CPU.
- Out of GPU memory: lower batch size and close other GPU workloads.
- Windows multiprocessing issue: leave --workers at 0; scripts already use main guards.
- Weight download blocked: allow the PyTorch weight download or use --from-scratch (changes the experiment).
- torchvision operator/import error: reinstall a matched torch/torchvision pair using the official installer.
- Very slow training: CPU at 224 x 224 can be slow; a CUDA GPU is preferable. Changing image size changes the experiment.

## 10. Verification and sources

The supplied scripts were syntax-checked. Split logic was exercised with 27,000 synthetic file records: sizes, disjoint membership, complete coverage, and class balance were checked. Actual image loading, training and inference were not run in the authoring environment because PyTorch and your dataset were unavailable. No trained accuracy or runtime is claimed.

Official sources:
- EuroSAT dataset: https://github.com/phelber/EuroSAT
- ResNet-18 API and ImageNet weights: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- PyTorch installation selector: https://pytorch.org/get-started/locally/
- EuroSAT paper: https://arxiv.org/abs/1709.00029
