"""Run after model selection. Do not tune hyperparameters on test results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from tqdm import tqdm

from data import EuroSATDataset, load_manifest, make_transform, manifest_hash
from model import load_model
from utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--checkpoint', default='outputs/run1/best_model.pt')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = p.parse_args()
    device = get_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    output = Path(args.checkpoint).parent
    manifest = load_manifest(output / 'splits.json')
    if manifest_hash(manifest) != checkpoint['split_hash']:
        raise ValueError('Split manifest does not match the trained checkpoint.')
    classes = checkpoint['classes']
    records = manifest['splits']['test']
    dataset = EuroSATDataset(args.data_dir, records, make_transform(checkpoint['image_size']))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    targets, predictions = [], []
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc='Test'):
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    labels = list(range(len(classes)))
    report = classification_report(targets, predictions, labels=labels, target_names=classes, zero_division=0)
    metrics = {'accuracy': accuracy_score(targets, predictions),
               'macro_f1': f1_score(targets, predictions, labels=labels, average='macro', zero_division=0),
               'classification_report': classification_report(targets, predictions, labels=labels,
                                                              target_names=classes, output_dict=True, zero_division=0)}
    print(report)
    print(f'Test accuracy: {metrics["accuracy"]:.2%}; macro F1: {metrics["macro_f1"]:.4f}')
    (output / 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (output / 'classification_report.txt').write_text(report, encoding='utf-8')
    with (output / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'actual', 'predicted'])
        writer.writerows((r['path'], classes[y], classes[pred]) for r, y, pred in zip(records, targets, predictions))
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(confusion_matrix(targets, predictions, labels=labels), display_labels=classes).plot(
        ax=ax, xticks_rotation=90, cmap='Blues', colorbar=False)
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
