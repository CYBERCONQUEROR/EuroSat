"""Offline replay of a captured failing batch; never trains or accesses test data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from model import build_model
from losses import make_loss
from numerical_diagnostics import tensor_stats, precision_settings, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    directory = Path(args.capture)
    snapshot = torch.load(directory/'batch_and_state.pt', map_location='cpu', weights_only=True)
    metadata = json.loads((directory/'capture.json').read_text())
    cfg = snapshot['config']
    classes = len(snapshot['model_state']['fc.weight'])
    reports = []
    modes = [('captured_settings', args.device), ('explicit_float32', args.device),
             ('strict_float32_no_tf32', args.device), ('cpu_float32_reference', 'cpu')]
    for name, device in modes:
        if device == 'cuda' and not torch.cuda.is_available():
            reports.append(dict(mode=name, error='CUDA unavailable'))
            continue
        original = metadata['precision']
        torch.backends.cudnn.enabled = original['cudnn_enabled']
        torch.backends.cudnn.benchmark = original['cudnn_benchmark']
        torch.backends.cudnn.deterministic = original['cudnn_deterministic']
        strict = name in ['strict_float32_no_tf32', 'cpu_float32_reference']
        torch.backends.cuda.matmul.allow_tf32 = False if strict else original['cuda_matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = False if strict else original['cudnn_allow_tf32']
        model = build_model(classes, pretrained=False, norm=cfg['norm'])
        model.load_state_dict(snapshot['model_state'])
        model = model.to(device)
        inputs = snapshot['inputs'].to(device)
        if name != 'captured_settings':
            model = model.float(); inputs = inputs.float()
        model.train(snapshot['training'])
        targets = snapshot['targets'].to(device)
        criterion = make_loss(cfg['loss'], cfg['train_counts'], cfg['beta'], cfg['gamma']).to(device)
        nonfinite_layers = []
        handles = []
        def hook(layer_name):
            def observe(module, module_inputs, output):
                if isinstance(output, torch.Tensor) and not torch.isfinite(output).all():
                    if len(nonfinite_layers) < 5:
                        nonfinite_layers.append(dict(layer=layer_name, output=tensor_stats(output)))
            return observe
        for layer_name, module in model.named_modules():
            if not list(module.children()):
                handles.append(module.register_forward_hook(hook(layer_name)))
        enabled = name == 'captured_settings' and original['autocast'][device]['enabled']
        dtype = getattr(torch, original['autocast'][device]['dtype'].split('.')[-1])
        try:
            with torch.no_grad(), torch.autocast(device_type=device, enabled=enabled, dtype=dtype):
                logits = model(inputs)
                value = criterion(logits, targets) if snapshot['training'] else F.cross_entropy(logits, targets)
                raw = F.cross_entropy(logits, targets, reduction='none')
                saved_logits = snapshot['logits'].to(device)
                saved_loss = F.cross_entropy(saved_logits, targets, reduction='none')
            finite_pairs = torch.isfinite(logits.cpu()) & torch.isfinite(snapshot['logits'])
            difference = (logits.cpu().float()-snapshot['logits'].float()).abs()[finite_pairs]
            reports.append(dict(mode=name, device=device, model_training=model.training,
                precision=precision_settings(), logits=tensor_stats(logits), loss=tensor_stats(value),
                unreduced_ce=raw.cpu().tolist(), ce_on_saved_logits=tensor_stats(saved_loss),
                first_nonfinite_layers=nonfinite_layers,
                maximum_finite_logit_difference_from_capture=float(difference.max()) if difference.numel() else None))
        except Exception as error:
            reports.append(dict(mode=name, device=device, error=repr(error)))
        finally:
            for handle in handles:handle.remove()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = directory/f'replay-{stamp}.json'
    write_json(output, dict(capture=str(directory.resolve()), reports=reports,
                           note='Offline diagnostic comparisons only; no experiment metric is replaced.'))
    print(output)
    for report in reports:
        print(json.dumps({k:v for k,v in report.items() if k not in ['unreduced_ce','precision']}, allow_nan=True))


if __name__ == '__main__':
    main()
