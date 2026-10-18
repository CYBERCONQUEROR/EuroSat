"""Failure-only capture inserted into the frozen trainer without editing it.

Healthy-path tensor calculations and RNG calls are unchanged. Instrumentation
provenance is stored separately; captures are diagnostic, not resume checkpoints.
"""
import ast
from collections import deque
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_context = {}
_samples = {}
_seen_modes = set()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def tensor_stats(tensor):
    value = tensor.detach().cpu()
    finite = torch.isfinite(value)
    selected = value[finite].double()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                original_device=str(tensor.device), elements=value.numel(),
                nonfinite=int((~finite).sum()), nan=int(torch.isnan(value).sum()),
                positive_inf=int(torch.isposinf(value).sum()),
                negative_inf=int(torch.isneginf(value).sum()),
                minimum=float(selected.min()) if selected.numel() else None,
                maximum=float(selected.max()) if selected.numel() else None,
                mean=float(selected.mean()) if selected.numel() else None,
                std=float(selected.std(unbiased=False)) if selected.numel() else None)


def precision_settings():
    return dict(default_dtype=str(torch.get_default_dtype()),
                autocast={device: dict(enabled=torch.is_autocast_enabled(device),
                                      dtype=str(torch.get_autocast_dtype(device)))
                          for device in ['cpu', 'cuda']},
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_enabled=torch.backends.cudnn.enabled,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def parent_training_context():
    frame = sys._getframe(1)
    while frame is not None:
        if 'config' in frame.f_locals and 'epoch' in frame.f_locals:
            return frame.f_locals
        frame = frame.f_back
    return {}


def begin_epoch(local):
    """Record mode once per process; never consume RNG or change model mode."""
    training = local['training']
    if training in _seen_modes:
        return
    _seen_modes.add(training)
    output = Path(_context['output']) / '_diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'instrumentation.json'
    record = json.loads(path.read_text()) if path.exists() else dict(_context)
    record.setdefault('observed_modes', []).append(dict(
        training=training, model_training=local['model'].training,
        gradients_enabled=training,
        active_loss='configured_training_objective' if training else 'unweighted_cross_entropy',
        precision=precision_settings()))
    write_json(path, record)


def unreduced_loss(logits, targets, training, criterion):
    if not training or criterion is None:
        return F.cross_entropy(logits, targets, reduction='none')
    if isinstance(criterion, nn.CrossEntropyLoss):
        return F.cross_entropy(logits, targets, weight=criterion.weight,
                               ignore_index=criterion.ignore_index,
                               label_smoothing=criterion.label_smoothing, reduction='none')
    if hasattr(criterion, 'weights'):
        return F.cross_entropy(logits, targets, reduction='none') * criterion.weights[targets]
    if hasattr(criterion, 'gamma'):
        logpt = F.log_softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        return -logpt * (-torch.expm1(logpt)).clamp_min(0).pow(criterion.gamma)
    raise TypeError('Unsupported training criterion: ' + type(criterion).__name__)


def capture_failure(local):
    """Called immediately before the original FloatingPointError is raised."""
    destination = None
    try:
        main = parent_training_context()
        cfg = main.get('config', _context.get('config', {}))
        epoch = main.get('epoch')
        training = local['training']
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        destination = Path(_context['output']) / '_diagnostics' / f'nonfinite-{stamp}'
        destination.mkdir(parents=True, exist_ok=False)
        model, loader = local['model'], local['loader']
        inputs, targets, logits, loss = (local[k] for k in ['images', 'labels', 'logits', 'loss'])
        batch_size = len(targets)
        samples = list(_samples.get(id(loader.dataset), []))[-batch_size:]
        if len(samples) != batch_size:
            start = len(local['targets'])
            records = getattr(loader.dataset, 'records', None)
            samples = [dict(index=i, path=records[i]['path'] if records else f'index:{i}',
                            label=int(targets[i-start])) for i in range(start, start+batch_size)]
        if [sample['label'] for sample in samples] != targets.detach().cpu().tolist():
            raise ValueError('Diagnostic image ID mapping does not match actual targets')
        state = cpu_copy(model.state_dict())
        parameters = {name: tensor_stats(value) for name, value in model.named_parameters()}
        buffers = {name: tensor_stats(value) for name, value in model.named_buffers()}
        batchnorm = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                batchnorm[name] = dict(training=module.training, eps=module.eps,
                    momentum=module.momentum, track_running_stats=module.track_running_stats,
                    running_mean=tensor_stats(module.running_mean) if module.running_mean is not None else None,
                    running_var=tensor_stats(module.running_var) if module.running_var is not None else None,
                    variance_plus_eps_nonpositive=int((module.running_var + module.eps <= 0).sum())
                    if module.running_var is not None else None)
        payload = dict(inputs=cpu_copy(inputs), targets=cpu_copy(targets),
                       logits=cpu_copy(logits), loss=cpu_copy(loss), model_state=state,
                       config=cfg, epoch=epoch, training=training,
                       classes=main.get('manifest', {}).get('classes', []),
                       optimizer_state=cpu_copy(main['optimizer'].state_dict()) if 'optimizer' in main else None,
                       scheduler_state=cpu_copy(main['scheduler'].state_dict()) if 'scheduler' in main else None,
                       torch_rng_state=torch.get_rng_state(),
                       cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                       python_rng_state=random.getstate())
        numpy_rng = np.random.get_state()
        payload['numpy_rng_state'] = dict(kind=numpy_rng[0], keys=numpy_rng[1].tolist(),
            position=numpy_rng[2], has_gauss=numpy_rng[3], cached_gaussian=numpy_rng[4])
        if local['criterion'] is not None:
            payload['criterion_state'] = cpu_copy(local['criterion'].state_dict())
        with torch.no_grad():
            per_sample = unreduced_loss(logits, targets, training, local['criterion'])
            payload['unreduced_loss'] = cpu_copy(per_sample)
            # This compares loss arithmetic on existing logits, not model precision.
            with torch.autocast(device_type=logits.device.type, enabled=False):
                ce32 = F.cross_entropy(logits.float(), targets, reduction='none')
            payload['float32_unweighted_ce_on_captured_logits'] = cpu_copy(ce32)
        metadata = dict(captured_utc=datetime.now(timezone.utc).isoformat(), epoch=epoch,
            training=training, model_training=model.training, batch_size=batch_size,
            prior_samples=len(local['targets']), image_ids=samples,
            targets=targets.detach().cpu().tolist(), inputs=tensor_stats(inputs),
            logits=tensor_stats(logits), logits_values=logits.detach().cpu().tolist(),
            loss=tensor_stats(loss), unreduced_loss=tensor_stats(per_sample),
            unreduced_loss_values=per_sample.detach().cpu().tolist(),
            float32_ce=tensor_stats(ce32), loss_configuration=dict(
                configured_loss=cfg.get('loss'), beta=cfg.get('beta'), gamma=cfg.get('gamma'),
                criterion_type=type(local['criterion']).__name__ if training else 'unweighted_cross_entropy',
                validation_is_unweighted=not training),
            precision=precision_settings(), parameters=parameters, buffers=buffers,
            nonfinite_parameters=[name for name, stats in parameters.items() if stats['nonfinite']],
            nonfinite_buffers=[name for name, stats in buffers.items() if stats['nonfinite']],
            batchnorm=batchnorm, model_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
            provenance=_context, snapshot_is_resumable=False,
            snapshot_note='Failure-time diagnostic state, not a supported exact mid-epoch resume checkpoint.')
        torch.save(payload, destination / 'batch_and_state.pt')
        write_json(destination / 'capture.json', metadata)
        print(f'Non-finite diagnostic capture: {destination}', file=sys.stderr, flush=True)
    except Exception as error:
        # The original guard still raises; capture failure must never hide it.
        print(f'Diagnostic capture error: {error!r}', file=sys.stderr, flush=True)
        if destination is not None:
            try:
                write_json(destination / 'capture_error.json', dict(error=repr(error)))
            except Exception:
                pass


def install(train_module, output, original_command):
    """Add only metadata/exception hooks; verify all original AST nodes remain."""
    _context.clear(); _samples.clear(); _seen_modes.clear()
    base = Path(train_module.__file__).resolve().parent
    _context.update(output=str(Path(output).resolve()), actual_launch_command=original_command,
        original_train_sha256=file_hash(train_module.__file__),
        instrumentation_sha256=file_hash(__file__),
        launcher_sha256=file_hash(base / 'diagnostic_workflow.py'),
        torch_version=str(torch.__version__), torchvision_version=str(train_module.torchvision.__version__),
        method='AST insertion of mode metadata and failure capture; original healthy-path arithmetic unchanged')
    original_epoch = train_module.run_epoch
    original_getitem = train_module.EuroSATDataset.__getitem__

    def tracked_getitem(dataset, index):
        result = original_getitem(dataset, index)
        _samples.setdefault(id(dataset), deque(maxlen=4096)).append(dict(
            index=index, path=dataset.records[index]['path'], label=int(result[1])))
        return result

    source, first = inspect.getsourcelines(original_epoch)
    tree = ast.parse(''.join(source)); original_dump = ast.dump(tree, include_attributes=False)
    function = tree.body[0]
    # Immediately after the original model.train(training), which is eval for val.
    function.body.insert(2, ast.parse('_diagnostics_begin_epoch(locals())').body[0])
    inserted = 0
    for node in ast.walk(function):
        if isinstance(node, ast.If) and any(isinstance(item, ast.Raise) and
                isinstance(item.exc, ast.Call) and isinstance(item.exc.func, ast.Name) and
                item.exc.func.id == 'FloatingPointError' for item in node.body):
            node.body.insert(0, ast.parse('_diagnostics_capture_failure(locals())').body[0])
            inserted += 1
    if inserted != 1:
        raise RuntimeError('Cannot identify the unique original non-finite guard')
    audit_tree = ast.parse(ast.unparse(tree))
    class RemoveHooks(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in {
                    '_diagnostics_begin_epoch', '_diagnostics_capture_failure'}:
                return None
            return self.generic_visit(node)
    audit_tree = RemoveHooks().visit(audit_tree)
    if ast.dump(audit_tree, include_attributes=False) != original_dump:
        raise RuntimeError('Instrumentation would change original training calculations')
    tree = ast.fix_missing_locations(tree); ast.increment_lineno(tree, first-1)
    train_module._diagnostics_begin_epoch = begin_epoch
    train_module._diagnostics_capture_failure = capture_failure
    exec(compile(tree, train_module.__file__, 'exec'), train_module.__dict__)
    train_module.EuroSATDataset.__getitem__ = tracked_getitem

    def uninstall():
        train_module.run_epoch = original_epoch
        train_module.EuroSATDataset.__getitem__ = original_getitem
    return uninstall
