"""Launch the frozen workflow/trainer with transparent numerical diagnostics.

No protocol hashes are rewritten and no completed run is changed. Only trainer
subprocesses are redirected through the instrumentation entry point.
"""
import json
from pathlib import Path
import runpy
import subprocess
import sys

BASE = Path(__file__).resolve().parent


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'workflow']:
        raise SystemExit('Usage: diagnostic_workflow.py train|workflow ORIGINAL_ARGUMENTS...')
    mode, arguments = sys.argv[1], sys.argv[2:]
    actual_command = list(sys.argv)
    if mode == 'train':
        import train
        from numerical_diagnostics import install
        if '--output' not in arguments:
            raise ValueError('Explicit --output is required for diagnostic provenance')
        output = arguments[arguments.index('--output')+1]
        workers = arguments[arguments.index('--workers')+1] if '--workers' in arguments else '0'
        if int(workers) != 0:
            raise ValueError('Exact diagnostic image mapping currently requires workers=0')
        install(train, output, actual_command)
        sys.argv = [str(BASE/'train.py'), *arguments]
        train.main()
    else:
        original_run = subprocess.run
        def diagnostic_run(command, *args, **kwargs):
            if isinstance(command, (tuple, list)) and len(command) > 1 and Path(command[1]).resolve() == BASE/'train.py':
                command = [command[0], str(BASE/'diagnostic_workflow.py'), 'train', *command[2:]]
            return original_run(command, *args, **kwargs)
        subprocess.run = diagnostic_run
        sys.argv = [str(BASE/'review_workflow.py'), *arguments]
        runpy.run_path(str(BASE/'review_workflow.py'), run_name='__main__')


if __name__ == '__main__':
    main()
