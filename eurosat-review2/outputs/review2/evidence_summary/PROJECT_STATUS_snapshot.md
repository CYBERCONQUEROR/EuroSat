# Project status

Updated: 2026-10-18, after verifying successful completion of final training, sealing, test evaluation and reports.

## Progress

- GN search: 9/9 runs completed; selected learning rate `0.0003`, mean validation macro-F1 95.2151% across three seeds.
- Loss search: **774/774 jobs completed; 0 remain.** Every canonical loss-search run has a completion receipt and completed status. The final detached invocation exited successfully with code 0. No training process remains active.
- Loss hyperparameter selection: **completed successfully**, exit code 0. All 774 validation-evidence hashes were verified against the selection record. The user has now explicitly authorized final training, sealing, test evaluation and reports, replacing the earlier hold.
- Authorized final pipeline: **completed**, stage **report (completed)**; final training 680/680, test evaluation 680/680. See the current lifecycle record below.
- Output root: `eurosat-review2/outputs/review2`.
- Environment: root `.venv/bin/python`, Python 3.12.13, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, CUDA runtime 13.0, NVIDIA RTX 4050 Laptop GPU. New run environment records match the original batch.

## Final completion verified

**All authorized stages completed successfully.** Final training: **680/680**, with all completion/checkpoint checksums and finite histories checked before sealing. The sealed plan contains all 680 models. Test evaluation: **680/680**. Report generation and final protected-file checks succeeded. Every stage and the full pipeline exited with code **0**, with no training/evaluation process left active.

Pipeline completion recorded at **2026-10-18 01:54:28 IST** (2026-10-17 20:24:28 UTC). Read-only follow-up verified that report provenance matches `results.csv` and the sealed plan, all 680 final CSV rows have finite test accuracy/macro-F1/balanced accuracy, and the CSV contains 1,073 unique training-result rows in total. Final seed set: 7, 42, 99, 123, 2024. Selected hyperparameters and frozen inputs were preserved; test results were not used for tuning. Historical crash/recovery notes below remain part of the audit trail.

Final output root: `eurosat-review2/outputs/review2/`.

- `final_plan.json`: sealed final requirements and selection fingerprints.
- `results.csv`: combined training/validation/test results.
- `runs/<run_id>/`: model checkpoints, completion evidence, test predictions, classification reports and confusion matrices.
- `generated/summary.csv`, `generated/seed_level_results.csv`, `generated/per_client_results.csv`: final summaries and client-level results.
- `generated/table1.tex`–`table3.tex`, `generated/figure1.pdf`–`figure4.pdf` (also PNG), supplementary learning curves: report tables and figures.
- `generated/interpretations.md`, `generated/results_discussion.tex`, `generated/trend_diagnostics.json`, `generated/scratch_gn_ratio100.csv`, `generated/provenance.json`: interpretation, diagnostics and report provenance.
- `operations/final-after-reboot-20261017T103449Z/`: final controller, process/stage records, all four console logs, preflight and exit-code evidence.

## Diagnostic recovery of seed 7, alpha 10, client 5

Post-reboot continuation authorized and launched at **16:05:24 IST** (10:35:24 UTC) on 2026-10-17. All **915** completed training runs passed full completion/checkpoint checksum verification; **522/680 final models** were complete at launch. No competing process was present, frozen source/selection/partition hashes and runtime versions matched, and the CUDA GPU was available. The reboot-interrupted client-6 focal run was archived at `eurosat-review2/outputs/review2/interrupted/bcc82aff60cae2c9e08e7bdc_20261017T103526848168`; all original file hashes matched after archival. It restarted from its original seed/configuration and was verified training on CUDA at epoch 3. No hyperparameter, model precision or GPU setting was changed.

Current supervisor/session/group: **8053**; initial workflow PID: **8055**. Persistent controller, verified preflight, launch metadata, stage logs and live `pipeline.json` are retained under `eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/`, rather than only in temporary storage. The existing numerical diagnostics remain active. The controller continues sequentially through all remaining training, verified sealing, evaluation and reporting, and stops on the first real error with no retry loop. See the current lifecycle block below for live stage details and safe-stop instructions.

Subsequent system event: the user reported a graphical crash and restarted the laptop. The new boot began at approximately 2026-10-17 **15:53 IST** (10:23 UTC). The detached session did not survive the reboot. Actual saved progress is now **522/680**, with interrupted run `bcc82aff60cae2c9e08e7bdc` (seed 7, alpha 10, client 6, focal gamma 1) showing a stale running status and zero completed epochs. It has no completion receipt or numerical failure capture. No seal or test evaluation exists. The last training log update was approximately **15:50:42 IST**. Accessible previous-boot logs did not identify a GPU/desktop crash cause. The GPU responded after reboot at 46 C with approximately 15 MiB used; that current reading does not establish its state at the crash.

The newest four completed checkpoint/config/metric/history receipt checksums and all frozen shared-input hashes passed post-reboot checks. Completed run files were left intact. The pipeline record is annotated `interrupted_by_reboot`; no exit-code file was written because the reboot prevented the supervisor's exit trap. Crash observation evidence: `eurosat-review2/outputs/review2/diagnostics/system-crash-20261017.json`. No training was relaunched during this inspection.

**Cause remains unknown.** The original non-finite validation loss did not recur in a single isolated restart with the original configuration. No scientific/numerical fix, learning-rate change, class-weight change or precision change was applied.

All 911 previously completed run directories passed completion/checkpoint checksum verification before recovery. The failed attempt was preserved byte-for-byte at `eurosat-review2/outputs/review2/interrupted/8953d19d54840571685e0d64_diagnostic_original_20261017T100645092169`. Its original best-model checkpoint contained no optimizer/scheduler state, so recovery was a **fresh restart from seed 7 and the original ImageNet initialization/configuration**, not a continuation from epoch 13 or from best weights.

The isolated restart completed 40 epochs, selected epoch 30, and produced finite history metrics, finite saved model tensors and nonnegative BatchNorm running variances. Best validation CE: **0.1409099249**; checkpoint validation macro-F1: **95.6025%**. Every recorded metric from original epochs 1–12 matched the restart exactly, excluding elapsed times. Its canonical completion receipt was written only after these checks passed. Updated final count: **519/680**; **161** trainings remain.

Validation mode was observed as `model.training == False`, with ordinary unweighted cross-entropy. Inputs/model tensors use float32, CUDA autocast is disabled, matmul precision is `highest`, CUDA matmul TF32 is disabled and the existing cuDNN TF32 permission remains enabled. No precision setting was changed. The scalar-loss check still raises the original exception; no bad batch is skipped and no NaN/Inf value is replaced.

Diagnostic additions are separate from the frozen CORE source files:

- `eurosat-review2/numerical_diagnostics.py`: adds mode/provenance metadata and a failure-capture call immediately before the original `FloatingPointError`. An AST equivalence check verifies that removing those two diagnostic calls recovers the original function exactly. Dataset reads record exact image IDs without changing samples, labels, augmentation or RNG calls.
- `eurosat-review2/diagnostic_workflow.py`: executes the original trainer/workflow with those hooks; completed run IDs, selected hyperparameters and source/protocol hashes remain unchanged. New runs record actual diagnostic entrypoint and code hashes under `runs/<id>/_diagnostics/instrumentation.json`.
- Failure capture stores image IDs/targets, input statistics and raw tensors, logits, unreduced loss, loss configuration, full model state, failure-time optimizer/scheduler/RNG state, parameter/buffer finiteness, BatchNorm statistics and precision/autocast settings under `runs/<id>/_diagnostics/nonfinite-*/`. These snapshots are explicitly diagnostic, not supported exact mid-epoch resume checkpoints.
- `eurosat-review2/replay_numerical_capture.py`: offline batch replay for captured settings, explicit float32, float32 without TF32, and CPU float32. It compares loss arithmetic and records the first non-finite layer outputs. No real failing batch was available for replay because the isolated reproduction did not fail; the original failed batch/model state was not saved.
- `eurosat-review2/test_numerical_diagnostics.py`: verifies identical healthy-path metrics/model updates/RNG and verifies image-ID/state capture before the original exception, with evaluation mode and unweighted validation loss. All **14** tests in `test_phase1`, `test_review2` and `test_numerical_diagnostics` passed.

Diagnostic bookkeeping adds timing overhead to newly instrumented runs; their elapsed-time measurements include that overhead. Completed measurements remain untouched. No experimental calculation, partition, selected parameter or checkpoint-selection rule was changed, so no scientific-protocol migration is required.

Evidence: `eurosat-review2/outputs/review2/diagnostics/recovery-20261017T100521Z/` contains `preflight.json`, `reproduction.json`, `reproduction.log`, `reproduction.exitcode` and `recovery_report.json`. The report includes checksum evidence and diagnostic code hashes. New continuation uses the diagnostic launcher, reuses completed runs, and stops on a new error without automatic retries. Before sealing it checks every required final completion/checkpoint checksum and finite history; evaluation and report remain gated on all prior requirements.

## Validation-only loss selection

Ran the existing `.venv/bin/python -u review_workflow.py select-losses --output outputs/review2` from `eurosat-review2`. No competing training or workflow process was active. The command completed with exit code 0 using only saved validation results. All 774 canonical loss-search completion records are present and their statuses are completed; the selection contains 774 matching validation-metric hashes. Frozen source hashes remain unchanged.

| Regime | Selected class-balanced beta | Selected focal gamma |
|---|---:|---:|
| Dirichlet | 0.999 | 1 |
| Long-tail | 0.9999 | 1 |

The frozen selection rule equally weights clients, severity settings and tuning seeds using validation macro-F1. Selected candidate mean validation macro-F1: Dirichlet class-balanced 67.9934%, Dirichlet focal 65.0483%, long-tail class-balanced 95.4661%, long-tail focal 94.7744%.

The non-CE loss selected for the later scratch-GN ratio-100 comparison is class-balanced CE with beta 0.9999, using mean pretrained ratio-100 validation macro-F1 93.6260% across three seeds.

- Selection record: `eurosat-review2/outputs/review2/loss_selection.json`.
- Selection SHA-256: `ad9d7bcfe7b4dca38b374e5437d8c81b7f590abcfd66db5ad950a8e0f8fb5ed5`.
- Console log: `eurosat-review2/outputs/review2/select-losses-20261016.log`.

No final training, plan sealing or test evaluation was started during validation-only selection. The user has subsequently replaced the earlier hold with explicit authorization for final-train → seal → evaluate → report. The live lifecycle block below records that new pipeline; its selections remain frozen and test results will not be used for tuning.

## Current detached session: restart after trainer crash

Final outcome checked on 2026-10-16: **completed successfully**. All 774 canonical jobs have completion records and completed status, with no unfinished canonical run records. The exit-code file contains `0`; supervisor 404972 and workflow 404973 have exited. The final log update was at approximately 2026-10-16 **00:21:25 IST** (2026-10-15 18:51:25 UTC). Last run `f60e5431151bc217eeb45b91` completed 38 epochs, with best epoch 28. Validation-only loss selection subsequently completed as recorded above. No final training or test evaluation was started; no `test_metrics.json` exists in the output root.

The following launch/process details are historical records of the successfully completed invocation.

User-authorized restart launched 2026-10-15 **13:35:16 IST** (08:05:16 UTC). No competing training process was active before launch. All 491 completion records, including checkpoint hashes, passed verification, and frozen code/split hashes were unchanged.

- Supervisor PID, session ID and process group ID: **404972**.
- Workflow PID: **404973**.
- Log: `eurosat-review2/outputs/review2/loss-search-restart-20261015T080516Z.log`.
- Launch record: `eurosat-review2/outputs/review2/loss-search-restart-20261015T080516Z.json`.
- Exit-code file, written on exit: `eurosat-review2/outputs/review2/loss-search-restart-20261015T080516Z.exitcode`.
- Invocation: same `.venv/bin/python`, dataset and `outputs/review2`, stage `loss-search`, `--device cuda --restart-incomplete`, without `--limit`.
- Interrupted evidence preserved at `eurosat-review2/outputs/review2/interrupted/d483604d021d6d36da15919d_20261015T080517403074`.
- Run `d483604d021d6d36da15919d` restarted from scratch under its canonical ID; new CUDA training was verified at epoch 1. Previously completed runs remain reusable.
- Scope: sequential remaining loss-search jobs only, stopping at completion or first error. No retry loop, frozen-code/protocol change, final training or test evaluation.

Follow progress from the repository root:

```bash
tail -f eurosat-review2/outputs/review2/loss-search-restart-20261015T080516Z.log
ps -o pid,ppid,sid,pgid,etime,stat,args -p 404972,404973 --ppid 404973
```

The canonical completion-count command retained below still reports current progress. Safely stop this current process group with:

```bash
kill -INT -- -404972
```

Wait for cleanup. Interrupted jobs restart from scratch after explicit archival; completed runs are not overwritten. The following older session details are historical.

## Previous detached session: first restart

Latest outcome: **stopped on error**, 2026-10-15 at approximately **11:38:39 IST** (06:08:39 UTC). Supervisor 358260 and workflow 358261 are no longer active; the exit-code file contains `1`. The log explicitly reports that `train.py` died with SIGSEGV (signal 11). Canonical completion records show 491/774 completed, leaving 283 jobs. There are no test evaluation artifacts.

Interrupted job: `d483604d021d6d36da15919d`, Dirichlet alpha 10, seed 123, client 8, focal loss gamma 2. Its last saved state records 36 completed epochs and best epoch 33. The native crash bypassed Python cleanup, leaving a stale `status: running` record but no active process or completion receipt. This run is incomplete and must not be counted as completed. Further continuation would require an explicitly authorized archive/restart of this incomplete run; it is not resumed mid-epoch. No completed run was overwritten, and no retry was launched during the status check.

The following launch details are retained for the now-terminated resumed invocation.

User-authorized restart launched 2026-10-15 **05:08:38 IST** (2026-10-14 23:38:38 UTC). No training or workflow process was active before launch. All 272 completed canonical loss-search jobs passed completion-record and checkpoint-checksum verification; no incomplete run required restarting. Frozen source and split hashes were unchanged.

- Supervisor PID, session ID and process group ID: **358260**.
- Workflow PID: **358261**; trainer PIDs change between sequential jobs.
- Log: `eurosat-review2/outputs/review2/loss-search-resume-20261014T233838Z.log`.
- Process record: `eurosat-review2/outputs/review2/loss-search-resume-20261014T233838Z.json`.
- Exit-code file, written when this invocation exits: `eurosat-review2/outputs/review2/loss-search-resume-20261014T233838Z.exitcode`.
- Invocation: the existing `.venv/bin/python -u review_workflow.py loss-search`, using the same dataset and `outputs/review2`, `--device cuda`, and **no `--limit`**.
- Persistence: detached POSIX session with stdin closed and console output saved. A shell supervisor records the exit code, including a native Python crash. No repository helper was added.
- Scope: continue sequentially until 774/774 or the first failure. Reuse completed runs; no retry loop, environment change, protocol change, final training or test evaluation.

From the repository root, follow the log and check the supervisor/workflow:

```bash
tail -f eurosat-review2/outputs/review2/loss-search-resume-20261014T233838Z.log
ps -o pid,ppid,sid,pgid,etime,stat,args -p 358260 --ppid 358260
```

The canonical completion-count command in the historical session section below still reports current progress.

Safely stop this **current** session, interrupting the supervisor, workflow and trainer together:

```bash
kill -INT -- -358260
```

Allow cleanup to finish. Completed runs remain reusable; an interrupted individual run may require an explicitly authorized archive/restart later. This is one user-authorized restart, not an automatic retry loop. The PID/group 295496 mentioned below belongs to the terminated original invocation.

## Audit of the first five loss-search runs

No training or review workflow Python process was running before the batch started. No concrete implementation or data error was found.

For each original run, verified completed status, canonical recipe and protocol settings, all four completion-record SHA-256 hashes (checkpoint, config, best validation metrics, history), and checkpoint metadata. Histories and model tensors contain no NaN/Inf. Checkpoint epoch and loss match the minimum validation cross-entropy in the history.

The source hashes match the frozen protocol. Copied splits and partition records match the frozen originals. The Dirichlet partition was independently regenerated using its recorded parameters and matched exactly. Reconstructing the loader's client filter reproduces every training path and recorded class count. Training and validation paths are disjoint. Audited training and validation image files are available and pass image verification.

All original runs use Dirichlet alpha 0.1, seed 42, client ID 0, ImageNet initialization, BatchNorm, image size 96, learning rate 0.0001, batch size 32 and patience 10.

| Loss setting | Run ID | Client | Train images | Best epoch | Completed epochs | Best validation CE | Validation macro-F1 at saved checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|
| Cross-entropy | ae7b11e1cfc66611e2f6427f | 0 | 1,048 | 1 | 11 | 3.218223 | 21.2109% |
| Class-balanced, beta 0.99 | 3bfcbb6ff2154afcd6d1c0d7 | 0 | 1,048 | 1 | 11 | 3.478534 | 22.1563% |
| Class-balanced, beta 0.999 | 706920af1ba6e86e54e3115a | 0 | 1,048 | 1 | 11 | 3.423670 | 22.1756% |
| Class-balanced, beta 0.9999 | 67f48d87dfb8ebe76ba4fb9f | 0 | 1,048 | 1 | 11 | 3.446953 | 22.2214% |
| Focal, gamma 1 | b0677606927ad791dcd7a9f6 | 0 | 1,048 | 1 | 11 | 3.000139 | 21.4273% |

Epoch 1 is best by validation CE for all five. Epochs 2–11 do not improve that loss, so ten consecutive non-improving epochs trigger early stopping at epoch 11. Macro-F1 sometimes improves later, but the frozen checkpoint/early-stopping rule monitors unweighted validation CE. The table reports macro-F1 at the saved minimum-CE checkpoint, not the maximum macro-F1 observed in the history.

Client 0 contains only four of the ten classes. Its training accuracy approaches 100% while global validation CE rises; this is consistent with overfitting the sparse client subset. Missing classes are allowed by the frozen Dirichlet protocol. Global validation retains all ten classes.

## Frozen subsets and class counts

Class order is identical across manifests, partitions, configs and checkpoints. Validation uses the unchanged global 4,050-image set for every run, with deterministic transforms and no client filtering. Its records match the original split exactly, and the validation classification-report supports match the counts below.

| Class | Client 0 training | Client 1 training | Global validation |
|---|---:|---:|---:|
| AnnualCrop | 0 | 32 | 450 |
| Forest | 0 | 898 | 450 |
| HerbaceousVegetation | 0 | 257 | 450 |
| Highway | 0 | 195 | 375 |
| Industrial | 0 | 0 | 375 |
| Pasture | 24 | 0 | 300 |
| PermanentCrop | 0 | 396 | 375 |
| Residential | 594 | 0 | 450 |
| River | 21 | 0 | 375 |
| SeaLake | 409 | 4 | 450 |
| Total | 1,048 | 1,782 | 4,050 |

Frozen split fingerprint: `314a000080506dac74ddf07a196d1d76311f847418149073f17c9628e901e7bf`.

Frozen protocol fingerprint: `d16e26df1f37b4180115e5cd60a082a7d8419bfbc9bbe0e2fb1733c4b567829b`.

## Second loss-search batch

Executed exactly the next five unfinished jobs using the same environment and output root, with `--device cuda --limit 5`. All use alpha 0.1 and seed 42. Completed jobs were reused.

| Loss setting | Run ID | Client | Train images | Best epoch | Completed epochs | Best validation CE | Validation macro-F1 at saved checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|
| Focal, gamma 2 | 14eac033425868af310175d0 | 0 | 1,048 | 1 | 11 | 2.911036 | 21.4883% |
| Cross-entropy | b2c97d82c767129464550711 | 1 | 1,782 | 5 | 15 | 2.338989 | 44.4543% |
| Class-balanced, beta 0.99 | 8e63676d8bfbc01152fb2369 | 1 | 1,782 | 1 | 11 | 1.954635 | 43.2645% |
| Class-balanced, beta 0.999 | 36621c0f69b96f2904b9baa0 | 1 | 1,782 | 1 | 11 | 1.929227 | 43.9101% |
| Class-balanced, beta 0.9999 | 55fb80a606db453660b6d8ee | 1 | 1,782 | 1 | 11 | 1.969530 | 44.0320% |

Post-batch checks verified all five completion records and checkpoint hashes, finite histories/model tensors, saved best epochs, intended client subsets, unchanged global validation records/supports and matching environment records. Every new run stopped after ten epochs without validation-CE improvement. All 242 protected files—including every previously completed run file and the frozen protocol, partitions, split, runtime, requirements and GN selection—have unchanged SHA-256 hashes.

Evidence: [audit JSON](eurosat-review2/outputs/review2/loss-search-audit-20261014.json), including full checkpoint hashes and original subset fingerprints; [batch log](eurosat-review2/outputs/review2/loss-search-batch2-limit5-gpu.log).

## Persistent full loss-search process

Latest check: 2026-10-15 at 05:05 IST (2026-10-14 23:35 UTC). Historical workflow PID/session/group 295496 is no longer active, and no replacement trainer or workflow was found. There are 272 canonical completed loss-search runs and no failed/incomplete canonical run records. The log last changed at 2026-10-15 02:26:24 IST (2026-10-14 20:56:24 UTC), ending after run `70481bb110e5ed332cae649b` completed at epoch 15, best epoch 5. The full log contains no traceback, KeyboardInterrupt, or chunk-limit stop. Accessible recent kernel logs did not identify a cause. The original detached launcher did not retain an exit code, so the reason for termination remains unknown. No restart was performed during this status check. No test evaluation artifacts exist.

The following PID/session details and launch commands are historical; the process is currently stopped.

Launched 2026-10-14 at 12:35:03 UTC (18:05:03 IST). Preflight verified actual canonical completion records: 10/774 loss-search jobs, with 9/9 GN runs and GN selection complete. All saved completion checksums, including checkpoints, matched; frozen source hashes and split matched the protocol. No incomplete run or active training process was present before launch.

Neither `tmux` nor `screen` is installed. The existing workflow was launched directly using a detached POSIX session (`start_new_session=True`), closed stdin, and console output redirected to a log. It survives the launcher and does not depend on keeping this chat turn open. No repository helper or training-code change was retained.

- Workflow PID: **295496**.
- Session ID: **295496**.
- Process group ID: **295496**. Each sequential trainer inherits this group.
- Trainer PID observed immediately after launch: 295508; this PID changes for subsequent jobs.
- First active job observed: `91b2b82e8c8949d57ae014ab`, focal gamma 1, client 1, Dirichlet alpha 0.1, seed 42; CUDA training was verified.
- Log: `eurosat-review2/outputs/review2/loss-search-full-20261014T123503Z.log`.
- Launch record: `eurosat-review2/outputs/review2/loss-search-full-20261014T123503Z.json`.

The command runs sequentially and reuses completed jobs. It continues through all 774 canonical loss-search jobs or exits on the first failure. There is no retry loop and no automatic restart of incomplete runs. This invocation does not start loss selection, final training, or test evaluation. The frozen protocol and completed run directories are preserved.

Exact invocation (working directory `eurosat-review2`):

```bash
'/media/phoenix/New Volume/krishnav/eurosat-resnet18/.venv/bin/python' -u review_workflow.py loss-search \
  --data-dir '/media/phoenix/New Volume/krishnav/eurosat-resnet18/EuroSAT_RGB' \
  --output outputs/review2 --device cuda
```

Check progress from the repository root:

```bash
tail -f eurosat-review2/outputs/review2/loss-search-full-20261014T123503Z.log
ps -o pid,ppid,sid,pgid,etime,stat,args -p 295496 --ppid 295496
```

Count canonical completed loss-search jobs without modifying any output:

```bash
.venv/bin/python -B - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, 'eurosat-review2')
from experiment_utils import fingerprint
from review_workflow import tuning_jobs
root = Path('eurosat-review2/outputs/review2')
protocol_hash = fingerprint(json.loads((root / 'protocol.json').read_text()))
jobs = tuning_jobs()
completed = 0
for job in jobs:
    rid = fingerprint({'protocol_hash': protocol_hash, 'job': job})[:24]
    directory = root / 'runs' / rid
    if (directory / 'completion.json').exists():
        status = json.loads((directory / 'status.json').read_text())
        completed += status['status'] == 'completed'
print(f'{completed}/{len(jobs)} completed; {len(jobs) - completed} remaining')
PY
```

Stop safely from a local terminal by sending SIGINT to the entire process group:

```bash
kill -INT -- -295496
```

This interrupts the workflow and its current trainer together. Allow cleanup to finish, then confirm the process group has exited and inspect the current run's `status.json`. Completed runs remain reusable. An interrupted individual job may need an explicitly authorized `--restart-incomplete` invocation to archive and restart it later; it is not resumed mid-epoch. Do not delete completed runs or start a competing workflow while this one is active.

After the process exits, use the canonical count and log to distinguish full completion from a failure: success requires 774/774 completion records; a failure produces a traceback and may leave a failed/incomplete run. The launch record and this document describe the launch, not a claim that the full stage has finished.

## Crash cause confirmed

Follow-up investigation supersedes the initial unknown-cause observation above. The system journal records a general protection fault in workflow process **295496** at **2026-10-15 02:26:27 IST** (2026-10-14 20:56:27 UTC). `coredumpctl info 295496` confirms **signal 11 (SIGSEGV)** and the executable `/home/phoenix/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12` underlying the project virtual environment.

The first native stack frames resolve to `_PyObject_Malloc`, `PyDict_SetItem`, and `scan_once_unicode`: the crash occurred while allocating/building a dictionary during JSON decoding. This identifies the crash location, not the underlying source of memory corruption or a proven dependency/interpreter bug. The last trainer completed successfully, and its completion receipt was written before the crash. Timing and the workflow's next operation are consistent with the parent crashing during results export; a full core backtrace would be needed to establish that Python call site conclusively.

Core dump retained by the system: `/var/lib/systemd/coredump/core.python.1000.bf32cddb28714743957e4a2dabfd8b7b.295496.1792011387000000.zst` (17.0 MB compressed). No Python traceback was expected from this native crash. The logs show no recorded out-of-memory kill or reboot around the crash, and the data volume has approximately 85.6 GiB available at the follow-up check.

No environment, training-code or frozen-protocol change was made. No automatic retry or test evaluation was started.

<!-- final-pipeline:start -->
## Current final pipeline after reboot

- State: **completed**, stage **report (completed)**; updated UTC 2026-10-17T20:24:28.546516+00:00.
- Final training: **680/680 complete**; **0 remain**; test evaluation **680/680 complete**.
- Supervisor PID / session / process group: **8053**; workflow PID: `59071`.
- Process/state record: `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/pipeline.json`.
- Controller and preflight: `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/controller.py`, `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/preflight.json` (preserved across reboots).
- Lifecycle log / exit code: `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/pipeline.log`, `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/pipeline.exitcode`.
- Detached session; continues independently of the chat, but a full laptop reboot terminates it.
- Same environment, CUDA, dataset, frozen selections/protocol, with existing numerical diagnostic hooks. No automatic retries.
- Stage order: final-train → verified seal → evaluate → report, without --limit. The reboot-interrupted run is archived and restarted through --restart-incomplete.
- Sealing requires all 680 completed training records/checkpoint hashes and finite histories; report requires all 680 evaluations.

| Stage | State | Exit | Log |
|---|---|---|---|
| final-train | completed | 0 | `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/final-train.log` |
| seal | completed | 0 | `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/seal.log` |
| evaluate | completed | 0 | `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/evaluate.log` |
| report | completed | 0 | `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/report.log` |

Progress and safe stop:

```bash
cat '/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/pipeline.json'
tail -f '/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/operations/final-after-reboot-20261017T103449Z/report.log'
kill -INT -- -8053
```

Wait for cleanup after SIGINT; completed results remain reusable.
Final outputs: `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/final_plan.json`, `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/results.csv`, `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/runs` (per-model evidence and test predictions), `/media/phoenix/New Volume/krishnav/eurosat-resnet18/eurosat-review2/outputs/review2/generated` (reports, tables, figures and provenance).
<!-- final-pipeline:end -->

## Final-training failure inspection

The failure in run `8953d19d54840571685e0d64` occurred during **epoch 13 validation**, after all 12 recorded epochs had finite metrics. The traceback points to the validation `run_epoch` call and its non-finite-loss guard. Validation progress reached 33/127 batches before the guard raised `FloatingPointError`. Validation uses ordinary unweighted cross-entropy; class-balanced weights apply only during training.

Read-only inspection found that the saved epoch-12 checkpoint has finite tensors and validation CE 0.151353. Frozen split and partition hashes match their copies, all 4,050 validation images pass image verification, and training class counts are 129–261 with finite class-balanced weights approximately 0.704–1.336. No matching GPU, out-of-memory or hardware fault was found in accessible kernel logs for the failure window.

The immediate cause was a non-finite validation loss. The underlying source was not established: the failing batch's logits and the epoch-13 model/optimizer state were not saved. These initial checks did not establish that the selected learning rate or loss weighting was erroneous. The later isolated diagnostic recovery and continuation are documented above; no scientific protocol or selected hyperparameter was changed.

## Phase 2 closure audit — 2026-10-18

The saved-data audit passed for **680/680 final models and 680/680 test evaluations** in the frozen **N=10-client plan**. GN search remains 9/9 and loss search 774/774 complete. No training, model inference, hyperparameter selection, sealing or evaluation was rerun for this audit. Existing scientific outputs were preserved.

**Corrected scope assessment:** the earlier interpretation of short user replies as introducing an unconditional new N=5 requirement is withdrawn. The now-supplied original `review.pdf` explicitly requests N=5 and N=10. The later `Reviews_2.pdf` specifies N=10 but does not explicitly remove N=5. Whether the later plan replaces the original scope, and whether any retained N=5 condition covers CE only or loss comparisons too, remains ambiguous. No N=5 jobs were launched or added to the sealed N=10 protocol. Five independent seeds is a separate quantity from client count. See the primary-review correction below; the earlier package is retained as a historical snapshot, not the current scope decision.

Audit checks included all checkpoint SHA-256 receipts and CPU-loaded finite tensors/BatchNorm state, exact canonical final manifest/selections, all 35 deterministically regenerated frozen partitions, intended training counts/subsets, unchanged global held-out membership, finite histories/checkpoint/early-stop epochs, and 2,754,000 saved prediction rows. Independently recomputed test metrics and classification reports agree with saved per-run files. The independently aggregated 28 five-seed summaries agree with the original generated report. The existing pipeline records confirm successful final-train → seal → evaluate → report ordering.

Recovery of `8953d19d54840571685e0d64` was a fresh restart from seed 7 and its original configuration, not an epoch-13 resume. The original first 12 numeric epochs match the restarted run. The recovered run completed 40 epochs, best epoch 30, validation CE 0.14090992494681734, validation macro-F1 0.9560252843458066. No scientific/precision/hyperparameter change occurred; the underlying non-finite-loss cause remains unknown. Separate diagnostic hooks were used for 162 final runs and add timing overhead.

New saved-data exports cover all four alpha values and all five partition seeds (20 heatmap panels), accuracy and macro-F1 versus alpha with client means within seeds then five-seed sample SD, 600 per-client metric records, all 680 run records/predictions/text classification reports, and 16,320 class/report rows across validation and test. The corrected summary explains the local-only setup, degradation, limitations and unresolved N=5 scope. Existing manuscript pending/historical prose and external model delivery/release tags remain editorial/release gaps; they do not require restarting completed N=10 training.

Locations relative to this workspace:

- Audit/results package directory: `eurosat-review2/outputs/review2/phase2_closure_20261018/`.
- Audit console log: `eurosat-review2/outputs/review2/phase2_closure_audit_20261018.log`.
- Saved-data audit script: `eurosat-review2/audit_phase2_closure.py`; packaging script: `eurosat-review2/package_phase2_closure.py`.
- Closure summary: `phase2_closure_20261018/PHASE2_RESULTS_SUMMARY.md`; verification: `VERIFICATION_REPORT.md`, `verification.json`, `tables/final_run_verification.csv`; scope: `REVIEW_REQUIREMENTS.md`, `scope_decision.json`; recovery: `RECOVERY.md` (all under the experiment output root).
- Portable archive target: `eurosat-review2/outputs/review2/phase2_closure_20261018.zip`; SHA-256 sidecar: `.zip.sha256`; per-file integrity manifest inside the package: `SHA256SUMS`.
- Checkpoint weights remain intact in `eurosat-review2/outputs/review2/runs/`; the portable audit package contains their verified path/size/hash manifest, not the approximately 28 GiB of weights or dataset images.

Process check found no active experiment trainer/evaluator or CUDA compute process. The read-only audit ran in terminal tool session 31441 and exited 0; it is not a training supervisor. Historical pipeline PIDs are historical records and must not be used as current stop commands. No training needs to be stopped or resumed for this closure task. Packaging integrity is recorded in the archive checksum sidecar and the audit archive verification entry below.

### Closure archive verification

Packaging terminal session 87058 exited 0. The ZIP contains 11,600 files, is 35,875,581 bytes (approximately 34.2 MiB), and passed its archive CRC check. All 11,599 content entries listed in `SHA256SUMS` were independently rehashed successfully; the remaining archive entry is the checksum list itself. Archive SHA-256 also verified: `9c1f0db7d1dfe9d651bae621ce9c9715fb7f7f162f36f6f12d59b637163e917e`. Packaging output is recorded in `eurosat-review2/outputs/review2/phase2_closure_package_20261018.log`.

The package includes a status snapshot taken before archive creation; this archive-verification entry and the archive's own checksum necessarily remain outside that snapshot. **N=10 results are audited and packaged. The earlier archive's scope interpretation is superseded by the primary-review correction below. No additional experiments were started.**

## Primary-review scope correction — 2026-10-18

Both attached PDFs are now available at the workspace root and have been read visually in full (image-only PDFs, rendered with `pdftoppm`; original page 3 is blank). The original `review.pdf`, page 2, section 3.A, states: **“Dirichlet split into 5 and 10 clients with alpha values 0.1, 0.5, 1 and 10.”** `Reviews_2.pdf`, page 1, Part A, states: **“Split the training set among 10 clients with alpha = 0.1, 0.5, 1 and 10.”** Its page-3 Table 3 also specifies ten clients. The later document never explicitly says that N=5 is removed or that its N=10 instruction replaces the original both-count requirement. Thus explicit supersession is **not established**, but continued N=5 scope must not be asserted as resolved either.

The exact reviewer question is whether the later N=10 plan replaces the original N=5-and-N=10 request; if N=5 remains, whether it covers only Part B CE baselines or also Part C class-balanced/focal comparisons, and whether already locked Dirichlet parameters transfer or N=5 needs separate validation selection. Original section 3.B follows the both-count data request with loss comparisons, while the later Part C refers back to a Dirichlet setting defined with ten clients. That is the unresolved scope ambiguity. The user's present instruction is to verify this scope and present a plan, not to launch extra training.

Conditional additional **unique training counts**, calculated from four alphas × five clients × five final seeds:

- N=10-only replacement confirmed: **0** additional jobs.
- N=5 CE baseline only: **100** jobs; no tuning.
- N=5 CE/class-balanced/focal using locked Dirichlet beta 0.999 and gamma 1: **300** jobs (100 CE + 200 non-CE); no extra search under the existing one-pair-per-regime selection rule.
- N=5-specific validation tuning, only under an explicitly agreed selection-scope amendment: **480** unique jobs = 360 search + 300 selected finals − 180 exact reuses; only 300 final tests. Search is four alphas × five clients × three tuning seeds × six candidate objectives. The other two final seeds require 120 new trainings. Do not tune from observed test results or change N=10 settings.

All variants preserve the existing split, alpha grid, seed protocol, optimizer, stopping/checkpoint rule, unweighted global validation CE, local-only/global-test procedure and five-seed aggregation. Any N=5 extension needs twenty independent five-client partitions and a separate protocol/runner/manifest/output root; it cannot select five clients from N=10 or silently reuse a runner hard-coded for ten clients. Pooled, long-tail and GN evidence have no N=5 axis and need no retraining. No N=5 partitions, training, inference or hyperparameter selection were run in preparing the plan.

Current corrected package directory: `eurosat-review2/outputs/review2/phase2_closure_20261018_review_correction/`; archive: the same path with `.zip`; checksum sidecar: `.zip.sha256`. Primary PDFs/quoted excerpts, `REVIEW_REQUIREMENTS.md`, `N5_EXPERIMENT_PLAN.md`, proposed job lists and corrected results summary are included. The prior archive is preserved with its original hash; scientific evidence in the revision is checked byte-for-byte against that previously verified package.

**N=10 experimental status remains 680/680 verified. Overall closure is pending scope clarification and remaining editorial/release requirements.** Even if N=10-only is confirmed, the existing pending/historical manuscript prose and external downloadable-checkpoint/tagged-commit evidence still prevent an unconditional complete-submission claim. The later review explicitly requests `phase1-final` in its Phase 1 repository checklist, rather than explicitly demanding a `phase2-final` tag. Both reviews' exact-balance wording also conflicts with preserving the established unequal-support stratified holdouts; the discrepancy is disclosed and was not silently fixed.

Primary source SHA-256: `review.pdf` = `4465aa5fed277787a0f028fcaa05654b4ecbf228ed459c861f672f802d626068`; `Reviews_2.pdf` = `7f72e9916dcace2c1615a3f41037cb7ed2bdfa05dbe701ea9cb59285df8f6cad`.
