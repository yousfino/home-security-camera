# Local Windows edge operation

Development target: Windows laptop integrated webcam, up to **10 simultaneous
people**. This is a capacity target, not measured recognition accuracy or throughput.
Initial soft budgets: 2-second frame age, 0.5-second per-person recognition interval,
2 ArcFace embeddings per frame, 2 GiB resident process memory, and 75% CPU normalized
by logical CPU count. All are configurable. Oldest recognition evidence is scheduled
first to prevent starvation. Low-quality/unscheduled observations remain unresolved;
stale inference never generates an event. If the device cannot meet the budgets,
health reports the condition; the software does not silently lower identity safety.

Use a 1280x720 camera mode if supported; verify the negotiated image quality on your
laptop. Ten distant faces may be too small for the minimum 80-pixel face-size gate.
Increasing capacity does not fix optics, lighting, occlusion, or available CPU.
When detections exceed `--max-people`, the strongest person detections are processed
and `person_capacity` is reported. People outside this limit are not recognized.

## Windows setup and launcher

Create a Python 3.11 venv and install the locked dependencies as described in README.
If InsightFace needs a native build, install Microsoft C++ Build Tools with the C++
workload and Windows SDK. Use appropriately licensed local models and a protected,
preferably encrypted data directory. For preview/enrollment install the GUI OpenCV
build last using the README command, and enable Windows webcam permissions.

```powershell
.\scripts\run-local.ps1 -DataDir "$env:LOCALAPPDATA\SecurityAI" `
  -PersonModel 'C:\PrivateModels\person.onnx' `
  -FaceModelRoot 'C:\PrivateModels\insightface' `
  -PersonSha256 '<verified 64-character SHA256>' -MaxPeople 10 -MaxRuntime 3600
```

The launcher is headless, disables crop collection by default and stops after one
hour unless configured otherwise. It does not elevate permissions, download models,
change execution policy, configure encryption, or register background services.
Ctrl+C requests graceful shutdown. Native driver/inference calls can delay shutdown;
capture join has a bounded wait and reports a blocked driver. Force-killing a Windows
process bypasses graceful handlers; startup reconciliation closes previous-session
episodes and handles interrupted writes. Only one monitor can own a data directory.

For unattended **local** development, configure Windows Task Scheduler manually:
run under your dedicated account with the project's Python executable, full local
model/data arguments and a working directory pointing at the checkout. Disable
parallel task instances; configure bounded retries and a maximum run duration. Avoid
capturing sensitive output in a shared task log. No scheduled task is created by
setup. A future production service needs separate deployment acceptance.

## Health and metrics

```powershell
.\.venv\Scripts\python.exe -m security_ai --data-dir "$env:LOCALAPPDATA\SecurityAI" health
```

This reads local `health.json` and exits 0 only for a fresh `healthy` snapshot; stale,
degraded, starting, failed, stopped or missing health exits 1. No network listener
is introduced. Health updates approximately once per second between operations.
A blocked inference cannot update health; a separate health command detects its
staleness. `--stall-timeout 10` detects lack of acquired frames when the consumer can
still execute. Maximum runtime/shutdown checks are cooperative between operations,
not hard preemption of native calls. Health is monitoring progress, not proof of
camera coverage or identity accuracy.

`--metrics` emits JSON aggregates to stdout: processed/acquired/stale/drop/reconnect
and event/failure counters; current process memory; normalized CPU percent since
the previous sample; p50/p95 timing over the latest 256 observations. This is a
bounded window, not whole-run p95. Logs on stderr contain static event/error-class
codes, never names, paths, embeddings or raw exception messages. Third-party model
startup output may contain model paths; keep any captured console output private.
The local `events --show-identities` command explicitly opts into sensitive display.

Performance faults are advisory; memory/CPU are not forcibly constrained. Storage
admission and episode-memory capacity are separately bounded. If episode checkpoints
keep failing, memory cannot grow indefinitely: the monitor fails explicitly once
its 10,000-entry checkpoint capacity is exhausted. Fix storage before restarting.

## Recovery, upgrades and rollback

1. Stop monitoring and confirm `health` reports stopped or the process has exited.
2. Make a protected backup of the data directory; never add it to Git.
3. Install the reviewed revision/dependency lock; run the camera-free test suite and
   `doctor`, then model checks and a short live run.
4. Database migration adds schema version 3 episode/event-idempotency fields. Do not
   open the upgraded database with earlier versions that reject this schema.
5. For rollback, stop the new process and restore both the previous code/environment
   and its matching pre-upgrade private data backup. Do not overwrite newer data
   without an explicit retention/recovery decision. Recheck model fingerprints.

Hardware acceptance, calibration and a multi-hour soak remain required before
claiming reliable 10-person operation. See HARDWARE_ACCEPTANCE.md and SPRINT3.md.
