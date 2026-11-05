# Sprint 3 review, testing and release guide

## Branches and review order

These branches are stacked on the merged Sprint 2 base (`7f34c84` on `main`):

| PR | Head | Initial base | Main change |
| --- | --- | --- | --- |
| 1 | `sprint3/01-acceptance-regressions` | `main` | Offline doctor, acceptance worksheet, bounded stale-frame runs/runtime |
| 2 | `sprint3/02-recognition-calibration` | `sprint3/01-acceptance-regressions` | Score calibration, quality/landmark policy and gallery aggregation |
| 3 | `sprint3/03-event-episodes` | `sprint3/02-recognition-calibration` | Episode lifecycle, safe reacquisition, idempotent event persistence, schema 3 |
| 4 | `sprint3/04-privacy-lifecycle` | `sprint3/03-event-episodes` | Profile deletion/hot reload, opt-in crops, redaction, storage limits, encryption design |
| 5 | `sprint3/05-edge-operation` | `sprint3/04-privacy-lifecycle` | Windows launcher, local health, bounded metrics, shutdown and fair 10-person scheduling |
| 6 | `sprint3/06-release-validation` | `sprint3/05-edge-operation` | Cross-platform dependency lock/CI, synthetic soak, recovery/privacy regressions and release guide |

Create PRs manually. Compare each head with its listed base to isolate the change.
Merge in order using merge commits, then retarget the next PR to `main`. If using
squash/rebase merging, restack remaining branches onto updated `main` to avoid
showing previously merged changes. Do not merge the final cumulative branch first
if six separate reviews are desired. CI on the final branch includes Linux and
Windows; earlier branches inherit the prior CI configuration until PR 6 is merged.

## Before testing on your Windows laptop

Use Python 3.11 **64-bit**. Create a dedicated disposable testing data folder outside
Git; do not test deletion/retention against your only copy of real enrollment data.
Make a private backup before upgrading existing data: schema version 3 is forward
migrated, not automatically downgraded.

Install Microsoft C++ Build Tools/Windows SDK if InsightFace builds from source.
Use Windows Settings to allow desktop camera access. Provision local, licensed
models and trusted SHA256 values as described in README. Use an account-private,
preferably encrypted directory (see PRIVACY_DESIGN.md).

```powershell
git fetch origin
git switch sprint3/06-release-validation
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip 'setuptools>=75,<81' wheel
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-windows-dev.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format --check src tests
.\.venv\Scripts\python.exe -m security_ai doctor
```

Linux uses `requirements-dev.lock` and `.venv/bin/python`, with the README's native
libraries. Do not use the Linux-resolved lock to install Windows dependencies.
The Windows lock includes platform dependencies such as `pyreadline3`. Native
source-build tools are prerequisites, not installed by application code.

Local validation on Python 3.11.17/Linux: **86 tests passed**, including a fresh
installation from the hash-pinned Linux lock. Lint, formatting, dependency checks
and offline diagnostics passed. Windows lock resolution succeeded; Windows CI,
PowerShell execution and native laptop checks are pending execution on Windows.

Tests never need a physical camera, display, trained weights or real biometric data.
The suite includes existing Sprint 1/2 tests, quality/calibration checks, a
400-frame synthetic 10-person soak with bounded state, gallery deletion/hot reload,
real ByteTrack/SQLite behavior, mocked capture/inference, event deduplication,
health failure/staleness, resource budgets and signal handling. A Linux subprocess
receives SIGTERM and exits cleanly while using a fake camera. Windows force-kill
semantics differ, so that subprocess test is explicitly skipped there; the signal
handler itself has a platform-independent unit test.

## Live model and camera checks (pending until you run them)

1. Run `doctor --check-models --person-model ... --person-sha256 ...
   --face-model-root ...`. This loads local model files but never opens a camera.
   Compare checksums against trusted publisher/provisioning records; computing a
   local hash alone does not authenticate a publisher.
2. Install GUI OpenCV last for enrollment/preview:
   `python -m pip install --force-reinstall --no-deps opencv-python==4.10.0.84`.
   Replace `python` with your venv executable if not activated.
3. Enroll consenting people using `enroll --name NAME --samples 5` and the local
   model/data flags. Vary pose/lighting. Cancel halfway and confirm nothing saves;
   repeat a sample and confirm rejection; replacing a profile requires `--replace`.
4. Start a short monitoring run with `--max-runtime 120 --metrics`. No crops should
   be created unless `--save-crops` is explicit. Test one person, then 2, 5 and 10.
   Increase camera resolution to 1280x720 where supported; check face sizes and image
   quality. Ten-person support is a target, not a demonstrated webcam capability.
5. Read `health` from another terminal while the monitor runs. Fresh healthy state
   exits 0; overloaded, stale, stopped or failed state exits 1. Review processed,
   unresolved, stale, capacity, event and failure counters plus p50/p95 timings.
6. Cross paths, occlude faces and briefly leave/re-enter. Familiar reacquisition
   requires fresh stable identity evidence, spatial overlap and exactly one lost
   candidate. Unknown people never merge across track IDs. Concurrent same-name
   candidates and spatial ambiguity remain separate. Re-entry after `--episode-gap`
   creates a new episode. Geometric tracking can still switch IDs; record failures.
7. Unplug/replug where possible. Integrated cameras may require disabling/re-enabling
   the device or a mocked test rather than a physical unplug. Avoid modifying a
   device in use by another app. Verify retry exhaustion, health failure and
   Ctrl+C/Q behavior. Native calls are cooperatively bounded, not preemptible.
8. On the disposable testing directory, run `profile-delete --name NAME --yes` and
   confirm the profile and linked episode events disappear. The running monitor
   reloads the gallery before its next inference. Unknown unlinked crops cannot be
   attributed to a person; see the deletion limits in PRIVACY_DESIGN.md.
9. Run `maintain --yes --drop-crops` with crop collection stopped. Confirm rows remain
   while image references/files disappear. Exercise a low `--storage-budget-mb` on
   disposable data and verify refusal is visible and no failed-write cooldown is
   advanced. Restore a usable budget before continuing.

## Calibration procedure

Keep enrollment and evaluation images/embeddings in separate private datasets.
Generate cosine comparison scores locally using the same recognition model and
normalization. The CSV must contain exactly:

```text
score,same_identity
0.82,1
0.21,0
```

The values above are illustrative, not measured. `same_identity=1` denotes genuine
comparisons, `0` impostor comparisons. Do not include identity names, captured
images or embeddings in CSV reports, and keep even comparison scores outside Git.

```powershell
.\.venv\Scripts\python.exe -m security_ai calibrate `
  --scores 'C:\PrivateEvaluation\test.scores.csv' `
  --thresholds 0.35 0.4 0.45 0.5 0.55 0.6 --max-fmr 0.01 --enrollment-independent
```

The flag is an explicit attestation; the tool cannot prove dataset independence.
The report includes FMR/FNMR, denominators and 95% Wilson intervals. It suggests a
threshold only when the FMR interval's upper bound meets the requested target;
small samples often return `null`. It never writes runtime configuration. Evaluate
any selected threshold on a further held-out set. Correlated video frames invalidate
the independent-trial interpretation. Pairwise scores do not establish full-system
open-set recognition or event accuracy: separately evaluate quality rejection,
identity confusion, temporal decisions and per-person event recall/duplication.
Compare `--gallery-aggregation max` and `mean` empirically; neither is universally
safer. Defaults and quality geometry remain uncalibrated heuristics, not liveness
or definitive occlusion detection.

## Development acceptance targets and soak

Starting targets selected for the laptop: 10 people; at most 2-second accepted
frame age; 0.5-second recognition cadence; at most 2 embeddings/frame; 2 GiB process
resident memory; 75% normalized machine CPU; 512 MiB event-storage admission budget.
They are configurable development budgets, not achieved measurements or guaranteed
hard limits. Over-capacity detections are excluded from recognition and reported.
CPU/memory faults are advisory; frame staleness suppresses events; storage admission
refuses writes. The schedule prioritizes oldest recognition evidence across people.

Run the Windows launcher for at least an hour, then repeat a multi-hour soak with
lighting changes and outages. Record hardware details, sample protocol, occupancy,
p95 frame age/latency, CPU/RAM, unresolved rate, identity switches, false matches,
false non-matches, event recall/duplication, reconnects and storage growth. Adjust
budgets based on evidence, not by weakening identity thresholds simply to make the
system look responsive. Keep deployment approval pending until explicit targets
and error tolerances are met. Complete HARDWARE_ACCEPTANCE.md in a private report.

## Release and rollback gates

- Review diffs and pass automated checks on both Linux and Windows CI.
- Review dependency inventory and advisory results separately. Dependency hashes
  verify artifacts; no current CVE/advisory clearance is claimed by this work.
- Record model source, license, SHA256 and intended usage; weights are not bundled.
- Verify Windows encryption/ACLs and backup/recovery policy; no application-level
  encryption or secure physical-erasure guarantee is introduced.
- Complete laptop hardware acceptance, calibration and sustained-operation checks.
- Stop monitoring, back up private data, and apply the reviewed revision. For a
  rollback restore matching old code, environment and its pre-migration data backup;
  older versions can reject the upgraded schema. See EDGE_OPERATION.md.

Firebase, Android, remote notifications/networking, liveness and confirmed-intruder
classification remain outside this sprint.
