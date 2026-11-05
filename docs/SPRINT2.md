# Sprint 2 implementation and test guide

## Six stacked branches

Each branch includes its predecessor. Create PRs in this order, using the listed
base to review only that feature's diff:

| PR | Head | Initial base | Scope |
| --- | --- | --- | --- |
| 1 | `sprint2/01-foundation` | `main` | Configuration, injectable boundaries, cleanup, stale-decision regression fix |
| 2 | `sprint2/02-person-detection` | `sprint2/01-foundation` | Local CPU YOLO ONNX detection, letterboxing, class filtering, NMS, optional hash verification |
| 3 | `sprint2/03-person-tracking` | `sprint2/02-person-detection` | ByteTrack adapter, expiry, non-reused public IDs, conservative face association |
| 4 | `sprint2/04-multi-face-recognition` | `sprint2/03-person-tracking` | Per-track recognition and cooldowns, quality gating, ambiguity checks, enrollment consistency, model compatibility |
| 5 | `sprint2/05-camera-performance` | `sprint2/04-multi-face-recognition` | Reconnection, latest-frame buffering, freshness checks, recognition scheduling and stage timing |
| 6 | `sprint2/06-security-integration` | `sprint2/05-camera-performance` | Private persistence, migrations, retention, interrupted-write recovery, dependency lock, CI and integration tests |

After merging a predecessor, retarget the next PR to `main`. Prefer merge commits
for this stack. If you squash or rebase merge, rebase the remaining branches onto
updated `main` before review to avoid showing already-merged commits. Do not merge
the final cumulative branch first if you want six separate reviews. No PR is
created or merged automatically.

## Automated checks without camera hardware

Follow the root README installation instructions. On the final branch:

```bash
source .venv/bin/activate
python --version                         # Must be Python 3.11.x
python -m pip check
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
python -m security_ai --help
python -m security_ai monitor --help
```

Test areas:

- Configuration rejects nonfinite thresholds, invalid ranges and sample counts.
- Current low-quality evidence breaks resolved identity decisions.
- A synthetic ONNX graph exercises actual ONNX Runtime CPU execution without trained
  weights; adapter tests check YOLO input color conversion, letterbox coordinate mapping, class filtering,
  duplicate suppression, unsupported exports and missing/hash-mismatched models.
- Real ByteTrack implementation driven by synthetic detections: multiple tracks,
  low-confidence second-stage association, brief occlusion, expiry and reset.
- Ambiguous face containment and multiple faces per person remain unresolved.
- Familiar/unfamiliar people have independent evidence windows and cooldowns.
- Recognition ambiguity, evidence expiry, inference scheduling, and invalid or
  model-incompatible galleries are tested without biometric samples.
- Mocked capture reconnects, stops during backoff and discards pre-outage frames.
- SQLite migration preserves legacy rows; failed image writes roll back; orphaned
  images are reconciled; retention and private permissions are verified.
- Headless CLI integration uses fake frames, detector and face inference with real
  ByteTrack/SQLite. It checks event contents, retry after persistence failure,
  reset on reconnect, raw-frame integrity and resource cleanup.

Validation at implementation completion: **49 tests passed** on Python 3.11.17,
including a fresh environment installed from the dependency lock; Ruff lint and
format checks also passed. These results do not establish hardware accuracy or FPS.

Existing Sprint 1 tests remain included. CI runs on Python 3.11 with Linux native
libraries and the hash-pinned dependency lock. No real camera/model/image fixture
is bundled or downloaded by tests.

## Architecture and decision policy

`cli.py` composes `capture.py`, `detection.py`, `tracking.py`, `pipeline.py`,
`vision.py`, `display.py` and `storage.py`. `config.py` validates recognition policy;
`core.py` owns normalization, cached gallery matching and decision windows.

The YOLO detector retains confidence down to 0.1 by default so ByteTrack can use
low-confidence detections to maintain an existing track. New tracks require the
backend's stronger activation evidence. Frame-rate assumptions affect ByteTrack's
Kalman prediction; wall-clock expiry supplements its frame-based lifetime.

A face must be at least 90% contained in exactly one current person box, with only
one candidate face in that person box. Otherwise it cannot contribute evidence.
This conservative geometry deliberately trades recognition availability for reduced
association errors. It does not solve identity switches at person crossings.

Only quality-approved, associated faces run ArcFace. Gallery vectors are validated
and normalized once. A familiar candidate must exceed the cosine threshold and
best-versus-second-identity margin. Three consistent observations are required by
default. Unresolved evidence breaks continuity, and old evidence expires. Cooldown
advances only after event persistence succeeds. Track IDs are scoped by the stored
session ID and never imply an enrolled identity.

## Local hardware acceptance (not established by cloud tests)

Use consenting subjects, private storage and appropriately licensed model weights.
Keep all images, videos, embeddings and measurements containing identities outside
Git. You do not need hardware to review or merge the unit-tested implementation,
but target-device acceptance remains a separate required deployment check.

1. **Provision models:** verify source, license and checksums. Confirm local startup
   works with networking unavailable and missing files fail without downloads.
   Third-party augmentation update checks are disabled before loading InsightFace.
2. **Enroll:** capture five varied samples. Confirm cancellation saves nothing;
   duplicates/inconsistent subjects are rejected; replacement requires `--replace`.
3. **Single person:** validate familiar, unfamiliar and unresolved observations across
   lighting, distance, blur, exposure and pose. Hidden/absent faces stay unresolved.
4. **Multiple people:** test an enrolled and unenrolled consenting subject together.
   Confirm separate track IDs, decisions and event cooldowns. Test crossings,
   overlapping boxes, re-entry and occlusion; record identity switches and misses.
5. **Recovery:** unplug/replug the camera within the retry budget. Confirm reopen,
   fresh track/evidence state and no old event replay. Leave it unplugged and verify
   a clear failure after the configured attempt limit. Test Ctrl+C/Q during outage.
6. **Storage failures:** on a disposable test directory, simulate disk-full/permission
   errors. Confirm failures are visible, cooldown does not advance on failed writes,
   and startup reconciliation handles interrupted writes. Test retention without
   using personal production data.
7. **Performance:** run `monitor --headless --metrics --max-frames 1000` with the model
   flags from the README. Record device CPU/RAM, frame age, dropped frames, reconnects,
   and person/tracking/face/recognition timings. Aggregate p50/p95 latency and compare
   `--recognition-interval 0` versus `0.2`. Choose frame age, detection resolution and
   inference cadence against measured accuracy/latency requirements; no fixed FPS
   target is claimed. Longer capture timestamps describe read completion, not sensor
   exposure time.
8. **Calibration and soak:** use separate enrollment/test sets; measure false-match
   and false-non-match rates, per-person event recall and duplicate rates. Run for
   several hours, including lighting changes and outages. Inspect memory, CPU and
   storage growth and establish explicit deployment limits.

## Security and persistence boundaries

- SQL is parameterized, NPZ pickle loading is disabled, profile names are restricted,
  and schema/model metadata is checked before matching.
- Gallery writers use a file lock and temporary-file replacement. Enrollment reloads
  inside the lock to preserve other profiles added during capture.
- SQLite schema version 2 adds track, session and model identifiers without discarding
  Sprint 1 rows. Crop filenames are unique; writes stage before commit. Ordinary
  exceptions clean files; power loss can leave orphans recovered by maintenance.
- Local cooperating processes share advisory locks. This is not a hostile-user
  sandbox. Use trusted storage roots and OS account controls; Unix modes do not
  configure Windows ACLs or prevent a privileged attacker.
- Storage remains unencrypted; use filesystem/device encryption and protected
  backups. Retention removes local event rows and generated crops, not external
  backups or enrolled profiles. SQLite deleted pages may remain until separately
  compacted; deletion is not a cryptographic erasure guarantee.
- Model authenticity depends on verified provisioning. Optional YOLO SHA256 checking
  detects mismatches against a trusted expected hash. The face model fingerprint
  establishes gallery compatibility, not publisher authenticity.
- No remote notification, Firebase, Android, liveness, or confirmed-intruder decision
  is included. Security hardening is incremental and not a certification.

## Dependency refresh

The lock is for Python 3.11 and includes tracking/dev extras. To refresh deliberately:

```bash
uv pip compile pyproject.toml --python .venv/bin/python --extra dev \
  --generate-hashes --output-file requirements-dev.lock
```

Review the lock diff, reinstall, then rerun checks. Do not silently substitute model
weights or change recognition models without re-enrollment and recalibration.
