# Home Security Camera — Sprint 2

Local-first indoor person tracking and face recognition on Python **3.11**.
An experimental prototype, not a certified security or emergency system.

The pipeline uses a local YOLOv8/YOLO11 COCO ONNX person detector, ByteTrack,
InsightFace SCRFD/ArcFace, ONNX Runtime CPU, OpenCV, NumPy, and SQLite.
Identity states are **familiar**, **unfamiliar**, and **unresolved**. Unfamiliar
means no accepted gallery match; it does **not** mean a confirmed intruder.
A person without an unambiguously associated, usable face remains unresolved.

## Install and run automated tests

Linux prerequisites: Python 3.11 with headers/venv support, a C++ compiler,
`libgl1`, and GLib (`libglib2.0-0` or the distribution's `libglib2.0-0t64`).
On Ubuntu with Python 3.11 available:

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev build-essential libgl1 libglib2.0-0
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=75,<81' wheel
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps -e .
python -m pytest -q
python -m ruff check src tests
```

The lock includes development and tracking dependencies and was resolved for
Python 3.11. If your distribution lacks Python 3.11, use an approved version
manager. Windows users can create the environment with `py -3.11 -m venv .venv`
and activate `.venv\Scripts\Activate.ps1` before running the Python commands.

Automated tests require **no camera, model weights, graphical display, or real
biometric data**. They use synthetic arrays and mocked camera/inference boundaries.

InsightFace's augmentation dependencies install `opencv-python-headless` alongside
`opencv-python`. Both distributions share the `cv2` namespace. The lock pins both
to 4.10.0.84. For an interactive desktop preview, install the GUI build last:

```bash
python -m pip install --force-reinstall --no-deps opencv-python==4.10.0.84
```

A working local desktop display is still required for GUI enrollment/preview.
Cloud validation uses headless tests and does not call GUI functions.

## Local model provisioning

The application never downloads weights automatically. Review model provenance,
license, and checksums before provisioning:

- YOLO: a **static batch-1 float32 raw COCO YOLOv8 or YOLO11 ONNX export**, with
  `[1,3,H,W]` input and `[1,84,N]` or `[1,N,84]` output. COCO class 0 is person.
  YOLOv5, custom class sets, segmentation, dynamic shapes, and exports with built-in
  NMS are unsupported. Choose an appropriately licensed model/export, for example
  a small model for CPU evaluation. Ultralytics model licensing may require AGPL
  compliance or a commercial license; this repository's MIT license does not
  grant rights to third-party model weights.
- InsightFace: provision licensed `buffalo_l` detection and recognition `.onnx`
  files beneath `<face-root>/models/buffalo_l/`. Its pretrained model terms can
  restrict use to non-commercial research. The runtime enables detection and
  recognition only and fingerprints the recognition file for gallery compatibility.

Store weights outside Git, preferably outside the checkout. No detector accuracy
or target-device throughput is claimed by the synthetic tests.

## Enroll and monitor on your own device

Only enroll consenting people. Enrollment is interactive and rejects inconsistent
or near-duplicate samples; vary lighting and pose without changing the subject.

```bash
security-ai --data-dir /private/security-ai enroll \
  --name alice --samples 5 --face-model-root /private/face-models

security-ai --data-dir /private/security-ai monitor \
  --person-model /private/models/yolo.onnx \
  --person-sha256 YOUR_VERIFIED_SHA256 \
  --face-model-root /private/face-models --metrics

security-ai --data-dir /private/security-ai events
```

Use `--replace` to replace an existing profile explicitly. Sprint 1 galleries lack
model fingerprints: archive them securely and enroll into a new private data
directory. Do not silently label old embeddings with the current model hash.

For local headless monitoring, add `--headless`; stop with Ctrl+C. A camera is
required for live monitoring, never for tests. `--max-frames 300` limits successful
processed frames; it is not a wall-clock timeout. Use `monitor --help` for controls.

The default recognition threshold **0.45**, ambiguity margin **0.05**, enrollment
consistency **0.6**, and duplicate threshold **0.995** are illustrative and
uncalibrated. Evaluate enrollment-independent samples for your device and risk
requirements. See [Sprint 2 testing and review guide](docs/SPRINT2.md).

## Data and limitations

- Templates, crops, and SQLite metadata remain **unencrypted**. Use device/disk
  encryption, a trusted account, and protected backups. Local permissions are
  enforced where the OS supports Unix modes.
- Default event retention is 7 days and 10,000 events, enforced at startup and
  every 300 processed frames. These are periodic limits, not a hard disk quota.
  Missing crops and interrupted writes are reconciled at the same points.
- Tracks are geometric continuity, not authenticated identities. Occlusion,
  crossings, pose, masks, and crowded scenes can fragment or switch tracks.
- No liveness or anti-spoofing protection. Recognition is not an access-control
  guarantee. Repeated frames are not independent biometric evidence.
- Capture uses a latest-frame buffer, bounded reconnect attempts, and expiring
  evidence. Native driver reads can still block; shutdown reports a timeout.
- Frames skipped by recognition scheduling produce unresolved current decisions;
  old recognition results never generate new events without new evidence.
- Model execution and storage are synchronous in the processing consumer; capture
  continues independently. Device benchmarks and sustained hardware testing remain
  required. No Firebase, Android, server, or remote application networking is added.

Never commit embeddings, crops, databases, weights, credentials, or captured video.
Git ignore patterns are defense in depth and do not replace reviewing staged files.
