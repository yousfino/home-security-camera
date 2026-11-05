# Home Security Camera — Sprint 1

**Local-only experimental prototype. Not a certified security or emergency system.**

## Requirements
- Windows 10/11 or Linux x86_64, Python **3.11**, USB webcam, 8 GB RAM recommended.
- No server, public port, cloud service, or Android app in this sprint.
- InsightFace `buffalo_l` pretrained weights may be restricted to non-commercial research. Check licensing before use; the first run may download model weights from the internet. Offline operation starts after the weights are present.
- Face templates and photos in `.local/` are **not encrypted** in Sprint 1. Only enroll consenting people, protect the device, and delete test data after evaluation.

## Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
python -m pytest -q
security-ai enroll --name alice --samples 5
security-ai monitor --threshold 0.45
security-ai events
```
If activation is blocked, invoke `.\.venv\Scripts\python.exe` directly and use `python -m security_ai` instead of `security-ai`.

## Ubuntu Linux
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential cmake libgl1 libglib2.0-0
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
python -m pytest -q
security-ai enroll --name alice --samples 5
security-ai monitor --threshold 0.45
security-ai events
```
On newer Ubuntu releases without `python3.11` in official repositories, install Python 3.11 through an approved version manager such as `pyenv`, rather than assuming the apt package exists.

## Usage
- `security-ai enroll --name alice --samples 5`: only one consenting face in view; press SPACE to collect each sharp sample, Q to cancel.
- `security-ai monitor --camera 0 --width 640 --height 480`: run the local monitoring window; Q exits.
- `security-ai events`: list recent local events.
- `security-ai --data-dir /some/private/path monitor`: choose another storage directory (global option before command).
- `security-ai monitor --threshold 0.45`: **example threshold, not validated**. Calibrate against your own authorized and unknown test samples.

## Data
- `.local/gallery.npz`: biometric embeddings, unencrypted.
- `.local/events.db`: local event metadata, unencrypted.
- `.local/crops/`: JPEG crops of unfamiliar faces, unencrypted.
- Delete `.local/` to erase local Sprint 1 data (verify backups separately).

## Important MVP limitations
- Supports **one face at a time**. Multiple faces result in unresolved; Sprint 2 needs person tracking and per-track decision windows.
- Repeated observations are not independent; the decision logic is illustrative, not statistically calibrated.
- No liveness detection, anti-spoofing, privacy masking, disk encryption, secure retention automation, or production authentication.
- Does not reliably detect persons whose faces are hidden; it is face-first, not full person detection.
- No camera auto-reconnect beyond frame-read retry. Reopen/recovery is a Sprint 2 task.
- Event cooldown is global, not per person. Avoid using this for actual security decisions.
- The capture pipeline is synchronous and CPU-only; actual FPS depends on hardware.
- The current event crop is drawn from the annotated frame, so it may contain the bounding-box overlay; a production version must crop the original frame before annotation.

## Acceptance checks
1. `python -m pytest -q` passes all tests.
2. Enrollment requires one sharp face and saves exactly 5 embeddings.
3. Enrolled person remains familiar under tested lighting.
4. Non-enrolled consenting test subject produces unfamiliar event and JPEG crop.
5. Low-quality or multiple-face frames show unresolved.
6. `security-ai events` lists UTC timestamps and image paths.
7. Disconnect/reconnect and multi-hour stability are **manual experiments**, not guaranteed behavior in this MVP.

## Suggested next work
Track faces per person, implement frame buffering and reconnection, tune thresholds with labeled data, encrypt stored templates, and add a durable event outbox before mobile notifications.
