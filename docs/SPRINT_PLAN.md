# Sprint 1 implementation plan

| Day | Tasks | Acceptance |
|---|---|---|
| 1 | Install Python 3.11, create venv, install deps, run tests | Tests pass |
| 2 | Check webcam permissions and OpenCV capture | Preview opens and exits cleanly |
| 3 | Load InsightFace models, inspect first-run downloads and license | Detects a face locally |
| 4 | Capture 5 quality-filtered embeddings for each consenting subject | Gallery saves/loads |
| 5 | Implement cosine similarity, threshold experiments | Familiar vs unfamiliar evaluation recorded |
| 6 | Add repeated observation decision logic | No event from one isolated observation |
| 7 | Add local SQLite event log and JPEG crops | Event and image stored |
| 8 | Test lighting, occlusion, multiple faces, and camera failure | Known limitations documented |
| 9 | Profile CPU latency, tune resolution | Benchmark log captured |
| 10 | Review security, erase unnecessary data, prepare Sprint 2 | Review complete |

## Calibration worksheet
For each consenting test subject, gather a distinct enrollment set and test set. Record pairwise similarity scores, lighting, face angle, image quality, false match rate, and false non-match rate. Choose a threshold based on the acceptable false-match risk; 0.45 in the CLI is an **unvalidated example**.

## Hardware notes
Windows: prefer DirectShow camera backend, close Teams/Zoom, allow camera permissions. Linux: check `/dev/video0` and user `video` group permissions; if running headless, OpenCV GUI windows will fail. Use an attached desktop display for this sprint. CPU-only `buffalo_l` can be slow; reduce capture size or evaluate a smaller appropriately licensed model if necessary.
