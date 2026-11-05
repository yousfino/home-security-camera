# Device acceptance record

Status: **PENDING**. Synthetic/cloud checks are not live-device validation.
Target device: Windows laptop integrated webcam. Development occupancy target: 10
people. Starting configurable budgets: 2-second frame age, 0.5-second recognition
interval, 2 embeddings/frame, 2 GiB process RAM and 75% normalized CPU. These are
chosen targets, not measurements; set recognition error tolerances using calibration.
Use a private copy of this worksheet for results that contain personal data.

Record device/OS, Python version, camera/backend, input dimensions, model source,
license/hash, configuration, run duration, and enrollment-independent evaluation
protocol. Set explicit limits for concurrent people, p95 frame age, CPU/RAM, false
matches, false non-matches, duplicate events and recovery time before deployment.
No arbitrary FPS or accuracy requirement is assumed by the software.

| Check | Result | Evidence |
| --- | --- | --- |
| Offline licensed model startup | Pending | |
| Five varied enrollment samples and cancellation | Pending | |
| Familiar/unfamiliar/unresolved under changed lighting | Pending | |
| Two or more people, crossings and occlusion | Pending | |
| Unplug/replug within retry budget | Pending | |
| Exhausted retry budget and graceful termination | Pending | |
| Storage failure/recovery on disposable data | Pending | |
| Enrollment-independent calibration/held-out evaluation | Pending | |
| Resource limits and multi-hour soak | Pending | |

Run `security-ai doctor` without hardware; use `doctor --check-models` with local
model flags to additionally check model loading. Neither command verifies camera
hardware or recognition accuracy. See the sprint guide for the complete procedure.
