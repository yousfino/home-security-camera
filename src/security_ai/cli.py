"""Single-face local webcam prototype; no server or network listeners."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
import cv2

from .config import RecognitionConfig, positive_int, finite_float
from .core import DecisionWindow, Observation, best_match, classify, normalized
from .storage import data_paths, init_db, load_gallery, save_event, save_gallery, safe_profile_name
from .vision import create_analyzer, crop_face, quality_ok


def camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(
        index, cv2.CAP_DSHOW if __import__("sys").platform == "win32" else cv2.CAP_ANY
    )
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open webcam index {index}; check permissions and camera index")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def enroll(args: argparse.Namespace, gallery_path: Path) -> None:
    name = safe_profile_name(args.name)
    model = create_analyzer(args.det_size)
    gallery = load_gallery(gallery_path)
    cap = camera(args.camera, args.width, args.height)
    samples = []
    print("Enrollment: one consenting person only. Press SPACE for each usable sample; Q quits.")
    try:
        while len(samples) < args.samples:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            faces = model.get(frame)
            ready = len(faces) == 1 and quality_ok(frame, faces[0])
            cv2.putText(
                frame,
                f"{name}: {len(samples)}/{args.samples} | SPACE capture | Q quit",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0) if ready else (0, 0, 255),
                2,
            )
            cv2.imshow("Enrollment", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32 and ready:
                samples.append(normalized(faces[0].embedding))
                print(f"Captured {len(samples)}/{args.samples}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if len(samples) != args.samples:
        print("Enrollment incomplete; nothing saved")
        return
    gallery[name] = samples
    save_gallery(gallery_path, gallery)
    print(f"Saved {len(samples)} templates for {name}. Protect {gallery_path} as biometric data.")


def monitor(args: argparse.Namespace, gallery_path: Path, db_path: Path, crops: Path) -> None:
    gallery = load_gallery(gallery_path)
    if not gallery:
        raise RuntimeError("No profiles enrolled; run 'enroll' first")
    init_db(db_path)
    model = create_analyzer(args.det_size)
    cap = camera(args.camera, args.width, args.height)
    decisions = DecisionWindow(minimum=3, window=5)
    last_alert = 0.0
    last_familiar_log = 0.0
    print("Monitoring locally. Q to quit. One face at a time for Sprint 1.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed; retrying")
                time.sleep(0.2)
                continue
            started = time.monotonic()
            faces = model.get(frame)
            label = "unresolved"
            score = -1.0
            name = None
            # Multiple faces are deliberately unresolved in this first sprint;
            # a tracker is needed to avoid mixing their observations.
            if len(faces) == 1:
                face = faces[0]
                good = quality_ok(frame, face)
                name, score = best_match(face.embedding, gallery) if good else (None, -1.0)
                status = classify(name, score, args.threshold, good)
                label, stable_name = decisions.add(
                    Observation(status, name if status == "familiar" else None, score)
                )
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0) if label == "familiar" else (0, 165, 255),
                    2,
                )
                now = time.monotonic()
                if label == "unfamiliar" and now - last_alert >= args.cooldown:
                    event_id = save_event(
                        db_path, crops, "unfamiliar", None, score, crop_face(frame, face)
                    )
                    print(f"UNFAMILIAR event #{event_id}")
                    last_alert = now
                    decisions.clear()
                elif label == "familiar" and now - last_familiar_log >= args.familiar_log_interval:
                    save_event(db_path, crops, "familiar", stable_name, score, None)
                    print(f"FAMILIAR: {stable_name}")
                    last_familiar_log = now
            else:
                decisions.clear()
            latency_ms = (time.monotonic() - started) * 1000
            cv2.putText(
                frame,
                f"{label} | {latency_ms:.0f} ms | faces={len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Home Security Camera - Local Sprint 1", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def list_events(db_path: Path) -> None:
    if not db_path.exists():
        print("No events yet")
        return
    with sqlite3.connect(db_path) as db:
        for row in db.execute(
            "SELECT id,captured_at,status,profile_name,image_path "
            "FROM events ORDER BY id DESC LIMIT 30"
        ):
            print(*row, sep=" | ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-only security AI prototype")
    parser.add_argument("--data-dir", type=Path, default=Path(".local"))
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ("enroll", "monitor"):
        p = sub.add_parser(cmd)
        p.add_argument("--camera", type=int, default=0)
        p.add_argument("--width", type=positive_int, default=640)
        p.add_argument("--height", type=positive_int, default=480)
        p.add_argument("--det-size", type=positive_int, default=640)
        if cmd == "enroll":
            p.add_argument("--name", required=True)
            p.add_argument("--samples", type=positive_int, default=5)
        else:
            p.add_argument(
                "--threshold",
                type=finite_float,
                default=0.45,
                help="Illustrative ONLY; calibrate before trusting",
            )
            p.add_argument("--cooldown", type=finite_float, default=30)
            p.add_argument("--familiar-log-interval", type=finite_float, default=60)
    sub.add_parser("events")
    args = parser.parse_args()
    if args.command == "monitor":
        try:
            RecognitionConfig(
                threshold=args.threshold,
                cooldown=args.cooldown,
                familiar_log_interval=args.familiar_log_interval,
            )
        except ValueError as exc:
            parser.error(str(exc))
    gallery, db, crops = data_paths(args.data_dir)
    if args.command == "enroll":
        enroll(args, gallery)
    elif args.command == "monitor":
        monitor(args, gallery, db, crops)
    else:
        list_events(db)


if __name__ == "__main__":
    main()
