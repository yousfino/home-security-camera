"""Local CLI composition; capture, inference and identity policies are separate."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import logging
import json
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from pathlib import Path
import cv2
import numpy as np

from .capture import LatestFrameCapture
from .config import RecognitionConfig, positive_int, finite_float
from .core import normalized, GalleryIndex
from .quality import QualityConfig
from .detection import YoloPersonDetector
from .display import HeadlessDisplay, OpenCVDisplay
from .pipeline import RecognitionPipeline
from .episodes import EpisodeManager
from .storage import (
    data_paths,
    init_db,
    load_gallery,
    save_event,
    save_profile,
    maintain_events,
    safe_profile_name,
    save_episodes,
    file_lock,
    delete_profile,
    drop_crops,
)
from .tracking import PersonTracker
from .vision import create_analyzer, crop_face, quality_ok


def camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open webcam index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def enrollment_sample(
    samples: list[np.ndarray], embedding: np.ndarray, consistency: float, duplicate: float
) -> np.ndarray:
    vector = normalized(embedding)
    if samples:
        scores = np.asarray([np.dot(vector, sample) for sample in samples])
        if scores.min() < consistency:
            raise ValueError("Sample differs from earlier enrollment; verify the same person")
        if scores.max() >= duplicate:
            raise ValueError("Duplicate sample; vary pose or lighting")
    return vector


def enroll(args: argparse.Namespace, gallery_path: Path) -> None:
    name = safe_profile_name(args.name)
    model = create_analyzer(args.det_size, args.face_model_root)
    gallery = load_gallery(gallery_path, expected_model=model.fingerprint)
    if name in gallery and not args.replace:
        raise ValueError("Profile exists; use --replace to explicitly replace it")
    samples: list[np.ndarray] = []
    cap = LatestFrameCapture(
        lambda: camera(args.camera, args.width, args.height), max_attempts=args.reconnect_attempts
    ).start()
    print("Enrollment: one consenting person; SPACE captures a sample, Q cancels.")
    try:
        while len(samples) < args.samples:
            packet = cap.read()
            if packet is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            frame = packet.image
            if time.monotonic() - packet.acquired_at > args.max_frame_age:
                continue
            faces = model.detect(frame)
            ready = len(faces) == 1 and quality_ok(frame, faces[0], quality_config(args))
            preview = frame.copy()
            cv2.putText(
                preview,
                f"{name}: {len(samples)}/{args.samples} | SPACE capture | Q quit",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0) if ready else (0, 0, 255),
                2,
            )
            cv2.imshow("Enrollment", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32 and ready:
                try:
                    sample = enrollment_sample(
                        samples,
                        model.embed(frame, faces[0]),
                        args.enrollment_consistency,
                        args.duplicate_threshold,
                    )
                except ValueError as exc:
                    print(f"Sample rejected: {exc}")
                else:
                    samples.append(sample)
    finally:
        try:
            cap.close()
        finally:
            cv2.destroyAllWindows()
    if len(samples) != args.samples:
        print("Enrollment incomplete; nothing saved")
        return
    save_profile(gallery_path, name, samples, model.fingerprint, replace=args.replace)
    print(f"Saved {len(samples)} enrollment templates in protected local storage")


def monitor(args: argparse.Namespace, gallery_path: Path, db_path: Path, crops: Path) -> None:
    model = create_analyzer(args.det_size, args.face_model_root)
    gallery = load_gallery(gallery_path, expected_model=model.fingerprint)
    detector = YoloPersonDetector(
        args.person_model, args.person_confidence, sha256=args.person_sha256
    )
    pipeline = RecognitionPipeline(
        detector, PersonTracker(args.track_max_age), model, gallery, recognition_config(args)
    )
    pipeline.quality = lambda frame, face: quality_ok(frame, face, quality_config(args))
    init_db(db_path)
    maintain_events(db_path, crops, args.retention_days, args.max_events)
    episodes = EpisodeManager(
        getattr(args, "episode_gap", 2.0), getattr(args, "reacquire_iou", 0.3)
    )
    session_id = uuid4().hex
    wall_start, monotonic_start = datetime.now(timezone.utc), time.monotonic()
    cap = LatestFrameCapture(
        lambda: camera(args.camera, args.width, args.height), max_attempts=args.reconnect_attempts
    ).start()
    display = HeadlessDisplay() if args.headless else OpenCVDisplay("Home Security Camera")
    gallery_stamp = None
    frames = 0
    generation = None
    runtime_start = time.monotonic()
    try:
        while args.max_frames is None or frames < args.max_frames:
            max_runtime = getattr(args, "max_runtime", None)
            if max_runtime is not None and time.monotonic() - runtime_start >= max_runtime:
                break
            packet = cap.read()
            if packet is None:
                if not args.headless and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            frames += 1
            if packet.generation != generation:
                pipeline.reset()
                generation = packet.generation
                episodes.close()
            frame, timestamp = packet.image, packet.acquired_at
            if time.monotonic() - timestamp > args.max_frame_age:
                pipeline.reset()
                continue
            with file_lock(gallery_path):
                stamp = (
                    (gallery_path.stat().st_mtime_ns, gallery_path.stat().st_size)
                    if gallery_path.exists()
                    else None
                )
                if stamp != gallery_stamp:
                    pipeline.gallery = GalleryIndex(
                        load_gallery(gallery_path, model.fingerprint),
                        recognition_config(args).aggregation,
                    )
                    pipeline.reset()
                    episodes.close()
                    gallery_stamp = stamp
                decisions = pipeline.process(frame, timestamp)
                if time.monotonic() - timestamp > args.max_frame_age:
                    # Slow inference must not emit an event for stale evidence.
                    pipeline.reset()
                    continue
                decisions = episodes.update(decisions, timestamp)
                try:
                    save_episodes(
                        db_path, episodes.pending(), session_id, wall_start, monotonic_start
                    )
                except (OSError, sqlite3.Error):
                    logging.error("Episode checkpoint failed")
                else:
                    episodes.checkpointed()
                for decision in decisions:
                    if decision.emit:
                        crop = (
                            crop_face(frame, decision.face)
                            if decision.status == "unfamiliar"
                            and getattr(args, "save_crops", False)
                            else None
                        )
                        try:
                            save_event(
                                db_path,
                                crops,
                                decision.status,
                                decision.name,
                                decision.score,
                                crop,
                                track_id=decision.track_id,
                                session_id=session_id,
                                model_id=model.fingerprint,
                                episode_id=decision.episode_id,
                                event_key=decision.event_key,
                                max_storage_bytes=getattr(args, "storage_budget_mb", 512)
                                * 1024
                                * 1024,
                                captured_at=wall_start
                                + timedelta(seconds=timestamp - monotonic_start),
                            )
                        except (OSError, sqlite3.Error, RuntimeError, cv2.error):
                            logging.exception("Event persistence failed; cooldown was not advanced")
                        else:
                            pipeline.acknowledge(decision, timestamp)
                            episodes.acknowledge(decision)
            preview = frame.copy()
            for decision in decisions:
                x1, y1, x2, y2 = map(int, decision.box)
                cv2.rectangle(
                    preview,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0) if decision.status == "familiar" else (0, 165, 255),
                    2,
                )
                cv2.putText(
                    preview,
                    f"#{decision.track_id} {decision.status} {decision.name or ''}",
                    (x1, max(20, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            if frames % 300 == 0:
                try:
                    maintain_events(db_path, crops, args.retention_days, args.max_events)
                except (OSError, sqlite3.Error):
                    logging.exception("Event maintenance failed")
            if args.metrics and frames % 30 == 0:
                print(
                    {
                        **pipeline.timings,
                        "frame_age_ms": (time.monotonic() - timestamp) * 1000,
                        "dropped_frames": cap.dropped,
                        "reconnects": cap.reconnects,
                    }
                )
            if not display.show(preview):
                break
    finally:
        try:
            cap.close()
        finally:
            display.close()
            episodes.close()
            try:
                save_episodes(db_path, episodes.pending(), session_id, wall_start, monotonic_start)
            except (OSError, sqlite3.Error):
                logging.error("Final episode checkpoint failed")


def list_events(db_path: Path, show_identities: bool = False) -> None:
    if not db_path.exists():
        print("No events yet")
        return
    with sqlite3.connect(db_path) as db:
        for row in db.execute(
            "SELECT id,captured_at,status,profile_name,image_path "
            "FROM events ORDER BY id DESC LIMIT 30"
        ):
            print(
                *row if show_identities else (row[0], row[1], row[2], "[identity/path hidden]"),
                sep=" | ",
            )


def recognition_config(args: argparse.Namespace) -> RecognitionConfig:
    return RecognitionConfig(
        threshold=args.threshold,
        aggregation=getattr(args, "gallery_aggregation", "max"),
        recognition_interval=args.recognition_interval,
        evidence_ttl=args.evidence_ttl,
        ambiguity_margin=args.ambiguity_margin,
        cooldown=args.cooldown,
        familiar_log_interval=args.familiar_log_interval,
    )


def quality_config(args: argparse.Namespace) -> QualityConfig:
    return QualityConfig(
        **{
            name: getattr(args, name, field.default)
            for name, field in QualityConfig.__dataclass_fields__.items()
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-only security camera prototype")
    parser.add_argument("--data-dir", type=Path, default=Path(".local"))
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("enroll", "monitor"):
        p = sub.add_parser(command)
        p.add_argument("--camera", type=int, default=0)
        p.add_argument("--min-face-px", type=positive_int, default=80)
        p.add_argument("--min-blur-variance", type=finite_float, default=55.0)
        p.add_argument("--min-brightness", type=finite_float, default=35.0)
        p.add_argument("--max-brightness", type=finite_float, default=220.0)
        p.add_argument("--max-roll", type=finite_float, default=25.0)
        p.add_argument("--max-yaw-asymmetry", type=finite_float, default=0.6)
        p.add_argument("--reconnect-attempts", type=positive_int, default=8)
        p.add_argument("--max-frame-age", type=finite_float, default=0.5)
        p.add_argument("--width", type=positive_int, default=640)
        p.add_argument("--height", type=positive_int, default=480)
        p.add_argument("--det-size", type=positive_int, default=640)
        p.add_argument("--face-model-root", type=Path, default=Path("~/.insightface"))
        if command == "enroll":
            p.add_argument("--name", required=True)
            p.add_argument("--samples", type=positive_int, default=5)
            p.add_argument("--replace", action="store_true")
            p.add_argument("--enrollment-consistency", type=finite_float, default=0.6)
            p.add_argument("--duplicate-threshold", type=finite_float, default=0.995)
        else:
            p.add_argument(
                "--person-model",
                type=Path,
                required=True,
                help="Local raw YOLOv8/11 COCO ONNX export",
            )
            p.add_argument("--person-sha256")
            p.add_argument(
                "--person-confidence",
                type=finite_float,
                default=0.1,
                help="Keep low-score detections for ByteTrack second-stage association",
            )
            p.add_argument("--track-max-age", type=finite_float, default=2.0)
            p.add_argument("--episode-gap", type=finite_float, default=2.0)
            p.add_argument("--reacquire-iou", type=finite_float, default=0.3)
            p.add_argument(
                "--threshold",
                type=finite_float,
                default=0.45,
                help="Unvalidated cosine threshold; calibrate locally",
            )
            p.add_argument("--gallery-aggregation", choices=("max", "mean"), default="max")
            p.add_argument("--ambiguity-margin", type=finite_float, default=0.05)
            p.add_argument("--evidence-ttl", type=finite_float, default=2.0)
            p.add_argument("--cooldown", type=finite_float, default=30.0)
            p.add_argument("--familiar-log-interval", type=finite_float, default=60.0)
            p.add_argument("--headless", action="store_true")
            p.add_argument("--metrics", action="store_true")
            p.add_argument(
                "--save-crops",
                action="store_true",
                help="Opt in to retaining unfamiliar face images",
            )
            p.add_argument("--storage-budget-mb", type=positive_int, default=512)
            p.add_argument("--recognition-interval", type=finite_float, default=0.2)
            p.add_argument(
                "--max-frames",
                type=positive_int,
                help="Maximum acquired frames, including stale frames",
            )
            p.add_argument(
                "--max-runtime",
                type=finite_float,
                help="Maximum elapsed seconds, including outages",
            )
            p.add_argument("--retention-days", type=finite_float, default=7.0)
            p.add_argument("--max-events", type=positive_int, default=10000)
    p = sub.add_parser("events")
    p.add_argument("--show-identities", action="store_true")
    p = sub.add_parser(
        "profile-delete", help="Remove profile and linked events; stop enrollment before deletion"
    )
    p.add_argument("--name", required=True)
    p.add_argument("--keep-events", action="store_true")
    p.add_argument("--yes", action="store_true", required=True)
    p = sub.add_parser("maintain", help="Run retention/reconciliation on local data")
    p.add_argument("--retention-days", type=finite_float, default=7.0)
    p.add_argument("--max-events", type=positive_int, default=10000)
    p.add_argument("--drop-crops", action="store_true")
    p.add_argument("--yes", action="store_true", required=True)
    p = sub.add_parser("doctor", help="Offline runtime diagnostics; no camera required")
    p.add_argument("--check-models", action="store_true")
    p.add_argument("--person-model", type=Path)
    p.add_argument("--person-sha256")
    p.add_argument("--face-model-root", type=Path)
    p = sub.add_parser("calibrate", help="Evaluate local enrollment-independent pair scores")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--thresholds", type=finite_float, nargs="+", required=True)
    p.add_argument("--max-fmr", type=finite_float, default=0.01)
    p.add_argument(
        "--enrollment-independent",
        action="store_true",
        required=True,
        help="Attest evaluation samples were not used for enrollment",
    )
    args = parser.parse_args()
    try:
        if args.command == "calibrate":
            from .calibration import evaluate, load_scores

            print(json.dumps(evaluate(load_scores(args.scores), args.thresholds, args.max_fmr)))
            return
        if args.command == "doctor":
            from .diagnostics import diagnose

            report = diagnose(
                args.person_model, args.face_model_root, args.person_sha256, args.check_models
            )
            print(json.dumps(report, sort_keys=True))
            if not report["ready"]:
                parser.exit(1)
            return
        if args.command in ("monitor", "enroll"):
            quality_config(args)
        if args.command in ("monitor", "enroll") and args.max_frame_age <= 0:
            raise ValueError("Maximum frame age must be positive")
        if args.command == "monitor":
            recognition_config(args)
            EpisodeManager(args.episode_gap, args.reacquire_iou)
            if args.max_runtime is not None and args.max_runtime <= 0:
                raise ValueError("Maximum runtime must be positive")
            if args.retention_days <= 0:
                raise ValueError("Retention days must be positive")
            if args.track_max_age <= 0 or not 0 < args.person_confidence <= 1:
                raise ValueError("Require positive track age and person confidence in (0,1]")
        elif args.command == "enroll":
            if not -1 <= args.enrollment_consistency < args.duplicate_threshold <= 1:
                raise ValueError("Require -1 <= consistency < duplicate threshold <= 1")
        gallery, db, crops = data_paths(args.data_dir)
        if args.command == "profile-delete":
            if db.exists():
                init_db(db)
            print(json.dumps(delete_profile(gallery, db, crops, args.name, not args.keep_events)))
        elif args.command == "maintain":
            init_db(db)
            maintain_events(db, crops, args.retention_days, args.max_events)
            if args.drop_crops:
                drop_crops(db, crops)
            print("Local data maintenance completed")
        elif args.command == "enroll":
            enroll(args, gallery)
        elif args.command == "monitor":
            monitor(args, gallery, db, crops)
        else:
            list_events(db, args.show_identities)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main()
