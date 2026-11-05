from types import SimpleNamespace
import numpy as np
import pytest
from security_ai.detection import PersonDetection
from security_ai.tracking import PersonTracker, PersonTrack, associate_faces


def detection(x=0, confidence=0.9):
    return PersonDetection((x, 0, x + 100, 200), confidence)


def test_bytetrack_continuity_occlusion_and_expiry():
    tracker = PersonTracker(max_age=1)
    first = tracker.update([detection()], 0)[0].track_id
    assert tracker.update([detection(2)], 0.1)[0].track_id == first
    assert tracker.update([], 0.2) == []
    assert tracker.update([detection(4)], 0.3)[0].track_id == first
    new = tracker.update([detection(4)], 2)[0].track_id
    assert new != first
    tracker.reset()
    assert tracker.update([detection()], 3)[0].track_id != new
    with pytest.raises(ValueError):
        tracker.update([], 3)


def test_low_confidence_second_stage_keeps_track():
    tracker = PersonTracker()
    first = tracker.update([detection()], 0)[0].track_id
    assert tracker.update([detection(1, 0.2)], 0.1)[0].track_id == first


def test_two_people_and_ambiguous_faces():
    tracker = PersonTracker()
    tracks = tracker.update([detection(), detection(200)], 0)
    assert len({t.track_id for t in tracks}) == 2

    def face(x):
        return SimpleNamespace(bbox=np.array([x, 10, x + 30, 40]))

    assert len(associate_faces(tracks, [face(10), face(210)])) == 2
    assert associate_faces(tracks, [face(10), face(20)]) == {}
    overlap = [PersonTrack(1, (0, 0, 100, 200), 0.9), PersonTrack(2, (0, 0, 100, 200), 0.9)]
    assert associate_faces(overlap, [face(10)]) == {}


def test_ambiguous_face_blocks_other_face_in_same_person():
    tracks = [PersonTrack(1, (0, 0, 200, 200), 0.9), PersonTrack(2, (100, 0, 300, 200), 0.9)]
    faces = [
        SimpleNamespace(bbox=np.array([120, 10, 150, 40])),
        SimpleNamespace(bbox=np.array([10, 10, 40, 40])),
    ]
    assert associate_faces(tracks, faces) == {}
