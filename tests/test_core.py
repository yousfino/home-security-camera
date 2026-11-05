import numpy as np
import pytest
from security_ai.core import normalized, best_match, classify, Observation, DecisionWindow


def test_normalization():
    assert np.allclose(normalized(np.array([3,4])), [0.6,0.8])
    with pytest.raises(ValueError):
        normalized(np.zeros(2))


def test_matching():
    gallery = {"alice": [np.array([1.,0.])], "bob": [np.array([0.,1.])]}
    name, score = best_match(np.array([0.99,0.01]), gallery)
    assert name == "alice" and score > 0.99


def test_unknown_and_quality():
    assert classify("alice", 0.2, 0.45, True) == "unfamiliar"
    assert classify("alice", 0.99, 0.45, False) == "unresolved"


def test_stability():
    window = DecisionWindow(minimum=3)
    for _ in range(2):
        assert window.add(Observation("unfamiliar", None, 0.1))[0] == "unresolved"
    assert window.add(Observation("unfamiliar", None, 0.1))[0] == "unfamiliar"
    window.clear()
    assert window.add(Observation("familiar", "alice", 0.9))[0] == "unresolved"
