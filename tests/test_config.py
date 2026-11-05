import pytest
from security_ai.config import RecognitionConfig, positive_int, finite_float


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": float("nan")},
        {"threshold": 2},
        {"cooldown": -1},
        {"evidence_ttl": 0},
        {"minimum": 6},
        {"ambiguity_margin": 3},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RecognitionConfig(**kwargs)


def test_numeric_arguments():
    assert positive_int("5") == 5
    with pytest.raises(ValueError):
        positive_int("0")
    with pytest.raises(ValueError):
        finite_float("inf")
