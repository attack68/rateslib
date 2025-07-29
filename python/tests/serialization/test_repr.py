import pytest
from rateslib.scheduling import Imm, StubInference


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (Imm.Wed1_Post9_HMUZ, "Imm.Wed1_Post9_HMUZ"),
        (StubInference.ShortFront, "StubInference.ShortFront"),
    ],
)
def test_json_round_trip(obj, expected) -> None:
    repr_ = obj.__repr__()
    assert f"<rl: {expected}" in repr_


def test_unique_repr_simple_enum():
    a = Imm.Wed1_Post9_HMUZ
    b = Imm.Wed1_Post9_HMUZ
    assert a.__repr__() == b.__repr__()
    assert hex(id(a)) == hex(id(b))
