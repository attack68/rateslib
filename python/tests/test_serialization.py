from datetime import datetime as dt

import pytest
from rateslib.calendars import get_calendar
from rateslib.curves import Curve, LineCurve
from rateslib.curves.utils import (
    _CurveInterpolator,
    _CurveMeta,
    _CurveNodes,
    _CurveSpline,
    _CurveType,
)
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, Variable
from rateslib.serialization import from_json
from rateslib.serialization.utils import _enum_to_json


@pytest.mark.parametrize("calendar", [get_calendar("tgt"), get_calendar(NoInput(0))])
@pytest.mark.parametrize(
    "index_base",
    [
        100.0,
        Dual(100.0, ["v"], []),
        Dual2(100.0, ["v"], [], []),
        NoInput(0),
    ],
)
@pytest.mark.parametrize("collateral", [None, "usd"])
def test_curvemeta_json_round_trip(calendar, index_base, collateral):
    obj = _CurveMeta(
        calendar=calendar,
        convention="act365f",
        modifier="MF",
        index_base=index_base,
        index_lag=1,
        collateral=collateral,
    )
    json_text = obj.to_json()
    round_trip = from_json(json_text)
    assert round_trip == obj


@pytest.mark.parametrize(
    "obj",
    [
        _CurveSpline(t=[dt(2000, 1, 1), dt(2002, 1, 1)], endpoints=("natural", "natural")),
        _CurveNodes({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98}),
        # _CurveNodes({dt(2000,1,1): Dual(1.0, ["x"], []), dt(2001, 1, 1): Dual(0.98, ["s"], [])}),
    ],
)
def test_curvespline_json_round_trip(obj):
    json_text = obj.to_json()
    round_trip = from_json(json_text)
    assert round_trip == obj


@pytest.mark.parametrize("local", ["linear", "spline"])
@pytest.mark.parametrize("t", [NoInput(0), [dt(2000, 1, 1), dt(2002, 1, 1)]])
def test_curveinterpolator_json_round_trip(local, t):
    if not isinstance(t, NoInput) and local == "spline":
        with pytest.raises(ValueError, match="When defining 'spline' interpola"):
            _CurveInterpolator(local, t, None, None, None, None)
        return None

    obj = _CurveInterpolator(
        local=local,
        t=t,
        endpoints=("natural", "natural"),
        node_dates=[dt(2000, 1, 1), dt(2002, 1, 1)],
        convention="act365f",
        curve_type=_CurveType.dfs,
    )
    json_text = obj.to_json()
    round_trip = from_json(json_text)
    assert round_trip == obj


@pytest.mark.parametrize("value", [-1, 0, 1])
def test_no_input_round_trip(value):
    obj = NoInput(value)
    json = _enum_to_json(obj)
    result = from_json(json)
    assert result == obj


@pytest.fixture
def curve():
    return Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolation="linear",
        id="v",
        convention="Act360",
        ad=1,
    )


@pytest.fixture
def line_curve():
    return LineCurve(
        nodes={
            dt(2022, 3, 1): 2.00,
            dt(2022, 3, 31): 2.01,
        },
        interpolation="linear",
        id="v",
        ad=1,
    )


@pytest.fixture
def index_curve():
    return Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.999,
        },
        interpolation="linear_index",
        id="v",
        ad=1,
        index_base=110.0,
    )


class TestCurve:
    def test_serialization(self, curve) -> None:
        expected = (
            '{"PyNative": '
            '{"Curve": '
            '{"meta": "{\\"PyNative\\": '
            '{\\"_CurveMeta\\": '
            '{\\"calendar\\": '
            '\\"{\\\\\\"NamedCal\\\\\\":{\\\\\\"name\\\\\\":\\\\\\"all\\\\\\"}}\\", '
            '\\"convention\\": \\"act360\\", \\"modifier\\": \\"MF\\", '
            '\\"index_base\\": '
            '\\"{\\\\\\"PyNative\\\\\\":{\\\\\\"NoInput\\\\\\":0}}\\", '
            '\\"index_lag\\": 3, '
            '\\"collateral\\": null}}}", '
            '"interpolator": '
            '"{\\"PyNative\\": {\\"_CurveInterpolator\\": '
            '{\\"local\\": \\"linear\\", \\"spline\\": \\"null\\", '
            '\\"convention\\": \\"act360\\"}}}", '
            '"id": "v", "ad": 1, '
            '"nodes": "{\\"PyNative\\": {\\"_CurveNodes\\": '
            '{\\"_nodes\\": {\\"2022-03-01\\": 1.0, \\"2022-03-31\\": 0.99}}}}"}}}'
        )
        result = curve.to_json()
        assert result == expected

    @pytest.mark.parametrize("c", ["curve", "line_curve", "index_curve"])
    def test_serialization_round_trip(self, c, curve, line_curve, index_curve) -> None:
        if c == "curve":
            obj = curve
        elif c == "line_curve":
            obj = line_curve
        elif c == "index_curve":
            obj = index_curve
        serial = obj.to_json()
        constructed = from_json(serial)
        assert constructed == obj

    def test_serialization_round_trip_spline(self) -> None:
        curve = Curve(
            nodes={
                dt(2022, 3, 1): 1.00,
                dt(2022, 3, 31): 0.99,
                dt(2022, 5, 1): 0.98,
                dt(2022, 6, 4): 0.97,
                dt(2022, 7, 4): 0.96,
            },
            interpolation="linear",
            id="v",
            convention="Act360",
            ad=1,
            t=[
                dt(2022, 5, 1),
                dt(2022, 5, 1),
                dt(2022, 5, 1),
                dt(2022, 5, 1),
                dt(2022, 6, 4),
                dt(2022, 7, 4),
                dt(2022, 7, 4),
                dt(2022, 7, 4),
                dt(2022, 7, 4),
            ],
        )

        serial = curve.to_json()
        constructed = from_json(serial)
        assert constructed == curve

    def test_serialization_curve_str_calendar(self) -> None:
        curve = Curve(
            nodes={
                dt(2022, 3, 1): 1.00,
                dt(2022, 3, 31): 0.99,
            },
            interpolation="linear",
            id="v",
            convention="Act360",
            modifier="F",
            calendar="LDN",
            ad=1,
        )
        serial = curve.to_json()
        constructed = from_json(serial)
        assert constructed == curve

    def test_serialization_curve_custom_calendar(self) -> None:
        calendar = get_calendar("ldn")
        curve = Curve(
            nodes={
                dt(2022, 3, 1): 1.00,
                dt(2022, 3, 31): 0.99,
            },
            interpolation="linear",
            id="v",
            convention="Act360",
            modifier="F",
            calendar=calendar,
            ad=1,
        )
        serial = curve.to_json()
        constructed = from_json(serial)
        assert constructed == curve
