import pytest
from datetime import datetime as dt

import context
from rateslib import default_context
from rateslib.fx import FXRates, FXForwards
from rateslib.default import Defaults, NoInput
from rateslib.curves import Curve, IndexCurve
from rateslib.json import Serialise


class TestFXRates:

    def test_to_json(self):
        fxr = FXRates({"usdnok": 8.0, "eurusd": 1.05})
        result = Serialise(fxr).to_json()
        expected = '{"fx_rates": {"usdnok": 8.0, "eurusd": 1.05}, "settlement": null, "base": "usd"}'
        assert result == expected

        fxr = FXRates({"usdnok": 8.0, "eurusd": 1.05}, dt(2022, 1, 3))
        result = Serialise(fxr).to_json()
        expected = (
            '{"fx_rates": {"usdnok": 8.0, "eurusd": 1.05}, "settlement": "2022-01-03", "base": "usd"}'
        )
        assert result == expected

    def test_from_json_and_equality(self):
        fxr1 = FXRates({"usdnok": 8.0, "eurusd": 1.05})
        fxr2 = FXRates({"usdnok": 12.0, "eurusd": 1.10})
        assert fxr1 != fxr2

        fxr2 = Serialise.from_json(FXRates,
            '{"fx_rates": {"usdnok": 8.0, "eurusd": 1.05}, "settlement": null, "base": "usd"}'
        )
        assert fxr2 == fxr1

        fxr3 = FXRates({"usdnok": 8.0, "eurusd": 1.05}, base="NOK")
        assert fxr1 != fxr3  # base is different


class TestCurve:

    def test_to_json(self):
        curve = Curve(
            nodes={
                dt(2022, 3, 1): 1.00,
                dt(2022, 3, 31): 0.99,
            },
            interpolation="linear",
            id="v",
            convention="Act360",
            ad=1,
        )
        expected = (
            '{"nodes": {"2022-03-01": 1.0, "2022-03-31": 0.99}, '
            '"interpolation": "linear", "t": null, "c": null, "id": "v", '
            '"convention": "Act360", "endpoints": ["natural", "natural"], "modifier": "MF", '
            '"calendar_type": "null", "ad": 1, "calendar": null}'
        )
        result = Serialise(curve).to_json()
        assert result == expected

    def test_round_trip(self):
        curve = Curve(
            nodes={
                dt(2022, 3, 1): 1.00,
                dt(2022, 3, 31): 0.99,
            },
            interpolation="linear",
            id="v",
            convention="Act360",
            ad=1,
        )
        data = Serialise(curve).to_json()
        constructed = Serialise.from_json(Curve, data)
        assert constructed == curve


# def test_round_trip(curve, line_curve, index_curve):
#     curve = Curve(
#         nodes={
#             dt(2022, 3, 1): 1.00,
#             dt(2022, 3, 31): 0.99,
#         },
#         interpolation="linear",
#         id="v",
#         convention="Act360",
#         ad=1,
#     )
#     serial = curve.to_json()
#     constructed = Curve.from_json(serial)
#     assert constructed == curve
#
#     serial = line_curve.to_json()
#     constructed = LineCurve.from_json(serial)
#     assert constructed == line_curve
#
#     serial = index_curve.to_json()
#     constructed = IndexCurve.from_json(serial)
#     assert constructed == index_curve
#
# def test_serialization_round_trip_spline():
#     curve = Curve(
#         nodes={
#             dt(2022, 3, 1): 1.00,
#             dt(2022, 3, 31): 0.99,
#             dt(2022, 5, 1): 0.98,
#             dt(2022, 6, 4): 0.97,
#             dt(2022, 7, 4): 0.96,
#         },
#         interpolation="linear",
#         id="v",
#         convention="Act360",
#         ad=1,
#         t=[
#             dt(2022, 5, 1),
#             dt(2022, 5, 1),
#             dt(2022, 5, 1),
#             dt(2022, 5, 1),
#             dt(2022, 6, 4),
#             dt(2022, 7, 4),
#             dt(2022, 7, 4),
#             dt(2022, 7, 4),
#             dt(2022, 7, 4),
#         ],
#     )
#
#     serial = curve.to_json()
#     constructed = Curve.from_json(serial)
#     assert constructed == curve
#
#
# def test_serialization_curve_str_calendar():
#     curve = Curve(
#         nodes={
#             dt(2022, 3, 1): 1.00,
#             dt(2022, 3, 31): 0.99,
#         },
#         interpolation="linear",
#         id="v",
#         convention="Act360",
#         modifier="F",
#         calendar="LDN",
#         ad=1,
#     )
#     serial = curve.to_json()
#     constructed = Curve.from_json(serial)
#     assert constructed == curve
#
#
# def test_serialization_curve_custom_calendar():
#     calendar = get_calendar("ldn")
#     curve = Curve(
#         nodes={
#             dt(2022, 3, 1): 1.00,
#             dt(2022, 3, 31): 0.99,
#         },
#         interpolation="linear",
#         id="v",
#         convention="Act360",
#         modifier="F",
#         calendar=calendar,
#         ad=1,
#     )
#     serial = curve.to_json()
#     constructed = Curve.from_json(serial)
#     assert constructed == curve