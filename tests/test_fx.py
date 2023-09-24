import pytest
from datetime import datetime as dt
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np

import context
from rateslib.fx import (
    FXForwards,
    FXRates,
    forward_fx,
)
from rateslib.dual import Dual, Dual2
from rateslib.curves import Curve, LineCurve, CompositeCurve
from rateslib.default import NoInput


@pytest.mark.parametrize(
    "fx_rates",
    [
        {"eurusd": 1.0, "seknok": 1.0},
        {"eurusd": 1.0, "usdeur": 1.0, "usdgbp": 1.0},
        {"eurusd": 1.0, "usdeur": 1.0, "seknok": 1.0},
    ],
)
def test_ill_constrained(fx_rates):
    with pytest.raises(ValueError):
        FXRates(fx_rates)


def test_rates():
    fxr = FXRates({"usdeur": 2.0, "usdgbp": 2.5})
    assert fxr.currencies == {"usd": 0, "eur": 1, "gbp": 2}
    assert fxr.currencies_list == ["usd", "eur", "gbp"]
    assert fxr.pairs == ["usdeur", "usdgbp"]
    assert fxr.q == 3
    assert fxr.fx_array[1, 2].real == 1.25
    assert fxr.fx_array[1, 2] == Dual(1.25, ["fx_usdeur", "fx_usdgbp"], [-0.625, 0.50])
    assert fxr.rate("eurgbp") == Dual(1.25, ["fx_usdeur", "fx_usdgbp"], [-0.625, 0.50])


def test_fx_update_blank():
    fxr = FXRates({"usdeur": 2.0, "usdgbp": 2.5})
    result = fxr.update()
    assert result is None


def test_convert_and_base():
    fxr = FXRates({"usdnok": 8.0})
    expected = Dual(125000, "fx_usdnok", [-15625])
    result = fxr.convert(1e6, "nok", "usd")
    result2 = fxr.convert_positions([0, 1e6], "usd")
    assert result == expected
    assert result2 == expected
    result3 = fxr.positions(expected, "usd")
    assert np.all(result3 == np.array([0, 1e6]))


def test_convert_none():
    fxr = FXRates({"usdnok": 8.0})
    assert fxr.convert(1, "usd", "gbp") is None


def test_convert_warn():
    fxr = FXRates({"usdnok": 8.0})
    with pytest.warns(UserWarning):
        fxr.convert(1, "usd", "gbp", on_error="warn")


def test_convert_error():
    fxr = FXRates({"usdnok": 8.0})
    with pytest.raises(ValueError):
        fxr.convert(1, "usd", "gbp", on_error="raise")


def test_positions_value():
    fxr = FXRates({"usdnok": 8.0})
    result = fxr.positions(80, "nok")
    assert all(result == np.array([0, 80.0]))


def test_fxrates_set_order():
    fxr = FXRates({"usdnok": 8.0})
    fxr._set_ad_order(order=2)
    expected = np.array([Dual2(1.0, "fx_usdnok", [0.0]), Dual2(8.0, "fx_usdnok", [1.0])])
    assert all(fxr.fx_vector == expected)


def test_update_raises():
    fxr = FXRates({"usdnok": 8.0})
    with pytest.raises(ValueError, match="`fx_rates` must contain"):
        fxr.update({"usdnok": 9.0, "gbpnok": 10.0})


def test_restate():
    fxr = FXRates({"usdnok": 8.0, "gbpnok": 10})
    fxr2 = fxr.restate(["gbpusd", "usdnok"])
    assert fxr2.pairs == ["gbpusd", "usdnok"]
    assert fxr2.fx_rates == {
        "gbpusd": Dual(1.25, "fx_gbpusd", [1.0]),
        "usdnok": Dual(8.0, "fx_usdnok", [1.0]),
    }


def test_restate_return_self():
    # test a new object is always returned even if nothing is restated
    fxr = FXRates({"usdnok": 8.0, "gbpnok": 10})
    assert id(fxr) != id(fxr.restate(["gbpnok", "usdnok"], True))


def test_rates_table():
    fxr = FXRates({"EURNOK": 10.0})
    result = fxr.rates_table()
    expected = DataFrame([[1.0, 10.0], [0.1, 1.0]], index=["eur", "nok"], columns=["eur", "nok"])
    assert_frame_equal(result, expected)


def test_fxrates_to_json():
    fxr = FXRates({"usdnok": 8.0, "eurusd": 1.05})
    result = fxr.to_json()
    expected = '{"fx_rates": {"usdnok": 8.0, "eurusd": 1.05}, "settlement": null, "base": "usd"}'
    assert result == expected

    fxr = FXRates({"usdnok": 8.0, "eurusd": 1.05}, dt(2022, 1, 3))
    result = fxr.to_json()
    expected = (
        '{"fx_rates": {"usdnok": 8.0, "eurusd": 1.05}, "settlement": "2022-01-03", "base": "usd"}'
    )
    assert result == expected


def test_from_json_and_equality():
    fxr1 = FXRates({"usdnok": 8.0, "eurusd": 1.05})
    fxr2 = FXRates({"usdnok": 12.0, "eurusd": 1.10})
    assert fxr1 != fxr2

    fxr2 = FXRates.from_json(
        '{"fx_rates": {"usdnok": 8.0, "eurusd": 1.05}, "settlement": null, "base": "usd"}'
    )
    assert fxr2 == fxr1

    fxr3 = FXRates({"usdnok": 8.0, "eurusd": 1.05}, base="NOK")
    assert fxr1 != fxr3  # base is different


def test_copy():
    fxr1 = FXRates({"usdnok": 8.0, "eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxr2 = fxr1.copy()
    assert fxr1 == fxr2
    assert id(fxr1) != id(fxr2)


def test_set_ad_order():
    fxr = FXRates({"usdnok": 10.0})
    fxr._set_ad_order(1)

    fxr._set_ad_order(2)
    assert fxr._ad == 2
    assert type(fxr.fx_vector[0]) is Dual2
    assert type(fxr.fx_vector[1]) is Dual2

    fxr._set_ad_order(0)
    assert fxr._ad == 0
    assert fxr.fx_vector[0] == 1.0
    assert fxr.fx_vector[1] == 10.0

    with pytest.raises(ValueError, match="`order` can only be in"):
        fxr._set_ad_order("bad arg")


@pytest.fixture()
def usdusd():
    nodes = {dt(2022, 1, 1): 1.00, dt(2022, 4, 1): 0.99}
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture()
def eureur():
    nodes = {dt(2022, 1, 1): 1.00, dt(2022, 4, 1): 0.997}
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture()
def usdeur():
    nodes = {dt(2022, 1, 1): 1.00, dt(2022, 4, 1): 0.996}
    return Curve(nodes=nodes, interpolation="log_linear")


def test_fxforwards_rates_unequal(usdusd, eureur, usdeur):
    fxf = FXForwards(
        FXRates({"usdeur": 2.0}, settlement=dt(2022, 1, 3)),
        {"usdusd": usdusd, "eureur": eureur, "usdeur": usdeur},
    )
    fxr = FXRates({"usdeur": 2.0}, settlement=dt(2022, 1, 3))
    assert fxf != fxr
    assert fxr != fxf

    fxf_other = FXForwards(
        FXRates({"usdeur": 3.0}, settlement=dt(2022, 1, 3)),
        {"usdusd": usdusd, "eureur": eureur, "usdeur": usdeur},
    )
    assert fxf != fxf_other

    fxf2 = fxf.copy()
    assert fxf2 == fxf
    fxf2.base = "eur"
    assert fxf2 != fxf


def test_fxforwards_without_settlement_raise():
    fxr = FXRates({"usdeur": 1.0})
    crv = Curve({dt(2022, 1, 1): 1.0})
    with pytest.raises(ValueError, match="`fx_rates` as FXRates supplied to FXForwards must cont"):
        FXForwards(
            fx_rates=fxr,
            fx_curves={
                "usdusd": crv,
                "usdeur": crv,
                "eureur": crv
            }
        )


def test_fxforwards_set_order(usdusd, eureur, usdeur):
    fxf = FXForwards(
        FXRates({"usdeur": 2.0}, settlement=dt(2022, 1, 3)),
        {"usdusd": usdusd, "eureur": eureur, "usdeur": usdeur},
    )
    fxf._set_ad_order(order=2)
    expected = np.array([Dual2(1.0, "fx_usdeur", [0.0]), Dual2(2.0, "fx_usdeur", [1.0])])
    assert all(fxf.fx_rates.fx_vector == expected)
    assert usdusd.ad == 2
    assert eureur.ad == 2
    assert usdeur.ad == 2


def test_fxforwards_set_order_list(usdusd, eureur, usdeur):
    fxf = FXForwards(
        [
            FXRates({"usdeur": 2.0}, settlement=dt(2022, 1, 3)),
            FXRates({"usdgbp": 3.0}, settlement=dt(2022, 1, 4)),
        ],
        {
            "usdusd": usdusd,
            "eureur": eureur,
            "usdeur": usdeur,
            "usdgbp": usdeur.copy(),
            "gbpgbp": eureur.copy(),
        },
    )
    fxf._set_ad_order(order=2)
    # expected = np.array(
    #     [
    #         Dual2(1.0, "fx_usdeur", [0.0]),
    #         Dual2(2.0, "fx_usdeur", [1.0]),
    #     ]
    # )
    assert type(fxf.fx_rates_immediate.fx_vector[0]) is Dual2
    assert usdusd.ad == 2
    assert eureur.ad == 2
    assert usdeur.ad == 2
    assert fxf.curve("usd", "gbp").ad == 2


def test_fxforwards_and_swap(usdusd, eureur, usdeur):
    fxf = FXForwards(
        FXRates({"usdeur": 0.9}, settlement=dt(2022, 1, 3)),
        {"usdusd": usdusd, "eureur": eureur, "usdeur": usdeur},
    )
    result = fxf.rate("usdeur", dt(2022, 3, 25))
    expected = Dual(0.8991875219289739, "fx_usdeur", [0.99909725])
    assert result == expected

    # test fx_swap price
    result = fxf.swap("usdeur", [dt(2022, 1, 3), dt(2022, 3, 25)])
    expected = (expected - fxf.rate("usdeur", dt(2022, 1, 3))) * 10000
    assert result == expected

    result = fxf.rate("eurusd", dt(2022, 3, 25))
    expected = Dual(1.1121150767915007, "fx_usdeur", [-1.23568342])
    assert result == expected


def test_fxforwards2():
    fx_rates = FXRates({"usdeur": 0.9, "eurnok": 8.888889}, dt(2022, 1, 3))
    fx_curves = {
        "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.96}),
        "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}),
        "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.991}),
        "noknok": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}),
        "nokeur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.978}),
    }
    fxf = FXForwards(fx_rates, fx_curves)
    result = fxf.rate("usdnok", dt(2022, 8, 16))
    expected = Dual(7.9039924628096845, ["fx_eurnok", "fx_usdeur"], [0.88919914, 8.78221385])
    assert result == expected


def test_fxforwards_immediate():
    fx_rates = FXRates({"usdeur": 0.95}, dt(2022, 1, 3))
    fx_curves = {
        "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 3): 0.95}),
        "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 3): 1.0}),
        "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 3): 1.0}),
    }
    fxf = FXForwards(fx_rates, fx_curves)
    F0_usdeur = 0.95 * 1.0 / 0.95  # f_usdeur * w_eurusd / v_usdusd
    assert fxf.fx_rates_immediate.fx_array[0, 1].real == F0_usdeur
    assert fxf.rate("usdeur").real == F0_usdeur

    result = fxf.rate("usdeur", dt(2022, 1, 1))
    assert result == Dual(1, "fx_usdeur", [1 / 0.95])

    result = fxf.rate("usdeur", dt(2022, 1, 3))
    assert result == Dual(0.95, "fx_usdeur", [1.0])


def test_fxforwards_immediate2():
    fx_rates = FXRates({"usdeur": 0.9, "eurnok": 8.888889}, dt(2022, 1, 3))
    fx_curves = {
        "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 3): 0.999}),
        "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 3): 0.998}),
        "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 3): 0.997}),
        "noknok": Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 3): 0.996}),
        "nokeur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 3): 0.995}),
    }
    fxf = FXForwards(fx_rates, fx_curves)
    F0_usdeur = 0.9 * 0.997 / 0.999  # f_usdeur * v_eurusd / w_usdusd
    F0_eurnok = 8.888889 * 0.995 / 0.998  # f_eurnok * w_nokeur / v_eureur
    assert abs(fxf.fx_rates_immediate.fx_array[0, 1].real - F0_usdeur) < 1e-15
    assert abs(fxf.fx_rates_immediate.fx_array[1, 2].real - F0_eurnok) < 1e-15


def test_fxforwards_bad_curves_raises(usdusd, eureur, usdeur):
    bad_curve = Curve({dt(2000, 1, 1): 1.00, dt(2023, 1, 1): 0.99})
    with pytest.raises(ValueError, match="`fx_curves` do not have the same initial"):
        FXForwards(
            FXRates({"usdeur": 0.9}, settlement=dt(2022, 1, 3)),
            {"usdusd": usdusd, "eureur": eureur, "usdeur": bad_curve},
        )

    bad_curve = LineCurve({dt(2022, 1, 1): 1.00, dt(2023, 1, 1): 0.99})
    with pytest.raises(TypeError, match="`fx_curves` must be DF based, not type Line"):
        FXForwards(
            FXRates({"usdeur": 0.9}, settlement=dt(2022, 1, 3)),
            {"usdusd": usdusd, "eureur": eureur, "usdeur": bad_curve},
        )

    # SHOULD NOT NECESSARILY FAIL
    # with pytest.raises(ValueError):
    #     FXForwards(
    #         FXRates({"usdeur": 0.9, "eurgbp": 0.9}, fx_settlement=dt(2022, 1, 3)),
    #         {"usdusd": usdusd,
    #          "eureur": eureur,
    #          "usdeur": usdeur,
    #          "usdgbp": usdeur,
    #          "gbpgbp": eureur
    #          }
    #     )


def test_fxforwards_convert(usdusd, eureur, usdeur):
    fxf = FXForwards(
        FXRates({"usdeur": 0.9}, settlement=dt(2022, 1, 3)),
        {"usdusd": usdusd, "eureur": eureur, "usdeur": usdeur},
    )
    result = fxf.convert(
        100,
        domestic="usd",
        foreign="eur",
        settlement=dt(2022, 1, 15),
        value_date=dt(2022, 1, 30)
    )
    expected = Dual(90.12374519723947, "fx_usdeur", [100.13749466359941])
    assert result == expected

    result = fxf.convert(
        100,
        domestic="usd",
        foreign="eur",
        settlement=NoInput(0),  # should imply immediate settlement
        value_date=NoInput(0),  # should imply same as settlement
    )
    expected = Dual(90.00200704713323, "fx_usdeur", [100.00223005237025])
    assert result == expected


def test_fxforwards_convert_not_in_ccys(usdusd, eureur, usdeur):
    fxf = FXForwards(
        FXRates({"usdeur": 0.9}, settlement=dt(2022, 1, 3)),
        {"usdusd": usdusd, "eureur": eureur, "usdeur": usdeur},
    )
    ccy = "gbp"
    with pytest.raises(ValueError, match=f"'{ccy}' not in FXForwards.currencies"):
        fxf.convert(
            100,
            domestic=ccy,
            foreign="eur",
            settlement=dt(2022, 1, 15),
            value_date=dt(2022, 1, 30),
            on_error="raise",
        )

    result = fxf.convert(
            100,
            domestic=ccy,
            foreign="eur",
            settlement=dt(2022, 1, 15),
            value_date=dt(2022, 1, 30),
            on_error="ignore",
        )
    assert result is None

    with pytest.warns(UserWarning):
        result = fxf.convert(
            100,
            domestic=ccy,
            foreign="eur",
            settlement=dt(2022, 1, 15),
            value_date=dt(2022, 1, 30),
            on_error="warn",
        )
        assert result is None


def test_fxforwards_position_not_dual(usdusd, eureur, usdeur):
    fxf = FXForwards(
        FXRates({"usdeur": 0.9}, settlement=dt(2022, 1, 3)),
        {"usdusd": usdusd, "eureur": eureur, "usdeur": usdeur},
    )
    result = fxf.positions(100)
    expected = DataFrame(
        {dt(2022, 1, 1): [100.0, 0.], dt(2022, 1, 3): [0.0, 0.0]},
        index=["usd", "eur"]
    )
    assert_frame_equal(result, expected)

    result = fxf.positions(100, aggregate=True)
    expected = Series(
        [100.0, 0.],
        index=["usd", "eur"],
        name=dt(2022, 1, 1),
    )
    assert_series_equal(result, expected)


def test_recursive_chain():
    T = np.array([[1, 1], [0, 1]])
    result = FXForwards._get_recursive_chain(T, 1, 0)
    expected = True, [{"col": 0}]
    assert result == expected

    result = FXForwards._get_recursive_chain(T, 0, 1)
    expected = True, [{"row": 1}]
    assert result == expected


def test_recursive_chain3():
    T = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    result = FXForwards._get_recursive_chain(T, 2, 0)
    expected = True, [{"col": 1}, {"col": 0}]
    assert result == expected

    result = FXForwards._get_recursive_chain(T, 0, 2)
    expected = True, [{"row": 1}, {"row": 2}]
    assert result == expected


def test_recursive_chain_interim_broken_path():
    T = np.array([[1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    result = FXForwards._get_recursive_chain(T, 0, 3)
    expected = True, [{"row": 2}, {"row": 3}]
    assert result == expected


def test_multiple_currencies_number_raises(usdusd):
    fxr1 = FXRates({"eurusd": 0.95}, settlement=dt(2022, 1, 3))
    fxr2 = FXRates({"gbpcad": 1.1}, settlement=dt(2022, 1, 2))
    with pytest.raises(ValueError, match="`fx_curves` is underspecified."):
        FXForwards([fxr1, fxr2], {})

    with pytest.raises(ValueError, match="`fx_curves` is overspecified."):
        FXForwards(
            fxr1,
            {
                "eureur": usdusd,
                "usdusd": usdusd,
                "usdeur": usdusd,
                "eurusd": usdusd,
            },
        )


def test_forwards_unexpected_curve_raise(usdusd):
    fxr = FXRates({"eurusd": 0.95}, settlement=dt(2022, 1, 3))
    with pytest.raises(ValueError, match="`fx_curves` contains an unexpected currency"):
        FXForwards(
            fxr,
            {
                "eureur": usdusd,
                "usdusd": usdusd,
                "usdeur": usdusd,
                "usdcad": usdusd,
            },
        )


def test_forwards_codependent_curve_raise(usdusd):
    fxr = FXRates({"eurusd": 0.95, "usdnok": 10.0}, settlement=dt(2022, 1, 3))
    with pytest.raises(ValueError, match="`fx_curves` contains co-dependent rates"):
        FXForwards(
            fxr,
            {
                "eureur": usdusd,
                "usdusd": usdusd,
                "usdeur": usdusd,
                "eurusd": usdusd,
                "noknok": usdusd,
            },
        )


def test_multiple_settlement_forwards():
    fxr1 = FXRates({"usdeur": 0.95}, dt(2022, 1, 3))
    fxr2 = FXRates({"usdcad": 1.1}, dt(2022, 1, 2))

    fxf = FXForwards(
        [fxr1, fxr2],
        {
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 3): 0.95}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 3): 1.0}),
            "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 3): 1.0}),
            "cadusd": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.97}),
            "cadcad": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.969}),
        },
    )
    F0_usdeur = 0.95 * 1.0 / 0.95  # f_usdeur * w_eurusd / v_usdusd
    F0_usdeur_result = fxf.rate("usdeur", dt(2022, 1, 1))
    assert F0_usdeur_result.real == F0_usdeur
    assert fxf.rate("usdeur", dt(2022, 1, 3)) == Dual(0.95, "fx_usdeur", [1.0])


def test_generate_proxy_curve():
    fxr1 = FXRates({"usdeur": 0.95}, dt(2022, 1, 3))
    fxr2 = FXRates({"usdcad": 1.1}, dt(2022, 1, 2))

    fxf = FXForwards(
        [fxr1, fxr2],
        {
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.95}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 1.0}),
            "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.99}),
            "cadusd": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.97}),
            "cadcad": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.969}),
        },
    )
    c1 = fxf.curve("cad", "cad")
    assert c1[dt(2022, 10, 1)] == 0.969

    c2 = fxf.curve("cad", "usd")
    assert c2[dt(2022, 10, 1)] == 0.97

    c3 = fxf.curve("cad", "eur")
    assert type(c3) is not Curve  # should be ProxyCurve
    assert c3[dt(2022, 10, 1)] == Dual(0.9797979797979798, ["fx_usdcad", "fx_usdeur"], [0, 0])


def test_generate_multi_csa_curve():
    fxr1 = FXRates({"usdeur": 0.95}, dt(2022, 1, 3))
    fxr2 = FXRates({"usdcad": 1.1}, dt(2022, 1, 2))

    fxf = FXForwards(
        [fxr1, fxr2],
        {
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.95}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 1.0}),
            "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.99}),
            "cadusd": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.97}),
            "cadcad": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.969}),
        },
    )
    c1 = fxf.curve("cad", ["cad","usd","eur"])
    assert isinstance(c1, CompositeCurve)


def test_proxy_curves_update_with_underlying():
    # Test ProxyCurves update after construction and underlying update
    fxr1 = FXRates({"usdeur": 0.95}, dt(2022, 1, 3))
    fxr2 = FXRates({"usdcad": 1.1}, dt(2022, 1, 2))

    fxf = FXForwards(
        [fxr1, fxr2],
        {
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.95}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 1.0}),
            "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.99}),
            "cadusd": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.97}),
            "cadcad": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.969}),
        },
    )

    proxy_curve = fxf.curve("cad", "eur")
    prev_value = proxy_curve[dt(2022, 10, 1)]
    fxf.fx_curves["eureur"].nodes[dt(2022, 10, 1)] = 0.90
    new_value = proxy_curve[dt(2022, 10, 1)]

    assert prev_value != new_value


def test_full_curves(usdusd, eureur, usdeur):
    usdusd = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.999})
    eureur = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.998})
    eurusd = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.9985})
    noknok = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.997})
    nokeur = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.9965})
    fxr = FXRates({"usdnok": 8.0, "eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxf = FXForwards(
        fxr,
        {
            "usdusd": usdusd,
            "eureur": eureur,
            "eurusd": eurusd,
            "noknok": noknok,
            "nokeur": nokeur,
        },
    )
    curve = fxf._full_curve("usd", "nok")
    assert type(curve) is Curve
    assert len(curve.nodes) == 10  # constructed with DF on every date


@pytest.mark.parametrize("settlement", [dt(2022, 1, 1), dt(2022, 1, 3), dt(2022, 1, 7)])
def test_rate_path_immediate(settlement):
    usdusd = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.999})
    eureur = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.998})
    eurusd = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.9985})
    noknok = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.997})
    nokeur = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.9965})
    fxr = FXRates({"usdnok": 8.0, "eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxf = FXForwards(
        fxr,
        {
            "usdusd": usdusd,
            "eureur": eureur,
            "eurusd": eurusd,
            "noknok": noknok,
            "nokeur": nokeur,
        },
    )
    _, result = fxf.rate("nokusd", settlement, return_path=True)
    expected = [{"col": 1}, {"col": 2}]
    assert result == expected


@pytest.mark.parametrize(
    "left",
    [
        NoInput(0),
        dt(2022, 1, 1),
        "0d",
    ],
)
@pytest.mark.parametrize(
    "right",
    [
        NoInput(0),
        dt(2022, 1, 10),
        "9d",
    ],
)
def test_fx_plot(left, right):
    usdusd = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.999})
    eureur = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.998})
    eurusd = Curve({dt(2022, 1, 1): 1.0, dt(2022, 1, 10): 0.9985})
    fxr = FXRates({"usdeur": 1.05}, settlement=dt(2022, 1, 3))
    fxf = FXForwards(
        fxr,
        {
            "usdusd": usdusd,
            "eureur": eureur,
            "eurusd": eurusd,
        },
    )
    result = fxf.plot("eurusd", left=left, right=right)
    assert len(result) == 3
    y_data = result[2][0].get_data()[1]
    assert abs(float(y_data[8]) - 0.9520631477714822) < 1e-10
    plt.close("all")


def test_delta_risk_equivalence():
    start, end = dt(2022, 1, 1), dt(2023, 1, 1)
    fx_curves = {
        "usdusd": Curve({start: 1.0, end: 0.96}, id="uu", ad=1),
        "eureur": Curve({start: 1.0, end: 0.99}, id="ee", ad=1),
        "eurusd": Curve({start: 1.0, end: 0.991}, id="eu", ad=1),
        "noknok": Curve({start: 1.0, end: 0.98}, id="nn", ad=1),
        "nokeur": Curve({start: 1.0, end: 0.978}, id="ne", ad=1),
    }
    fx_rates = FXRates({"usdeur": 0.9, "eurnok": 8.888889}, dt(2022, 1, 3))
    fxf = FXForwards(fx_rates, fx_curves)

    discounted_nok = fx_curves["nokeur"][dt(2022, 8, 15)] * 1000
    result1 = discounted_nok * fxf.rate("nokusd", dt(2022, 1, 1))

    forward_eur = fxf.rate("nokeur", dt(2022, 8, 15)) * 1000
    discounted_eur = forward_eur * fx_curves["eureur"][dt(2022, 8, 15)]
    result2 = discounted_eur * fxf.rate("eurusd", dt(2022, 1, 1))

    assert result1.vars == (
        "ee0",
        "ee1",
        "eu0",
        "eu1",
        "fx_eurnok",
        "fx_usdeur",
        "ne0",
        "ne1",
        "uu0",
        "uu1",
    )
    assert result1 == result2


def test_oo_update_rates_and_id():
    # Test the FXRates object can be updated with new FX Rates without creating new
    fxr = FXRates({"usdeur": 2.0, "usdgbp": 2.5})
    id_ = id(fxr)
    assert fxr.rate("eurgbp") == Dual(1.25, ["fx_usdeur", "fx_usdgbp"], [-0.625, 0.5])
    fxr.update({"usdGBP": 3.0})
    assert fxr.rate("eurgbp") == Dual(1.5, ["fx_usdeur", "fx_usdgbp"], [-0.75, 0.5])
    assert id(fxr) == id_


def test_oo_update_forwards_rates():
    # Test the FXForwards object update method will react to an update of FXRates
    start, end = dt(2022, 1, 1), dt(2023, 1, 1)
    fx_curves = {
        "usdusd": Curve({start: 1.0, end: 0.96}, id="uu", ad=1),
        "eureur": Curve({start: 1.0, end: 0.99}, id="ee", ad=1),
        "eurusd": Curve({start: 1.0, end: 0.991}, id="eu", ad=1),
        "noknok": Curve({start: 1.0, end: 0.98}, id="nn", ad=1),
        "nokeur": Curve({start: 1.0, end: 0.978}, id="ne", ad=1),
    }
    fx_rates = FXRates({"usdeur": 0.9, "eurnok": 8.888889}, dt(2022, 1, 3))
    fxf = FXForwards(fx_rates, fx_curves)
    original_fwd = fxf.rate("usdnok", dt(2022, 7, 15))  # 7.917 = 0.9 * 8.888
    fx_rates.update({"usdeur": 1.0})
    fxf.update()
    updated_fwd = fxf.rate("usdnok", dt(2022, 7, 15))  # 8.797 = 1.0 * 8.888
    assert original_fwd != updated_fwd


def test_oo_update_forwards_rates_list():
    # Test the FXForwards object update method will react to an update of FXRates
    start, end = dt(2022, 1, 1), dt(2023, 1, 1)
    fx_curves = {
        "usdusd": Curve({start: 1.0, end: 0.96}, id="uu", ad=1),
        "eureur": Curve({start: 1.0, end: 0.99}, id="ee", ad=1),
        "eurusd": Curve({start: 1.0, end: 0.991}, id="eu", ad=1),
        "noknok": Curve({start: 1.0, end: 0.98}, id="nn", ad=1),
        "nokeur": Curve({start: 1.0, end: 0.978}, id="ne", ad=1),
    }
    fx_rates1 = FXRates({"usdeur": 0.9}, dt(2022, 1, 2))
    fx_rates2 = FXRates({"eurnok": 8.888889}, dt(2022, 1, 3))
    fxf = FXForwards([fx_rates1, fx_rates2], fx_curves)
    original_fwd = fxf.rate("usdnok", dt(2022, 7, 15))  # 7.917 = 0.9 * 8.888
    assert abs(original_fwd - 7.917) < 1e-3
    fx_rates1.update({"usdeur": 1.0})
    fxf.update()
    updated_fwd = fxf.rate("usdnok", dt(2022, 7, 15))  # 8.797 = 1.0 * 8.888
    assert abs(updated_fwd - 8.797) < 1e-3
    assert original_fwd != updated_fwd


def test_oo_update_forwards_rates_equivalence():
    # Test the FXForwards object update method is equivalent to an FXRates update
    start, end = dt(2022, 1, 1), dt(2023, 1, 1)
    fx_curves = {
        "usdusd": Curve({start: 1.0, end: 0.96}, id="uu", ad=1),
        "eureur": Curve({start: 1.0, end: 0.99}, id="ee", ad=1),
        "eurusd": Curve({start: 1.0, end: 0.991}, id="eu", ad=1),
        "noknok": Curve({start: 1.0, end: 0.98}, id="nn", ad=1),
        "nokeur": Curve({start: 1.0, end: 0.978}, id="ne", ad=1),
    }
    fx_rates1 = FXRates({"usdeur": 0.9, "eurnok": 8.888889}, dt(2022, 1, 3))
    fx_rates2 = FXRates({"usdeur": 0.9, "eurnok": 8.888889}, dt(2022, 1, 3))
    fxf1 = FXForwards(fx_rates1, fx_curves)
    fxf2 = FXForwards(fx_rates2, fx_curves)

    fx_rates1.update({"usdeur": 1.0})
    fxf1.update()

    fxf2.update(FXRates({"usdeur": 1.0, "eurnok": 8.888889}, dt(2022, 1, 3)))

    assert fxf1.rate("usdnok", dt(2022, 7, 15)) == fxf2.rate("usdnok", dt(2022, 7, 15))


@pytest.mark.parametrize(
    "fxr",
    [
        FXRates({"usdeur": 0.9}, settlement=dt(2022, 1, 3)),
        [
            FXRates({"usdeur": 0.9}, settlement=dt(2022, 1, 3)),
        ],
    ],
)
def test_fxforwards_to_json_round_trip(fxr, usdusd, eureur, usdeur):
    fxc = {"usdusd": usdusd, "eureur": eureur, "usdeur": usdeur}
    fxf = FXForwards(fxr, fxc)

    result = fxf.to_json()
    fxf1 = FXForwards.from_json(result)
    fxr1, fxc1 = fxf1.fx_rates, fxf1.fx_curves

    assert fxc1 == fxc
    assert fxr1 == fxr
    assert fxf1 == fxf


def test_bad_settlement_date(usdusd, usdeur, eureur):
    fxf = FXForwards(
        FXRates({"usdeur": 0.9}, settlement=dt(2022, 1, 3)),
        {"usdusd": usdusd, "eureur": eureur, "usdeur": usdeur},
    )
    with pytest.raises(ValueError, match="`settlement` cannot"):
        fxf.rate("usdeur", dt(1999, 1, 1))  # < date before curves


def test_fxforwards_separable_system():
    fxr1 = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxr2 = FXRates({"usdcad": 1.1}, settlement=dt(2022, 1, 2))
    fxf = FXForwards(
        fx_rates=[fxr1, fxr2],
        fx_curves={
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "cadcad": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "usdeur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "cadusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
        },
    )
    result = fxf.rate("eurcad", dt(2022, 2, 1))
    expected = 1.05 * 1.10
    assert abs(result - expected) < 1e-2


def test_fxforwards_acyclic_system():
    fxr1 = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxr2 = FXRates({"usdcad": 1.1}, settlement=dt(2022, 1, 2))
    fxf = FXForwards(
        fx_rates=[fxr1, fxr2],
        fx_curves={
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "cadcad": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "usdeur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "cadeur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
        },
    )
    result = fxf.rate("eurcad", dt(2022, 2, 1))
    expected = 1.05 * 1.10
    assert abs(result - expected) < 1e-2


def test_fxforwards_cyclic_system_fails():
    fxr1 = FXRates({"eurusd": 1.05, "gbpusd": 1.2}, settlement=dt(2022, 1, 3))
    fxr2 = FXRates({"usdcad": 1.1}, settlement=dt(2022, 1, 2))
    with pytest.raises(ValueError, match="`fx_curves` is underspecified."):
        FXForwards(
            fx_rates=[fxr1, fxr2],
            fx_curves={
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
                "cadcad": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
                "usdeur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
                "cadeur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
                "gbpcad": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
                "gbpgbp": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            },
        )


def test_fxforwards_cyclic_system_restructured():
    fxr1 = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxr2 = FXRates({"usdcad": 1.1}, settlement=dt(2022, 1, 2))
    fxr3 = FXRates({"gbpusd": 1.2}, settlement=dt(2022, 1, 3))
    fxf = FXForwards(
        fx_rates=[fxr1, fxr2, fxr3],
        fx_curves={
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "cadcad": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "usdeur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "cadeur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "gbpcad": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "gbpgbp": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
        },
    )
    result = fxf.rate("eurcad", dt(2022, 2, 1))
    expected = 1.05 * 1.10
    assert abs(result - expected) < 1e-2


def test_fxforwards_positions_when_immediate_aligns_with_settlement():
    fxr1 = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 1))
    fxr2 = FXRates({"usdcad": 1.1}, settlement=dt(2022, 1, 1))
    fxf = FXForwards(
        fx_rates=[fxr1, fxr2],
        fx_curves={
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "cadcad": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "usdeur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "cadusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
        },
    )
    pv = Dual(100000, ["fx_eurusd", "fx_usdcad"], [-100000, -150000])
    result = fxf.positions(pv, base="usd")
    expected = DataFrame(
        index=["cad", "eur", "usd"],
        columns=[dt(2022, 1, 1)],
        data=[[181500.0], [-100000.0], [40000]],
    )
    assert_frame_equal(result, expected)


def test_fxforwards_positions_multiple_fx_rates():
    fxr1 = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxr2 = FXRates({"usdcad": 1.1}, settlement=dt(2022, 1, 2))
    fxf = FXForwards(
        fx_rates=[fxr1, fxr2],
        fx_curves={
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "cadcad": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "usdeur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
            "cadusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}),
        },
    )
    pv = Dual(100000, ["fx_eurusd", "fx_usdcad"], [-100000, -150000])
    result = fxf.positions(pv, base="usd")
    expected = DataFrame(
        index=["cad", "eur", "usd"],
        columns=[dt(2022, 1, 1), dt(2022, 1, 2), dt(2022, 1, 3)],
        data=[[0.0, 181500.0, 0.0], [0.0, 0.0, -100000.0], [100000, -165000, 105000]],
    )
    assert_frame_equal(result, expected)


def test_forward_fx_immediate():
    d_curve = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, interpolation="log_linear")
    f_curve = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.95})
    result = forward_fx(dt(2022, 4, 1), d_curve, f_curve, 10.0)
    assert abs(result - 10.102214) < 1e-6

    result = forward_fx(dt(2022, 1, 1), d_curve, f_curve, 10.0, dt(2022, 1, 1))
    assert abs(result - 10.0) < 1e-6

    result = forward_fx(dt(2022, 1, 1), d_curve, f_curve, 10.0, None)
    assert abs(result - 10.0) < 1e-6


def test_forward_fx_spot_equivalent():
    d_curve = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, interpolation="log_linear")
    f_curve = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.95})
    result = forward_fx(dt(2022, 7, 1), d_curve, f_curve, 10.102214, dt(2022, 4, 1))
    assert abs(result - 10.206626) < 1e-6
