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
from rateslib.dual import Dual, Dual2, gradient
from rateslib.curves import Curve, LineCurve, CompositeCurve
from rateslib.default import NoInput
from rateslib.fx_volatility import FXDeltaVolSmile
from rateslib.periods import FXPutPeriod

@pytest.fixture()
def fxfo():
    # FXForwards for FX Options tests
    eureur = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.9851909811629752}, calendar="tgt", id="eureur"
    )
    usdusd = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.976009366603271}, calendar="nyc", id="usdusd"
    )
    eurusd = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.987092591908283}, id="eurusd"
    )
    fxr = FXRates({"eurusd": 1.0615}, settlement=dt(2023, 3, 20))
    fxf = FXForwards(
        fx_curves={"eureur": eureur, "eurusd": eurusd, "usdusd": usdusd},
        fx_rates=fxr
    )
    # fxf.swap("eurusd", [dt(2023, 3, 20), dt(2023, 6, 20)]) = 60.10
    return fxf


class TestFXDeltaVolSmile:

    @pytest.mark.parametrize("k", [0.2, 0.8, 0.9, 1.0, 1.05, 1.10, 1.25, 1.5, 9.0])
    def test_get_from_strike(self, fxfo, k):
        fxvs = FXDeltaVolSmile(
            nodes={
                0.25: 10.15,
                0.5: 7.8,
                0.75: 8.9,
            },
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
        )
        put_vol = fxvs.get_from_strike(
            k=k,
            phi=-1.0,
            f=fxfo.rate("eurusd", dt(2023, 6, 20)),
            w_deli=fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
            w_spot=fxfo.curve("eur", "usd")[dt(2023, 3, 20)]
        )
        call_vol = fxvs.get_from_strike(
            k=k,
            phi=1.0,
            f=fxfo.rate("eurusd", dt(2023, 6, 20)),
            w_deli=fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
            w_spot=fxfo.curve("eur", "usd")[dt(2023, 3, 20)]
        )
        assert abs(put_vol[1] - call_vol[1]) < 1e-9

    @pytest.mark.parametrize("var, idx, val", [
        ("vol0", 0.25, 10.15),
        ("vol1", 0.5, 7.8),
        ("vol2", 0.75, 8.9)
    ])
    @pytest.mark.parametrize("k", [0.9, 1.0, 1.05, 1.10, 1.4])
    def test_get_from_strike_ad(self, fxfo, var, idx, val, k):
        fxvs = FXDeltaVolSmile(
            nodes={
                0.25: 10.15,
                0.5: 7.8,
                0.75: 8.9,
            },
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
            ad=1,
        )
        args = (
            k,
            -1.0,
            fxfo.rate("eurusd", dt(2023, 6, 20)),
            fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
            fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
        )
        put_vol = fxvs.get_from_strike(*args)

        fxvs.nodes[idx] = Dual(val + 0.0000001, [var], [])
        fxvs.csolve()
        put_vol_plus = fxvs.get_from_strike(*args)

        finite_diff = (put_vol_plus[1]-put_vol[1]) * 10000000.0
        ad_grad = gradient(put_vol[1], [var])[0]

        assert abs(finite_diff - ad_grad) < 1e-7

    @pytest.mark.parametrize("k", [0.9, 1.0, 1.05, 1.10, 1.4])
    @pytest.mark.parametrize("cross", [
        (["vol0", 10.15, 0.25], ["vol1", 7.8, 0.5]),
        (["vol0", 10.15, 0.25], ["vol2", 8.9, 0.75]),
        (["vol1", 7.8, 0.5], ["vol2", 8.9, 0.75]),
    ])
    def test_get_from_strike_ad_2(self, fxfo, k, cross):
        fxvs = FXDeltaVolSmile(
            nodes={
                0.25: 10.15,
                0.5: 7.8,
                0.75: 8.9,
            },
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
            ad=2,
        )
        fxfo._set_ad_order(2)
        args = (
            k,
            -1.0,
            fxfo.rate("eurusd", dt(2023, 6, 20)),
            fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
            fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
        )
        pv00 = fxvs.get_from_strike(*args)

        fxvs.nodes[cross[0][2]] = Dual2(cross[0][1] + 0.00001, [cross[0][0]], [], [])
        fxvs.nodes[cross[1][2]] = Dual2(cross[1][1] + 0.00001, [cross[1][0]], [], [])
        fxvs.csolve()
        pv11 = fxvs.get_from_strike(*args)

        fxvs.nodes[cross[0][2]] = Dual2(cross[0][1] + 0.00001, [cross[0][0]], [], [])
        fxvs.nodes[cross[1][2]] = Dual2(cross[1][1] - 0.00001, [cross[1][0]], [], [])
        fxvs.csolve()
        pv1_1 = fxvs.get_from_strike(*args)

        fxvs.nodes[cross[0][2]] = Dual2(cross[0][1] - 0.00001, [cross[0][0]], [], [])
        fxvs.nodes[cross[1][2]] = Dual2(cross[1][1] - 0.00001, [cross[1][0]], [], [])
        fxvs.csolve()
        pv_1_1 = fxvs.get_from_strike(*args)

        fxvs.nodes[cross[0][2]] = Dual2(cross[0][1] - 0.00001, [cross[0][0]], [], [])
        fxvs.nodes[cross[1][2]] = Dual2(cross[1][1] + 0.00001, [cross[1][0]], [], [])
        fxvs.csolve()
        pv_11 = fxvs.get_from_strike(*args)

        finite_diff = (pv11[1] + pv_1_1[1] - pv1_1[1] - pv_11[1]) * 1e10 / 4.0
        ad_grad = gradient(pv00[1], [cross[0][0], cross[1][0]], 2)[0, 1]

        assert abs(finite_diff - ad_grad) < 5e-5

    # @pytest.mark.parametrize("delta, type, smile_type, k, phi, exp", [
    #     (-0.2, "forward", "forward", 1.02, -1, 0.2),
    #     (-0.2, "spot", "spot", 1.02, -1, 0.2),
    #     (-0.2, "forward_pa", "forward_pa", 1.02, -1, 0.2),
    #     (-0.2, "spot_pa", "spot_pa", 1.02, -1, 0.2),
    #     (-0.2, "forward", "spot", 1.02, -1, 0.19870506706),
    #     (-0.2, "spot", "forward", 1.02, -1, 0.20130337183),
    #     (-0.2, "forward_pa", "spot_pa", 1.02, -1, 0.19870506706),
    #     (-0.2, "spot_pa", "forward_pa", 1.02, -1, 0.20130337183),
    #     (-0.2, "forward", "forward_pa", 1.02, -1, 0.22081326385),
    #     (-0.2, "forward", "spot_pa", 1.02, -1, 0.2194459183655),
    # ])
    # def test_convert_delta_put(self, fxfo, delta, type, smile_type, k, phi, exp):
    #     fxvs = FXDeltaVolSmile(
    #         nodes={
    #             0.25: 10.15,
    #             0.5: 7.8,
    #             0.75: 8.9
    #         },
    #         delta_type=smile_type,
    #         eval_date=dt(2023, 3, 16),
    #         expiry=dt(2023, 6, 16),
    #         id="vol",
    #     )
    #     result = fxvs._convert_delta(
    #         delta,
    #         type,
    #         phi,
    #         fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
    #         fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
    #         k / fxfo.rate("eurusd", dt(2023, 6, 20))
    #     )
    #
    #     assert abs(result - exp) < 1e-10

    @pytest.mark.parametrize("delta_type, smile_type, k", [
        ("forward", "forward_pa", 0.8),
        ("forward", "spot_pa", 0.8),
        ("spot", "forward_pa", 0.8),
        ("spot", "spot_pa", 0.8),
        ("forward", "forward_pa", 1.0),
        ("forward", "spot_pa", 1.0),
        ("spot", "forward_pa", 1.0),
        ("spot", "spot_pa", 1.0),
        ("forward", "forward_pa", 1.10),
        ("forward", "spot_pa", 1.10),
        ("spot", "forward_pa", 1.10),
        ("spot", "spot_pa", 1.10),
        ("forward", "forward_pa", 1.2),
        ("forward", "spot_pa", 1.2),
        ("spot", "forward_pa", 1.2),
        ("spot", "spot_pa", 1.2),
        ("forward_pa", "forward", 0.8),
        ("forward_pa", "spot", 0.8),
        ("spot_pa", "forward", 0.8),
        ("spot_pa", "spot", 0.8),
        ("forward_pa", "forward", 1.0),
        ("forward_pa", "spot", 1.0),
        ("spot_pa", "forward", 1.0),
        ("spot_pa", "spot", 1.0),
        ("forward_pa", "forward", 1.10),
        ("forward_pa", "spot", 1.10),
        ("spot_pa", "forward", 1.10),
        ("spot_pa", "spot", 1.10),
        ("forward_pa", "forward", 1.19),
        ("forward_pa", "spot", 1.19),
        ("spot_pa", "forward", 1.2),
        ("spot_pa", "spot", 1.2),
    ])
    def test_convert_delta_put2(self, fxfo, delta_type, smile_type, k):
        # Test the _convert_delta method of a DeltaVolSmile

        fxo1 = FXPutPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=k,
            delta_type=smile_type,
        )
        fxo2 = FXPutPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=k,
            delta_type=delta_type,
        )
        smile_delta = fxo1.analytic_greeks(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fxfo,
            vol=0.10,
        )["delta"]
        delta = fxo2.analytic_greeks(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fxfo,
            vol=0.10,
        )["delta"]

        if -smile_delta < 0.5:
            nodes = {float(-smile_delta): 10.0, float(-smile_delta) + 0.25: 12.5}
        else:
            nodes = {float(-smile_delta) - 0.25: 7.5, float(-smile_delta): 10}

        fxvs = FXDeltaVolSmile(
            nodes=nodes,
            delta_type=smile_type,
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
        )
        result = fxvs._convert_delta(
            delta,
            delta_type,
            -1.0,
            fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
            fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
            k / fxfo.rate("eurusd", dt(2023, 6, 20))
        )
        assert abs(result + smile_delta) < 1e-10


    def test_get_from_unsimilar_delta(self, ):
        fxvs = FXDeltaVolSmile(
            nodes={0.25: 10.0, 0.5: 10.0, 0.75: 11.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
        )
        result = fxvs.get(0.65, "spot_pa", 1.0, 0.99, 0.999, 0.9)
        expected = 10.0
        assert (result - expected) < 0.01

    def test_call_to_put_delta_raises(self):
        fxvs = FXDeltaVolSmile(
            nodes={0.25: 10.0, 0.5: 10.0, 0.75: 11.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
        )
        with pytest.raises(ValueError, match="`delta_type` must be in"):
            fxvs._call_to_put_delta(0.5, "bad_type")

    def test_iter_raises(self):
        fxvs = FXDeltaVolSmile(
            nodes={0.5: 1.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
        )
        with pytest.raises(TypeError, match="`FXDeltaVolSmile` is not iterable."):
            fxvs.__iter__()

    # @pytest.mark.parametrize("smile_type, delta_type, exp", [
    #     ("forward", "forward", 0.25),
    #     ("forward", "spot", 0.25),
    # ])
    # def test_get(self, fxfo):
    #     pass


