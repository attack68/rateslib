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
    def test_get_from_strike_ad(self, fxfo, var, idx, val):
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
            1.05,
            -1.0,
            fxfo.rate("eurusd", dt(2023, 6, 20)),
            fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
            fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
        )
        put_vol = fxvs.get_from_strike(*args)

        fxvs.nodes[idx] = Dual(val + 0.00001, [var], [])
        fxvs.csolve()
        put_vol2 = fxvs.get_from_strike(*args)
        finite_diff = (put_vol2[1]-put_vol[1]) * 100000.0
        ad_grad = gradient(put_vol[1], [var])[0]

        assert abs(finite_diff - ad_grad) < 1e-8
