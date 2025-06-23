from datetime import datetime as dt

import numpy as np
import pytest
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal
from rateslib.calendars import get_calendar
from rateslib.curves import CompositeCurve, Curve, LineCurve
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, gradient
from rateslib.fx import (
    FXForwards,
    FXRates,
    forward_fx,
)
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, _validate_delta_type
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
    eurusd = Curve({dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.987092591908283}, id="eurusd")
    fxr = FXRates({"eurusd": 1.0615}, settlement=dt(2023, 3, 20))
    fxf = FXForwards(fx_curves={"eureur": eureur, "eurusd": eurusd, "usdusd": usdusd}, fx_rates=fxr)
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
            f=fxfo.rate("eurusd", dt(2023, 6, 20)),
            w_deli=fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
            w_spot=fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
        )
        call_vol = fxvs.get_from_strike(
            k=k,
            f=fxfo.rate("eurusd", dt(2023, 6, 20)),
            w_deli=fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
            w_spot=fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
        )
        assert abs(put_vol[1] - call_vol[1]) < 1e-9

    @pytest.mark.parametrize(
        "var, idx, val", [("vol0", 0.25, 10.15), ("vol1", 0.5, 7.8), ("vol2", 0.75, 8.9)]
    )
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
            fxfo.rate("eurusd", dt(2023, 6, 20)),
            fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
            fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
        )
        put_vol = fxvs.get_from_strike(*args)

        fxvs.nodes[idx] = Dual(val + 0.0000001, [var], [])
        fxvs.csolve()
        put_vol_plus = fxvs.get_from_strike(*args)

        finite_diff = (put_vol_plus[1] - put_vol[1]) * 10000000.0
        ad_grad = gradient(put_vol[1], [var])[0]

        assert abs(finite_diff - ad_grad) < 1e-7

    @pytest.mark.parametrize("k", [0.9, 1.0, 1.05, 1.10, 1.4])
    @pytest.mark.parametrize(
        "cross",
        [
            (["vol0", 10.15, 0.25], ["vol1", 7.8, 0.5]),
            (["vol0", 10.15, 0.25], ["vol2", 8.9, 0.75]),
            (["vol1", 7.8, 0.5], ["vol2", 8.9, 0.75]),
        ],
    )
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

    @pytest.mark.parametrize(
        "delta_type, smile_type, k",
        [
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
        ],
    )
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
            k / fxfo.rate("eurusd", dt(2023, 6, 20)),
        )
        assert abs(result + smile_delta) < 1e-10

    def test_get_from_unsimilar_delta(self):
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

    @pytest.mark.parametrize("delta_type, exp", [("spot", 10.00000489), ("forward", 10.0)])
    def test_get_from_similar_delta(self, delta_type, exp):
        fxvs = FXDeltaVolSmile(
            nodes={0.25: 11.0, 0.5: 10.0, 0.75: 11.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
        )
        result = fxvs.get(0.5, delta_type, 1.0, 0.99, 0.991, 1.02)
        assert abs(result - exp) < 1e-6

    def test_set_same_ad_order(self):
        fxvs = FXDeltaVolSmile(
            nodes={0.25: 10.0, 0.5: 10.0, 0.75: 11.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
            ad=1,
        )
        assert fxvs._set_ad_order(1) is None
        assert fxvs.nodes[0.25] == Dual(10.0, ["vol0"], [])

    def test_set_ad_order_raises(self):
        fxvs = FXDeltaVolSmile(
            nodes={0.25: 10.0, 0.5: 10.0, 0.75: 11.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
            ad=1,
        )
        with pytest.raises(ValueError, match="`order` can only be in"):
            fxvs._set_ad_order(10)

    def test_iter_raises(self):
        fxvs = FXDeltaVolSmile(
            nodes={0.5: 1.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
        )
        with pytest.raises(TypeError, match="`FXDeltaVolSmile` is not iterable."):
            fxvs.__iter__()


class TestFXDeltaVolSurface:
    def test_expiry_before_eval(self):
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2025, 1, 1)],
            node_values=[[11, 10, 12], [8, 7, 9]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
        )
        with pytest.raises(ValueError, match="`expiry` before the `eval_date` of"):
            fxvs.get_smile(dt(2022, 1, 1))

    def test_smile_0_no_interp(self):
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2025, 1, 1)],
            node_values=[[11, 10, 12], [8, 7, 9]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
        )
        result = fxvs.get_smile(dt(2023, 2, 1))
        expected = FXDeltaVolSmile(
            nodes={0.25: 11, 0.5: 10, 0.75: 12},
            eval_date=dt(2023, 1, 1),
            expiry=dt(2023, 2, 1),
            delta_type="forward",
        )
        assert result.nodes == expected.nodes
        assert result.expiry == expected.expiry
        assert result.delta_type == expected.delta_type
        assert result.eval_date == expected.eval_date

    def test_smile_end_no_interp(self):
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2025, 1, 1)],
            node_values=[[11, 10, 12], [8, 7, 9]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
        )
        result = fxvs.get_smile(dt(2029, 2, 1))
        expected = FXDeltaVolSmile(
            nodes={0.25: 8, 0.5: 7, 0.75: 9},
            eval_date=dt(2023, 1, 1),
            expiry=dt(2029, 2, 1),
            delta_type="forward",
        )
        assert result.nodes == expected.nodes
        assert result.expiry == expected.expiry
        assert result.delta_type == expected.delta_type
        assert result.eval_date == expected.eval_date

    def test_smile_tot_var_lin_interp(self):
        # See Foreign Exchange Option Pricing: Iain Clarke Table 4.5
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2025, 1, 1)],
            node_values=[[19.590, 18.250, 18.967], [18.801, 17.677, 18.239]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
        )
        result = fxvs.get_smile(dt(2024, 7, 1))
        expected = FXDeltaVolSmile(
            nodes={0.25: 19.0693, 0.5: 17.8713, 0.75: 18.4864},
            eval_date=dt(2023, 1, 1),
            expiry=dt(2024, 7, 1),
            delta_type="forward",
        )
        for (k1, v1), (k2, v2) in zip(result.nodes.items(), expected.nodes.items()):
            assert abs(v1 - v2) < 0.0001
        assert result.expiry == expected.expiry
        assert result.delta_type == expected.delta_type
        assert result.eval_date == expected.eval_date

    def test_smile_from_exact_expiry(self):
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2025, 1, 1)],
            node_values=[[19.590, 18.250, 18.967], [18.801, 17.677, 18.239]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
            id="surf",
        )
        expected = FXDeltaVolSmile(
            nodes={0.25: 19.590, 0.5: 18.25, 0.75: 18.967},
            eval_date=dt(2023, 1, 1),
            expiry=dt(2024, 1, 1),
            delta_type="forward",
            id="surf_0_",
        )
        result = fxvs.get_smile(dt(2024, 1, 1))
        for (k1, v1), (k2, v2) in zip(result.nodes.items(), expected.nodes.items()):
            assert abs(v1 - v2) < 0.0001
        assert result.expiry == expected.expiry
        assert result.delta_type == expected.delta_type
        assert result.eval_date == expected.eval_date
        assert result.id == expected.id

    def test_get_vol_from_strike(self):
        # from a surface creates a smile and then re-uses methods
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2025, 1, 1)],
            node_values=[[19.590, 18.250, 18.967], [18.801, 17.677, 18.239]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
        )
        result = fxvs.get_from_strike(1.05, 1.03, 0.99, 0.999, dt(2024, 7, 1))[1]
        # expected close to delta index of 0.5 i.e around 17.87% vol
        expected = 17.882603173
        assert abs(result - expected) < 1e-8

    def test_get_vol_from_strike_raises(self):
        # from a surface creates a smile and then re-uses methods
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2025, 1, 1)],
            node_values=[[19.590, 18.250, 18.967], [18.801, 17.677, 18.239]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
        )
        with pytest.raises(ValueError, match="`expiry` required to get cross-section"):
            fxvs.get_from_strike(1.05, 1.03, 0.99, 0.999)

    def test_set_node_vector(self):
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2025, 1, 1)],
            node_values=[[19.590, 18.250, 18.967], [18.801, 17.677, 18.239]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
        )
        vec = np.array([3, 2, 4, 5, 4, 6])
        fxvs._set_node_vector(vec, 1)
        for v1, v2 in zip(vec[:3], fxvs.smiles[0].nodes.values()):
            assert abs(v1 - v2) < 1e-10
        for v1, v2 in zip(vec[3:], fxvs.smiles[1].nodes.values()):
            assert abs(v1 - v2) < 1e-10

    def test_expiries_unsorted(self):
        with pytest.raises(ValueError, match="Surface `expiries` are not sorted or"):
            FXDeltaVolSurface(
                delta_indexes=[0.25, 0.5, 0.75],
                expiries=[dt(2024, 1, 1), dt(2024, 1, 1)],
                node_values=[[19.590, 18.250, 18.967], [18.801, 17.677, 18.239]],
                eval_date=dt(2023, 1, 1),
                delta_type="forward",
            )

    def test_set_weights(self):
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2024, 2, 1), dt(2024, 3, 1)],
            node_values=[[11, 10, 12], [8, 7, 9], [9, 7.5, 10]],
            eval_date=dt(2023, 12, 1),
            delta_type="forward",
            weights=Series(2.0, index=[dt(2024, 1, 5), dt(2024, 1, 12), dt(2024, 2, 5)]),
        )
        assert fxvs.weights.loc[dt(2023, 12, 15)] == 1.0
        assert fxvs.weights.loc[dt(2024, 1, 4)] == 0.9393939393939394
        assert fxvs.weights.loc[dt(2024, 1, 5)] == 1.878787878787879
        assert fxvs.weights.loc[dt(2024, 2, 2)] == 0.9666666666666667
        assert fxvs.weights.loc[dt(2024, 2, 5)] == 1.9333333333333333
        assert fxvs.weights.loc[dt(2027, 12, 15)] == 1.0

        # test that the sum of weights to each expiry node is as expected.
        for e in fxvs.expiries:
            assert abs(fxvs.weights[fxvs.eval_date : e].sum() - (e - fxvs.eval_date).days) < 1e-13

    @pytest.mark.parametrize("scalar", [1.0, 0.5])
    def test_weights_get_vol(self, scalar):
        # from a surface creates a smile and then re-uses methods
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2023, 2, 1), dt(2023, 3, 1)],
            node_values=[[19.590, 18.250, 18.967], [18.801, 17.677, 18.239]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
        )
        fxvs_weights = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2023, 2, 1), dt(2023, 3, 1)],
            node_values=[[19.590, 18.250, 18.967], [18.801, 17.677, 18.239]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
            weights=Series(scalar, index=[dt(2023, 2, 2), dt(2023, 2, 3)]),
        )
        result = fxvs.get_from_strike(1.03, 1.03, 0.99, 0.999, dt(2023, 2, 3))
        result2 = fxvs_weights.get_from_strike(1.03, 1.03, 0.99, 0.999, dt(2023, 2, 3))
        w = fxvs_weights.weights

        expected = result[1] * (w[: dt(2023, 2, 3)].sum() / 33.0) ** 0.5
        # This result is not exact because the shape of the spline changes
        assert abs(expected - result2[1]) < 5e-2

    def test_weights_get_vol_clark(self):
        cal = get_calendar("bus")
        weights = Series(0.0, index=cal.cal_date_range(dt(2024, 2, 9), dt(2024, 3, 9)))
        weights.update(Series(1.0, index=cal.bus_date_range(dt(2024, 2, 9), dt(2024, 3, 8))))
        fxvs_weights = FXDeltaVolSurface(
            delta_indexes=[0.5],
            expiries=[
                dt(2024, 2, 12),
                dt(2024, 2, 16),
                dt(2024, 2, 23),
                dt(2024, 3, 1),
                dt(2024, 3, 8),
            ],
            node_values=[[8.15], [11.95], [11.97], [11.75], [11.80]],
            eval_date=dt(2024, 2, 9),
            delta_type="forward",
            weights=weights,
        )

        # Clark FX Option Pricing Table 4.7
        expected = [
            0.0,
            0.0,
            8.15,
            9.99,
            10.95,
            11.54,
            11.95,
            11.18,
            10.54,
            10.96,
            11.29,
            11.56,
            11.78,
            11.97,
            11.56,
            11.20,
            11.34,
            11.46,
            11.57,
            11.66,
            11.75,
            11.48,
            11.23,
            11.36,
            11.49,
            11.60,
            11.70,
            11.80,
            11.59,
        ]

        for i, date in enumerate(cal.cal_date_range(dt(2024, 2, 10), dt(2024, 3, 9))):
            smile = fxvs_weights.get_smile(date)
            assert abs(smile.nodes[0.5] - expected[i]) < 5e-3


def test_validate_delta_type():
    with pytest.raises(ValueError, match="`delta_type` must be in"):
        _validate_delta_type("BAD_TYPE")
