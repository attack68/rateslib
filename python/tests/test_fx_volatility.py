from datetime import datetime as dt
from itertools import combinations

import numpy as np
import pytest
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal
from rateslib import default_context
from rateslib.calendars import get_calendar
from rateslib.curves import CompositeCurve, Curve, LineCurve
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, gradient
from rateslib.fx import (
    FXForwards,
    FXRates,
    forward_fx,
)
from rateslib.fx_volatility import (
    FXDeltaVolSmile,
    FXDeltaVolSurface,
    FXSabrSmile,
    _d_sabr_d_k,
    _sabr,
    _validate_delta_type,
)
from rateslib.periods import FXPutPeriod


@pytest.fixture
def fxfo():
    # FXForwards for FX Options tests
    eureur = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.9851909811629752},
        calendar="tgt",
        id="eureur",
    )
    usdusd = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.976009366603271},
        calendar="nyc",
        id="usdusd",
    )
    eurusd = Curve({dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.987092591908283}, id="eurusd")
    fxr = FXRates({"eurusd": 1.0615}, settlement=dt(2023, 3, 20))
    fxf = FXForwards(fx_curves={"eureur": eureur, "eurusd": eurusd, "usdusd": usdusd}, fx_rates=fxr)
    # fxf.swap("eurusd", [dt(2023, 3, 20), dt(2023, 6, 20)]) = 60.10
    return fxf


class TestFXDeltaVolSmile:
    @pytest.mark.parametrize("k", [0.2, 0.8, 0.9, 1.0, 1.05, 1.10, 1.25, 1.5, 9.0])
    def test_get_from_strike(self, fxfo, k) -> None:
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
        ("var", "idx", "val"),
        [("vol0", 0.25, 10.15), ("vol1", 0.5, 7.8), ("vol2", 0.75, 8.9)],
    )
    @pytest.mark.parametrize("k", [0.9, 1.0, 1.05, 1.10, 1.4])
    def test_get_from_strike_ad(self, fxfo, var, idx, val, k) -> None:
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
    def test_get_from_strike_ad_2(self, fxfo, k, cross) -> None:
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
        ("delta_type", "smile_type", "k"),
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
    def test_convert_delta_put2(self, fxfo, delta_type, smile_type, k) -> None:
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

    def test_get_from_unsimilar_delta(self) -> None:
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

    @pytest.mark.parametrize(("delta_type", "exp"), [("spot", 10.00000489), ("forward", 10.0)])
    def test_get_from_similar_delta(self, delta_type, exp) -> None:
        fxvs = FXDeltaVolSmile(
            nodes={0.25: 11.0, 0.5: 10.0, 0.75: 11.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
        )
        result = fxvs.get(0.5, delta_type, 1.0, 0.99, 0.991, 1.02)
        assert abs(result - exp) < 1e-6

    def test_set_same_ad_order(self) -> None:
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

    def test_set_ad_order_raises(self) -> None:
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

    def test_iter_raises(self) -> None:
        fxvs = FXDeltaVolSmile(
            nodes={0.5: 1.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
        )
        with pytest.raises(TypeError, match="`FXDeltaVolSmile` is not iterable."):
            fxvs.__iter__()

    def test_update_node(self):
        fxvs = FXDeltaVolSmile(
            nodes={0.5: 1.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
        )
        with pytest.raises(KeyError, match=r"`key` is not in Curve ``nodes``"):
            fxvs.update_node(0.4, 10.0)

        fxvs.update_node(0.5, 12.0)
        assert fxvs[0.5] == 12.0

    @pytest.mark.parametrize(
        "nodes", [{0.5: 10.0}, {0.35: 10.0, 0.65: 9.0}, {0.25: 10.0, 0.5: 8.0, 0.75: 11.0}]
    )
    def test_delta_index_range_for_spot(self, nodes):
        # spot delta type can lead to a delta index greater than 1.0
        # test ensures extrapolation of a DeltaVolSmile is possible, but it is a flat function
        fxv = FXDeltaVolSmile(
            eval_date=dt(2000, 1, 1),
            expiry=dt(2001, 1, 1),
            nodes=nodes,
            delta_type="spot",
        )
        result = fxv[1.025]
        assert result == fxv[1.0]


class TestFXDeltaVolSurface:
    def test_expiry_before_eval(self) -> None:
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2025, 1, 1)],
            node_values=[[11, 10, 12], [8, 7, 9]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
        )
        with pytest.raises(ValueError, match="`expiry` before the `eval_date` of"):
            fxvs.get_smile(dt(2022, 1, 1))

    def test_smile_0_no_interp(self) -> None:
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

    def test_smile_end_no_interp(self) -> None:
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

    def test_smile_tot_var_lin_interp(self) -> None:
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

    def test_smile_from_exact_expiry(self) -> None:
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

    def test_get_vol_from_strike(self) -> None:
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

    def test_get_vol_from_strike_raises(self) -> None:
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

    def test_set_node_vector(self) -> None:
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

    def test_expiries_unsorted(self) -> None:
        with pytest.raises(ValueError, match="Surface `expiries` are not sorted or"):
            FXDeltaVolSurface(
                delta_indexes=[0.25, 0.5, 0.75],
                expiries=[dt(2024, 1, 1), dt(2024, 1, 1)],
                node_values=[[19.590, 18.250, 18.967], [18.801, 17.677, 18.239]],
                eval_date=dt(2023, 1, 1),
                delta_type="forward",
            )

    def test_set_weights(self) -> None:
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
    def test_weights_get_vol(self, scalar) -> None:
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

    def test_weights_get_vol_clark(self) -> None:
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

    def test_cache_clear_and_defaults(self):
        fxvs = FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2025, 1, 1)],
            node_values=[[19.590, 18.250, 18.967], [18.801, 17.677, 18.239]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
        )
        fxvs.get_smile(dt(2024, 7, 1))
        assert dt(2024, 7, 1) in fxvs._cache

        fxvs._clear_cache()
        assert dt(2024, 7, 1) not in fxvs._cache

        with default_context("curve_caching", False):
            fxvs.get_smile(dt(2024, 7, 1))
            # no clear cache required, but value will re-calc anyway
            assert dt(2024, 7, 1) not in fxvs._cache


class TestFXSabrSmile:
    @pytest.mark.parametrize(
        ("strike", "vol"),
        [
            (1.2034, 19.49),
            (1.2050, 19.47),
            (1.3395, 18.31),  # f == k
            (1.3620, 18.25),
            (1.5410, 18.89),
            (1.5449, 18.93),
        ],
    )
    def test_sabr_vol(self, strike, vol):
        # test the SABR function using Clark 'FX Option Pricing' Table 3.7 as benchmark.
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="vol",
        )
        # F_0,T is stated in section 3.5.4 as 1.3395
        result = fxss.get_from_strike(strike, 1.3395)[1]
        assert abs(result - vol) < 1e-2

    @pytest.mark.parametrize(("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33)])
    def test_sabr_vol_finite_diff_first_order(self, k, f):
        # Test all of the first order gradients using finite diff, for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="vol",
            ad=2,
        )
        # F_0,T is stated in section 3.5.4 as 1.3395
        base = fxss.get_from_strike(Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []))[1]

        # k
        _up = fxss.get_from_strike(Dual2(k + 1e-5, ["k"], [], []), Dual2(f, ["f"], [], []))[1]
        _dw = fxss.get_from_strike(Dual2(k - 1e-5, ["k"], [], []), Dual2(f, ["f"], [], []))[1]
        assert abs((_up - _dw) / 2e-5 - gradient(base, ["k"])[0]) < 1e-5

        # f
        _up = fxss.get_from_strike(Dual2(k, ["k"], [], []), Dual2(f + 1e-5, ["f"], [], []))[1]
        _dw = fxss.get_from_strike(Dual2(k, ["k"], [], []), Dual2(f - 1e-5, ["f"], [], []))[1]
        assert abs((_up - _dw) / 2e-5 - gradient(base, ["f"])[0]) < 1e-5

        # SABR params
        for i, key in enumerate(["alpha", "rho", "nu"]):
            fxss.nodes[key] = fxss.nodes[key] + 1e-5
            _up = fxss.get_from_strike(Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []))[1]
            fxss.nodes[key] = fxss.nodes[key] - 2e-5
            _dw = fxss.get_from_strike(Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []))[1]
            fxss.nodes[key] = fxss.nodes[key] + 1e-5
            assert abs((_up - _dw) / 2e-5 - gradient(base, [f"vol{i}"])[0]) < 1e-5

    @pytest.mark.parametrize(
        ("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33), (1.3399, 1.34), (1.34, 1.3401)]
    )
    @pytest.mark.parametrize("pair", list(combinations(["k", "f", "alpha", "rho", "nu"], 2)))
    def test_sabr_vol_cross_finite_diff_second_order(self, k, f, pair):
        # Test all of the second order cross gradients using finite diff,
        # for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="v",
            ad=2,
        )

        a = fxss.nodes["alpha"]
        p = fxss.nodes["rho"]
        v = fxss.nodes["nu"]

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = fxss.get_from_strike(Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []))[1]

        def inc_(key1, key2, inc1, inc2):
            k_ = k
            f_ = f
            if key1 == "k":
                k_ = k + inc1
            elif key1 == "f":
                f_ = f + inc1
            else:
                fxss.nodes[key1] = fxss.nodes[key1] + inc1

            if key2 == "k":
                k_ = k + inc2
            elif key2 == "f":
                f_ = f + inc2
            else:
                fxss.nodes[key2] = fxss.nodes[key2] + inc2

            _ = fxss.get_from_strike(Dual2(k_, ["k"], [], []), Dual2(f_, ["f"], [], []))[1]

            fxss.nodes["alpha"] = a
            fxss.nodes["rho"] = p
            fxss.nodes["nu"] = v

            return _

        v_map = {"k": "k", "f": "f", "alpha": "v0", "rho": "v1", "nu": "v2"}

        upup = inc_(pair[0], pair[1], 1e-3, 1e-3)
        updown = inc_(pair[0], pair[1], 1e-3, -1e-3)
        downup = inc_(pair[0], pair[1], -1e-3, 1e-3)
        downdown = inc_(pair[0], pair[1], -1e-3, -1e-3)
        expected = (upup + downdown - updown - downup) / 4e-6
        result = gradient(base, [v_map[pair[0]], v_map[pair[1]]], order=2)[0][1]
        assert abs(result - expected) < 1e-2

    @pytest.mark.parametrize(
        ("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33), (1.3399, 1.34), (1.34, 1.3401)]
    )
    @pytest.mark.parametrize("var", ["k", "f", "alpha", "rho", "nu"])
    def test_sabr_vol_same_finite_diff_second_order(self, k, f, var):
        # Test all of the second order cross gradients using finite diff,
        # for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="v",
            ad=2,
        )

        a = fxss.nodes["alpha"]
        p = fxss.nodes["rho"]
        v = fxss.nodes["nu"]

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = fxss.get_from_strike(Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []))[1]

        def inc_(key1, inc1):
            k_ = k
            f_ = f
            if key1 == "k":
                k_ = k + inc1
            elif key1 == "f":
                f_ = f + inc1
            else:
                fxss.nodes[key1] = fxss.nodes[key1] + inc1

            _ = fxss.get_from_strike(Dual2(k_, ["k"], [], []), Dual2(f_, ["f"], [], []))[1]

            fxss.nodes["alpha"] = a
            fxss.nodes["rho"] = p
            fxss.nodes["nu"] = v

            return _

        v_map = {"k": "k", "f": "f", "alpha": "v0", "rho": "v1", "nu": "v2"}

        up = inc_(var, 1e-4)
        down = inc_(var, -1e-4)
        expected = (up + down - 2 * base) / 1e-8
        result = gradient(base, [v_map[var]], order=2)[0][0]
        assert abs(result - expected) < 5e-3

    def test_sabr_vol_root_multi_duals_neighbourhood(self):
        # test the SABR function when regular arithmetic operations produce an undefined 0/0
        # value so AD has to be hard coded into the solution. This occurs when f == k.
        # test by comparing derivatives with those captured at a nearby valid point
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="vol",
            ad=2,
        )
        # F_0,T is stated in section 3.5.4 as 1.3395
        base = fxss.get_from_strike(Dual2(1.34, ["k"], [], []), Dual2(1.34, ["f"], [], []))[1]
        comparison1 = fxss.get_from_strike(Dual2(1.341, ["k"], [], []), Dual2(1.34, ["f"], [], []))[
            1
        ]

        assert np.all(abs(base.dual - comparison1.dual) < 1e-1)
        diff = base.dual2 - comparison1.dual2
        dual2 = abs(diff) < 5e-1
        assert np.all(dual2)

    @pytest.mark.parametrize("param", ["alpha", "beta", "rho", "nu"])
    def test_missing_param_raises(self, param):
        nodes = {
            "alpha": 0.17431060,
            "beta": 1.0,
            "rho": -0.11268306,
            "nu": 0.81694072,
        }
        nodes.pop(param)
        with pytest.raises(ValueError):
            FXSabrSmile(
                nodes=nodes,
                eval_date=dt(2001, 1, 1),
                expiry=dt(2002, 1, 1),
                id="vol",
            )

    def test_non_iterable(self):
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="vol",
        )
        with pytest.raises(TypeError):
            _ = list(fxss)

    def test_update_node_raises(self):
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="vol",
        )
        with pytest.raises(KeyError, match="`key` is not in ``nodes``."):
            fxss.update_node("bananas", 12.0)

    def test_set_ad_order_raises(self):
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="vol",
        )
        with pytest.raises(ValueError, match="`order` can only be in {0, 1, 2} "):
            fxss._set_ad_order(12)

    def test_get_node_vars_and_vector(self):
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.20,
                "beta": 1.0,
                "rho": -0.10,
                "nu": 0.80,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="myid",
        )
        result = fxss._get_node_vars()
        expected = ("myid0", "myid1", "myid2")
        assert result == expected

        result = fxss._get_node_vector()
        expected = np.array([0.20, -0.1, 0.80])
        assert np.all(result == expected)

    def test_get_from_strike_expiry_raises(self):
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.20,
                "beta": 1.0,
                "rho": -0.10,
                "nu": 0.80,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="vol",
            ad=2,
        )
        with pytest.raises(ValueError, match="`expiry` of VolSmile and OptionPeriod do not match"):
            fxss.get_from_strike(1.0, 1.0, 1.0, 1.0, dt(1999, 1, 1))

    @pytest.mark.parametrize("k", [1.2034, 1.2050, 1.3620, 1.5410, 1.5449])
    def test_get_from_strike_ad_2(self, fxfo, k) -> None:
        # Use finite diff to validate the 2nd order AD of the SABR function in alpha and rho.
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.20,
                "beta": 1.0,
                "rho": -0.10,
                "nu": 0.80,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="vol",
            ad=2,
        )
        fxfo._set_ad_order(2)
        args = (
            k,
            fxfo.rate("eurusd", dt(2023, 6, 20)),
        )
        pv00 = fxss.get_from_strike(*args)

        fxss.update_node("alpha", 0.20 + 0.00001)
        fxss.update_node("rho", -0.10 + 0.00001)
        pv11 = fxss.get_from_strike(*args)

        fxss.update_node("alpha", 0.20 + 0.00001)
        fxss.update_node("rho", -0.10 - 0.00001)
        pv1_1 = fxss.get_from_strike(*args)

        fxss.update_node("alpha", 0.20 - 0.00001)
        fxss.update_node("rho", -0.10 - 0.00001)
        pv_1_1 = fxss.get_from_strike(*args)

        fxss.update_node("alpha", 0.20 - 0.00001)
        fxss.update_node("rho", -0.10 + 0.00001)
        pv_11 = fxss.get_from_strike(*args)

        finite_diff = (pv11[1] + pv_1_1[1] - pv1_1[1] - pv_11[1]) * 1e10 / 4.0
        ad_grad = gradient(pv00[1], ["vol0", "vol1"], 2)[0, 1]

        assert abs(finite_diff - ad_grad) < 1e-4

    @pytest.mark.parametrize("p", [-0.1, 0.15])
    @pytest.mark.parametrize("a", [0.05, 0.2])
    @pytest.mark.parametrize("k_", [1.15, 1.3620, 1.45, 1.3395])
    def test_sabr_derivative(self, a, p, k_):
        # test the analytic derivative of the SABR function with respect to k created by sympy
        b = 1.0
        v = 0.8
        f = 1.3395
        t = 1.0
        k = Dual(k_, ["k"], [1.0])

        sabr_vol, result = _d_sabr_d_k(k, f, t, a, b, p, v)
        expected = gradient(sabr_vol, ["k"])[0]

        assert abs(result - expected) < 1e-13

    @pytest.mark.parametrize(("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33)])
    def test_sabr_derivative_finite_diff_first_order(self, k, f):
        # Test all of the first order gradients using finite diff, for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="vol",
            ad=2,
        )
        t = dt(2002, 1, 1)
        base = fxss._d_sabr_d_k(Dual2(k, ["k"], [1.0], []), Dual2(f, ["f"], [1.0], []), t)[1]

        # k
        _up = fxss._d_sabr_d_k(Dual2(k + 1e-4, ["k"], [], []), Dual2(f, ["f"], [], []), t)[1]
        _dw = fxss._d_sabr_d_k(Dual2(k - 1e-4, ["k"], [], []), Dual2(f, ["f"], [], []), t)[1]
        result = gradient(base, ["k"])[0]
        expected = (_up - _dw) / 2e-4
        assert abs(result - expected) < 1e-5

        # f
        _up = fxss._d_sabr_d_k(Dual2(k, ["k"], [], []), Dual2(f + 1e-4, ["f"], [], []), t)[1]
        _dw = fxss._d_sabr_d_k(Dual2(k, ["k"], [], []), Dual2(f - 1e-4, ["f"], [], []), t)[1]
        result = gradient(base, ["f"])[0]
        expected = (_up - _dw) / 2e-4
        assert abs(result - expected) < 1e-5

        # SABR params
        for i, key in enumerate(["alpha", "rho", "nu"]):
            fxss.nodes[key] = fxss.nodes[key] + 1e-5
            _up = fxss._d_sabr_d_k(Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []), t)[1]
            fxss.nodes[key] = fxss.nodes[key] - 2e-5
            _dw = fxss._d_sabr_d_k(Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []), t)[1]
            fxss.nodes[key] = fxss.nodes[key] + 1e-5
            result = gradient(base, [f"vol{i}"])[0]
            expected = (_up - _dw) / 2e-5
            assert abs(result - expected) < 1e-5

    @pytest.mark.parametrize(
        ("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33), (1.3395, 1.34), (1.34, 1.3405)]
    )
    @pytest.mark.parametrize("pair", list(combinations(["k", "f", "alpha", "rho", "nu"], 2)))
    def test_sabr_derivative_cross_finite_diff_second_order(self, k, f, pair):
        # Test all of the second order cross gradients using finite diff,
        # for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="v",
            ad=2,
        )

        a = fxss.nodes["alpha"]
        p = fxss.nodes["rho"]
        v = fxss.nodes["nu"]

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = fxss._d_sabr_d_k(Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []), dt(2002, 1, 1))[1]

        def inc_(key1, key2, inc1, inc2):
            k_ = k
            f_ = f
            if key1 == "k":
                k_ = k + inc1
            elif key1 == "f":
                f_ = f + inc1
            else:
                fxss.nodes[key1] = fxss.nodes[key1] + inc1

            if key2 == "k":
                k_ = k + inc2
            elif key2 == "f":
                f_ = f + inc2
            else:
                fxss.nodes[key2] = fxss.nodes[key2] + inc2

            _ = fxss._d_sabr_d_k(Dual2(k_, ["k"], [], []), Dual2(f_, ["f"], [], []), dt(2002, 1, 1))[1]

            fxss.nodes["alpha"] = a
            fxss.nodes["rho"] = p
            fxss.nodes["nu"] = v

            return _

        v_map = {"k": "k", "f": "f", "alpha": "v0", "rho": "v1", "nu": "v2"}

        upup = inc_(pair[0], pair[1], 1e-3, 1e-3)
        updown = inc_(pair[0], pair[1], 1e-3, -1e-3)
        downup = inc_(pair[0], pair[1], -1e-3, 1e-3)
        downdown = inc_(pair[0], pair[1], -1e-3, -1e-3)
        expected = (upup + downdown - updown - downup) / 4e-6
        result = gradient(base, [v_map[pair[0]], v_map[pair[1]]], order=2)[0][1]
        assert abs(result - expected) < 5e-3

    @pytest.mark.parametrize(
        ("k", "f"),
        [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33), (1.3395, 1.34), (1.34, 1.3405)],
    )
    @pytest.mark.parametrize("var", ["k", "f", "alpha", "rho", "nu"])
    def test_sabr_derivative_same_finite_diff_second_order(self, k, f, var):
        # Test all of the second order cross gradients using finite diff,
        # for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="v",
            ad=2,
        )

        a = fxss.nodes["alpha"]
        p = fxss.nodes["rho"]
        v = fxss.nodes["nu"]

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = fxss._d_sabr_d_k(Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []), dt(2002, 1, 1))[1]

        def inc_(key1, inc1):
            k_ = k
            f_ = f
            if key1 == "k":
                k_ = k + inc1
            elif key1 == "f":
                f_ = f + inc1
            else:
                fxss.nodes[key1] = fxss.nodes[key1] + inc1

            _ = fxss._d_sabr_d_k(Dual2(k_, ["k"], [], []), Dual2(f_, ["f"], [], []), dt(2002, 1, 1))[1]

            fxss.nodes["alpha"] = a
            fxss.nodes["rho"] = p
            fxss.nodes["nu"] = v

            return _

        v_map = {"k": "k", "f": "f", "alpha": "v0", "rho": "v1", "nu": "v2"}

        up = inc_(var, 1e-3)
        down = inc_(var, -1e-3)
        expected = (up + down - 2 * base) / 1e-6
        result = gradient(base, [v_map[var]], order=2)[0][0]
        assert abs(result - expected) < 3e-3

    def test_sabr_derivative_root_multi_duals_neighbourhood(self):
        # test the SABR function when regular arithmetic operations produce an undefined 0/0
        # value so AD has to be hard coded into the solution. This occurs when f == k.
        # test by comparing derivatives with those captured at a nearby valid point
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="vol",
            ad=2,
        )
        # F_0,T is stated in section 3.5.4 as 1.3395
        base = fxss._d_sabr_d_k(Dual2(1.34, ["k"], [], []), Dual2(1.34, ["f"], [], []), dt(2002, 1, 1))[1]
        comparison1 = fxss._d_sabr_d_k(
            Dual2(1.341, ["k"], [], []), Dual2(1.34, ["f"], [], []), dt(2002, 1, 1)
        )[1]

        assert np.all(abs(base.dual - comparison1.dual) < 5e-3)
        diff = base.dual2 - comparison1.dual2
        dual2 = abs(diff) < 3e-2
        assert np.all(dual2)

    def test_sabr_derivative_ad(self):
        # Test is probably superceded by test_sabr_derivative_same/cross_finite_diff

        # test the analytic derivative of the SABR function and its preservation of AD.
        a = 0.10
        b = 1.0
        p = Dual2(-0.20, ["p"], [1.0], [0.0])
        v = 0.8
        f = 1.3395
        t = 1.0
        k = Dual2(1.45, ["k"], [1.0], [0.0])

        _, result = _d_sabr_d_k(k, f, t, a, b, p, v)
        _, r1 = _d_sabr_d_k(k, f, t, a, b, p + 1e-4, v)
        _, r_1 = _d_sabr_d_k(k, f, t, a, b, p - 1e-4, v)
        expected = (r1 - r_1) / (2e-4)
        result = gradient(result, ["p"])[0]
        assert abs(result - expected) < 1e-9

        _, result = _d_sabr_d_k(k, f, t, a, b, p, v)
        _, r1 = _d_sabr_d_k(k, f, t, a, b, p + 1e-4, v)
        _, r_1 = _d_sabr_d_k(k, f, t, a, b, p - 1e-4, v)
        expected = (r1 - 2 * result + r_1) / (1e-8)
        result = gradient(result, ["p"], order=2)[0][0]
        assert abs(result - expected) < 1e-8

    def test_sabr_derivative_root(self):
        # Test is probably superceded by test_sabr_derivative_same/cross_finite_diff

        # test the analytic derivative of the SABR function when f == k
        a = 0.10
        b = 1.0
        p = -0.20
        v = 0.8
        f = 1.3395
        t = 1.0
        k = Dual(1.3395, ["k"], [1.0])

        sabr_vol, result = _d_sabr_d_k(k, f, t, a, b, p, v)
        expected = gradient(sabr_vol, ["k"])[0]

        assert abs(result - expected) < 1e-13

    def test_sabr_derivative_root_ad(self):
        # Test is probably superceded by test_sabr_derivative_same/cross_finite_diff

        # test the analytic derivative of the SABR function when f == k, and its preservation of AD.
        a = 0.10
        b = 1.0
        p = Dual2(-0.20, ["p"], [1.0], [0.0])
        v = 0.8
        f = 1.3395
        t = 1.0
        k = Dual2(1.3395, ["k"], [1.0], [0.0])

        _, result = _d_sabr_d_k(k, f, t, a, b, p, v)
        _, r1 = _d_sabr_d_k(k, f, t, a, b, p + 1e-4, v)
        _, r_1 = _d_sabr_d_k(k, f, t, a, b, p - 1e-4, v)
        expected = (r1 - r_1) / (2e-4)
        result = gradient(result, ["p"])[0]
        assert abs(result - expected) < 1e-9

        _, result = _d_sabr_d_k(k, f, t, a, b, p, v)
        _, r1 = _d_sabr_d_k(k, f, t, a, b, p + 1e-4, v)
        _, r_1 = _d_sabr_d_k(k, f, t, a, b, p - 1e-4, v)
        expected = (r1 - 2 * result + r_1) / (1e-8)
        result = gradient(result, ["p"], order=2)[0][0]
        assert abs(result - expected) < 1e-8

    def test_f_with_fxforwards(self, fxfo):
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 4, 16),
            id="v",
            ad=1,
            pair="eurusd",
            calendar="tgt|fed",
        )
        result = fxss.get_from_strike(1.02, fxfo)[1]
        expected = 17.803563
        assert abs(result - expected) < 1e-6

    def test_f_with_fxrates_raises(self, fxfo):
        fxss = FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 4, 16),
            id="v",
            ad=1,
            pair="eurusd",
            calendar="tgt|fed",
        )
        with pytest.raises(ValueError):
            fxss.get_from_strike(1.02, FXRates({"eurusd": 1.06}))


class TestStateAndCache:
    @pytest.mark.parametrize(
        "curve",
        [
            FXDeltaVolSmile(
                nodes={0.25: 10.0, 0.5: 10.0, 0.75: 11.0},
                delta_type="forward",
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                id="vol",
            ),
            FXSabrSmile(
                nodes={
                    "alpha": 0.17431060,
                    "beta": 1.0,
                    "rho": -0.11268306,
                    "nu": 0.81694072,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                id="vol",
            ),
        ],
    )
    @pytest.mark.parametrize(("method", "args"), [("_set_ad_order", (1,))])
    def test_method_does_not_change_state(self, curve, method, args):
        before = curve._state
        getattr(curve, method)(*args)
        after = curve._state
        assert before == after

    @pytest.mark.parametrize(
        "curve",
        [
            FXDeltaVolSmile(
                nodes={0.25: 10.0, 0.5: 10.0, 0.75: 11.0},
                delta_type="forward",
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                id="vol",
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98, 0.99], 1)),
            ("update_node", (0.25, 0.98)),
            ("update", ({0.25: 10.0, 0.5: 10.0, 0.75: 10.1},)),
            ("csolve", tuple()),
        ],
    )
    def test_method_changes_state(self, curve, method, args):
        before = curve._state
        getattr(curve, method)(*args)
        after = curve._state
        assert before != after

    @pytest.mark.parametrize(
        "curve",
        [
            FXSabrSmile(
                nodes={
                    "alpha": 0.17431060,
                    "beta": 1.0,
                    "rho": -0.11268306,
                    "nu": 0.81694072,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                id="vol",
            )
        ],
    )
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98, 0.99], 1)),
            ("update_node", ("alpha", 0.98)),
        ],
    )
    def test_method_changes_state_sabr(self, curve, method, args):
        before = curve._state
        getattr(curve, method)(*args)
        after = curve._state
        assert before != after

    def test_populate_cache(self):
        # objects have yet to implement cache handling
        pass

    def test_method_clears_cache(self):
        # objects have yet to implement cache handling
        pass

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98], 1)),
            ("_set_ad_order", (2,)),
        ],
    )
    def test_surface_clear_cache(self, method, args):
        surf = FXDeltaVolSurface(
            expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
            delta_indexes=[0.5],
            node_values=[[10.0], [9.0]],
            eval_date=dt(1999, 1, 1),
            delta_type="forward",
        )
        surf.get_smile(dt(2000, 3, 1))
        assert dt(2000, 3, 1) in surf._cache

        getattr(surf, method)(*args)
        assert len(surf._cache) == 0

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("get_from_strike", (1.0, 1.0, NoInput(0), NoInput(0), dt(2000, 5, 3))),
            ("_get_index", (0.9, dt(2000, 5, 3))),
            ("get_smile", (dt(2000, 5, 3),)),
        ],
    )
    def test_surface_populate_cache(self, method, args):
        surf = FXDeltaVolSurface(
            expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
            delta_indexes=[0.5],
            node_values=[[10.0], [9.0]],
            eval_date=dt(1999, 1, 1),
            delta_type="forward",
        )
        before = surf._cache_len
        getattr(surf, method)(*args)
        assert surf._cache_len == before + 1

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98], 1)),
        ],
    )
    def test_surface_change_state(self, method, args):
        surf = FXDeltaVolSurface(
            expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
            delta_indexes=[0.5],
            node_values=[[10.0], [9.0]],
            eval_date=dt(1999, 1, 1),
            delta_type="forward",
        )
        pre_state = surf._state
        getattr(surf, method)(*args)
        assert surf._state != pre_state

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_ad_order", (2,)),
        ],
    )
    def test_surface_maintain_state(self, method, args):
        surf = FXDeltaVolSurface(
            expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
            delta_indexes=[0.5],
            node_values=[[10.0], [9.0]],
            eval_date=dt(1999, 1, 1),
            delta_type="forward",
        )
        pre_state = surf._state
        getattr(surf, method)(*args)
        assert surf._state == pre_state

    def test_surface_validate_states(self):
        # test the get_smile method validates the states after a mutation
        surf = FXDeltaVolSurface(
            expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
            delta_indexes=[0.5],
            node_values=[[10.0], [9.0]],
            eval_date=dt(1999, 1, 1),
            delta_type="forward",
        )
        pre_state = surf._state
        surf.smiles[0].update_node(0.5, 11.0)
        surf.get_smile(dt(2000, 1, 9))
        post_state = surf._state
        assert pre_state != post_state  # validate states has been run and updated the state.

    @pytest.mark.parametrize(
        "smile",
        [
            FXDeltaVolSmile(
                nodes={0.25: 10.0, 0.5: 10.0, 0.75: 11.0},
                delta_type="forward",
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                id="vol",
            ),
            FXSabrSmile(
                nodes={
                    "alpha": 0.17431060,
                    "beta": 1.0,
                    "rho": -0.11268306,
                    "nu": 0.81694072,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                id="vol",
            ),
        ],
    )
    def test_initialisation_state_smile(self, smile):
        assert smile._state != 0

    def test_initialisation_state_surface(self):
        surf = FXDeltaVolSurface(
            expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
            delta_indexes=[0.5],
            node_values=[[10.0], [9.0]],
            eval_date=dt(1999, 1, 1),
            delta_type="forward",
        )
        assert surf._state != 0


def test_validate_delta_type() -> None:
    with pytest.raises(ValueError, match="`delta_type` must be in"):
        _validate_delta_type("BAD_TYPE")
