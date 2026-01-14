# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################

from datetime import datetime as dt
from itertools import combinations

import numpy as np
import pytest
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal
from rateslib import default_context
from rateslib.curves import CompositeCurve, Curve, LineCurve
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, Variable, gradient
from rateslib.enums.parameters import _get_fx_delta_type
from rateslib.fx import (
    FXForwards,
    FXRates,
    forward_fx,
)
from rateslib.fx_volatility import (
    FXDeltaVolSmile,
    FXDeltaVolSurface,
    FXSabrSmile,
    FXSabrSurface,
)
from rateslib.fx_volatility.utils import (
    _d_sabr_d_k_or_f,
    _FXSabrSmileNodes,
)
from rateslib.periods import FXCallPeriod
from rateslib.scheduling import get_calendar


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
            z_w=fxfo.curve("eur", "usd")[dt(2023, 6, 20)]
            / fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
        )
        call_vol = fxvs.get_from_strike(
            k=k,
            f=fxfo.rate("eurusd", dt(2023, 6, 20)),
            z_w=fxfo.curve("eur", "usd")[dt(2023, 6, 20)]
            / fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
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
        kwargs = dict(
            k=k,
            f=fxfo.rate("eurusd", dt(2023, 6, 20)),
            z_w=fxfo.curve("eur", "usd")[dt(2023, 6, 20)]
            / fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
        )
        put_vol = fxvs.get_from_strike(**kwargs)

        fxvs.update_node(idx, Dual(val + 0.0000001, [var], []))
        put_vol_plus = fxvs.get_from_strike(**kwargs)

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
        kwargs = dict(
            k=k,
            f=fxfo.rate("eurusd", dt(2023, 6, 20)),
            z_w=fxfo.curve("eur", "usd")[dt(2023, 6, 20)]
            / fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
        )
        pv00 = fxvs.get_from_strike(**kwargs)

        fxvs.update_node(cross[0][2], Dual2(cross[0][1] + 0.00001, [cross[0][0]], [], []))
        fxvs.update_node(cross[1][2], Dual2(cross[1][1] + 0.00001, [cross[1][0]], [], []))
        pv11 = fxvs.get_from_strike(**kwargs)

        fxvs.update_node(cross[0][2], Dual2(cross[0][1] + 0.00001, [cross[0][0]], [], []))
        fxvs.update_node(cross[1][2], Dual2(cross[1][1] - 0.00001, [cross[1][0]], [], []))
        pv1_1 = fxvs.get_from_strike(**kwargs)

        fxvs.update_node(cross[0][2], Dual2(cross[0][1] - 0.00001, [cross[0][0]], [], []))
        fxvs.update_node(cross[1][2], Dual2(cross[1][1] - 0.00001, [cross[1][0]], [], []))
        pv_1_1 = fxvs.get_from_strike(**kwargs)

        fxvs.update_node(cross[0][2], Dual2(cross[0][1] - 0.00001, [cross[0][0]], [], []))
        fxvs.update_node(cross[1][2], Dual2(cross[1][1] + 0.00001, [cross[1][0]], [], []))
        pv_11 = fxvs.get_from_strike(**kwargs)

        finite_diff = (pv11[1] + pv_1_1[1] - pv1_1[1] - pv_11[1]) * 1e10 / 4.0
        ad_grad = gradient(pv00[1], [cross[0][0], cross[1][0]], 2)[0, 1]

        assert abs(finite_diff - ad_grad) < 5e-5

    def test_get_from_unsimilar_delta(self) -> None:
        fxvs = FXDeltaVolSmile(
            nodes={0.25: 10.0, 0.5: 10.0, 0.75: 11.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
        )
        result = fxvs.get(0.65, "spot_pa", 1.0, 0.99 / 0.999)
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
        result = fxvs.get(0.5, delta_type, 1.0, 0.99 / 0.991)
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize(
        ("delta_type", "exp"), [("spot_pa", 10.000085036853598), ("forward_pa", 10.0)]
    )
    def test_get_from_similar_delta_pa(self, delta_type, exp) -> None:
        fxvs = FXDeltaVolSmile(
            nodes={0.25: 11.0, 0.5: 10.0, 0.75: 11.0},
            delta_type="forward_pa",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
        )
        result = fxvs.get(-0.5, delta_type, -1.0, 0.99 / 0.991)
        assert abs(result - exp) < 1e-6

    def test_get_from_unsimilar_delta2(self):
        # GH 730
        fdvs = FXDeltaVolSmile(
            nodes={
                0.1: 5,
                0.25: 4,
                0.5: 3,
                0.75: 4,
                0.9: 5,
            },
            expiry=dt(2025, 5, 10),
            eval_date=dt(2025, 4, 10),
            delta_type="forward",
        )
        result = fdvs.get(delta=0.1, delta_type="forward_pa", phi=1, z_w=1.0)
        expected = 4.995304045589985
        assert abs(result - expected) < 1e-9

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
        assert fxvs.nodes.nodes[0.25] == Dual(10.0, ["vol0"], [])

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
        with pytest.raises(TypeError, match="`Smile` types are not iterable."):
            fxvs.__iter__()

    def test_update_node(self):
        fxvs = FXDeltaVolSmile(
            nodes={0.5: 1.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
        )
        with pytest.raises(KeyError, match=r"`key`: '0.4' is not in Curve ``nodes``"):
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

    def test_update_csolve(self):
        import rateslib

        anchor = rateslib.dt(2025, 5, 22)
        expiry = rateslib.dt(2025, 6, 24)

        test_smile = rateslib.FXDeltaVolSmile(
            nodes={
                0.1: 5,
                0.25: 4,
                0.5: 3,
                0.75: 4,
                0.9: 5,
            },
            expiry=expiry,
            eval_date=anchor,
            delta_type="forward",
            id="test_vol",
        )

        prior_c = test_smile.nodes.spline.spline.c
        # update node
        nodes_bump = {k: v + 0.5 for k, v in test_smile.nodes.nodes.items()}
        test_smile.update(nodes_bump)
        after_c = test_smile.nodes.spline.spline.c

        assert after_c != prior_c

    def test_flat_smile_with_zero_delta_index_input(self):
        smile = FXDeltaVolSmile(
            nodes={0.0: 10.0},
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            id="vol",
            expiry=dt(2023, 6, 16),
        )
        assert abs(smile[0.5] - 10.0) < 1e-14


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
        assert result.meta.expiry == expected.meta.expiry
        assert result.meta.delta_type == expected.meta.delta_type
        assert result.meta.eval_date == expected.meta.eval_date

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
        assert result.meta.expiry == expected.meta.expiry
        assert result.meta.delta_type == expected.meta.delta_type
        assert result.meta.eval_date == expected.meta.eval_date

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
        for v1, v2 in zip(result.nodes.values, expected.nodes.values):
            assert abs(v1 - v2) < 0.0001
        assert result.meta.expiry == expected.meta.expiry
        assert result.meta.delta_type == expected.meta.delta_type
        assert result.meta.eval_date == expected.meta.eval_date

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
        for v1, v2 in zip(result.nodes.values, expected.nodes.values):
            assert abs(v1 - v2) < 0.0001
        assert result.meta.expiry == expected.meta.expiry
        assert result.meta.delta_type == expected.meta.delta_type
        assert result.meta.eval_date == expected.meta.eval_date
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
        result = fxvs.get_from_strike(k=1.05, f=1.03, z_w=0.99 / 0.999, expiry=dt(2024, 7, 1))[1]
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
            fxvs.get_from_strike(k=1.05, f=1.03, z_w=0.99 / 0.999)

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
        for v1, v2 in zip(vec[:3], fxvs.smiles[0].nodes.values):
            assert abs(v1 - v2) < 1e-10
        for v1, v2 in zip(vec[3:], fxvs.smiles[1].nodes.values):
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
        assert fxvs.meta.weights.loc[dt(2023, 12, 15)] == 1.0
        assert fxvs.meta.weights.loc[dt(2024, 1, 4)] == 0.9393939393939394
        assert fxvs.meta.weights.loc[dt(2024, 1, 5)] == 1.878787878787879
        assert fxvs.meta.weights.loc[dt(2024, 2, 2)] == 0.9666666666666667
        assert fxvs.meta.weights.loc[dt(2024, 2, 5)] == 1.9333333333333333
        assert fxvs.meta.weights.loc[dt(2027, 12, 15)] == 1.0

        # test that the sum of weights to each expiry node is as expected.
        for e in fxvs.meta.expiries:
            assert (
                abs(
                    fxvs.meta.weights[fxvs.meta.eval_date : e].sum()
                    - (e - fxvs.meta.eval_date).days
                )
                < 1e-13
            )

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
        kwargs = dict(k=1.03, f=1.03, z_w=0.99 / 0.999, expiry=dt(2023, 2, 3))
        result = fxvs.get_from_strike(**kwargs)
        result2 = fxvs_weights.get_from_strike(**kwargs)
        w = fxvs_weights.meta.weights

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
            assert abs(smile.nodes.nodes[0.5] - expected[i]) < 5e-3

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

    @pytest.mark.parametrize("smile_expiry", [dt(2026, 5, 1), dt(2026, 6, 9), dt(2026, 7, 1)])
    def test_flat_surface_and_get_smile_one_expiry(self, smile_expiry):
        # gh 911
        anchor = dt(2025, 6, 9)
        expiry = dt(2026, 6, 9)

        surf = FXDeltaVolSurface(
            eval_date=anchor,
            expiries=[expiry],
            delta_indexes=[0.5],
            node_values=[[10]],
            delta_type="forward",
        )

        smile = surf.get_smile(smile_expiry)
        assert abs(smile[0.3] - 10.0) < 1e-13


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

        a = fxss.nodes.alpha
        p = fxss.nodes.rho
        v = fxss.nodes.nu

        def inc_(key1, inc1):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1

            fxss._nodes = _FXSabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = (
                fxss._d_sabr_d_k_or_f(
                    Dual2(in_["k"], ["k"], [], []),
                    Dual2(in_["f"], ["f"], [], []),
                    dt(2002, 1, 1),
                    False,
                    1,
                )[0]
                * 100.0
            )

            # reset
            fxss._nodes = _FXSabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
            return _

        for key in ["k", "f", "alpha", "rho", "nu"]:
            map_ = {"k": "k", "f": "f", "alpha": "vol0", "rho": "vol1", "nu": "vol2"}
            up_ = inc_(key, 1e-5)
            dw_ = inc_(key, -1e-5)
            assert abs((up_ - dw_) / 2e-5 - gradient(base, [map_[key]])[0]) < 1e-5

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

        a = fxss.nodes.alpha
        p = fxss.nodes.rho
        v = fxss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = fxss.get_from_strike(Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []))[1]

        def inc_(key1, key2, inc1, inc2):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1
            in_[key2] += inc2

            fxss._nodes = _FXSabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = (
                fxss._d_sabr_d_k_or_f(
                    Dual2(in_["k"], ["k"], [], []),
                    Dual2(in_["f"], ["f"], [], []),
                    dt(2002, 1, 1),
                    False,
                    1,
                )[0]
                * 100.0
            )

            # reset
            fxss._nodes = _FXSabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
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

        a = fxss.nodes.alpha
        p = fxss.nodes.rho
        v = fxss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = fxss.get_from_strike(Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []))[1]

        def inc_(key1, inc1):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1

            fxss._nodes = _FXSabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = (
                fxss._d_sabr_d_k_or_f(
                    Dual2(in_["k"], ["k"], [], []),
                    Dual2(in_["f"], ["f"], [], []),
                    dt(2002, 1, 1),
                    False,
                    1,
                )[0]
                * 100.0
            )

            # reset
            fxss._nodes = _FXSabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
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
            fxss.get_from_strike(k=1.0, f=1.0, z_w=1.0, expiry=(1999, 1, 1))

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

        sabr_vol, result = _d_sabr_d_k_or_f(k, f, t, a, b, p, v, 1)
        expected = gradient(sabr_vol, ["k"])[0]

        assert abs(result - expected) < 1e-13

    @pytest.mark.parametrize("p", [-0.1, 0.15])
    @pytest.mark.parametrize("a", [0.05, 0.2])
    @pytest.mark.parametrize("f_", [1.15, 1.3620, 1.45, 1.3395])
    def test_sabr_derivative_f(self, a, p, f_):
        # test the analytic derivative of the SABR function with respect to f created by sympy
        # tests the regular case as well as the limit z->0 where a separate AD calculation o
        # is branched.
        b = 1.0
        v = 0.8
        k = 1.3395
        t = 1.0
        f = Dual(f_, ["f"], [1.0])

        sabr_vol, result = _d_sabr_d_k_or_f(k, f, t, a, b, p, v, 2)
        expected = gradient(sabr_vol, ["f"])[0]

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
        base = fxss._d_sabr_d_k_or_f(
            Dual2(k, ["k"], [1.0], []), Dual2(f, ["f"], [1.0], []), t, False, 1
        )[1]

        a = fxss.nodes.alpha
        p = fxss.nodes.rho
        v = fxss.nodes.nu

        def inc_(key1, inc1):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1

            fxss._nodes = _FXSabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = fxss._d_sabr_d_k_or_f(
                Dual2(in_["k"], ["k"], [], []),
                Dual2(in_["f"], ["f"], [], []),
                dt(2002, 1, 1),
                False,
                1,
            )[1]

            # reset
            fxss._nodes = _FXSabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
            return _

        for key in ["k", "f", "alpha", "rho", "nu"]:
            map_ = {"k": "k", "f": "f", "alpha": "vol0", "rho": "vol1", "nu": "vol2"}
            up_ = inc_(key, 1e-5)
            dw_ = inc_(key, -1e-5)
            assert abs((up_ - dw_) / 2e-5 - gradient(base, [map_[key]])[0]) < 2e-3

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

        a = fxss.nodes.alpha
        p = fxss.nodes.rho
        v = fxss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = fxss._d_sabr_d_k_or_f(
            Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []), dt(2002, 1, 1), False, 1
        )[1]

        def inc_(key1, key2, inc1, inc2):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1
            in_[key2] += inc2

            fxss._nodes = _FXSabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = fxss._d_sabr_d_k_or_f(
                Dual2(in_["k"], ["k"], [], []),
                Dual2(in_["f"], ["f"], [], []),
                dt(2002, 1, 1),
                False,
                1,
            )[1]

            # reset
            fxss._nodes = _FXSabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
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

        a = fxss.nodes.alpha
        p = fxss.nodes.rho
        v = fxss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = fxss._d_sabr_d_k_or_f(
            Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []), dt(2002, 1, 1), False, 1
        )[1]

        def inc_(key1, inc1):
            k_ = k
            f_ = f
            if key1 == "k":
                k_ = k + inc1
            elif key1 == "f":
                f_ = f + inc1
            else:
                fxss.update_node(key1, getattr(fxss.nodes, key1) + inc1)
                # fxss.nodes[key1] = fxss.nodes[key1] + inc1

            _ = fxss._d_sabr_d_k_or_f(
                Dual2(k_, ["k"], [], []), Dual2(f_, ["f"], [], []), dt(2002, 1, 1), False, 1
            )[1]

            fxss._nodes = _FXSabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
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
        base = fxss._d_sabr_d_k_or_f(
            Dual2(1.34, ["k"], [], []), Dual2(1.34, ["f"], [], []), dt(2002, 1, 1), False, 1
        )[1]
        comparison1 = fxss._d_sabr_d_k_or_f(
            Dual2(1.341, ["k"], [], []), Dual2(1.34, ["f"], [], []), dt(2002, 1, 1), False, 1
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

        _, result = _d_sabr_d_k_or_f(k, f, t, a, b, p, v, 1)
        _, r1 = _d_sabr_d_k_or_f(k, f, t, a, b, p + 1e-4, v, 1)
        _, r_1 = _d_sabr_d_k_or_f(k, f, t, a, b, p - 1e-4, v, 1)
        expected = (r1 - r_1) / (2e-4)
        result = gradient(result, ["p"])[0]
        assert abs(result - expected) < 1e-9

        _, result = _d_sabr_d_k_or_f(k, f, t, a, b, p, v, 1)
        _, r1 = _d_sabr_d_k_or_f(k, f, t, a, b, p + 1e-4, v, 1)
        _, r_1 = _d_sabr_d_k_or_f(k, f, t, a, b, p - 1e-4, v, 1)
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

        sabr_vol, result = _d_sabr_d_k_or_f(k, f, t, a, b, p, v, 1)
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

        _, result = _d_sabr_d_k_or_f(k, f, t, a, b, p, v, 1)
        _, r1 = _d_sabr_d_k_or_f(k, f, t, a, b, p + 1e-4, v, 1)
        _, r_1 = _d_sabr_d_k_or_f(k, f, t, a, b, p - 1e-4, v, 1)
        expected = (r1 - r_1) / (2e-4)
        result = gradient(result, ["p"])[0]
        assert abs(result - expected) < 1e-9

        _, result = _d_sabr_d_k_or_f(k, f, t, a, b, p, v, 1)
        _, r1 = _d_sabr_d_k_or_f(k, f, t, a, b, p + 1e-4, v, 1)
        _, r_1 = _d_sabr_d_k_or_f(k, f, t, a, b, p - 1e-4, v, 1)
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

    def test_plot_domain(self):
        ss = FXSabrSmile(
            eval_date=dt(2024, 5, 28),
            expiry=dt(2054, 5, 28),
            nodes={"alpha": 0.02, "beta": 1.0, "rho": 0.01, "nu": 0.05},
        )
        ax, fig, lines = ss.plot(f=1.60)
        assert abs(lines[0]._x[0] - 1.3427) < 1e-4
        assert abs(lines[0]._x[-1] - 1.9299) < 1e-4
        assert abs(lines[0]._y[0] - 2.0698) < 1e-4
        assert abs(lines[0]._y[-1] - 2.0865) < 1e-4

    def test_get_from_strike_raises_fx(self, fxfo):
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
            calendar="tgt|fed",
        )
        with pytest.raises(ValueError, match="`FXSabrSmile` must be specified with a `pair` arg"):
            fxss.get_from_strike(1.02, fxfo)

    def test_solver_variable_numbers(self):
        from rateslib import IRS, FXBrokerFly, FXCall, FXRiskReversal, FXStraddle, FXSwap, Solver

        usdusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="nyc", id="usdusd")
        eureur = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="tgt", id="eureur")
        eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, id="eurusd")

        # Create an FX Forward market with spot FX rate data
        fxr = FXRates({"eurusd": 1.0760}, settlement=dt(2024, 5, 9))
        fxf = FXForwards(
            fx_rates=fxr,
            fx_curves={"eureur": eureur, "usdusd": usdusd, "eurusd": eurusd},
        )

        pre_solver = Solver(
            curves=[eureur, eurusd, usdusd],
            instruments=[
                IRS(dt(2024, 5, 9), "3W", spec="eur_irs", curves="eureur"),
                IRS(dt(2024, 5, 9), "3W", spec="usd_irs", curves="usdusd"),
                FXSwap(
                    dt(2024, 5, 9), "3W", pair="eurusd", curves=[None, "eurusd", None, "usdusd"]
                ),
            ],
            s=[3.90, 5.32, 8.85],
            fx=fxf,
            id="rates_sv",
        )

        dv_smile = FXSabrSmile(
            nodes={"alpha": 0.05, "beta": 1.0, "rho": 0.01, "nu": 0.03},
            eval_date=dt(2024, 5, 7),
            expiry=dt(2024, 5, 28),
            id="eurusd_3w_smile",
            pair="eurusd",
        )
        option_args = dict(
            pair="eurusd",
            expiry=dt(2024, 5, 28),
            calendar="tgt|fed",
            delta_type="spot",
            curves=["eurusd", "usdusd"],
            vol="eurusd_3w_smile",
        )

        dv_solver = Solver(
            pre_solvers=[pre_solver],
            curves=[dv_smile],
            instruments=[
                FXStraddle(strike="atm_delta", **option_args),
                FXRiskReversal(strike=("-25d", "25d"), **option_args),
                FXRiskReversal(strike=("-10d", "10d"), **option_args),
                FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **option_args),
                FXBrokerFly(strike=(("-10d", "10d"), "atm_delta"), **option_args),
            ],
            s=[5.493, -0.157, -0.289, 0.071, 0.238],
            fx=fxf,
            id="dv_solver",
        )

        fc = FXCall(
            expiry=dt(2024, 5, 28),
            pair="eurusd",
            strike=1.07,
            notional=100e6,
            curves=["eurusd", "usdusd"],
            vol="eurusd_3w_smile",
            premium=98.216647 * 1e8 / 1e4,
            premium_ccy="usd",
            delta_type="spot",
        )
        fc.delta(solver=dv_solver)

    @pytest.mark.parametrize("a", [0.02, 0.06])
    @pytest.mark.parametrize("b", [0.0, 0.4, 0.65, 1.0])
    @pytest.mark.parametrize("p", [-0.1, 0.1])
    @pytest.mark.parametrize("v", [0.05, 0.15])
    @pytest.mark.parametrize("k", [1.05, 1.25, 1.6])
    def test_sabr_function_values(self, a, b, p, v, k):
        fxs = FXSabrSmile(
            nodes={"alpha": a, "beta": b, "rho": p, "nu": v},
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            ad=0,
        )

        # this code is taken from PySabr, another library implementing SABR.
        # it is used as a benchmark
        def _x(rho, z):
            """Return function x used in Hagan's 2002 SABR lognormal vol expansion."""
            a = (1 - 2 * rho * z + z**2) ** 0.5 + z - rho
            b = 1 - rho
            return np.log(a / b)

        def lognormal_vol(k, f, t, alpha, beta, rho, volvol):
            """
            Hagan's 2002 SABR lognormal vol expansion.

            The strike k can be a scalar or an array, the function will return an array
            of lognormal vols.
            """
            # Negative strikes or forwards
            if k <= 0 or f <= 0:
                return 0.0
            eps = 1e-07
            logfk = np.log(f / k)
            fkbeta = (f * k) ** (1 - beta)
            a = (1 - beta) ** 2 * alpha**2 / (24 * fkbeta)
            b = 0.25 * rho * beta * volvol * alpha / fkbeta**0.5
            c = (2 - 3 * rho**2) * volvol**2 / 24
            d = fkbeta**0.5
            v = (1 - beta) ** 2 * logfk**2 / 24
            w = (1 - beta) ** 4 * logfk**4 / 1920
            z = volvol * fkbeta**0.5 * logfk / alpha
            # if |z| > eps
            if abs(z) > eps:
                vz = alpha * z * (1 + (a + b + c) * t) / (d * (1 + v + w) * _x(rho, z))
                return vz
            # if |z| <= eps
            else:
                v0 = alpha * (1 + (a + b + c) * t) / (d * (1 + v + w))
                return v0

        expected = lognormal_vol(k, 1.25, 1.0, a, b, p, v)
        result = fxs.get_from_strike(k, 1.25)[1] / 100.0

        assert abs(result - expected) < 1e-4


class TestFXSabrSurface:
    @pytest.mark.parametrize(
        "expiries",
        [
            [dt(2024, 5, 29), dt(2024, 7, 29), dt(2024, 6, 29)],
            [dt(2024, 5, 29), dt(2024, 6, 29), dt(2024, 6, 29)],
        ],
    )
    def test_unsorted_expiries(self, expiries):
        with pytest.raises(ValueError, match="Surface `expiries` are not sorted or contain dupl"):
            FXSabrSurface(
                eval_date=dt(2024, 5, 28),
                expiries=expiries,
                node_values=[[0.05, 1.0, 0.01, 0.15]] * 3,
                pair="eurusd",
                delivery_lag=2,
                calendar="tgt|fed",
                id="eurusd_vol",
            )

    def test_z_eurusd_surface_cookbook(self):
        from rateslib import (
            IRS,
            XCS,
            FXBrokerFly,
            FXRiskReversal,
            FXStraddle,
            FXSwap,
            Solver,
            add_tenor,
        )

        fxr = FXRates({"eurusd": 1.0867}, settlement=dt(2024, 5, 30))
        mkt_data = DataFrame(
            data=[
                [
                    "1w",
                    3.9035,
                    5.3267,
                    3.33,
                ],
                [
                    "2w",
                    3.9046,
                    5.3257,
                    6.37,
                ],
                [
                    "3w",
                    3.8271,
                    5.3232,
                    9.83,
                ],
                [
                    "1m",
                    3.7817,
                    5.3191,
                    13.78,
                ],
                [
                    "2m",
                    3.7204,
                    5.3232,
                    30.04,
                ],
                ["3m", 3.667, 5.3185, 45.85, -2.5],
                [
                    "4m",
                    3.6252,
                    5.3307,
                    61.95,
                ],
                [
                    "5m",
                    3.587,
                    5.3098,
                    78.1,
                ],
                ["6m", 3.5803, 5.3109, 94.25, -3.125],
                [
                    "7m",
                    3.5626,
                    5.301,
                    110.82,
                ],
                [
                    "8m",
                    3.531,
                    5.2768,
                    130.45,
                ],
                ["9m", 3.5089, 5.2614, 145.6, -7.25],
                [
                    "10m",
                    3.4842,
                    5.2412,
                    162.05,
                ],
                [
                    "11m",
                    3.4563,
                    5.2144,
                    178,
                ],
                ["1y", 3.4336, 5.1936, None, -6.75],
                ["15m", 3.3412, 5.0729, None, -6.75],
                ["18m", 3.2606, 4.9694, None, -6.75],
                ["21m", 3.1897, 4.8797, None, -7.75],
                ["2y", 3.1283, 4.8022, None, -7.875],
                ["3y", 2.9254, 4.535, None, -9],
                ["4y", 2.81, 4.364, None, -10.125],
                ["5y", 2.7252, 4.256, None, -11.125],
                ["6y", 2.6773, 4.192, None, -12.125],
                ["7y", 2.6541, 4.151, None, -13],
                ["8y", 2.6431, 4.122, None, -13.625],
                ["9y", 2.6466, 4.103, None, -14.25],
                ["10y", 2.6562, 4.091, None, -14.875],
                ["12y", 2.6835, 4.084, None, -16.125],
                ["15y", 2.7197, 4.08, None, -17],
                ["20y", 2.6849, 4.04, None, -16],
                ["25y", 2.6032, 3.946, None, -12.75],
                ["30y", 2.5217, 3.847, None, -9.5],
            ],
            columns=["tenor", "estr", "sofr", "fx_swap", "xccy"],
        )
        eur = Curve(
            nodes={
                dt(2024, 5, 28): 1.0,
                **{add_tenor(dt(2024, 5, 30), _, "F", "tgt"): 1.0 for _ in mkt_data["tenor"]},
            },
            calendar="tgt",
            interpolation="log_linear",
            convention="act360",
            id="estr",
        )
        usd = Curve(
            nodes={
                dt(2024, 5, 28): 1.0,
                **{add_tenor(dt(2024, 5, 30), _, "F", "nyc"): 1.0 for _ in mkt_data["tenor"]},
            },
            calendar="nyc",
            interpolation="log_linear",
            convention="act360",
            id="sofr",
        )
        eurusd = Curve(
            nodes={
                dt(2024, 5, 28): 1.0,
                **{add_tenor(dt(2024, 5, 30), _, "F", "tgt"): 1.0 for _ in mkt_data["tenor"]},
            },
            interpolation="log_linear",
            convention="act360",
            id="eurusd",
        )
        fxf = FXForwards(fx_rates=fxr, fx_curves={"eureur": eur, "eurusd": eurusd, "usdusd": usd})
        estr_swaps = [
            IRS(dt(2024, 5, 30), _, spec="eur_irs", curves="estr") for _ in mkt_data["tenor"]
        ]
        estr_rates = mkt_data["estr"].tolist()
        labels = mkt_data["tenor"].to_list()
        sofr_swaps = [
            IRS(dt(2024, 5, 30), _, spec="usd_irs", curves="sofr") for _ in mkt_data["tenor"]
        ]
        sofr_rates = mkt_data["sofr"].tolist()
        eur_solver = Solver(
            curves=[eur],
            instruments=estr_swaps,
            s=estr_rates,
            fx=fxf,
            instrument_labels=labels,
            id="eur",
        )
        usd_solver = Solver(
            curves=[usd],
            instruments=sofr_swaps,
            s=sofr_rates,
            fx=fxf,
            instrument_labels=labels,
            id="usd",
        )
        fxswaps = [
            FXSwap(dt(2024, 5, 30), _, pair="eurusd", curves=["eurusd", "sofr"])
            for _ in mkt_data["tenor"][0:14]
        ]
        fxswap_rates = mkt_data["fx_swap"][0:14].tolist()
        xcs = [
            XCS(dt(2024, 5, 30), _, spec="eurusd_xcs", curves=["estr", "eurusd", "sofr", "sofr"])
            for _ in mkt_data["tenor"][14:]
        ]
        xcs_rates = mkt_data["xccy"][14:].tolist()
        fx_solver = Solver(
            pre_solvers=[eur_solver, usd_solver],
            curves=[eurusd],
            instruments=fxswaps + xcs,
            s=fxswap_rates + xcs_rates,
            fx=fxf,
            instrument_labels=labels,
            id="eurusd_xccy",
        )
        vol_data = DataFrame(
            data=[
                ["1w", 4.535, -0.047, 0.07, -0.097, 0.252],
                ["2w", 5.168, -0.082, 0.077, -0.165, 0.24],
                ["3w", 5.127, -0.175, 0.07, -0.26, 0.233],
                ["1m", 5.195, -0.2, 0.07, -0.295, 0.235],
                ["2m", 5.237, -0.28, 0.087, -0.535, 0.295],
                ["3m", 5.257, -0.363, 0.1, -0.705, 0.35],
                ["4m", 5.598, -0.47, 0.123, -0.915, 0.422],
                ["5m", 5.776, -0.528, 0.133, -1.032, 0.463],
                ["6m", 5.92, -0.565, 0.14, -1.11, 0.49],
                ["9m", 6.01, -0.713, 0.182, -1.405, 0.645],
                ["1y", 6.155, -0.808, 0.23, -1.585, 0.795],
                ["18m", 6.408, -0.812, 0.248, -1.588, 0.868],
                ["2y", 6.525, -0.808, 0.257, -1.58, 0.9],
                ["3y", 6.718, -0.733, 0.265, -1.45, 0.89],
                ["4y", 7.025, -0.665, 0.265, -1.31, 0.885],
                ["5y", 7.26, -0.62, 0.26, -1.225, 0.89],
                ["6y", 7.508, -0.516, 0.27, -0.989, 0.94],
                ["7y", 7.68, -0.442, 0.278, -0.815, 0.975],
                ["10y", 8.115, -0.267, 0.288, -0.51, 1.035],
                ["15y", 8.652, -0.325, 0.362, -0.4, 1.195],
                ["20y", 8.651, -0.078, 0.343, -0.303, 1.186],
                ["25y", 8.65, -0.029, 0.342, -0.218, 1.178],
                ["30y", 8.65, 0.014, 0.341, -0.142, 1.171],
            ],
            columns=["tenor", "atm", "25drr", "25dbf", "10drr", "10dbf"],
        )
        vol_data["expiry"] = [add_tenor(dt(2024, 5, 28), _, "MF", "tgt") for _ in vol_data["tenor"]]
        surface = FXSabrSurface(
            eval_date=dt(2024, 5, 28),
            expiries=list(vol_data["expiry"]),
            node_values=[[0.05, 1.0, 0.01, 0.15]] * 23,
            pair="eurusd",
            delivery_lag=2,
            calendar="tgt|fed",
            id="eurusd_vol",
        )
        fx_args = dict(
            pair="eurusd",
            curves=["eurusd", "sofr"],
            calendar="tgt",
            delivery_lag=2,
            payment_lag=2,
            eval_date=dt(2024, 5, 28),
            modifier="MF",
            premium_ccy="usd",
            vol="eurusd_vol",
        )

        instruments_le_1y, rates_le_1y, labels_le_1y = [], [], []
        for row in range(11):
            instruments_le_1y.extend(
                [
                    FXStraddle(
                        strike="atm_delta",
                        expiry=vol_data["expiry"][row],
                        delta_type="spot",
                        **fx_args,
                    ),
                    FXRiskReversal(
                        strike=("-25d", "25d"),
                        expiry=vol_data["expiry"][row],
                        delta_type="spot",
                        **fx_args,
                    ),
                    FXBrokerFly(
                        strike=(("-25d", "25d"), "atm_delta"),
                        expiry=vol_data["expiry"][row],
                        delta_type="spot",
                        **fx_args,
                    ),
                    FXRiskReversal(
                        strike=("-10d", "10d"),
                        expiry=vol_data["expiry"][row],
                        delta_type="spot",
                        **fx_args,
                    ),
                    FXBrokerFly(
                        strike=(("-10d", "10d"), "atm_delta"),
                        expiry=vol_data["expiry"][row],
                        delta_type="spot",
                        **fx_args,
                    ),
                ]
            )
            rates_le_1y.extend(
                [
                    vol_data["atm"][row],
                    vol_data["25drr"][row],
                    vol_data["25dbf"][row],
                    vol_data["10drr"][row],
                    vol_data["10dbf"][row],
                ]
            )
            labels_le_1y.extend(
                [f"atm_{row}", f"25drr_{row}", f"25dbf_{row}", f"10drr_{row}", f"10dbf_{row}"]
            )

        instruments_gt_1y, rates_gt_1y, labels_gt_1y = [], [], []
        for row in range(11, 23):
            instruments_gt_1y.extend(
                [
                    FXStraddle(
                        strike="atm_delta",
                        expiry=vol_data["expiry"][row],
                        delta_type="forward",
                        **fx_args,
                    ),
                    FXRiskReversal(
                        strike=("-25d", "25d"),
                        expiry=vol_data["expiry"][row],
                        delta_type="forward",
                        **fx_args,
                    ),
                    FXBrokerFly(
                        strike=(("-25d", "25d"), "atm_delta"),
                        expiry=vol_data["expiry"][row],
                        delta_type="forward",
                        **fx_args,
                    ),
                    FXRiskReversal(
                        strike=("-10d", "10d"),
                        expiry=vol_data["expiry"][row],
                        delta_type="forward",
                        **fx_args,
                    ),
                    FXBrokerFly(
                        strike=(("-10d", "10d"), "atm_delta"),
                        expiry=vol_data["expiry"][row],
                        delta_type="forward",
                        **fx_args,
                    ),
                ]
            )
            rates_gt_1y.extend(
                [
                    vol_data["atm"][row],
                    vol_data["25drr"][row],
                    vol_data["25dbf"][row],
                    vol_data["10drr"][row],
                    vol_data["10dbf"][row],
                ]
            )
            labels_gt_1y.extend(
                [f"atm_{row}", f"25drr_{row}", f"25dbf_{row}", f"10drr_{row}", f"10dbf_{row}"]
            )

        Solver(
            surfaces=[surface],
            instruments=instruments_le_1y + instruments_gt_1y,
            s=rates_le_1y + rates_gt_1y,
            instrument_labels=labels_le_1y + labels_gt_1y,
            fx=fxf,
            pre_solvers=[fx_solver],
            id="eurusd_vol",
        )

    def test_k_derivative_interpolation(self, fxfo):
        # test the derivative of the k-interpolated volatility of a SabrSurface against Fwd diff
        # and AD.
        surface = FXSabrSurface(
            eval_date=dt(2023, 3, 16),
            expiries=[dt(2025, 5, 28), dt(2026, 5, 28)],
            node_values=[
                [0.05, 1.0, 0.01, 0.15],
                [0.06, 1.0, 0.02, 0.20],
            ],
            pair="eurusd",
            delivery_lag=2,
            calendar="tgt|fed",
            id="eurusd_vol",
        )
        k = Dual(1.10, ["k"], [1.0])
        base = surface.get_from_strike(k, fxfo, dt(2025, 12, 12))[1]
        expected_ad = gradient(base, vars=["k"])[0]
        expected_fwd_diff = (
            surface.get_from_strike(k + 0.0001, fxfo, dt(2025, 12, 12))[1] - base
        ) / 1e-4
        result = surface._d_sabr_d_k_or_f(k, fxfo, dt(2025, 12, 12), False, 1)[1] * 100.0
        assert abs(expected_fwd_diff - result) < 1e-3
        assert abs(expected_ad - result) < 1e-3

    @pytest.mark.parametrize(
        ("k", "expiry", "expected"),
        [
            (1.10, dt(2023, 4, 15), 5.011351023668074),
            (1.10, dt(2023, 6, 28), 5.011351023668074),
            (1.10, dt(2023, 7, 15), 5.333915841859923),
            (1.10, dt(2023, 9, 28), 6.021827601466909),
            (1.10, dt(2023, 10, 28), 6.022252380963102),
        ],
    )
    def test_get_from_strike(self, fxfo, k, expiry, expected):
        # test different branches for expiry
        surface = FXSabrSurface(
            eval_date=dt(2023, 3, 16),
            expiries=[dt(2023, 6, 28), dt(2023, 9, 28)],
            node_values=[
                [0.05, 1.0, 0.01, 0.15],
                [0.06, 1.0, 0.02, 0.20],
            ],
            pair="eurusd",
            delivery_lag=2,
            calendar="tgt|fed",
            id="eurusd_vol",
        )
        result = surface.get_from_strike(k, fxfo, expiry)
        assert result[0] == 0.0
        assert abs(result[1] - expected) < 1e-14
        assert result[2] == k

    def test_variables_on_extrapolated_sabr_smiles_before(self, fxfo):
        # assert that vars on extrapolated smiles reference the underlying smiles vars
        fxss = FXSabrSurface(
            eval_date=dt(2023, 3, 16),
            expiries=[dt(2023, 7, 15), dt(2023, 9, 15)],
            node_values=[[0.05, 1.0, 0.01, 0.15]] * 2,
            pair="eurusd",
            delivery_lag=2,
            calendar="tgt|fed",
            id="v",
            ad=1,
        )
        result = fxss.get_from_strike(1.10, fxfo, dt(2023, 4, 14))[1]
        assert result.vars == ["v_0_0", "v_0_1", "v_0_2", "fx_eurusd"]

    def test_variables_on_extrapolated_sabr_smiles_after(self, fxfo):
        # assert that vars on extrapolated smiles reference the underlying smiles vars
        fxss = FXSabrSurface(
            eval_date=dt(2023, 3, 16),
            expiries=[dt(2023, 7, 15), dt(2023, 9, 15)],
            node_values=[[0.05, 1.0, 0.01, 0.15]] * 2,
            pair="eurusd",
            delivery_lag=2,
            calendar="tgt|fed",
            id="v",
            ad=1,
        )
        result = fxss.get_from_strike(1.10, fxfo, dt(2024, 4, 14))[1]
        assert result.vars == ["v_1_0", "v_1_1", "v_1_2", "fx_eurusd"]

    def test_update_state(self):
        fxss = FXSabrSurface(
            eval_date=dt(2023, 3, 16),
            expiries=[dt(2023, 7, 15), dt(2023, 9, 15)],
            node_values=[[0.05, 1.0, 0.01, 0.15]] * 2,
            pair="eurusd",
            delivery_lag=2,
            calendar="tgt|fed",
            id="v",
            ad=1,
        )
        state_ = fxss._state
        fxss.smiles[1].update_node("alpha", 0.06)
        assert state_ != fxss._get_composited_state()

        # calling get from strike will validate
        fxss.get_from_strike(1.1, 1.1, dt(2023, 7, 15))
        assert fxss._state == fxss._get_composited_state()

    @pytest.mark.parametrize("smile_expiry", [dt(2026, 5, 1), dt(2026, 6, 9), dt(2026, 7, 1)])
    def test_flat_surface_and_get_smile_one_expiry(self, smile_expiry):
        # gh 911
        anchor = dt(2025, 6, 9)
        expiry = dt(2026, 6, 9)

        surf = FXSabrSurface(
            eval_date=anchor,
            expiries=[expiry],
            node_values=[[0.10, 1.0, 0.0, 0.0]],
        )

        result = surf.get_from_strike(1.0, 1.10, smile_expiry)[1]
        assert abs(result - 10.0) < 1e-13

    @pytest.mark.parametrize("option_expiry", [dt(2026, 5, 1), dt(2026, 6, 9), dt(2026, 7, 1)])
    def test_flat_surface_option_strike_delta(self, option_expiry):
        surf = FXSabrSurface(
            eval_date=dt(2025, 6, 9),
            expiries=[dt(2026, 6, 9)],
            node_values=[[0.10, 1.0, 0.0, 0.0]],
        )
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=option_expiry,
            delivery=option_expiry,
            strike=NoInput(0),
            delta_type="forward",
        )
        result = fxo._index_vol_and_strike_from_delta_sabr(0.25, "forward", surf, 1, 1.10)
        assert abs(result[1] - 10.0) < 1e-13

        result = fxo._index_vol_and_strike_from_atm_sabr(1.10, 0.50, surf)
        assert abs(result[1] - 10.0) < 1e-13


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
            ("get_from_strike", (1.0, 1.0, dt(2000, 5, 3), NoInput(0))),
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
            ("_set_node_vector", ([0.99, 0.98, 0.99, 0.99, 0.98, 0.99], 1)),
        ],
    )
    @pytest.mark.parametrize(
        "surface",
        [
            FXDeltaVolSurface(
                expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
                delta_indexes=[0.25, 0.5, 0.75],
                node_values=[[10.0, 9.0, 8.0], [9.0, 8.0, 7.0]],
                eval_date=dt(1999, 1, 1),
                delta_type="forward",
            ),
            FXSabrSurface(
                expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
                node_values=[[10.0, 1.0, 8.0, 9.0], [9.0, 1.0, 8.0, 7.0]],
                eval_date=dt(1999, 1, 1),
            ),
        ],
    )
    def test_surface_change_state(self, method, args, surface):
        pre_state = surface._state
        getattr(surface, method)(*args)
        assert surface._state != pre_state

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_ad_order", (2,)),
        ],
    )
    @pytest.mark.parametrize(
        "surface",
        [
            FXDeltaVolSurface(
                expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
                delta_indexes=[0.25, 0.5, 0.75],
                node_values=[[10.0, 9.0, 8.0], [9.0, 8.0, 7.0]],
                eval_date=dt(1999, 1, 1),
                delta_type="forward",
            ),
            FXSabrSurface(
                expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
                node_values=[[10.0, 1.0, 8.0, 9.0], [9.0, 1.0, 8.0, 7.0]],
                eval_date=dt(1999, 1, 1),
            ),
        ],
    )
    def test_surface_maintain_state(self, method, args, surface):
        pre_state = surface._state
        getattr(surface, method)(*args)
        assert surface._state == pre_state

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
    with pytest.raises(ValueError, match="`delta_type` as string: 'BAD_TYPE' i"):
        _get_fx_delta_type("BAD_TYPE")
