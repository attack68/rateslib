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

import sys
from datetime import datetime as dt
from itertools import combinations, product

import numpy as np
import pytest
from matplotlib import pyplot as plt
from pandas import DataFrame, Index, IndexSlice, Series
from pandas.testing import assert_frame_equal, assert_series_equal
from rateslib import calendars, default_context
from rateslib.curves import CompositeCurve, Curve, LineCurve
from rateslib.data.fixings import IRSSeries
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, Variable, gradient
from rateslib.instruments import IRS, IRSCall, IRSPut, IRSStraddle, IRVolValue
from rateslib.solver import Solver
from rateslib.splines import PPSplineF64
from rateslib.volatility import (
    IRSabrCube,
    IRSabrSmile,
    IRSplineCube,
    IRSplineSmile,
)
from rateslib.volatility.ir.utils import _bilinear_interp, _scale_weights
from rateslib.volatility.utils import _OptionModelBachelier, _OptionModelBlack76, _SabrSmileNodes


@pytest.mark.parametrize(
    ("h", "v", "expected"),
    [
        ((1, 1), (1, 1), 10),
        ((0.5, 0.5), (0.5, 0.5), 5.0),
        ((0.0, 0.0), (0.0, 0.0), 0.0),
        ((0.0, 0.5), (0.0, 0.0), 0.0),
        ((0.0, 0.0), (0.8, 0.4), 4.80),
        ((0.1, 0.2), (0.4, 0.5), 4.0 * 0.1 * 0.5 + 6.0 * 0.8 * 0.4 + 10.0 * 0.2 * 0.5),
    ],
)
def test_bilinear_interp(h, v, expected):
    result = _bilinear_interp(0.0, 4.0, 6.0, 10.0, h, v)
    assert abs(result - expected) < 1e-10


def test_numpy_ravel_for_dates_posix():
    a = np.array([[1, 1, 2], [3, 4, 5]])
    b = np.reshape(list(a.ravel()), (2, 3))
    assert np.all(a == b)


@pytest.fixture
def curve():
    return Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2032, 3, 31): 0.50,
        },
        interpolation="log_linear",
        id="v",
        convention="Act360",
        ad=1,
    )


class TestIRSabrSmile:
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
        # repeat the same test developed for FXSabrSmile
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
        )
        result = irss.get_from_strike(k=strike, f=1.3395).vol
        assert abs(result - vol) < 1e-2

    def test_sabr_vol_plot(self):
        # repeat the same test developed for FXSabrSmile
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
        )
        result = irss.plot(f=1.0)
        _x = result[2][0]._x
        _y = result[2][0]._y
        assert (_x[0], _y[0]) == (0.7524348790033292, 23.108399874378378)
        assert (_x[-1], _y[-1]) == (1.3743407823531082, 21.950871667495214)

    def test_sabr_vol_plot_fail(self):
        # repeat the same test developed for FXSabrSmile
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
        )
        with pytest.raises(
            ValueError,
            match=r"`f` \(ATM-forward interest rate\) is required by `_BaseIRSmile.plot`.",
        ):
            irss.plot()

    @pytest.mark.parametrize(("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33)])
    def test_sabr_vol_finite_diff_first_order(self, k, f):
        # Test all of the first order gradients using finite diff, for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
            ad=2,
        )
        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss.get_from_strike(k=Dual2(k, ["k"], [], []), f=Dual2(f, ["f"], [], [])).vol

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        def inc_(key1, inc1):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1

            irss._nodes = _SabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = (
                irss._d_sabr_d_k_or_f(
                    Dual2(in_["k"], ["k"], [], []),
                    Dual2(in_["f"], ["f"], [], []),
                    dt(2002, 1, 1),
                    False,
                    1,
                )[0]
                * 100.0
            )

            # reset
            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
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
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss.get_from_strike(k=Dual2(k, ["k"], [], []), f=Dual2(f, ["f"], [], [])).vol

        def inc_(key1, key2, inc1, inc2):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1
            in_[key2] += inc2

            irss._nodes = _SabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = (
                irss._d_sabr_d_k_or_f(
                    Dual2(in_["k"], ["k"], [], []),
                    Dual2(in_["f"], ["f"], [], []),
                    dt(2002, 1, 1),
                    False,
                    1,
                )[0]
                * 100.0
            )

            # reset
            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
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
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss.get_from_strike(k=Dual2(k, ["k"], [], []), f=Dual2(f, ["f"], [], [])).vol

        def inc_(key1, inc1):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1

            irss._nodes = _SabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = (
                irss._d_sabr_d_k_or_f(
                    Dual2(in_["k"], ["k"], [], []),
                    Dual2(in_["f"], ["f"], [], []),
                    dt(2002, 1, 1),
                    False,
                    1,
                )[0]
                * 100.0
            )

            # reset
            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
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
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss.get_from_strike(k=Dual2(1.34, ["k"], [], []), f=Dual2(1.34, ["f"], [], [])).vol
        comparison1 = irss.get_from_strike(
            k=Dual2(1.341, ["k"], [], []), f=Dual2(1.34, ["f"], [], [])
        ).vol

        assert np.all(abs(base.dual - comparison1.dual) < 1e-1)
        diff = base.dual2 - comparison1.dual2
        dual2 = abs(diff) < 5e-1
        assert np.all(dual2)

    @pytest.mark.parametrize("param", ["alpha", "rho", "nu"])
    def test_missing_param_raises(self, param):
        nodes = {
            "alpha": 0.17431060,
            "rho": -0.11268306,
            "nu": 0.81694072,
        }
        nodes.pop(param)
        with pytest.raises(ValueError):
            IRSabrSmile(
                nodes=nodes,
                beta=1.0,
                eval_date=dt(2001, 1, 1),
                expiry=dt(2002, 1, 1),
                irs_series="eur_irs6",
                tenor="2y",
                id="v",
                ad=2,
            )

    def test_non_iterable(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        with pytest.raises(TypeError):
            list(irss)

    def test_update_node_raises(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        with pytest.raises(KeyError, match="'bananas' is not in `nodes`."):
            irss.update_node("bananas", 12.0)

    def test_set_ad_order_raises(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        with pytest.raises(ValueError, match="`order` can only be in {0, 1, 2} "):
            irss._set_ad_order(12)

    def test_get_node_vars_and_vector(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "rho": -0.10,
                "nu": 0.80,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="myid",
        )
        result = irss._get_node_vars()
        expected = ("myid0", "myid1", "myid2")
        assert result == expected

        result = irss._get_node_vector()
        expected = np.array([0.20, -0.1, 0.80])
        assert np.all(result == expected)

    def test_get_from_strike_expiry_raises(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "rho": -0.10,
                "nu": 0.80,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="myid",
        )
        with pytest.raises(
            ValueError, match="`expiry` of _BaseIRSmile and intended price do not match"
        ):
            irss.get_from_strike(k=1.0, f=1.0, expiry=dt(1999, 1, 1))

    @pytest.mark.parametrize("k", [1.2034, 1.2050, 1.3620, 1.5410, 1.5449])
    def test_get_from_strike_ad_2(self, k) -> None:
        # Use finite diff to validate the 2nd order AD of the SABR function in alpha and rho.
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "rho": -0.10,
                "nu": 0.80,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
            ad=2,
        )

        kwargs = dict(
            k=k,
            f=1.350,
        )
        pv00 = irss.get_from_strike(**kwargs)

        irss.update_node("alpha", 0.20 + 0.00001)
        irss.update_node("rho", -0.10 + 0.00001)
        pv11 = irss.get_from_strike(**kwargs)

        irss.update_node("alpha", 0.20 + 0.00001)
        irss.update_node("rho", -0.10 - 0.00001)
        pv1_1 = irss.get_from_strike(**kwargs)

        irss.update_node("alpha", 0.20 - 0.00001)
        irss.update_node("rho", -0.10 - 0.00001)
        pv_1_1 = irss.get_from_strike(**kwargs)

        irss.update_node("alpha", 0.20 - 0.00001)
        irss.update_node("rho", -0.10 + 0.00001)
        pv_11 = irss.get_from_strike(**kwargs)

        finite_diff = (pv11.vol + pv_1_1.vol - pv1_1.vol - pv_11.vol) * 1e10 / 4.0
        ad_grad = gradient(pv00.vol, ["vol0", "vol1"], 2)[0, 1]

        assert abs(finite_diff - ad_grad) < 1e-4

    @pytest.mark.parametrize(("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33)])
    def test_sabr_derivative_finite_diff_first_order(self, k, f):
        # Test all of the first order gradients using finite diff, for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "rho": -0.10,
                "nu": 0.80,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
            ad=2,
        )
        t = dt(2002, 1, 1)
        base = irss._d_sabr_d_k_or_f(
            Dual2(k, ["k"], [1.0], []), Dual2(f, ["f"], [1.0], []), t, False, 1
        )[1]

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        def inc_(key1, inc1):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1

            irss._nodes = _SabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = irss._d_sabr_d_k_or_f(
                Dual2(in_["k"], ["k"], [], []),
                Dual2(in_["f"], ["f"], [], []),
                dt(2002, 1, 1),
                False,
                1,
            )[1]

            # reset
            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
            return _

        for key in ["k", "f", "alpha", "rho", "nu"]:
            map_ = {"k": "k", "f": "f", "alpha": "vol0", "rho": "vol1", "nu": "vol2"}
            up_ = inc_(key, 1e-5)
            dw_ = inc_(key, -1e-5)
            expected = (up_ - dw_) / 2e-5
            result = gradient(base, [map_[key]])[0]
            assert abs(expected - result) < 7e-3

    @pytest.mark.parametrize(
        ("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33), (1.3395, 1.34), (1.34, 1.3405)]
    )
    @pytest.mark.parametrize("pair", list(combinations(["k", "f", "alpha", "rho", "nu"], 2)))
    def test_sabr_derivative_cross_finite_diff_second_order(self, k, f, pair):
        # Test all of the second order cross gradients using finite diff,
        # for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "rho": -0.10,
                "nu": 0.80,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss._d_sabr_d_k_or_f(
            Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []), dt(2002, 1, 1), False, 1
        )[1]

        def inc_(key1, key2, inc1, inc2):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1
            in_[key2] += inc2

            irss._nodes = _SabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = irss._d_sabr_d_k_or_f(
                Dual2(in_["k"], ["k"], [], []),
                Dual2(in_["f"], ["f"], [], []),
                dt(2002, 1, 1),
                False,
                1,
            )[1]

            # reset
            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
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
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "rho": -0.10,
                "nu": 0.80,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss._d_sabr_d_k_or_f(
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
                irss.update_node(key1, getattr(irss.nodes, key1) + inc1)
                # irss.nodes[key1] = irss.nodes[key1] + inc1

            _ = irss._d_sabr_d_k_or_f(
                Dual2(k_, ["k"], [], []), Dual2(f_, ["f"], [], []), dt(2002, 1, 1), False, 1
            )[1]

            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
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
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "rho": -0.10,
                "nu": 0.80,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss._d_sabr_d_k_or_f(
            Dual2(1.34, ["k"], [], []), Dual2(1.34, ["f"], [], []), dt(2002, 1, 1), False, 1
        )[1]
        comparison1 = irss._d_sabr_d_k_or_f(
            Dual2(1.341, ["k"], [], []), Dual2(1.34, ["f"], [], []), dt(2002, 1, 1), False, 1
        )[1]

        assert np.all(abs(base.dual - comparison1.dual) < 5e-3)
        diff = base.dual2 - comparison1.dual2
        dual2 = abs(diff) < 3e-2
        assert np.all(dual2)

    #
    # def test_plot_domain(self):
    #     ss = FXSabrSmile(
    #         eval_date=dt(2024, 5, 28),
    #         expiry=dt(2054, 5, 28),
    #         nodes={"alpha": 0.02, "beta": 1.0, "rho": 0.01, "nu": 0.05},
    #     )
    #     ax, fig, lines = ss.plot(f=1.60)
    #     assert abs(lines[0]._x[0] - 1.3427) < 1e-4
    #     assert abs(lines[0]._x[-1] - 1.9299) < 1e-4
    #     assert abs(lines[0]._y[0] - 2.0698) < 1e-4
    #     assert abs(lines[0]._y[-1] - 2.0865) < 1e-4
    #

    #
    # def test_solver_variable_numbers(self):
    #     from rateslib import IRS, FXBrokerFly, FXCall, FXRiskReversal, FXStraddle, FXSwap, Solver
    #
    #     usdusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="nyc", id="usdusd")
    #     eureur = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="tgt", id="eureur")
    #     eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, id="eurusd")
    #
    #     # Create an FX Forward market with spot FX rate data
    #     fxr = FXRates({"eurusd": 1.0760}, settlement=dt(2024, 5, 9))
    #     fxf = FXForwards(
    #         fx_rates=fxr,
    #         fx_curves={"eureur": eureur, "usdusd": usdusd, "eurusd": eurusd},
    #     )
    #
    #     pre_solver = Solver(
    #         curves=[eureur, eurusd, usdusd],
    #         instruments=[
    #             IRS(dt(2024, 5, 9), "3W", spec="eur_irs", curves="eureur"),
    #             IRS(dt(2024, 5, 9), "3W", spec="usd_irs", curves="usdusd"),
    #             FXSwap(
    #                 dt(2024, 5, 9), "3W", pair="eurusd", curves=[None, "eurusd", None, "usdusd"]
    #             ),
    #         ],
    #         s=[3.90, 5.32, 8.85],
    #         fx=fxf,
    #         id="rates_sv",
    #     )
    #
    #     dv_smile = FXSabrSmile(
    #         nodes={"alpha": 0.05, "beta": 1.0, "rho": 0.01, "nu": 0.03},
    #         eval_date=dt(2024, 5, 7),
    #         expiry=dt(2024, 5, 28),
    #         id="eurusd_3w_smile",
    #         pair="eurusd",
    #     )
    #     option_args = dict(
    #         pair="eurusd",
    #         expiry=dt(2024, 5, 28),
    #         calendar="tgt|fed",
    #         delta_type="spot",
    #         curves=["eurusd", "usdusd"],
    #         vol="eurusd_3w_smile",
    #     )
    #
    #     dv_solver = Solver(
    #         pre_solvers=[pre_solver],
    #         curves=[dv_smile],
    #         instruments=[
    #             FXStraddle(strike="atm_delta", **option_args),
    #             FXRiskReversal(strike=("-25d", "25d"), **option_args),
    #             FXRiskReversal(strike=("-10d", "10d"), **option_args),
    #             FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **option_args),
    #             FXBrokerFly(strike=(("-10d", "10d"), "atm_delta"), **option_args),
    #         ],
    #         s=[5.493, -0.157, -0.289, 0.071, 0.238],
    #         fx=fxf,
    #         id="dv_solver",
    #     )
    #
    #     fc = FXCall(
    #         expiry=dt(2024, 5, 28),
    #         pair="eurusd",
    #         strike=1.07,
    #         notional=100e6,
    #         curves=["eurusd", "usdusd"],
    #         vol="eurusd_3w_smile",
    #         premium=98.216647 * 1e8 / 1e4,
    #         premium_ccy="usd",
    #         delta_type="spot",
    #     )
    #     fc.delta(solver=dv_solver)
    #
    @pytest.mark.parametrize("a", [0.02, 0.06])
    @pytest.mark.parametrize("b", [0.0, 0.4, 0.65, 1.0])
    @pytest.mark.parametrize("p", [-0.1, 0.1])
    @pytest.mark.parametrize("v", [0.05, 0.15])
    @pytest.mark.parametrize("k", [1.05, 1.25, 1.6])
    def test_sabr_function_values(self, a, b, p, v, k):
        irss = IRSabrSmile(
            nodes={
                "alpha": a,
                "rho": p,
                "nu": v,
            },
            beta=b,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
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
        result = irss.get_from_strike(k=k, f=1.25).vol / 100.0

        assert abs(result - expected) < 1e-4

    def test_init_raises_key(self):
        with pytest.raises(
            ValueError, match=r"'nu' is a required SABR parameter that must be inclu"
        ):
            IRSabrSmile(
                nodes={
                    "alpha": 0.05,
                    "rho": 0.1,
                    "bad": 0.1,
                },
                beta=-0.03,
                eval_date=dt(2001, 1, 1),
                expiry=dt(2002, 1, 1),
                irs_series="eur_irs6",
                tenor="2y",
                id="v",
                ad=2,
            )

    def test_attributes(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.05,
                "rho": 0.1,
                "nu": 0.1,
            },
            beta=1.0,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        assert irss._n == 4

    def test_get_from_strike_with_curves(self):
        curve = Curve({dt(2001, 1, 1): 1.0, dt(2003, 1, 1): 0.94})
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.05,
                "rho": 0.1,
                "nu": 0.1,
            },
            beta=-0.03,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
        )
        result = irss.get_from_strike(k=3.0, curves=[curve])
        assert abs(result.f - 3.142139380) < 1e-6
        assert abs(result.vol - 1.575277) < 1e-4

    def test_set_node_vector(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.05,
                "rho": 0.1,
                "nu": 0.1,
            },
            beta=-0.03,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            ad=2,
            id="v",
        )
        irss._set_node_vector(np.array([1.0, 2.0, 3.0]), ad=1)
        assert irss.nodes.alpha == Dual(1.0, ["v0"], [])
        assert irss.nodes.rho == Dual(2.0, ["v1"], [])
        assert irss.nodes.nu == Dual(3.0, ["v2"], [])

    @pytest.mark.skip(reason="SABR Smile cannot solve to parameters matching the target")
    def test_plot_normal_from_black_shift(self):
        # test that smiles with shift equate to the same normal vol graph
        smile1 = IRSabrSmile(
            eval_date=dt(2000, 1, 1),
            expiry=dt(2000, 7, 1),
            tenor="1y",
            irs_series="usd_irs",
            nodes={
                "alpha": 0.20,
                "rho": -0.05,
                "nu": 1.5,
            },
            beta=0.5,
            id="sofr_vol",
            shift=0.0,
        )
        smile2 = IRSabrSmile(
            eval_date=dt(2000, 1, 1),
            expiry=dt(2000, 7, 1),
            tenor="1y",
            irs_series="usd_irs",
            nodes={
                "alpha": 0.20,
                "rho": -0.06,
                "nu": 1.5,
            },
            beta=0.5,
            id="sofr_vol",
            shift=10.0,
        )

        from rateslib import IRS, IRSCall, Solver

        curve = Curve(nodes={dt(2000, 1, 1): 1.0, dt(2003, 1, 1): 0.90}, id="sofr")
        curve_solver = Solver(
            curves=[curve],
            instruments=[IRS(dt(2000, 1, 1), "1y", spec="usd_irs", curves="sofr")],
            s=[3.0],
            instrument_labels=["1Y IRS"],
        )
        option_args = dict(
            expiry=dt(2000, 7, 1),
            tenor="1y",
            irs_series="usd_irs",
            metric="NormalVol",
            curves="sofr",
            vol="sofr_vol",
        )

        instruments = [
            IRSCall(strike="-20bps", **option_args),
            IRSCall(strike="atm", **option_args),
            IRSCall(strike="+20bps", **option_args),
        ]

        def solver_factory(smile):
            solver = Solver(
                pre_solvers=[curve_solver],
                curves=[smile],
                instruments=instruments,
                s=[50.0, 47.0, 49.0],
                instrument_labels=["-20bps Vol", "ATM Vol", "+20bps Vol"],
                ini_lambda=(20000, 0.4, 2),
                conv_tol=1e-6,
            )
            return solver

        s2 = solver_factory(smile2)
        s1 = solver_factory(smile1)

        _res1_nvol = [_.rate(solver=s1) for _ in instruments]
        _res2_nvol = [_.rate(solver=s2) for _ in instruments]
        _res1_lnvol = [_.rate(solver=s1, metric="black_vol_shift_0") for _ in instruments]
        _res2_lnvol = [_.rate(solver=s2, metric="black_vol_shift_10") for _ in instruments]

        fig, ax, lines = smile1.plot(curves=curve, y_axis="normal_vol", comparators=[smile2])

        pp1 = PPSplineF64(k=2, t=[lines[0]._x[0]] + lines[0]._x.tolist() + [lines[0]._x[-1]])
        pp1.csolve(tau=lines[0]._x, y=lines[0]._y, left_n=0, right_n=0, allow_lsq=False)
        pp2 = PPSplineF64(k=2, t=[lines[1]._x[0]] + lines[1]._x.tolist() + [lines[1]._x[-1]])
        pp2.csolve(tau=lines[1]._x, y=lines[1]._y, left_n=0, right_n=0, allow_lsq=False)

        x = np.linspace(2.54, 2.83, 101)
        eps = [abs(pp1.ppev_single(_) - pp2.ppev_single(_)) for _ in x]

        assert all(_ < 0.001 for _ in eps)

    @pytest.mark.parametrize(
        "klass",
        [
            (IRSStraddle, IRSPut, IRSCall),
            (IRVolValue, IRVolValue, IRVolValue),
        ],
    )
    def test_plot_normal_from_black_shift2_with_IROption_Solving(self, klass):
        # klass denotes the instruments used in the solving process
        from rateslib import IRS, IRSCall, IRSPut, IRSStraddle, Solver

        # test that smiles with shift equate to the same normal vol graph
        smile_args = dict(
            eval_date=dt(2026, 3, 2),
            expiry="6m",
            tenor="1y",
            irs_series="usd_irs",
            id="sofr_vol",
        )

        curve = Curve(
            nodes={dt(2026, 3, 2): 1.0, dt(2029, 3, 2): 0.90},
            calendar="nyc",
            convention="act360",
            id="sofr",
        )

        curve_solver = Solver(
            curves=[curve],
            instruments=[IRS(dt(2026, 3, 4), "2y", spec="usd_irs", curves=["sofr"])],
            s=[3.90],
            instrument_labels=["US_2y"],
        )

        def smile_solver_factory(smile):
            _solver = Solver(
                pre_solvers=[curve_solver],  # <- contains the US SOFR Curve
                curves=[smile],  # <- mutates only the smile
                instruments=[
                    klass[0](
                        dt(2026, 9, 2),
                        "1y",
                        "atm",
                        "usd_irs",
                        curves="sofr",
                        vol="sofr_vol",
                        metric="normal_vol",
                    ),
                    klass[1](
                        dt(2026, 9, 2),
                        "1y",
                        "-20bps",
                        "usd_irs",
                        curves="sofr",
                        vol="sofr_vol",
                        metric="normal_vol",
                    ),
                    klass[2](
                        dt(2026, 9, 2),
                        "1y",
                        "+20bps",
                        "usd_irs",
                        curves="sofr",
                        vol="sofr_vol",
                        metric="normal_vol",
                    ),
                ],
                s=[50, 62, 60],
                instrument_labels=["ATM", "-20bps", "20bps"],
                id="sofr_sv",
            )

        smile1 = IRSabrSmile(
            shift=0, beta=0.5, nodes={"alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args
        )
        smile2 = IRSabrSmile(
            shift=0, beta=0.75, nodes={"alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args
        )
        smile3 = IRSabrSmile(
            shift=0, beta=0.25, nodes={"alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args
        )
        smile4 = IRSabrSmile(
            shift=100, beta=0.5, nodes={"alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args
        )
        smile5 = IRSabrSmile(
            shift=200, beta=0.5, nodes={"alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args
        )

        # calibrate each smile similarly
        smile_solver_factory(smile1)
        smile_solver_factory(smile2)
        smile_solver_factory(smile3)
        smile_solver_factory(smile4)
        smile_solver_factory(smile5)

        fig, ax, lines = smile1.plot(
            curves=curve, y_axis="normal_vol", comparators=[smile2, smile3, smile4, smile5]
        )

        pp1 = PPSplineF64(k=2, t=[lines[0]._x[0]] + lines[0]._x.tolist() + [lines[0]._x[-1]])
        pp1.csolve(tau=lines[0]._x, y=lines[0]._y, left_n=0, right_n=0, allow_lsq=False)
        pp2 = PPSplineF64(k=2, t=[lines[1]._x[0]] + lines[1]._x.tolist() + [lines[1]._x[-1]])
        pp2.csolve(tau=lines[1]._x, y=lines[1]._y, left_n=0, right_n=0, allow_lsq=False)
        pp3 = PPSplineF64(k=2, t=[lines[2]._x[0]] + lines[2]._x.tolist() + [lines[2]._x[-1]])
        pp3.csolve(tau=lines[2]._x, y=lines[2]._y, left_n=0, right_n=0, allow_lsq=False)
        pp4 = PPSplineF64(k=2, t=[lines[3]._x[0]] + lines[3]._x.tolist() + [lines[3]._x[-1]])
        pp4.csolve(tau=lines[3]._x, y=lines[3]._y, left_n=0, right_n=0, allow_lsq=False)
        pp5 = PPSplineF64(k=2, t=[lines[4]._x[0]] + lines[4]._x.tolist() + [lines[4]._x[-1]])
        pp5.csolve(tau=lines[4]._x, y=lines[4]._y, left_n=0, right_n=0, allow_lsq=False)

        x = np.linspace(3.50, 4.40, 101)
        comparators = [pp2, pp3, pp4, pp5]
        for pp in comparators:
            eps = np.array([abs(pp1.ppev_single(_) - pp.ppev_single(_)) for _ in x])
            assert eps.max() < 0.3
            assert eps.mean() < 0.08

    def test_d_sigma_d_f(self):
        irss = IRSabrSmile(
            eval_date=dt(2000, 1, 1),
            expiry=dt(2000, 7, 1),
            tenor="1y",
            irs_series="usd_irs",
            beta=0.5,
            nodes=dict(alpha=0.2, rho=-0.05, nu=0.65),
            shift=0.0,
        )
        result = irss._d_sigma_d_f(k=0.8, f=1.0)
        manual = irss.get_from_strike(k=0.8, f=Dual(1.0, ["f"], []))
        manual_gradient = gradient(manual.vol, ["f"])[0] / 100.0
        assert abs(result - manual_gradient) < 2e-3

    def test_time_scalar(self):
        irss = IRSabrSmile(
            eval_date=dt(2000, 1, 1),
            expiry=dt(2000, 7, 1),
            tenor="1y",
            irs_series="usd_irs",
            beta=0.5,
            nodes=dict(alpha=0.2, rho=-0.05, nu=0.65),
            shift=0.0,
            time_scalar=0.9,
        )
        assert irss.meta.t_expiry == 0.9 * (31 + 29 + 31 + 30 + 31 + 30) / 365


class TestIRSabrCube:
    def test_init(self):
        IRSabrCube(
            eval_date=dt(2026, 2, 16),
            expiries=["1m", "3m"],
            tenors=["1Y", "2y", "3y"],
            irs_series="usd_irs",
            id="usd_ir_vol",
            beta=0.5,
            alpha=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
            rho=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
            nu=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
        )
        pass

    @pytest.mark.parametrize(("ad", "klass"), [(1, Dual), (2, Dual2)])
    def test_constructed_sabr_smile_vars(self, ad, klass):
        irsc = IRSabrCube(
            eval_date=dt(2026, 2, 20),
            expiries=["1m", "3m"],
            tenors=["2y", "5y"],
            irs_series="usd_irs",
            beta=0.5,
            alpha=0.05,
            rho=-0.01,
            nu=0.01,
            ad=ad,
            id="my-c",
        )
        _ = irsc.get_from_strike(k=1.0, f=1.02, expiry=dt(2026, 3, 30), tenor=dt(2028, 8, 12))
        smile = irsc._cache[(dt(2026, 3, 30), dt(2028, 8, 12))]
        assert smile.nodes.alpha.vars == ["my-c_a_0_0", "my-c_a_0_1", "my-c_a_1_0", "my-c_a_1_1"]
        assert smile.nodes.rho.vars == ["my-c_p_0_0", "my-c_p_0_1", "my-c_p_1_0", "my-c_p_1_1"]
        assert smile.nodes.nu.vars == ["my-c_v_0_0", "my-c_v_0_1", "my-c_v_1_0", "my-c_v_1_1"]
        assert isinstance(smile.nodes.alpha, klass)

    @pytest.mark.parametrize(
        ("expiry", "tenor", "expected"),
        [
            # tests on a node directly
            (dt(2001, 1, 1), dt(2002, 1, 1), (0.1, 1.0, 10.0)),
            (dt(2002, 1, 1), dt(2003, 1, 1), (0.3, 3.0, 30.0)),
            (dt(2001, 1, 1), dt(2003, 1, 1), (0.2, 2.0, 20.0)),
            (dt(2002, 1, 1), dt(2004, 1, 1), (0.4, 4.0, 40.0)),
            # test within bounds
            (
                dt(2001, 4, 1),
                dt(2002, 7, 1),
                (0.17424657534246576, 1.7424657534246577, 17.424657534246577),
            ),
            (
                dt(2001, 4, 1),
                dt(2003, 1, 1),
                (0.22465753424657536, 2.2465753424657535, 22.46575342465753),
            ),
            (
                dt(2001, 10, 1),
                dt(2003, 1, 1),
                (0.27479452054794523, 2.747945205479452, 27.47945205479452),
            ),
            (
                dt(2001, 10, 1),
                dt(2003, 7, 1),
                (0.32438356164383564, 3.243835616438356, 32.43835616438356),
            ),
            # test out of bounds
            (dt(2000, 7, 1), dt(2001, 1, 1), (0.1, 1.0, 10.0)),  # 6m6m
            (
                dt(2000, 7, 1),
                dt(2002, 1, 1),
                (0.1504109589041096, 1.504109589041096, 15.04109589041096),
            ),  # 6m18m
            (dt(2000, 7, 1), dt(2003, 7, 1), (0.2, 2.0, 20.0)),  # 6m3y
            (
                dt(2001, 7, 1),
                dt(2002, 1, 1),
                (0.1991780821917808, 1.9917808219178081, 19.91780821917808),
            ),  # 18m6m
            (
                dt(2001, 7, 1),
                dt(2004, 7, 1),
                (0.2991780821917808, 2.991780821917808, 29.91780821917808),
            ),  # 18m3y
            (dt(2003, 1, 1), dt(2003, 7, 1), (0.30, 3.0, 30.0)),  # 3y6m
            (
                dt(2003, 1, 1),
                dt(2004, 7, 1),
                (0.34986301369863015, 3.4986301369863018, 34.986301369863014),
            ),  # 3y18m
            (dt(2003, 1, 1), dt(2006, 1, 1), (0.4, 4.0, 40.0)),  # 3y3y
        ],
    )
    def test_interpolation_boundaries(self, expiry, tenor, expected):
        # test that the SabrCube will interpolate the parameters if the expiry and tenors are
        # - exactly falling on node dates
        # - some elements within the node-mesh
        # - some elements outside the node-mesh which are mapped to nearest components.
        irsc = IRSabrCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y", "2y"],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            beta=0.5,
            alpha=np.array([[0.1, 0.2], [0.3, 0.4]]),
            rho=np.array([[1.0, 2.0], [3.0, 4.0]]),
            nu=np.array([[10.0, 20.0], [30.0, 40.0]]),
            id="my-c",
        )
        result = tuple(irsc._bilinear_interpolation(expiry=expiry, tenor=tenor))
        assert result == expected

    @pytest.mark.parametrize(
        ("expiry", "tenor", "expected"),
        [
            (dt(2000, 7, 1), dt(2001, 1, 1), (0.1, 1.0, 10.0)),
            (dt(2000, 7, 1), dt(2001, 7, 1), (0.1, 1.0, 10.0)),
            (
                dt(2000, 7, 1),
                dt(2002, 1, 1),
                (0.1504109589041096, 1.504109589041096, 15.04109589041096),
            ),
            (dt(2000, 7, 1), dt(2003, 7, 1), (0.2, 2.0, 20.0)),
            (dt(2001, 1, 1), dt(2001, 7, 1), (0.1, 1.0, 10.0)),
            (dt(2001, 1, 1), dt(2002, 1, 1), (0.1, 1.0, 10.0)),
            (
                dt(2001, 1, 1),
                dt(2002, 7, 1),
                (0.1495890410958904, 1.495890410958904, 14.95890410958904),
            ),
            (dt(2001, 1, 1), dt(2003, 7, 1), (0.2, 2.0, 20.0)),
            (dt(2002, 1, 1), dt(2002, 7, 1), (0.1, 1.0, 10.0)),
            (dt(2002, 1, 1), dt(2003, 1, 1), (0.1, 1.0, 10.0)),
            (
                dt(2002, 1, 1),
                dt(2003, 7, 1),
                (0.1495890410958904, 1.495890410958904, 14.95890410958904),
            ),
            (dt(2002, 1, 1), dt(2004, 7, 1), (0.2, 2.0, 20.0)),
        ],
    )
    def test_interpolation_single_expiry(self, expiry, tenor, expected):
        # test that the SabrCube will interpolate the parameters if the expiry and tenors are
        # - exactly falling on node dates
        # - some elements within the node-mesh
        # - some elements outside the node-mesh which are mapped to nearest components.
        irsc = IRSabrCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y"],
            tenors=["1y", "2y"],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            beta=0.5,
            alpha=np.array([[0.1, 0.2]]),
            rho=np.array([[1.0, 2.0]]),
            nu=np.array([[10.0, 20.0]]),
            id="my-c",
        )
        result = tuple(irsc._bilinear_interpolation(expiry=expiry, tenor=tenor))
        assert result == expected

    @pytest.mark.parametrize(
        ("expiry", "tenor", "expected"),
        [
            (dt(2000, 7, 1), dt(2001, 1, 1), (0.1, 1.0, 10.0)),
            (dt(2000, 7, 1), dt(2001, 7, 1), (0.1, 1.0, 10.0)),
            (dt(2000, 7, 1), dt(2002, 1, 1), (0.1, 1.0, 10.0)),
            (dt(2001, 1, 1), dt(2001, 7, 1), (0.1, 1.0, 10.0)),
            (dt(2001, 1, 1), dt(2002, 1, 1), (0.1, 1.0, 10.0)),
            (dt(2001, 1, 1), dt(2002, 7, 1), (0.1, 1.0, 10.0)),
            (
                dt(2001, 7, 1),
                dt(2002, 1, 1),
                (0.1495890410958904, 1.495890410958904, 14.95890410958904),
            ),
            (
                dt(2001, 7, 1),
                dt(2002, 7, 1),
                (0.1495890410958904, 1.495890410958904, 14.95890410958904),
            ),
            (
                dt(2001, 7, 1),
                dt(2003, 1, 1),
                (0.1495890410958904, 1.495890410958904, 14.95890410958904),
            ),
            (dt(2002, 7, 1), dt(2003, 1, 1), (0.2, 2.0, 20.0)),
            (dt(2002, 7, 1), dt(2003, 7, 1), (0.2, 2.0, 20.0)),
            (dt(2002, 7, 1), dt(2004, 7, 1), (0.2, 2.0, 20.0)),
        ],
    )
    def test_interpolation_single_tenor(self, expiry, tenor, expected):
        # test that the SabrCube will interpolate the parameters if the expiry and tenors are
        # - exactly falling on node dates
        # - some elements within the node-mesh
        # - some elements outside the node-mesh which are mapped to nearest components.
        irsc = IRSabrCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y"],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            beta=0.5,
            alpha=np.array([[0.1], [0.2]]),
            rho=np.array([[1.0], [2.0]]),
            nu=np.array([[10.0], [20.0]]),
            id="my-c",
        )
        result = tuple(irsc._bilinear_interpolation(expiry=expiry, tenor=tenor).tolist())
        assert result == expected

    def test_alpha(self):
        irsc = IRSabrCube(
            eval_date=dt(2026, 2, 16),
            expiries=["1m", "3m"],
            tenors=["1Y", "2Y"],
            irs_series="usd_irs",
            id="usd_ir_vol",
            beta=0.5,
            alpha=np.array([[0.1, 0.2], [0.11, 0.12]]),
            rho=np.array([[0.1, 0.3], [0.11, 0.12]]),
            nu=np.array([[0.1, 0.4], [0.11, 0.12]]),
        )
        expected = DataFrame(
            index=Index(["1m", "3m"], name="expiry"),
            columns=Index(["1Y", "2Y"], name="tenor"),
            data=[[0.1, 0.2], [0.11, 0.12]],
            dtype=object,
        )
        assert_frame_equal(expected, irsc.alpha)
        expected = DataFrame(
            index=Index(["1m", "3m"], name="expiry"),
            columns=Index(["1Y", "2Y"], name="tenor"),
            data=[[0.1, 0.3], [0.11, 0.12]],
            dtype=object,
        )
        assert_frame_equal(expected, irsc.rho)
        expected = DataFrame(
            index=Index(["1m", "3m"], name="expiry"),
            columns=Index(["1Y", "2Y"], name="tenor"),
            data=[[0.1, 0.4], [0.11, 0.12]],
            dtype=object,
        )
        assert_frame_equal(expected, irsc.nu)
        assert irsc._n == 12

    def test_cache(self):
        irsc = IRSabrCube(
            eval_date=dt(2026, 2, 16),
            expiries=["1m", "3m"],
            tenors=["1Y", "2Y"],
            irs_series="usd_irs",
            id="usd_ir_vol",
            beta=0.5,
            alpha=np.array([[0.1, 0.2], [0.11, 0.12]]),
            rho=np.array([[0.1, 0.3], [0.11, 0.12]]),
            nu=np.array([[0.1, 0.4], [0.11, 0.12]]),
        )
        irsc.get_from_strike(k=1.02, f=1.04, expiry=dt(2026, 3, 30), tenor=dt(2027, 8, 12))
        assert (dt(2026, 3, 30), dt(2027, 8, 12)) in irsc._cache

    def test_get_node_vector(self):
        irsc = IRSabrCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y", "2y"],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            beta=0.5,
            alpha=np.array([[0.1, 0.2], [0.3, 0.4]]),
            rho=np.array([[1.0, 2.0], [3.0, 4.0]]),
            nu=np.array([[10.0, 20.0], [30.0, 40.0]]),
            id="X",
        )
        result = irsc._get_node_vector()
        expected = np.array([0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3, 4, 10, 20, 30, 40])
        assert np.all(result == expected)

    def test_get_node_vector_ad1(self):
        irsc = IRSabrCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y", "2y"],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            beta=0.5,
            alpha=np.array([[0.1, 0.2], [0.3, 0.4]]),
            rho=np.array([[1.0, 2.0], [3.0, 4.0]]),
            nu=np.array([[10.0, 20.0], [30.0, 40.0]]),
            id="X",
            ad=1,
        )
        result = irsc._get_node_vector()
        assert result[2] == Dual(0.30, ["X_a_1_0"], [])
        assert result[9] == Dual(20.0, ["X_v_0_1"], [])

    def test_set_node_vector(self):
        irsc = IRSabrCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y", "2y"],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            beta=0.5,
            alpha=np.array([[0.1, 0.2], [0.3, 0.4]]),
            rho=np.array([[1.0, 2.0], [3.0, 4.0]]),
            nu=np.array([[10.0, 20.0], [30.0, 40.0]]),
            id="X",
        )
        irsc._set_node_vector(np.array([0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3, 4, 10, 20, 30, 40]), ad=1)
        result = irsc._get_node_vector()
        assert result[2] == Dual(0.30, ["X_a_1_0"], [])
        assert result[9] == Dual(20.0, ["X_v_0_1"], [])

    @pytest.mark.parametrize(
        ("weights", "expiries"),
        [
            (
                Series(index=[dt(2000, 1, 3), dt(2000, 1, 8), dt(2000, 1, 4)], data=0.0),
                [dt(2000, 1, 5), dt(2000, 1, 10), dt(2000, 1, 15)],
            ),
            (
                Series(index=[dt(2000, 1, 3), dt(2000, 1, 20), dt(2000, 1, 4)], data=0.0),
                [dt(2000, 1, 5), dt(2000, 1, 10), dt(2000, 1, 15)],
            ),
        ],
    )
    def test_weights_implementation(self, weights, expiries):
        result = _scale_weights(
            eval_date=dt(2000, 1, 1),
            weights=weights,
            expiries=expiries,
        )

        c = result.cumsum()
        for expiry in expiries:
            if expiry > c.index[-1]:
                assert c.iloc[-1] == (c.index[-1] - dt(2000, 1, 1)).days
            else:
                assert c[expiry] == (expiry - dt(2000, 1, 1)).days

        assert c.iloc[-1] == (c.index[-1] - dt(2000, 1, 1)).days

    def test_weights(self):
        nyc = calendars.get("nyc")
        irsc = IRSabrCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y", "2y"],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            beta=0.5,
            alpha=np.array([[0.1, 0.2], [0.3, 0.4]]),
            rho=np.array([[1.0, 2.0], [3.0, 4.0]]),
            nu=np.array([[10.0, 20.0], [30.0, 40.0]]),
            id="X",
            weights=Series(
                index=[
                    _
                    for _ in nyc.cal_date_range(dt(2000, 1, 1), dt(2001, 2, 3))
                    if nyc.is_non_bus_day(_)
                ],
                data=0.0,
            ),
        )
        result = irsc.meta.time_scalars
        assert abs(result.iloc[-1] - 1.0) < 1e-14


class TestIRSplineSmile:
    @pytest.mark.parametrize(
        ("strike", "vol"),
        [
            (1.2034, 51.0888),
            (1.2050, 51.07599999999999),
            (1.3395, 50.0),  # f == k
            (1.3620, 50.2475),
            (1.5410, 52.216499999999996),
            (1.5449, 52.2594),
        ],
    )
    def test_spline_vol(self, strike, vol):
        # repeat the same test developed for FXSabrSmile
        irss = IRSplineSmile(
            nodes={-200.0: 70.0, -100.0: 58, 0: 50.0, 100.0: 61, 200.0: 75.0},
            k=2,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
        )
        result = irss.get_from_strike(k=strike, f=1.3395).vol
        assert abs(result - vol) < 1e-2

    @pytest.mark.parametrize(
        ("strike", "vol"),
        [
            (1.01, 50.0),
            (1.85, 50.0),
            (1.3395, 50.0),  # f == k
        ],
    )
    @pytest.mark.parametrize("k", [2, 4])
    def test_spline_vol_flat(self, strike, vol, k):
        # repeat the same test developed for FXSabrSmile
        irss = IRSplineSmile(
            nodes={0: 50.0},
            k=k,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
        )
        result = irss.get_from_strike(k=strike, f=1.3395).vol
        assert abs(result - vol) < 1e-2

    @pytest.mark.parametrize("k", [2, 4])
    @pytest.mark.parametrize(
        ("nodes", "expected_k"),
        [
            ({0.0: 100.0}, 2),
            ({-10.0: 49.0, 10.0: 53.0}, 2),
            ({-25.0: 62, 0: 59, 25: 65}, None),
            ({-25.0: 64, -10: 60, 10: 61, 25: 66}, None),
        ],
    )
    def test_spline_construction(self, k, nodes, expected_k):
        irss = IRSplineSmile(
            nodes=nodes,
            k=k,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
        )
        expected_k = expected_k or k
        for key, v in nodes.items():
            result = irss.get_from_strike(k=key / 100.0, f=0.0).vol
            assert abs(result - v) < 1e-6
            assert irss.nodes.spline.k == expected_k

    @pytest.mark.parametrize(
        ("model", "metric"), [("black76", "black_vol_shift_0"), ("bachelier", "normal_vol")]
    )
    def test_pricing_model(self, model, metric):
        irss = IRSplineSmile(
            nodes={0: 20.0},
            k=2,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="usd_irs",
            tenor="3m",
            id="vol",
            pricing_model=model,
        )
        curve = Curve({dt(2001, 1, 1): 1.0, dt(2003, 1, 1): 0.94})
        iro = IRSCall(
            expiry=dt(2002, 1, 1),
            tenor="3m",
            irs_series="usd_irs",
            strike=3.0,
        )
        result = iro.rate(vol=irss, curves=curve, metric=metric)
        expected = 20.0
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize("model", ["black76", "bachelier"])
    @pytest.mark.parametrize("k", [2, 4])
    def test_d_sigma_d_f(self, model, k):
        irss = IRSplineSmile(
            nodes={-200.0: 70.0, -100.0: 58, 0: 50.0, 100.0: 61, 200.0: 75.0},
            k=k,
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
            pricing_model=model,
        )
        result = irss._d_sigma_d_f(k=0.8, f=1.0)
        dual = irss.nodes.spline.evaluate(x=(0.8 - Dual(1.0, ["f"], [])) * 100.0, m=0)
        manual_gradient = gradient(dual, ["f"])[0] / 100.0
        assert abs(result - manual_gradient) < 1e-10

    def test_time_scalar(self):
        irss = IRSplineSmile(
            nodes={-200.0: 70.0, -100.0: 58, 0: 50.0, 100.0: 61, 200.0: 75.0},
            k=2,
            eval_date=dt(2000, 1, 1),
            expiry=dt(2000, 7, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
            time_scalar=0.9,
        )
        assert irss.meta.t_expiry == 0.9 * (31 + 29 + 31 + 30 + 31 + 30) / 365


class TestIRSplineCube:
    def test_init(self):
        IRSplineCube(
            eval_date=dt(2026, 2, 16),
            expiries=["1m", "3m"],
            tenors=["1Y", "2y", "3y"],
            strikes=[-100.0, 0.0, 100.0],
            irs_series="usd_irs",
            id="usd_ir_vol",
            parameters=20.0,
        )
        pass

    @pytest.mark.parametrize(("ad", "klass"), [(1, Dual), (2, Dual2)])
    def test_constructed_spline_smile_vars(self, ad, klass):
        irsc = IRSplineCube(
            eval_date=dt(2026, 2, 20),
            expiries=["1m", "3m"],
            tenors=["2y", "5y"],
            strikes=[-10.0],
            irs_series="usd_irs",
            parameters=10.0,
            ad=ad,
            id="my-c",
        )
        _ = irsc.get_from_strike(k=1.0, f=1.02, expiry=dt(2026, 3, 30), tenor=dt(2028, 8, 12))
        smile = irsc._cache[(dt(2026, 3, 30), dt(2028, 8, 12))]
        vars_ = smile.pricing_params[0].vars
        assert vars_ == ["my-c0", "my-c1", "my-c2", "my-c3"]
        assert isinstance(smile.pricing_params[0], klass)

    @pytest.mark.parametrize(
        ("expiry", "tenor", "expected"),
        [
            # tests on a node directly
            (dt(2001, 1, 1), dt(2002, 1, 1), (10.0,)),
            (dt(2002, 1, 1), dt(2003, 1, 1), (30.0,)),
            (dt(2001, 1, 1), dt(2003, 1, 1), (20.0,)),
            (dt(2002, 1, 1), dt(2004, 1, 1), (40.0,)),
            # test within bounds
            (dt(2001, 4, 1), dt(2002, 7, 1), (17.424657534246577,)),
            (
                dt(2001, 4, 1),
                dt(2003, 1, 1),
                (22.46575342465753,),
            ),
            (
                dt(2001, 10, 1),
                dt(2003, 1, 1),
                (27.47945205479452,),
            ),
            (
                dt(2001, 10, 1),
                dt(2003, 7, 1),
                (32.43835616438356,),
            ),
            # test out of bounds
            (dt(2000, 7, 1), dt(2001, 1, 1), (10.0,)),  # 6m6m
            (
                dt(2000, 7, 1),
                dt(2002, 1, 1),
                (15.04109589041096,),
            ),  # 6m18m
            (dt(2000, 7, 1), dt(2003, 7, 1), (20.0,)),  # 6m3y
            (
                dt(2001, 7, 1),
                dt(2002, 1, 1),
                (19.91780821917808,),
            ),  # 18m6m
            (
                dt(2001, 7, 1),
                dt(2004, 7, 1),
                (29.91780821917808,),
            ),  # 18m3y
            (dt(2003, 1, 1), dt(2003, 7, 1), (30.0,)),  # 3y6m
            (
                dt(2003, 1, 1),
                dt(2004, 7, 1),
                (34.986301369863014,),
            ),  # 3y18m
            (dt(2003, 1, 1), dt(2006, 1, 1), (40.0,)),  # 3y3y
        ],
    )
    def test_interpolation_boundaries(self, expiry, tenor, expected):
        # test that the SplineCube will interpolate the parameters if the expiry and tenors are
        # - exactly falling on node dates
        # - some elements within the node-mesh
        # - some elements outside the node-mesh which are mapped to nearest components.
        irsc = IRSplineCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y", "2y"],
            strikes=[0.0],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            parameters=np.reshape(np.array([10.0, 20.0, 30.0, 40.0]), (2, 2, 1)),
            id="my-c",
        )
        result = tuple(irsc._bilinear_interpolation(expiry=expiry, tenor=tenor))
        assert result == expected

    @pytest.mark.parametrize(
        ("expiry", "tenor", "expected"),
        [
            (dt(2000, 7, 1), dt(2001, 1, 1), (10.0,)),
            (dt(2000, 7, 1), dt(2001, 7, 1), (10.0,)),
            (
                dt(2000, 7, 1),
                dt(2002, 1, 1),
                (15.04109589041096,),
            ),
            (dt(2000, 7, 1), dt(2003, 7, 1), (20.0,)),
            (dt(2001, 1, 1), dt(2001, 7, 1), (10.0,)),
            (dt(2001, 1, 1), dt(2002, 1, 1), (10.0,)),
            (
                dt(2001, 1, 1),
                dt(2002, 7, 1),
                (14.95890410958904,),
            ),
            (dt(2001, 1, 1), dt(2003, 7, 1), (20.0,)),
            (dt(2002, 1, 1), dt(2002, 7, 1), (10.0,)),
            (dt(2002, 1, 1), dt(2003, 1, 1), (10.0,)),
            (
                dt(2002, 1, 1),
                dt(2003, 7, 1),
                (14.95890410958904,),
            ),
            (dt(2002, 1, 1), dt(2004, 7, 1), (20.0,)),
        ],
    )
    def test_interpolation_single_expiry(self, expiry, tenor, expected):
        # test that the SplineCube will interpolate the parameters if the expiry and tenors are
        # - exactly falling on node dates
        # - some elements within the node-mesh
        # - some elements outside the node-mesh which are mapped to nearest components.
        irsc = IRSplineCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y"],
            tenors=["1y", "2y"],
            strikes=[0.0],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            parameters=np.reshape(np.array([10.0, 20.0]), (1, 2, 1)),
            id="my-c",
        )
        result = tuple(irsc._bilinear_interpolation(expiry=expiry, tenor=tenor))
        assert result == expected

    @pytest.mark.parametrize(
        ("expiry", "tenor", "expected"),
        [
            (dt(2000, 7, 1), dt(2001, 1, 1), (10.0,)),
            (dt(2000, 7, 1), dt(2001, 7, 1), (10.0,)),
            (dt(2000, 7, 1), dt(2002, 1, 1), (10.0,)),
            (dt(2001, 1, 1), dt(2001, 7, 1), (10.0,)),
            (dt(2001, 1, 1), dt(2002, 1, 1), (10.0,)),
            (dt(2001, 1, 1), dt(2002, 7, 1), (10.0,)),
            (
                dt(2001, 7, 1),
                dt(2002, 1, 1),
                (14.95890410958904,),
            ),
            (
                dt(2001, 7, 1),
                dt(2002, 7, 1),
                (14.95890410958904,),
            ),
            (
                dt(2001, 7, 1),
                dt(2003, 1, 1),
                (14.95890410958904,),
            ),
            (dt(2002, 7, 1), dt(2003, 1, 1), (20.0,)),
            (dt(2002, 7, 1), dt(2003, 7, 1), (20.0,)),
            (dt(2002, 7, 1), dt(2004, 7, 1), (20.0,)),
        ],
    )
    def test_interpolation_single_tenor(self, expiry, tenor, expected):
        # test that the SplineCube will interpolate the parameters if the expiry and tenors are
        # - exactly falling on node dates
        # - some elements within the node-mesh
        # - some elements outside the node-mesh which are mapped to nearest components.
        irsc = IRSplineCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y"],
            strikes=[0.0],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            parameters=np.reshape(np.array([10.0, 20.0]), (2, 1, 1)),
            id="my-c",
        )
        result = tuple(irsc._bilinear_interpolation(expiry=expiry, tenor=tenor).tolist())
        assert result == expected

    def test_cache(self):
        irsc = IRSplineCube(
            eval_date=dt(2026, 2, 16),
            expiries=["1m", "3m"],
            tenors=["1Y", "2Y"],
            strikes=[-10.0, 0.0, 10.0],
            irs_series="usd_irs",
            id="usd_ir_vol",
            parameters=10.0,
        )
        irsc.get_from_strike(k=1.02, f=1.04, expiry=dt(2026, 3, 30), tenor=dt(2027, 8, 12))
        assert (dt(2026, 3, 30), dt(2027, 8, 12)) in irsc._cache

    def test_get_node_vector(self):
        irsc = IRSplineCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y", "2y"],
            strikes=[-10.0, 0.0],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            parameters=np.reshape(np.array([1, 2, 3, 4, 5, 6, 7, 8]), (2, 2, 2)),
            id="X",
        )
        result = irsc._get_node_vector()
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        assert irsc.get_smile("1y", "1y").pricing_params == [np.float64(1.0), np.float64(2.0)]
        assert np.all(result == expected)

    def test_get_node_vector_ad1(self):
        irsc = IRSplineCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y", "2y"],
            strikes=[0.0, 10.0],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            parameters=10.0,
            id="X",
            ad=1,
        )
        result = irsc._get_node_vector()
        assert result[2] == Dual(10.0, ["X2"], [])
        assert result[7] == Dual(10.0, ["X7"], [])

    def test_set_node_vector(self):
        irsc = IRSplineCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y", "2y"],
            strikes=[-10.0, 0.0],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            parameters=10.0,
            id="X",
        )
        irsc._set_node_vector(np.array([0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3, 4]), ad=1)
        result = irsc._get_node_vector()
        assert result[2] == Dual(0.30, ["X2"], [])
        assert result[7] == Dual(4, ["X7"], [])

    @pytest.mark.skip(reason="no decision on how to use _set_ad_order for manually updated nodes.")
    def test_update_single_key(self):
        # TODO need to decide how _set_or_ad should work with update nodes.
        irsc = IRSplineCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y", "2y"],
            strikes=[-10.0, 0.0],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            parameters=10.0,
            id="X",
            ad=1,
        )
        irsc.update_node(("1y", "1y", -10.0), 20.0)
        result = irsc._get_node_vector()
        assert result[0] == Dual(20.0, ["X0"], [])

    @pytest.mark.parametrize(
        ("model", "metric"), [("black76", "black_vol_shift_0"), ("bachelier", "normal_vol")]
    )
    def test_pricing_model(self, model, metric):
        irss = IRSplineCube(
            parameters=[[[20.0]]],
            k=2,
            eval_date=dt(2001, 1, 1),
            irs_series="usd_irs",
            expiries=["1y"],
            tenors=["3m"],
            strikes=[0.0],
            id="vol",
            pricing_model=model,
        )
        curve = Curve({dt(2001, 1, 1): 1.0, dt(2003, 1, 1): 0.94})
        iro = IRSCall(
            expiry=dt(2002, 1, 1),
            tenor="3m",
            irs_series="usd_irs",
            strike=3.0,
        )
        result = iro.rate(vol=irss, curves=curve, metric=metric)
        expected = 20.0
        assert abs(result - expected) < 1e-6

    def test_business_day_time_and_weights(self):
        nyc = calendars.get("nyc")
        irsc = IRSplineCube(
            eval_date=dt(2000, 1, 3),
            expiries=["1m", "3m", "6m"],
            tenors=["1y"],
            strikes=[0],
            parameters=[[[30.0]], [[35.0]], [[38.0]]],
            irs_series="usd_irs",
        )
        irsc2 = IRSplineCube(
            eval_date=dt(2000, 1, 3),
            expiries=["1m", "3m", "6m"],
            tenors=["1y"],
            strikes=[0],
            parameters=[[[30.0]], [[35.0]], [[38.0]]],
            irs_series="usd_irs",
            weights=Series(
                index=[
                    _
                    for _ in nyc.cal_date_range(dt(2000, 1, 7), dt(2000, 7, 15))
                    if nyc.is_non_bus_day(_)
                ],
                data=0.0,
            ),
        )
        curve = Curve(
            nodes={dt(2000, 1, 3): 1.0, dt(2002, 1, 3): 0.93},
            convention="act360",
            calendar="nyc",
        )
        for expiry in irsc.meta.expiry_dates:
            # test at expiries time remapping does not exist because these are the natural pillars
            iro = IRSCall(
                expiry=expiry,
                strike="atm",
                irs_series="usd_irs",
                tenor="1y",
            )
            r1 = iro.rate(curves=curve, vol=irsc, metric="percentnotional") * 100.0
            r2 = iro.rate(curves=curve, vol=irsc2, metric="percentnotional") * 100.0
            assert abs(r1 - r2) < 1e-8

        for expiry in [dt(2000, 1, 14), dt(2000, 2, 18), dt(2000, 5, 12)]:
            # test at expiries inbetween the time remapping exists
            iro = IRSCall(
                expiry=expiry,
                strike="atm",
                irs_series="usd_irs",
                tenor="1y",
            )
            r1 = iro.rate(curves=curve, vol=irsc, metric="percentnotional") * 100.0
            r2 = iro.rate(curves=curve, vol=irsc2, metric="percentnotional") * 100.0
            assert abs(r1 - r2) > 1e-3

        for expiry in [dt(2000, 7, 20), dt(2000, 7, 25)]:
            # test after weights stop being defined
            iro = IRSCall(
                expiry=expiry,
                strike="atm",
                irs_series="usd_irs",
                tenor="1y",
            )
            r1 = iro.rate(curves=curve, vol=irsc, metric="percentnotional") * 100.0
            r2 = iro.rate(curves=curve, vol=irsc2, metric="percentnotional") * 100.0
            assert abs(r1 - r2) < 1e-8


class TestStateAndCache:
    @pytest.mark.parametrize(
        "obj",
        [
            IRSabrSmile(
                nodes={
                    "alpha": 0.1,
                    "rho": -0.05,
                    "nu": 0.1,
                },
                beta=0.5,
                eval_date=dt(2001, 1, 1),
                expiry=dt(2002, 1, 1),
                irs_series="eur_irs6",
                tenor="2y",
                id="v",
                ad=2,
            ),
            IRSabrCube(
                eval_date=dt(2026, 2, 16),
                expiries=["1m", "3m"],
                tenors=["1Y", "2y", "3y"],
                irs_series="usd_irs",
                id="usd_ir_vol",
                beta=0.5,
                alpha=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
                rho=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
                nu=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
            ),
        ],
    )
    @pytest.mark.parametrize(("method", "args"), [("_set_ad_order", (1,))])
    def test_method_does_not_change_state(self, obj, method, args):
        before = obj._state
        getattr(obj, method)(*args)
        after = obj._state
        assert before == after

    @pytest.mark.parametrize(
        "obj",
        [
            IRSabrSmile(
                nodes={
                    "alpha": 0.1,
                    "rho": -0.05,
                    "nu": 0.1,
                },
                beta=0.5,
                eval_date=dt(2001, 1, 1),
                expiry=dt(2002, 1, 1),
                irs_series="eur_irs6",
                tenor="2y",
                id="v",
                ad=2,
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98, 0.99], 1)),
            ("update_node", ("alpha", 0.98)),
        ],
    )
    def test_method_changes_state(self, obj, method, args):
        before = obj._state
        getattr(obj, method)(*args)
        after = obj._state
        assert before != after

    @pytest.mark.parametrize(
        "curve",
        [
            IRSabrSmile(
                nodes={
                    "alpha": 0.1,
                    "rho": -0.05,
                    "nu": 0.1,
                },
                beta=0.5,
                eval_date=dt(2001, 1, 1),
                expiry=dt(2002, 1, 1),
                irs_series="eur_irs6",
                tenor="2y",
                id="v",
                ad=2,
            ),
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

    @pytest.mark.parametrize(
        "curve",
        [
            IRSabrCube(
                eval_date=dt(2026, 2, 16),
                expiries=["1m"],
                tenors=["1Y"],
                irs_series="usd_irs",
                id="usd_ir_vol",
                beta=0.5,
                alpha=np.array([[0.1]]),
                rho=np.array([[0.2]]),
                nu=np.array([[0.3]]),
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98, 0.99], 1)),
            ("update_node", ((dt(2026, 3, 16), dt(2027, 3, 18), "alpha"), 0.98)),
        ],
    )
    def test_method_changes_state_sabr_cube(self, curve, method, args):
        before = curve._state
        getattr(curve, method)(*args)
        after = curve._state
        assert before != after

    #
    #     def test_populate_cache(self):
    #         # objects have yet to implement cache handling
    #         pass
    #
    #     def test_method_clears_cache(self):
    #         # objects have yet to implement cache handling
    #         pass
    #
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98, 1.0], 1)),
            ("_set_ad_order", (2,)),
        ],
    )
    def test_surface_clear_cache(self, method, args):
        surf = IRSabrCube(
            eval_date=dt(2026, 2, 16),
            expiries=["1m"],
            tenors=["1Y"],
            irs_series="usd_irs",
            id="usd_ir_vol",
            beta=0.5,
            alpha=np.array([[0.1]]),
            rho=np.array([[0.2]]),
            nu=np.array([[0.3]]),
        )
        surf.get_from_strike(f=1.0, k=1.01, expiry=dt(2026, 3, 1), tenor=dt(2027, 3, 1))
        assert (dt(2026, 3, 1), dt(2027, 3, 1)) in surf._cache

        getattr(surf, method)(*args)
        assert len(surf._cache) == 0


class TestPricingModelConversion:
    class TestBachelier:
        @pytest.mark.parametrize(
            ("vol", "k", "shift", "expected"),
            [
                (25.0, 2.99, 0.0, 8.3496780104),
                (25.0, 2.99, 50.0, 7.15460637959775),
                (25.0, 2.99, 200.0, 5.005529190687043),
                (25.0, 1.50, 0.0, 11.615241673583585),
                (25.0, 1.50, 50.0, 9.312911744191437),
                (25.0, 1.50, 200.0, 5.9394076088397645),
                (25.0, 4.50, 0.0, 6.753315378082834),
                (25.0, 4.50, 50.0, 5.9394076088397645),
                (25.0, 4.50, 200.0, 4.368303987428187),
            ],
        )
        def test_convert_to_black_no_shift(self, vol, k, shift, expected):
            result = _OptionModelBachelier.convert_to_black76(
                f=3.0, k=k, shift=shift, vol=vol, t_e=1.0
            )
            assert abs(result - expected) < 1e-6

    class TestBlack76:
        @pytest.mark.parametrize(
            ("vol", "k", "shift", "expected"),
            [
                (25.0, 2.99, 0.0, 74.68039981110007),
                (25.0, 2.99, 50.0, 87.14793380301037),
                (25.0, 2.99, 200.0, 124.55052385921005),
                (25.0, 1.50, 0.0, 53.96106256666565),
                (25.0, 1.50, 50.0, 66.8366143175683),
                (25.0, 1.50, 200.0, 104.86487953597288),
                (25.0, 4.50, 0.0, 92.24642085914786),
                (25.0, 4.50, 50.0, 104.86487953597292),
                (25.0, 4.50, 200.0, 142.55991748648242),
            ],
        )
        def test_convert_to_bachelier(self, vol, k, shift, expected):
            result = _OptionModelBlack76.convert_to_bachelier(
                f=3.0, k=k, shift=shift, vol=vol, t_e=1.0
            )
            assert abs(result - expected) < 1e-9

        @pytest.mark.parametrize(
            ("vol", "k", "shift", "tgt", "expected"),
            [
                (25.0, 2.99, 0.0, 50.0, 21.40861097419223),
                (25.0, 2.99, 50.0, 100.0, 21.85769609359381),
                (25.0, 2.99, 200.0, 100.0, 31.30396613960251),
                (25.0, 1.50, 0.0, 50.0, 20.16566976523089),
                (25.0, 1.50, 50.0, 100.0, 20.980647995758154),
                (25.0, 1.50, 200.0, 100.0, 33.00686423510773),
                (25.0, 4.50, 0.0, 50.0, 21.9787696869096),
                (25.0, 4.50, 50.0, 100.0, 22.309213489533068),
                (25.0, 4.50, 200.0, 100.0, 30.382178316599756),
            ],
        )
        def test_convert_to_new_shift(self, vol, k, shift, tgt, expected):
            result = _OptionModelBlack76.convert_to_new_shift(
                f=3.0, k=k, old_shift=shift, target_shift=tgt, vol=vol, t_e=1.0
            )
            assert abs(result - expected) < 1e-9


@pytest.mark.skipif(
    sys.version_info[:2] == (3, 10), reason="This test is incompatible with Python 3.10"
)
class TestCookbokReplicators:
    def test_z_ir_vol_risks(self):
        curve = Curve(
            nodes={
                dt(2026, 3, 17): 1.0,
                dt(2026, 9, 17): 1.0,
                dt(2027, 3, 17): 1.0,
                dt(2028, 3, 17): 1.0,
                dt(2029, 3, 17): 1.0,
                dt(2030, 3, 17): 1.0,
                dt(2031, 4, 17): 1.0,
            },
            convention="act360",
            calendar="nyc",
            interpolation="log_linear",
            id="sofr",
        )
        swap_tenors = ["6m", "1y", "2y", "3y", "4y", "5y"]
        curve_solver = Solver(
            curves=[curve],
            instruments=[
                IRS(dt(2026, 3, 17), _, spec="usd_irs", curves="sofr") for _ in swap_tenors
            ],
            s=[4.10, 4.02, 4.08, 4.12, 4.18, 4.22],
            instrument_labels=swap_tenors,
            id="us_rates",
        )

        expiries = ["6m", "1y", "2y"]
        tenors = ["3m", "1y", "2y"]
        pricing_cube = IRSabrCube(
            eval_date=dt(2026, 3, 17),
            expiries=expiries,
            tenors=tenors,
            irs_series="usd_irs",
            beta=0.5,  # <- beta is a hyper-parameter and applies globally to this Cube
            alpha=0.5,  # <- alpha as scalar applies the same value to each gridpoint automatically
            rho=[  # <- rho is provided in array format with a value at each gridpoint
                [0.4, 0.45, 0.29],
                [0.4, 0.4, 0.26],
                [0.3, 0.3, 0.25],
            ],
            nu=[  # <- nu is provided in array format with a value at each gridpoint
                [1.0, 0.98, 0.87],
                [0.9, 0.875, 0.7],
                [0.63, 0.6, 0.56],
            ],
            id="usd_cube",
        )
        pricing_solver = Solver(surfaces=[pricing_cube], pre_solvers=[curve_solver])
        iro = IRSPut(
            expiry=dt(2027, 3, 3),
            tenor="1y",
            irs_series="usd_irs",
            notional=125e6,
            strike=3.99,
            premium=400000,
            curves="sofr",
            vol="usd_cube",
            metric="normal_vol",
        )

        result = iro.npv(solver=pricing_solver)
        assert abs(result - 12988.135) < 1e-2

        result = iro.rate(solver=pricing_solver)
        assert abs(result - 103.889) < 1e-2

        expiries = ["3m", "1y", "2y"]  # <- expiries are different to those above
        tenors = ["1y", "2y"]  # <- tenors are also different
        strikes = [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0]  # <- strikes are bps to ATM
        risk_cube = IRSplineCube(
            eval_date=dt(2026, 3, 17),
            expiries=expiries,
            tenors=tenors,
            strikes=strikes,
            irs_series="usd_irs",
            parameters=25.0,  # <-  all normal vol values are initialised at 25bps
            id="usd_cube",
        )
        strikes_str = [f"{_}bps" for _ in strikes]
        args = dict(
            irs_series="eur_irs3",
            eval_date=dt(2026, 3, 11),
            metric="normal_vol",
            curves="sofr",
            vol="usd_cube",
        )
        instruments = [
            IRVolValue(e, t, k, **args) for e, t, k in product(expiries, tenors, strikes_str)
        ]
        instrument_labels = [f"{e}{t}_{k}" for e, t, k in product(expiries, tenors, strikes_str)]
        risk_solver = Solver.from_other(
            pricing_solver=pricing_solver,  # <- will determine our ``s`` rates directly
            surfaces=[risk_cube],
            instruments=instruments,
            instrument_labels=instrument_labels,
            id="us_vol",
            pre_solvers=[
                curve_solver
            ],  # <- the curve_solver is still needed to pass the SOFR Curve.
            grad_tol=1e-5,
            func_tol=1e-5,
            step_tol=1e-5,
        )
        df = iro.delta(solver=risk_solver)
        ix = IndexSlice
        delta = df.loc[ix[:, "us_rates"], :].sum(axis=None)
        vega = df.loc[ix[:, "us_vol"], :].sum(axis=None)
        gf = iro.gamma(solver=risk_solver)
        gamma = gf.loc[ix[:, :, :, "us_rates"], ix[:, "us_rates"]].sum(axis=None)
        vomma = gf.loc[ix[:, :, :, "us_vol"], ix[:, "us_vol"]].sum(axis=None)
        vanna = gf.loc[ix[:, :, :, "us_rates"], ix[:, "us_vol"]].sum(axis=None)

        agks = iro.analytic_greeks(solver=risk_solver)

        assert abs(agks["gamma_usd"] - gamma) < 5.5
        assert abs(agks["vega_usd"] - vega) < 1e-3
        assert abs(agks["vomma_usd"] - vomma) < 1e-3
        assert abs(agks["vanna_usd"] - vanna) < 1.1
        assert abs(agks["delta_sticky_usd"] - delta) < 42.0
