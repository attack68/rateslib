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

import os
from datetime import datetime as dt
from itertools import product

import numpy as np
import pytest
from pandas import DataFrame, Series, date_range
from pandas.testing import assert_frame_equal
from rateslib import defaults, fixings
from rateslib.curves import Curve, LineCurve
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, Variable, gradient
from rateslib.fx import FXForwards, FXRates
from rateslib.instruments import (
    IRS,
    Bill,
    BondFuture,
    FixedRateBond,
    FloatRateNote,
    IndexFixedRateBond,
)
from rateslib.instruments.bonds.conventions import US_GBB, BondCalcMode
from rateslib.instruments.protocols.pricing import _Curves
from rateslib.scheduling import dcf, get_calendar
from rateslib.solver import Solver


@pytest.fixture
def curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.99,
        dt(2022, 7, 1): 0.98,
        dt(2022, 10, 1): 0.97,
    }
    # convention = "Act360"
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture
def curve2():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.98,
        dt(2022, 7, 1): 0.97,
        dt(2022, 10, 1): 0.95,
    }
    return Curve(nodes=nodes, interpolation="log_linear")


class TestBondCalcMode:
    def test_custom_function(self):
        def _my_acc(*args):
            return 0.5

        my_calc = BondCalcMode(
            settle_accrual=_my_acc,
            ytm_accrual=_my_acc,
            v1="compounding",
            v2="regular",
            v3="compounding",
            c1="cashflow",
            ci="cashflow",
            cn="cashflow",
        )

        bond = FixedRateBond(dt(2022, 1, 1), "2y", spec="de_gb", fixed_rate=2.0, calc_mode=my_calc)
        de_bond = FixedRateBond(
            dt(2022, 1, 1),
            "2y",
            spec="de_gb",
            fixed_rate=2.0,
        )

        assert bond.accrued(dt(2022, 2, 4)) == 1.0  # 0.5 * 2.0
        assert bond.accrued(dt(2022, 2, 4)) != de_bond.accrued(dt(2022, 2, 4))

        assert bond.ytm(100.0, dt(2022, 2, 4)) != de_bond.ytm(100.0, dt(2022, 2, 4))

        assert my_calc.kwargs["settle_accrual"] == "custom"
        assert my_calc.kwargs["ytm_accrual"] == "custom"

    def test_custom_function_affects_ytm(self):
        def _my_acc(*args):
            return 0.4

        my_calc = BondCalcMode(
            settle_accrual="linear_days",
            ytm_accrual=_my_acc,
            v1="compounding_final_simple",
            v2="regular",
            v3="compounding",
            c1="cashflow",
            ci="cashflow",
            cn="cashflow",
        )

        bond = FixedRateBond(dt(2022, 1, 1), "2y", spec="de_gb", fixed_rate=2.0, calc_mode=my_calc)

        v2 = 1 / (1 + 0.02)
        v1 = v2 ** (1 - 0.4)
        expected = 2 * v1 + 102 * v1 * v2 - 0.4 * 2
        result = bond.price(ytm=2.00, settlement=dt(2022, 1, 1))

        assert abs(result - expected) < 1e-10

    def test_custom_ytm_disc_funcs(self):
        def _my_acc(*args):
            return 0.0

        def _v(*args):
            return 1 / (1 + 0.02)

        calc_mode = BondCalcMode(
            settle_accrual=_my_acc,
            ytm_accrual=_my_acc,
            v1=_v,
            v2=_v,
            v3=_v,
            c1="cashflow",
            ci="cashflow",
            cn="cashflow",
        )

        bond = FixedRateBond(
            effective=dt(2000, 1, 1),
            termination="2y",
            fixed_rate=2.00,
            spec="de_gb",
            calc_mode=calc_mode,
        )

        # custom funcs give the same clean price of 100 for any date
        for date in [dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 11, 1), dt(2001, 6, 1)]:
            result = bond.price(ytm=2.0, settlement=dt(2000, 1, 1))
            assert abs(result - 100.0) < 1e-10


class TestFixedRateBond:
    def test_metric_ytm_no_fx(self) -> None:
        # GH 193
        usd = Curve(nodes={dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.9, dt(2010, 1, 5): 0.8})
        gbp = Curve(nodes={dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.9, dt(2010, 1, 5): 0.8})
        fxf = FXForwards(
            fx_rates=FXRates({"gbpusd": 1.25}, settlement=dt(2000, 1, 1)),
            fx_curves={"gbpgbp": gbp, "usdusd": usd, "gbpusd": gbp},
        )
        expected = FixedRateBond(dt(2000, 1, 1), "10y", spec="uk_gb", fixed_rate=2.0).rate(
            curves=gbp,
            metric="ytm",
        )
        result = FixedRateBond(dt(2000, 1, 1), "10y", spec="uk_gb", fixed_rate=2.0).rate(
            curves=gbp,
            metric="ytm",
            fx=fxf,
        )
        assert abs(result - expected) < 1e-9

    def test_accrued_in_text(self) -> None:
        bond = FixedRateBond(
            effective=dt(2022, 1, 1),
            termination=dt(2023, 1, 1),
            fixed_rate=5.0,
            spec="ca_gb",
        )
        assert abs(bond.accrued(dt(2022, 4, 15)) - 1.42465753) < 1e-8

        bond = FixedRateBond(
            effective=dt(2022, 1, 1),
            termination=dt(2023, 1, 1),
            fixed_rate=5.0,
            spec="uk_gb",
        )
        assert abs(bond.accrued(dt(2022, 4, 15)) - 1.43646409) < 1e-8

    # UK Gilts Tests: Data from public DMO website.

    @pytest.mark.parametrize(
        ("settlement", "exp"),
        [
            (dt(1999, 5, 24), False),
            (dt(1999, 5, 26), False),
            (dt(1999, 5, 27), True),
            (dt(1999, 6, 7), True),  # on payment date the
        ],
    )
    def test_ex_div(self, settlement, exp) -> None:
        ukg = FixedRateBond(
            effective=dt(1998, 1, 1),
            termination=dt(2015, 12, 7),
            frequency="S",
            fixed_rate=8.0,
            convention="ActActICMA",
            calendar="ldn",
            ex_div=7,
            modifier="NONE",
        )
        assert ukg.ex_div(settlement) is exp

    def test_fixed_rate_bond_price_ukg(self) -> None:
        # test pricing functions against Gilt Example prices from UK DMO
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
        )
        assert abs(bond.price(4.445, dt(1999, 5, 24), True) - 145.012268) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 5, 26), True) - 145.047301) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 5, 27), True) - 141.070132) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 6, 7), True) - 141.257676) < 1e-6

        bond = FixedRateBond(
            dt(1997, 1, 1),
            dt(2004, 11, 26),
            "S",
            convention="ActActICMA",
            fixed_rate=6.75,
            ex_div=7,
            calendar="ldn",
            modifier="F",
        )
        assert abs(bond.price(4.634, dt(1999, 5, 10), True) - 113.315543) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 17), True) - 113.415969) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 18), True) - 110.058738) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 26), True) - 110.170218) < 1e-6

    def test_fixed_rate_bond_price_ukg_back_stub(self) -> None:
        bond = FixedRateBond(
            dt(1995, 12, 7),
            dt(2015, 1, 23),
            "S",
            stub="SHORTBACK",
            roll=7,
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
            calc_mode="ukg",
        )
        result = bond.price(ytm=8.00, settlement=dt(1995, 12, 7))
        expected = 100.00334028292  # compounded back stub does not yield par
        assert abs(result - expected) < 1e-9

    def test_fixed_rate_bond_yield_ukg(self) -> None:
        # test pricing functions against Gilt Example prices from UK DMO
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
        )
        assert bond.ytm(135.0, dt(1999, 5, 24), True) - 5.1620635 < 1e-6
        assert bond.ytm(135.0, dt(1999, 5, 26), True) - 5.1649111 < 1e-6
        assert bond.ytm(135.0, dt(1999, 5, 27), True) - 4.871425 < 1e-6
        assert bond.ytm(135.0, dt(1999, 6, 7), True) - 4.8856785 < 1e-6

        bond = FixedRateBond(
            dt(1997, 1, 1),
            dt(2004, 11, 26),
            "S",
            convention="ActActICMA",
            fixed_rate=6.75,
            ex_div=7,
            calendar="ldn",
            modifier="F",
        )
        assert bond.ytm(108.0, dt(1999, 5, 10), True) - 5.7009527 < 1e-6
        assert bond.ytm(108.0, dt(1999, 5, 17), True) - 5.7253361 < 1e-6
        assert bond.ytm(108.0, dt(1999, 5, 18), True) - 5.0413308 < 1e-6
        assert bond.ytm(108.0, dt(1999, 5, 26), True) - 5.0652248 < 1e-6

    def test_fixed_rate_bond_accrual(self) -> None:
        # test pricing functions against Gilt Example prices from UK DMO, with stub
        bond = FixedRateBond(
            dt(1999, 5, 7),
            dt(2002, 12, 7),
            "S",
            convention="ActActICMA",
            front_stub=dt(1999, 12, 7),
            fixed_rate=6,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
        )
        bond.accrued(dt(1999, 5, 8)) == 0.016484
        bond.accrued(dt(1999, 6, 8)) == 0.527382
        bond.accrued(dt(1999, 7, 8)) == 1.019186
        bond.accrued(dt(1999, 11, 8)) == 3.035579
        bond.accrued(dt(1999, 11, 26)) == 3.330661
        bond.accrued(dt(1999, 11, 27)) == -0.16393
        bond.accrued(dt(1999, 12, 6)) == -0.01639
        bond.accrued(dt(1999, 12, 7)) == 0.0

    def test_fixed_rate_bond_stub_ytm(self) -> None:
        # if a regular bond is set to stub similar output should be gotten
        bond = FixedRateBond(
            dt(1999, 6, 7),
            dt(2002, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=6,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
        )
        regular_ytm = bond.ytm(101, dt(1999, 11, 8), dirty=True)
        bond.leg1.periods[0].stub = True
        stubbed_ytm = bond.ytm(101, dt(1999, 11, 8), dirty=True)
        assert regular_ytm == stubbed_ytm

    # US Treasury Tests. Examples from Rulebook.

    @pytest.mark.parametrize(
        ("e", "t", "s", "fr", "ec", "ed", "y", "se"),
        [
            (
                dt(1990, 5, 15),
                dt(2020, 5, 15),
                NoInput(0),
                8.75,
                99.057893,
                99.057893,
                8.84,
                dt(1990, 5, 15),
            ),  # A
            (
                dt(1990, 4, 2),
                dt(1992, 3, 31),
                NoInput(0),
                8.5,
                99.838183,
                99.838183,
                8.59,
                dt(1990, 4, 2),
            ),  # B
            (
                dt(1990, 3, 1),
                dt(1995, 5, 15),
                dt(1990, 11, 15),
                8.5,
                99.805118,
                99.805118,
                8.53,
                dt(1990, 3, 1),
            ),  # C
            (
                dt(1985, 11, 15),
                dt(1995, 11, 15),
                NoInput(0),
                9.5,
                99.730918,
                100.098321,
                9.54,
                dt(1985, 11, 29),
            ),  # D
            (
                dt(1985, 7, 2),
                dt(2005, 8, 15),
                dt(1986, 2, 15),
                10.75,
                102.214586,
                105.887384,
                10.47,
                dt(1985, 11, 4),
            ),  # E
            (
                dt(1983, 5, 16),
                dt(1991, 5, 15),
                dt(1983, 11, 15),
                10.5,
                99.777074,
                102.373541,
                10.53,
                dt(1983, 8, 15),
            ),  # F
            (
                dt(1988, 10, 15),
                dt(1994, 12, 15),
                dt(1989, 6, 15),
                9.75,
                99.738045,
                100.563865,
                9.79,
                dt(1988, 11, 15),
            ),  # G
        ],
    )
    def test_fixed_rate_bond_price_ust(self, e, t, s, fr, ec, ed, y, se) -> None:
        # The UST tests are from:
        # https://www.ecfr.gov/current/title-31/subtitle-B/chapter-II/subchapter-A/part-356/appendix-Appendix%20B%20to%20Part%20356
        ust = FixedRateBond(
            effective=e,
            termination=t,
            front_stub=s,
            fixed_rate=fr,
            frequency="S",
            calendar="nyc",
            convention="ActActICMA",
            calc_mode="ust_31Bii",
            ex_div=1,
            modifier="NONE",
        )
        res1 = ust.price(ytm=y, settlement=se, dirty=False)
        res2 = ust.price(ytm=y, settlement=se, dirty=True)
        assert abs(res1 - ec) < 1e-6
        assert abs(res2 - ed) < 1e-6

    @pytest.mark.parametrize(
        ("s", "exp", "acc"),
        [
            (dt(2025, 2, 14), 99.106414, 1.926970),
            (dt(2025, 2, 18), 99.107179, 0.032113),
            (dt(2025, 8, 15), 99.151393, 0.0),
        ],
    )
    def test_ust_price_street(self, s, exp, acc) -> None:
        bond = FixedRateBond(
            effective=dt(2023, 8, 15),
            termination=dt(2033, 8, 15),
            fixed_rate=3.875,
            spec="us_gb",
        )
        result = bond.price(ytm=4, settlement=s)
        accrued = bond.accrued(settlement=s)
        assert abs(accrued - acc) < 1e-6
        assert abs(result - exp) < 1e-5

    def test_long_stub_first_cashflow(self):
        # test against 31.B.ii.A356.Appendix.B.I.A Example Long First
        note = FixedRateBond(
            effective=dt(1990, 12, 3),
            termination=dt(1996, 2, 15),
            stub="longfront",
            spec="us_gb",
            fixed_rate=7.875,
            notional=-7000,
        )
        assert abs(note.leg1.periods[0].cashflow() - 386.474184670) < 5e-7

    def test_calc_mode_ytm(self):
        b = FixedRateBond(dt(1985, 11, 15), dt(1995, 11, 15), fixed_rate=9.5, spec="us_gb_tsy")

        y1 = b.ytm(price=99.730918, settlement=dt(1985, 11, 29))
        assert abs(y1 - 9.54) < 1e-6

        b2 = FixedRateBond(dt(1985, 11, 15), dt(1995, 11, 15), fixed_rate=9.5, spec="us_gb")
        exp_y2 = b2.ytm(price=99.730918, settlement=dt(1985, 11, 29))

        # street convention
        y2 = b.ytm(price=99.730918, settlement=dt(1985, 11, 29), calc_mode="us_gb")
        assert abs(y2 - 9.54) > 1e-6
        assert abs(y2 - exp_y2) < 1e-6

    # Swedish Government Bond Tests. Data from alternative systems.

    @pytest.mark.parametrize(
        ("settlement", "exp_accrued", "exp_price"),
        [
            (dt(2024, 5, 3), 0.73125, 88.134),
            # (dt(2024, 5, 5), 0.735417, 88.150), # ambiguous Sunday
            (dt(2024, 5, 6), -0.0125, 88.158),
            (dt(2024, 5, 7), -0.0104, 88.165),
            (dt(2024, 5, 8), -0.008333, 88.173),
            (dt(2024, 5, 12), 0.0, 88.203),
            (dt(2024, 5, 13), 0.002083, 88.210),
        ],
    )
    def test_sgb_1060s_price_and_accrued(self, settlement, exp_accrued, exp_price) -> None:
        sgb = FixedRateBond(
            effective=dt(2023, 5, 12),
            termination=dt(2028, 5, 12),
            frequency="A",
            convention="ActActICMA",
            calendar="stk",
            ex_div=5,
            modifier="NONE",
            fixed_rate=0.75,
            calc_mode="sgb",
        )
        accrued = sgb.accrued(settlement)
        assert abs(accrued - exp_accrued) < 1e-4
        price = sgb.price(ytm=4.0, settlement=settlement, dirty=False)
        assert abs(price - exp_price) < 1e-3

    def test_sgb_ultra_short_ytm(self):
        # SE0010469205
        komins = FixedRateBond(
            effective=dt(2017, 10, 2), termination=dt(2024, 10, 2), fixed_rate=1.0, spec="se_gb"
        )
        dp = komins.price(ytm=3.42092, settlement=dt(2024, 9, 24), dirty=True)
        cp = komins.price(ytm=3.42092, settlement=dt(2024, 9, 24), dirty=False)
        assert abs(dp - cp - komins.accrued(settlement=dt(2024, 9, 24))) < 1e-10

        assert abs(cp - 99.9455205) < 1e-4

    def test_fixed_rate_bond_price_sgb_back_stub(self) -> None:
        bond = FixedRateBond(
            dt(1995, 12, 7),
            dt(2015, 1, 23),
            "A",
            stub="SHORTBACK",
            roll=7,
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
            calc_mode="sgb",
        )
        result = bond.price(ytm=8.00, settlement=dt(1995, 12, 7))
        expected = 100.0018153890108  # simple period back stub yields close to par
        assert abs(result - expected) < 1e-9

    # Canadian Government Bond Tests. Data from alternative systems
    # and from https://iiac-accvm.ca/wp-content/uploads/Canadian-Conventions-in-FI-Markets-Release-1.3.pdf

    @pytest.mark.parametrize(
        ("settlement", "exp"),
        [
            (dt(2005, 12, 1), 1.671232),
            (dt(2006, 1, 31), 2.486301),
        ],
    )
    def test_settlement_accrued(self, settlement, exp) -> None:
        bond = FixedRateBond(
            effective=dt(2004, 8, 1),
            termination=dt(2008, 2, 1),
            fixed_rate=5.0,
            modifier="NONE",
            frequency="S",
            convention="ActActICMA_stub365f",
            calc_mode="cadgb",
            ex_div=1,
        )
        result = bond.accrued(settlement=settlement)
        assert abs(result - exp) < 1e-6

    @pytest.mark.skip(reason="<1Y CAD bonds NotImplemented")
    @pytest.mark.parametrize(
        ("s", "exp", "acc"),
        [
            (dt(2024, 8, 1), 99.839907, 0.0),
            (dt(2024, 7, 17), 99.866051, 1.715753),
            (dt(2024, 8, 7), 99.842641, 0.061644),
        ],
    )
    def test_cadgb_price(self, s, exp, acc) -> None:
        bond = FixedRateBond(
            effective=dt(2022, 11, 2),
            termination=dt(2025, 2, 1),
            fixed_rate=3.75,
            modifier="NONE",
            convention="ActActICMA_STUB365f",
            frequency="S",
            calc_mode="cadgb",
            roll=1,
            stub="SHORTFRONT",
            ex_div=1,
        )
        result = bond.price(ytm=4.0, settlement=s)
        accrued = bond.accrued(settlement=s)
        assert abs(accrued - acc) < 1e-6
        # Price fails becuase bond is <1Y from maturity needs a branched formula.
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize(
        ("s", "exp", "acc"),
        [
            (dt(2024, 11, 26), 91.055145, 1.341096),
            (dt(2024, 12, 2), 91.069934, 0.007534),
            (dt(2024, 6, 3), 90.634570, 0.015068),
        ],
    )
    def test_cadgb_price2(self, s, exp, acc) -> None:
        bond = FixedRateBond(
            effective=dt(2023, 2, 2),
            termination=dt(2033, 6, 1),
            fixed_rate=2.75,
            modifier="NONE",
            convention="ActActICMA_STUB365f",
            frequency="S",
            calc_mode="ca_gb",
            roll=1,
            stub="SHORTFRONT",
            ex_div=1,
        )
        result = bond.price(ytm=4.0, settlement=s)
        accrued = bond.accrued(settlement=s)
        assert abs(accrued - acc) < 1e-6
        assert abs(result - exp) < 1e-6

    def test_cadgb_price3(self) -> None:
        bond = FixedRateBond(
            effective=dt(2018, 7, 27),
            termination=dt(2029, 6, 1),
            fixed_rate=2.25,
            modifier="NONE",
            convention="ActActICMA_STUB365f",
            frequency="S",
            calc_mode="cadgb",
            roll=1,
            stub="SHORTFRONT",
            ex_div=1,
        )
        result = bond.price(ytm=2.249977, settlement=dt(2018, 10, 16))
        accrued = bond.accrued(settlement=dt(2018, 10, 16))
        stub_cash = bond.leg1.periods[0].cashflow()
        assert abs(accrued - 0.499315) < 1e-6
        assert abs(result - 100.00) < 1e-5
        assert abs(stub_cash + 7828.77) < 1e-2

    ## German gov bonds comparison with official bundesbank publications.

    @pytest.mark.parametrize(
        ("sett", "price", "exp_ytm", "exp_acc"),
        [
            (dt(2024, 1, 10), 105.0, 1.208836, 0.321311),
            (
                dt(2024, 6, 12),
                97.180,
                2.66368627,
                1.204918,
            ),  # https://www.bundesbank.de/en/service/federal-securities/prices-and-yields
            (dt(2022, 12, 20), 99.31, 2.208075, 0.350959),
            # (dt(2022, 12, 20), 99.31, 2.20804175, 0.3452055),  # Bundesbank official data:
            # see link above (accrual is unexplained and does not match systems)
            (
                dt(2023, 11, 2),
                97.04,
                2.636708016,
                2.174795,
            ),  # Bundesbank official data: see link above (agrees with BXT)
            (dt(2028, 11, 15), 97.5, 4.717949, 0.0),  # YAS
        ],
    )
    def test_de_gb(self, sett, price, exp_ytm, exp_acc) -> None:
        frb = FixedRateBond(  # ISIN DE0001102622
            effective=dt(2022, 10, 20),
            termination=dt(2029, 11, 15),
            stub="LONGFRONT",
            fixed_rate=2.1,
            spec="de_gb",
        )
        result = frb.accrued(settlement=sett)
        assert abs(result - exp_acc) < 1e-6

        result = frb.ytm(price=price, settlement=sett)
        assert abs(result - exp_ytm) < 1e-6

    @pytest.mark.parametrize(
        ("sett", "price", "exp_ytm", "exp_acc"),
        [
            (
                dt(2024, 6, 12),
                99.555,
                3.5314195,
                0.825137,
            ),  # https://www.bundesbank.de/en/service/federal-securities/prices-and-yields
        ],
    )
    def test_de_gb_mm(self, sett, price, exp_ytm, exp_acc) -> None:
        # tests the MoneyMarket simple yield for the final period.
        frb = FixedRateBond(  # ISIN DE0001102366
            effective=dt(2014, 8, 15),
            termination=dt(2024, 8, 15),
            fixed_rate=1.0,
            spec="de_gb",
        )
        result = frb.accrued(settlement=sett)
        assert abs(result - exp_acc) < 1e-6

        result = frb.ytm(price=price, settlement=sett)
        assert abs(result - exp_ytm) < 1e-6

    def test_long_stub(self):
        # DE000BU2Z056
        bond = FixedRateBond(dt(2025, 7, 4), dt(2035, 8, 15), spec="de_gb", fixed_rate=2.60)
        assert bond.leg1.schedule.aschedule[0:2] == [dt(2025, 7, 4), dt(2026, 8, 15)]

    ## French OAT

    @pytest.mark.parametrize(
        ("sett", "price", "exp_ytm", "exp_acc"),
        [
            (dt(2024, 6, 14), 101.0, 2.886581, 1.655738),
            (dt(2033, 11, 25), 99.75, 3.258145, 0.0),
            (dt(2034, 6, 13), 101.0, 0.769200, 1.643836),
        ],
    )
    def test_fr_gb(self, sett, price, exp_ytm, exp_acc) -> None:
        frb = FixedRateBond(  # ISIN FR001400QMF9
            effective=dt(2023, 11, 25),
            termination=dt(2034, 11, 25),
            fixed_rate=3.0,
            spec="fr_gb",
        )
        result = frb.accrued(settlement=sett)
        assert abs(result - exp_acc) < 1e-6

        result = frb.ytm(price=price, settlement=sett)
        assert abs(result - exp_ytm) < 1e-6

    ## Italian BTP

    @pytest.mark.parametrize(
        ("sett", "price", "exp_ytm", "exp_acc"),
        [
            (dt(2024, 6, 14), 98.0, 4.730058, 0.526090),
            (dt(2026, 4, 14), 99.0, 4.617209, 1.993370),
        ],
    )
    def test_regular_it_gb(self, sett, price, exp_ytm, exp_acc) -> None:
        frb = FixedRateBond(  # ISIN IT0005518128
            effective=dt(2022, 11, 1),
            termination=dt(2033, 5, 1),
            fixed_rate=4.4,
            spec="it_gb",
        )
        result = frb.accrued(settlement=sett)
        assert abs(result - exp_acc) < 5e-6

        result = frb.price(ytm=exp_ytm, settlement=sett)
        result = frb.ytm(price=price, settlement=sett)
        assert abs(result - exp_ytm) < 2e-4

    @pytest.mark.parametrize(
        ("sett", "ytm", "exp_price", "exp_acc"),
        [
            (dt(2032, 11, 1), 6.5, 98.96593464, 0.0),
            (dt(2032, 11, 2), 6.5, 98.97099073, 0.01215),
            (dt(2033, 3, 15), 6.5, 99.69805695, 1.628730),
            (dt(2033, 4, 29), 6.5, 99.96938727, 2.175690),
        ],
    )
    def test_regular_it_gb_final_simple_vs_excel(self, sett, ytm, exp_price, exp_acc) -> None:
        # These values are not the same as for BBG YA.
        # BBG YA can be replicated here if the number of days between payments
        # in the last period 1/11/32 to 2/5/33 is 184 days and the pay_adj is 1/184 instead of
        # 1/182. Problem is, the actual number of days between payments are 182.
        frb = FixedRateBond(  # ISIN IT0005518128
            effective=dt(2022, 11, 1),
            termination=dt(2033, 5, 1),
            fixed_rate=4.4,
            spec="it_gb",
        )

        result = frb.accrued(settlement=sett)
        assert abs(result - exp_acc) < 5e-6

        result = frb.price(ytm=ytm, settlement=sett)
        assert abs(result - exp_price) < 1e-6

    @pytest.mark.parametrize(
        ("sett", "ytm", "exp_price", "exp_acc"),
        [
            (dt(2032, 11, 1), 6.429702, 99.00, 0.0),  # Last coupon simple rate
            (dt(2032, 11, 2), 6.439891, 99.00, 0.01215),  # Last coupon simple rate
            (dt(2033, 3, 15), 6.862519, 99.65, 1.628730),  # Last coupon simple rate
            (dt(2033, 4, 29), 6.450803, 99.97, 2.175690),  # Test accrual upto adjusted payment date
        ],
    )
    def test_regular_it_gb_final_simple(self, sett, ytm, exp_price, exp_acc) -> None:
        frb = FixedRateBond(  # ISIN IT0005518128
            effective=dt(2022, 11, 1),
            termination=dt(2033, 5, 1),
            fixed_rate=4.4,
            spec="it_gb",
        )

        result = frb.accrued(settlement=sett)
        assert abs(result - exp_acc) < 5e-6

        result = frb.price(ytm=ytm, settlement=sett)
        assert abs(result - exp_price) < 3e-3

    @pytest.mark.parametrize(
        ("sett", "ytm", "exp_price", "exp_acc"),
        [
            (dt(2026, 12, 13), 6.5, 98.77226353, 0.0),
            (dt(2027, 1, 11), 6.5, 98.95139167, 0.318681),
        ],
    )
    def test_regular_it_gb_final_simple_vs_excel2(self, sett, ytm, exp_price, exp_acc) -> None:
        frb = FixedRateBond(  # ISIN IT0005547408
            effective=dt(2023, 6, 13),
            termination=dt(2027, 6, 13),
            fixed_rate=4.00,
            spec="it_gb",
        )

        result = frb.accrued(settlement=sett)
        assert abs(result - exp_acc) < 5e-6

        result = frb.price(ytm=ytm, settlement=sett)
        assert abs(result - exp_price) < 1e-6

    ## Norwegian

    @pytest.mark.parametrize(
        ("set_", "price", "exp_ytm", "exp_acc"),
        [
            (dt(2026, 4, 13), 99.3, 3.727804, 0.0),  # YAS Coupon aligned
            (dt(2033, 4, 13), 99.9, 3.728729, 0.0),  # Last period
            (dt(2033, 9, 12), 99.9, 3.772713, 1.509589),  # Middle Last period
            (dt(2024, 2, 13), 99.9, 3.638007, 0.0),  # Start of bond
            (
                dt(2024, 3, 13),
                99.9,
                3.637518,
                0.288014,
            ),  # Mid stub period
        ],
    )
    def test_no_gb(self, set_, price, exp_ytm, exp_acc) -> None:
        frb = FixedRateBond(  # ISIN NO0013148338
            effective=dt(2024, 2, 13),
            termination=dt(2034, 4, 13),
            fixed_rate=3.625,
            spec="no_gb",
        )
        result = frb.accrued(settlement=set_)
        assert abs(result - exp_acc) < 5e-6

        result = frb.ytm(price=price, settlement=set_)
        assert abs(result - exp_ytm) < 1e-5

    ## Dutch

    @pytest.mark.parametrize(
        ("set_", "price", "exp_ytm", "exp_acc"),
        [
            (dt(2025, 6, 10), 98.0, 2.751162, 2.260274),  # YAS Coupon aligned
            (dt(2033, 7, 15), 99.8, 2.705411, 0.0),  # Last period
            (dt(2033, 7, 18), 99.9, 2.602897, 0.020548),  # Middle Last period
            (dt(2024, 2, 8), 99.0, 2.611616, 0.0),  # Start of bond
            (dt(2024, 3, 13), 99.0, 2.612194, 0.232240),  # Mid stub period
        ],
    )
    def test_nl_gb(self, set_, price, exp_ytm, exp_acc) -> None:
        frb = FixedRateBond(  # ISIN NL0015001XZ6
            effective=dt(2024, 2, 8),
            termination=dt(2034, 7, 15),
            fixed_rate=2.5,
            spec="nl_gb",
        )
        result = frb.accrued(settlement=set_)
        assert abs(result - exp_acc) < 5e-6

        result = frb.ytm(price=price, settlement=set_)
        assert abs(result - exp_ytm) < 1e-5

    # US Corp: BNY Mello

    @pytest.mark.parametrize(
        ("settlement", "price", "exp_ytm", "exp_acc"),
        [
            (dt(2025, 5, 6), 101.0, 3.493237, 0.08555556),
            (dt(2028, 4, 3), 100.05, 3.077448, 1.65763889),
        ],
    )
    def test_bny_mellon(self, settlement, price, exp_ytm, exp_acc) -> None:
        # BNY Mellon ISIN: US06406RAH03,
        b = FixedRateBond(
            effective=dt(2018, 4, 30),
            termination=dt(2028, 4, 28),
            fixed_rate=3.85,
            convention="30u360",
            spec="us_gb",
            calc_mode="us_corp",
        )
        ytm = b.ytm(price, settlement)
        acc = b.accrued(settlement)
        assert abs(ytm - exp_ytm) < 1e-6
        assert abs(acc - exp_acc) < 1e-8

    @pytest.mark.parametrize(
        ("settlement", "price", "exp_ytm", "exp_acc"),
        [
            (dt(2018, 5, 30), 101.0, 3.728114, 0.32083333),
            (dt(2018, 5, 31), 101.0, 3.728114, 0.32083333),
            (dt(2025, 5, 6), 101.0, 3.493237, 0.08555556),
            (dt(2028, 4, 3), 100.05, 3.077448, 1.65763889),
        ],
    )
    def test_bny_mellon_spec(self, settlement, price, exp_ytm, exp_acc) -> None:
        # BNY Mellon ISIN: US06406RAH03,
        b = FixedRateBond(
            effective=dt(2018, 4, 30),
            termination=dt(2028, 4, 28),
            fixed_rate=3.85,
            spec="us_corp",
        )
        assert abs(b.leg1.periods[0].cashflow() + 19036.1111) < 1e-4
        ytm = b.ytm(price, settlement)
        acc = b.accrued(settlement)
        assert abs(acc - exp_acc) < 1e-8
        assert abs(ytm - exp_ytm) < 1e-6

    # US MUNI: Cali State

    @pytest.mark.parametrize(
        ("settlement", "price", "exp_ytm", "exp_acc"),
        [
            (dt(2025, 5, 12), 102.35, 2.879, 1.819444),
            (dt(2025, 1, 31), 100.1, 4.923, 0.416667),
            (dt(2026, 1, 1), 102.35, 0.293, 0.0),
            (dt(2026, 5, 19), 100.10, 4.061, 1.916667),
            (dt(2026, 6, 30), 100.10, -30.219, 2.486111),
        ],
    )
    def test_cali_state_school(self, settlement, price, exp_ytm, exp_acc) -> None:
        # LA Unif ISIN: US544647CW89,
        b = FixedRateBond(
            effective=dt(2020, 11, 10),
            termination=dt(2026, 7, 1),
            fixed_rate=5.0,
            spec="us_muni",
        )
        ytm = b.ytm(price, settlement)
        acc = b.accrued(settlement)
        assert abs(ytm - exp_ytm) < 1e-3
        assert abs(acc - exp_acc) < 1e-6

    @pytest.mark.parametrize(
        ("settlement", "price", "exp_ytm", "exp_acc"),
        [
            (dt(2025, 3, 31), 110.0, -3.441, 1.356800),
            (dt(2025, 5, 12), 101.0, 3.662, 1.881600),
            (dt(2025, 5, 30), 100.02, 4.586, 2.11200),
            (dt(2025, 12, 15), 101.0, 2.582, 0.0),
            (dt(2026, 3, 19), 101.0, 0.413, 1.203200),
            (dt(2026, 6, 11), 100.02, 2.746, 2.2528),
        ],
    )
    def test_new_jersey_transport(self, settlement, price, exp_ytm, exp_acc) -> None:
        # NJ Transport ISIN: US64613CEZ77,
        b = FixedRateBond(
            effective=dt(2024, 10, 24),
            termination=dt(2026, 6, 15),
            fixed_rate=4.608,
            spec="us_muni",
        )
        ytm = b.ytm(price, settlement)
        acc = b.accrued(settlement)
        assert abs(acc - exp_acc) < 1e-6
        assert abs(ytm - exp_ytm) < 1e-3

    # Customised Thai Government Bonds

    def test_thai_example_a3(self):
        # see file in _static/thai_standard_formula.pdf
        def _v1_thb_gb(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx):
            r_u = (obj.leg1.schedule.uschedule[acc_idx + 1] - settlement).days
            return v2 ** (r_u * f / 365)

        def _v3_thb_gb(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx):
            r_u = (obj.leg1.schedule.uschedule[-1] - obj.leg1.schedule.uschedule[-2]).days
            return v2 ** (r_u * f / 365)

        thai_cm = BondCalcMode(
            settle_accrual="linear_days",
            ytm_accrual="linear_days",
            v1=_v1_thb_gb,
            v2="regular",
            v3=_v3_thb_gb,
            c1="full_coupon",
            ci="full_coupon",
            cn="cashflow",
        )

        b = FixedRateBond(
            effective=dt(1993, 1, 15),
            termination=dt(1996, 4, 30),
            stub="shortback",
            frequency="S",
            fixed_rate=11.25,
            convention="act365f",
            modifier="none",
            currency="thb",
            calendar="bus",
            calc_mode=thai_cm,
        )

        expected_acc = 4.86986301
        expected_clean = 103.1099263
        result_acc = b.accrued(settlement=dt(1994, 12, 20))
        result_clean = b.price(ytm=8.75, settlement=dt(1994, 12, 20))

        assert abs(result_acc - expected_acc) < 1e-8
        assert abs(result_clean - expected_clean) < 1e-7

    def test_thai_example_a3_exdiv(self):
        # see file in _static/thai_standard_formula.pdf
        def _v1_thb_gb(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx):
            r_u = (obj.leg1.schedule.uschedule[acc_idx + 1] - settlement).days
            return v2 ** (r_u * f / 365)

        def _v3_thb_gb(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx):
            r_u = (obj.leg1.schedule.uschedule[-1] - obj.leg1.schedule.uschedule[-2]).days
            return v2 ** (r_u * f / 365)

        thai_cm = BondCalcMode(
            settle_accrual="linear_days",
            ytm_accrual="linear_days",
            v1=_v1_thb_gb,
            v2="regular",
            v3=_v3_thb_gb,
            c1="cashflow",
            ci="full_coupon",
            cn="cashflow",
        )

        b = FixedRateBond(
            effective=dt(1993, 1, 15),
            termination=dt(1996, 4, 30),
            stub="shortback",
            frequency="S",
            fixed_rate=11.25,
            convention="act365f",
            modifier="none",
            currency="thb",
            calendar="bus",
            calc_mode=thai_cm,
            ex_div=21,
        )

        result_acc = b.accrued(dt(1994, 12, 20))
        expected_acc = -0.80136986
        assert abs(result_acc - expected_acc) < 1e-8

        result_clean = b.price(ytm=8.75, settlement=dt(1994, 12, 20))
        expected_clean = 103.19036939
        assert abs(result_clean - expected_clean) < 1e-8

    # Swiss GB

    @pytest.mark.parametrize(
        ("ytm", "sett", "exp"),
        [
            (2.01111, dt(2025, 5, 23), [92.724231, 0.095833333]),
            (2.01111, dt(2018, 5, 29), [90.369254, 0.120833333]),  # accrued DCF
            (2.01111, dt(2018, 5, 30), [90.370093, 0.125000000]),  # accrued DCF
            (2.01111, dt(2018, 5, 31), [90.370093, 0.125000000]),  # accrued DCF
            (2.01111, dt(2018, 6, 1), [90.370931, 0.129166666]),
            (2.01111, dt(2024, 4, 29), [92.343879, 1.49583333]),  # Ex div
            (2.01111, dt(2024, 4, 30), [92.344903, 0.000000000]),  # Ex div
            (2.01111, dt(2042, 4, 15), [99.978326, 1.43750000]),  # Final period
        ],
    )
    def test_ch_gb(self, ytm, sett, exp):
        # ISIN: CH0127181169
        bond = FixedRateBond(dt(2012, 4, 30), dt(2042, 4, 30), fixed_rate=1.5, spec="ch_gb")
        accrued = bond.accrued(sett)
        assert abs(accrued - exp[1]) < 1e-8
        price = bond.price(ytm=ytm, settlement=sett)
        assert abs(price - exp[0]) < 1e-6

    # New Zealand GB

    @pytest.mark.parametrize(
        ("ytm", "sett", "maturity", "coupon", "exp"),
        [
            (4.355, dt(2022, 11, 22), dt(2034, 5, 15), 4.25, [99.0583817412, 0.0821823204]),
            (
                5.348,
                dt(2051, 4, 15),
                dt(2051, 5, 15),
                2.75,
                [99.7842450753699, 1.1470994475],
            ),  # Last period simple_act365f
            (0.745, dt(2021, 2, 10), dt(2026, 5, 15), 0.50, [98.7384877998, 0.1201657459]),
        ],
    )
    def test_nz_gb(self, ytm, sett, maturity, coupon, exp):
        bond = FixedRateBond(dt(2020, 5, 15), maturity, fixed_rate=coupon, spec="nz_gb")
        accrued = bond.accrued(sett)
        assert abs(accrued - exp[1]) < 1e-8
        price = bond.price(ytm=ytm, settlement=sett)
        assert abs(price - exp[0]) < 1e-6

    # General Method Coverage

    def test_fixed_rate_bond_yield_domains(self) -> None:
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
        )
        assert bond.ytm(500.0, dt(1999, 5, 24), True) + 5.86484231333 < 1e-8
        assert bond.ytm(200, dt(1999, 5, 24), True) - 1.4366895440550 < 1e-8
        assert bond.ytm(100, dt(1999, 5, 24), True) - 8.416909601459 < 1e-8
        assert bond.ytm(50, dt(1999, 5, 24), True) - 18.486840866431 < 1e-6
        assert bond.ytm(1, dt(1999, 5, 24), True) - 13421775210.82037 < 1e-3

    def test_fixed_rate_bond_ytm_duals(self) -> None:
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
        )

        dPdy = bond.duration(4, dt(1995, 1, 1))
        P = bond.price(4, dt(1995, 1, 1))
        result = bond.ytm(Dual(P, ["a", "b"], [1, -0.5]), dt(1995, 1, 1))
        expected = Dual(4.00, ["a", "b"], [-1 / dPdy, 0.5 / dPdy])
        assert abs(result - expected) < 1e-11
        assert all(np.isclose(expected.dual, result.dual))

        d2ydP2 = -bond.convexity(4, dt(1995, 1, 1)) * -(dPdy**-3)
        result = bond.ytm(Dual2(P, ["a", "b"], [1, -0.5], []), dt(1995, 1, 1))
        expected = Dual2(
            4.00,
            ["a", "b"],
            [-1 / dPdy, 0.5 / dPdy],
            [d2ydP2 * 0.5, d2ydP2 * -0.25, d2ydP2 * -0.25, d2ydP2 * 0.125],
        )
        assert abs(result - expected) < 1e-11
        assert all(np.isclose(result.dual, expected.dual))
        assert all(np.isclose(result.dual2, expected.dual2).flat)

    @pytest.mark.skip(reason="Bills have Z frequency, this no longer raises")
    def test_fixed_rate_bond_zero_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="FixedRateBond `frequency`"):
            FixedRateBond(dt(1999, 5, 7), dt(2002, 12, 7), "Z", convention="ActActICMA")

    @pytest.mark.parametrize("metric", ["risk", "duration", "modified"])
    def test_fixed_rate_bond_duration(self, metric) -> None:
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
        )
        price0 = gilt.price(4.445, dt(1999, 5, 27), dirty=True)
        price1 = gilt.price(4.446, dt(1999, 5, 27), dirty=True)
        if metric == "risk":
            numeric = price0 - price1
        elif metric == "modified":
            numeric = (price0 - price1) / price0 * 100
        elif metric == "duration":
            numeric = (price0 - price1) / price0 * (1 + 4.445 / (100 * 2)) * 100

        result = gilt.duration(4.445, dt(1999, 5, 27), metric=metric)
        assert (result - numeric * 1000) < 1e-1

    def test_fixed_rate_bond_convexity(self) -> None:
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
        )
        numeric = gilt.duration(4.445, dt(1999, 5, 27)) - gilt.duration(4.446, dt(1999, 5, 27))
        result = gilt.convexity(4.445, dt(1999, 5, 27))
        assert (result - numeric * 1000) < 1e-3

        price = gilt.price(4.445, dt(1999, 5, 27), dirty=True)
        result2 = gilt.convexity(4.445, dt(1999, 5, 27), "convexity")
        assert abs(result2 - result * 100.0 / price) < 1e-6

    def test_convexity_traditional(self):
        aapl_bond = FixedRateBond(dt(2013, 5, 4), dt(2043, 5, 4), fixed_rate=3.85, spec="us_corp")
        # c1 = aapl_bond.convexity(4.653674794785435, dt(2014, 3, 5))
        c2 = aapl_bond.convexity(4.653674794785435, dt(2014, 3, 5), metric="convexity")
        assert abs(c2 - 3.803) < 1e-4

    def test_fixed_rate_bond_rate(self) -> None:
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            settle=0,
        )
        curve = Curve({dt(1998, 12, 9): 1.0, dt(2015, 12, 7): 0.50})
        clean_price = gilt.rate(curves=curve, metric="clean_price")
        result = gilt.rate(
            curves={"disc_curve": curve}, metric="clean_price", settlement=dt(1998, 12, 9)
        )
        assert abs(result - clean_price) < 1e-8

        result = gilt.rate(curves=_Curves(disc_curve=curve), metric="dirty_price")
        expected = clean_price + gilt.accrued(dt(1998, 12, 9))
        assert result == expected
        result = gilt.rate(curves=curve, metric="dirty_price", settlement=dt(1998, 12, 9))
        assert abs(result - clean_price - gilt.accrued(dt(1998, 12, 9))) < 1e-8

        result = gilt.rate(curves=curve, metric="ytm")
        expected = gilt.ytm(clean_price, dt(1998, 12, 9), False)
        assert abs(result - expected) < 1e-8

    def test_initialisation_rate_metric(self) -> None:
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            settle=0,
            metric="ytm",
        )
        curve = Curve({dt(1998, 12, 9): 1.0, dt(2015, 12, 7): 0.50})
        clean_price = gilt.rate(curves=curve, metric="clean_price")
        expected = gilt.ytm(price=clean_price, settlement=dt(1998, 12, 9))
        result = gilt.rate(curves=curve)  # default metric is "ytm"
        assert abs(result - expected) < 1e-8

    def test_fixed_rate_bond_npv(self) -> None:
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=0,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})
        result = gilt.npv(curves=curve)
        expected = 113.22198344812742
        assert abs(result - expected) < 1e-6

        gilt.kwargs.meta["settle"] = 2
        result = gilt.npv(curves=curve)  # bond is ex div on settlement 27th Nov 2010
        expected = 109.229489312983  # bond has dropped a coupon payment of 4.
        assert abs(result - expected) < 1e-6

        result = gilt.npv(curves=curve, local=True)
        assert abs(result["gbp"] - expected) < 1e-6

    def test_fixed_rate_bond_npv_private(self) -> None:
        # this test shadows 'fixed_rate_bond_npv' but extends it for projection on 27th Nov ex div.
        curve = Curve({dt(2004, 11, 25): 1.0, dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=0,
        )
        result = gilt.npv(curves=curve, settlement=dt(2010, 11, 27), forward=dt(2010, 11, 25))
        expected = 109.229489312983  # npv should match associated test
        assert abs(result - expected) < 1e-6

    def test_fixed_rate_bond_analytic_delta(self) -> None:
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-1000000,
            settle=0,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 1.0})
        result = gilt.analytic_delta(curves=curve)
        expected = -550.0
        assert abs(result - expected) < 1e-6

        gilt.kwargs.meta["settle"] = 2
        result = gilt.analytic_delta(curves=curve)  # bond is ex div on settle 27th Nov 2010
        expected = -500.0  # bond has dropped a 6m coupon payment
        assert abs(result - expected) < 1e-6

    def test_fixed_rate_bond_cashflows(self) -> None:
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=1,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})

        flows = gilt.cashflows(curves=curve)  # bond is ex div on 26th nov 2010
        result = flows[defaults.headers["npv"]].sum()
        expected = gilt.npv(curves=curve)
        assert abs(result - expected) < 1e-6

        gilt.settle = 0
        flows = gilt.cashflows(curves=curve)  # settlement from curve initial node
        result = flows[defaults.headers["npv"]].sum()
        expected = gilt.npv(curves=curve)
        assert abs(result - expected) < 1e-6

    def test_fixed_rate_bond_rate_raises(self) -> None:
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
        )
        curve = Curve({dt(1998, 12, 7): 1.0, dt(2015, 12, 7): 0.50})
        with pytest.raises(ValueError, match="`metric` must be in"):
            gilt.rate(curves=curve, metric="bad_metric")

    def test_fixed_rate_bond_no_amortization(self) -> None:
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'amortization"):
            FixedRateBond(
                effective=dt(1998, 12, 7),
                termination=dt(2015, 12, 7),
                frequency="S",
                calendar="ldn",
                currency="gbp",
                convention="ActActICMA",
                ex_div=7,
                fixed_rate=8.0,
                notional=-100,
                amortization=100,
            )

    @pytest.mark.parametrize(
        ("f_s", "exp"),
        [
            (dt(2001, 12, 31), 99.997513754),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_forward_price_analogue(self, f_s, exp) -> None:
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=NoInput(0),
            currency="gbp",
            convention="Act365f",
            ex_div=0,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.fwd_from_repo(100.0, dt(2001, 1, 1), f_s, 1.0, "act365f")
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize(
        ("f_s", "exp"),
        [
            (dt(2001, 12, 31), 100.49888361793),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_forward_price_analogue_dirty(self, f_s, exp) -> None:
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=NoInput(0),
            currency="gbp",
            convention="Act365f",
            ex_div=1,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.fwd_from_repo(100.0, dt(2001, 1, 1), f_s, 1.0, "act365f", dirty=True)
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize(
        ("s", "f_s", "exp"),
        [
            (dt(2010, 11, 25), dt(2011, 11, 25), 99.9975000187),  # div div
            (dt(2010, 11, 28), dt(2011, 11, 29), 99.997471945),  # ex-div ex-div
            (dt(2010, 11, 28), dt(2011, 11, 25), 99.997419419),  # ex-div div
            (dt(2010, 11, 25), dt(2011, 11, 29), 99.9975516607),  # div ex-div
        ],
    )
    def test_fixed_rate_bond_forward_price_analogue_ex_div(self, s, f_s, exp) -> None:
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="act365f",
            ex_div=7,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.fwd_from_repo(100.0, s, f_s, 1.0, "act365f")
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize(
        ("f_s", "f_p"),
        [
            (dt(2001, 12, 31), 99.997513754),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_implied_repo(self, f_s, f_p) -> None:
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=NoInput(0),
            currency="gbp",
            convention="Act365f",
            ex_div=0,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.repo_from_fwd(100.0, dt(2001, 1, 1), f_s, f_p, "act365f")
        assert abs(result - 1.00) < 1e-8

    @pytest.mark.parametrize(
        ("f_s", "f_p"),
        [
            (dt(2001, 12, 31), 100.49888361793),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_implied_repo_analogue_dirty(self, f_s, f_p) -> None:
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=NoInput(0),
            currency="gbp",
            convention="Act365f",
            ex_div=1,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.repo_from_fwd(100.0, dt(2001, 1, 1), f_s, f_p, "act365f", dirty=True)
        assert abs(result - 1.0) < 1e-8

    @pytest.mark.parametrize(
        ("price", "tol"),
        [(112.0, 5e-7), (104.0, 1e-8), (96.0, 1e-7), (91.0, 1e-6)],
    )
    def test_oaspread(self, price, tol) -> None:
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=0,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})
        # result = gilt.npv(curve) = 113.22198344812742
        result = gilt.oaspread(curves=curve, price=price)
        curve_z = curve.shift(result)
        result = gilt.rate(curves=curve_z, metric="clean_price")
        assert abs(result - price) < tol

    @pytest.mark.parametrize(
        ("price", "tol"),
        [
            (85, 5e-8),
            (75, 5e-8),
            (65, 1e-7),
            (55, 1e-7),
            (45, 5e-8),
            (35, 5e-8),
        ],
    )
    def test_oaspread_low_price(self, price, tol) -> None:
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        curve = Curve({dt(1999, 11, 25): 1.0, dt(2015, 12, 7): 0.85})
        # result = gilt.npv(curve) = 113.22198344812742
        result = gilt.oaspread(curves=curve, price=price)
        curve_z = curve.shift(result)
        result = gilt.rate(curves=curve_z, metric="clean_price")
        assert abs(result - price) < tol

    def test_oas_spread_with_solver(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            spec="uk_gb",
            fixed_rate=1.0,
        )
        curve = Curve({dt(1999, 11, 25): 1.0, dt(2015, 12, 7): 0.85})
        Solver(
            curves=[curve],
            instruments=[
                IRS(
                    effective=dt(1999, 12, 8),
                    termination="2y",
                    spec="gbp_irs",
                    curves=curve,
                    leg2_fixing_series="eur_rfr",
                )
            ],
            s=[2.5],
        )
        # gilt.rate(curve, metric="dirty_price") = 80.52025551638633
        result = gilt.oaspread(curves=curve, price=95.00)
        curve_z = curve.shift(result)
        result = gilt.rate(curves=curve_z, metric="clean_price")
        assert abs(result - 95.00) < 1e-8

    def test_oaspread_ift_fwddiff(self):
        bond = FixedRateBond(dt(2000, 1, 1), "3Y", fixed_rate=2.5, spec="us_gb")
        curve = Curve({dt(2000, 7, 1): 1.0, dt(2005, 7, 1): 0.80})
        # Add AD variables to the curve without a Solver
        curve._set_ad_order(1)

        result = bond.oaspread(curves=curve, price=Variable(95.0, ["price"], []))
        grad = gradient(result, ["price"])[0]

        assert abs(bond.oaspread(curves=curve, price=95.01) - result - 0.01 * grad) < 1e-3
        assert abs(bond.oaspread(curves=curve, price=94.99) - result + 0.01 * grad) < 1e-3

    def test_oas_spread_metric(self):
        gilt = FixedRateBond(dt(1998, 12, 7), dt(2015, 12, 7), spec="uk_gb", fixed_rate=1.0)
        curve = Curve({dt(1999, 11, 3): 1.0, dt(2015, 12, 7): 0.85})
        result1 = gilt.oaspread(curves=curve, price=95.0, metric="clean_price")
        result2 = gilt.oaspread(
            curves=curve, price=95.0 + gilt.accrued(dt(1999, 11, 4)), metric="dirty_price"
        )
        result3 = gilt.oaspread(curves=curve, price=gilt.ytm(95.0, dt(1999, 11, 4)), metric="ytm")
        assert abs(result1 - result2) < 1e-5
        assert abs(result1 - result3) < 1e-5

    def test_cashflows_no_curve(self) -> None:
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1), termination="1Y", spec="uk_gb", fixed_rate=5.0
        )
        result = gilt.cashflows()  # no curve argument is passed to cashflows
        assert isinstance(result, DataFrame)

    def test_schedule_start_non_business(self) -> None:
        frb = FixedRateBond(
            effective=dt(2000, 1, 1),
            termination="1y",
            spec="us_gb",
            notional=5e6,
            fixed_rate=2.0,
        )
        assert frb.leg1.periods[1].settlement_params.payment == dt(2001, 1, 2)

    def test_random_ytm_collection(self):
        NUMBER = 75
        START = dt(2000, 1, 1)
        TENORS = ["2y", "3y", "4y", "5y", "6y", "7y", "8y", "9y", "10y", "15y"]
        COUPS = [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
        RAND_PRICES = np.random.rand(NUMBER) * 150 + 25.0
        BONDS = [
            FixedRateBond(
                effective=START,
                termination=TENORS[i % 10],
                spec="us_gb",
                fixed_rate=COUPS[i % 4],
            )
            for i in range(NUMBER)
        ]
        for i in range(NUMBER):
            BONDS[i].ytm(price=RAND_PRICES[i], settlement=dt(2001, 8, 30))

    def test_custom_calc_mode(self):
        cm = BondCalcMode(
            settle_accrual="linear_days",
            ytm_accrual="linear_days",
            v1="compounding",
            v2="regular",
            v3="compounding",
            c1="cashflow",
            ci="cashflow",
            cn="cashflow",
        )
        bond = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination="10y",
            frequency="s",
            calendar="ldn",
            convention="ActActICMA",
            modifier="none",
            settle=1,
            calc_mode=cm,
            fixed_rate=1.0,
        )
        bond2 = FixedRateBond(dt(2001, 1, 1), "10y", spec="uk_gb", fixed_rate=1.0)
        assert bond.price(3.0, dt(2002, 3, 4)) == bond2.price(3.0, dt(2002, 3, 4))
        assert bond.accrued(dt(2002, 3, 4)) == bond2.accrued(dt(2002, 3, 4))

    def test_must_have_fixed_rate(self):
        with pytest.raises(ValueError, match=r"`fixed_rate` must be provided for FixedRateBond."):
            FixedRateBond(
                effective=dt(2001, 1, 1),
                termination="10y",
                frequency="s",
                calendar="ldn",
                convention="ActActICMA",
                modifier="none",
                settle=1,
            )

    def test_ytm_domains2(self):
        # the first pass in the quadratic approximator predicts a yield outside of the
        # interval so a bisection method is adopted instead.

        frb = FixedRateBond(
            effective=dt(2000, 1, 15),
            termination=dt(2030, 9, 25),
            spec="uk_gb",
            stub="shortfront",
            fixed_rate=0.57744089871129,
        )
        result = frb.ytm(price=173.80904334438674, settlement=dt(2000, 1, 20))
        assert abs(result + 1.3549202231746622) < 1e-10

    def test_oas_coupon_on_non_bus_day(self):
        # coupon falls on 30th Jun (sunday) and paid on 1st July. OAS spread now handles.
        # dev gh 17
        bond = FixedRateBond(dt(2023, 12, 31), "3y", fixed_rate=0.5, spec="us_gb")
        curve = Curve({dt(2024, 6, 24): 1.0, dt(2028, 6, 25): 1.0})
        for today in [
            dt(2024, 6, 25),
            dt(2024, 6, 26),
            dt(2024, 6, 27),
            dt(2024, 6, 28),
            dt(2024, 6, 29),
            dt(2024, 6, 30),
            dt(2024, 7, 1),
            dt(2024, 7, 2),
            dt(2024, 7, 3),
        ]:
            curve_ = curve.translate(today)
            assert 49.1 < bond.oaspread(curves=curve_, price=100.0) < 49.2

    def test_dirty_price_on_non_bus_day(self):
        # coupon falls on 30th Jun (sunday) and paid on 1st July. OAS spread now handles.
        # dev gh 17
        bond = FixedRateBond(dt(2023, 12, 31), "3y", fixed_rate=0.5, spec="us_gb")
        curve = Curve({dt(2024, 6, 24): 1.0, dt(2028, 6, 25): 1.0})
        for today in [
            dt(2024, 6, 25),
            dt(2024, 6, 26),
            dt(2024, 6, 27),
            dt(2024, 6, 28),
            dt(2024, 6, 29),
            dt(2024, 6, 30),
            dt(2024, 7, 1),
            dt(2024, 7, 2),
            dt(2024, 7, 3),
        ]:
            curve_ = curve.translate(today)
            if today <= dt(2024, 6, 27):  # settlement Friday 28th June
                assert bond.rate(curves=curve_, metric="dirty_price") == 101.5
            else:
                assert bond.rate(curves=curve_, metric="dirty_price") == 101.25

    @pytest.mark.parametrize(
        "bond",
        [
            FixedRateBond(dt(2023, 12, 31), dt(2025, 12, 31), fixed_rate=4.25, spec="us_gb"),
            FixedRateBond(
                dt(2023, 12, 31), dt(2025, 12, 31), fixed_rate=4.25, spec="us_gb", modifier="F"
            ),
        ],
    )
    def test_npv_and_oas_with_adjusted_accrual_on_non_bus_day(self, bond):
        curve = Curve({dt(2024, 6, 28): 1.0, dt(2026, 6, 30): 0.96})
        result = (
            bond.npv(curves=curve),
            bond.oaspread(curves=curve, price=97.0),
            bond.rate(curves=curve, metric="clean_price"),
        )
        for date in [dt(2024, 7, 1), dt(2024, 7, 2)]:
            curve_ = curve.translate(date)
            assert abs(bond.npv(curves=curve_) - result[0]) < 250.0
            assert abs(bond.oaspread(curves=curve_, price=97.0) - result[1]) < 0.75
            assert abs(bond.rate(curves=curve_, metric="clean_price") - result[2]) < 0.03

    @pytest.mark.parametrize(
        ("settlement", "forward_settlement", "expected"),
        [
            (dt(2024, 6, 27), dt(2024, 6, 28), 100.002503),
            (dt(2024, 6, 27), dt(2024, 6, 29), 100.005596),
            (dt(2024, 6, 27), dt(2024, 6, 30), 100.007805),
            (dt(2024, 6, 27), dt(2024, 7, 1), 100.010140),
            (dt(2024, 6, 27), dt(2024, 7, 2), 100.012475),
            (dt(2024, 6, 29), dt(2024, 7, 1), 100.004550),
        ],
    )
    def test_fwd_from_repo_ex_div_and_holidays(self, settlement, forward_settlement, expected):
        bond = FixedRateBond(dt(2023, 12, 31), dt(2025, 12, 31), fixed_rate=4.25, spec="us_gb")
        result = bond.fwd_from_repo(
            price=100.0,
            settlement=settlement,
            forward_settlement=forward_settlement,
            repo_rate=5.0,
            convention="Act360",
        )
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize(
        ("settlement", "forward_settlement", "fwd_price"),
        [
            (dt(2024, 6, 27), dt(2024, 6, 28), 100.002503),
            (dt(2024, 6, 27), dt(2024, 6, 29), 100.005596),
            (dt(2024, 6, 27), dt(2024, 6, 30), 100.007805),
            (dt(2024, 6, 27), dt(2024, 7, 1), 100.010140),
            (dt(2024, 6, 27), dt(2024, 7, 2), 100.012475),
            (dt(2024, 6, 29), dt(2024, 7, 1), 100.004550),
        ],
    )
    def test_repo_from_fwd_ex_div_and_holidays(self, settlement, forward_settlement, fwd_price):
        bond = FixedRateBond(dt(2023, 12, 31), dt(2025, 12, 31), fixed_rate=4.25, spec="us_gb")
        result = bond.repo_from_fwd(
            price=100.0,
            settlement=settlement,
            forward_settlement=forward_settlement,
            forward_price=fwd_price,
            convention="Act360",
        )
        assert abs(result - 5.00) < 2e-4

    def test_183d_ytm(self):
        bond_base = FixedRateBond(dt(2000, 1, 1), dt(2001, 1, 1), fixed_rate=5.0, spec="us_gb")
        bond_test = FixedRateBond(
            dt(2000, 1, 1), dt(2001, 1, 1), fixed_rate=5.0, spec="us_gb", frequency="183D"
        )
        expected = bond_base.ytm(100, dt(2000, 1, 1))
        result = bond_test.ytm(100, dt(2000, 1, 1))
        assert abs(expected - result) < 1e-5

    def test_long_back_stub_split_accrued(self):
        bond = FixedRateBond(
            dt(2000, 1, 1), dt(2001, 2, 15), fixed_rate=20.0, spec="us_gb", stub="LongBack"
        )
        accrued = bond.accrued(dt(2001, 1, 15))
        approximation = (dt(2001, 1, 15) - dt(2000, 7, 1)).days / 365 * 20.0
        assert abs(accrued - approximation) < 1e-1

    def test_long_back_front_stubs_split_accrued(self):
        bond = FixedRateBond(
            dt(2000, 1, 1),
            dt(2002, 2, 15),
            front_stub=dt(2000, 9, 8),
            fixed_rate=20.0,
            spec="us_gb",
            stub="LongBack",
        )
        accrued = bond.accrued(dt(2002, 1, 15))
        approximation = (dt(2002, 1, 15) - dt(2001, 3, 8)).days / 365 * 20.0
        assert abs(accrued - approximation) < 1e-1

        price = bond.price(ytm=20.0, settlement=dt(2002, 1, 15))
        assert abs(price - 100.0) < 5e-1

    def test_coupon_setter(self):
        frb = FixedRateBond(dt(2000, 1, 1), dt(2005, 1, 1), fixed_rate=2.0, spec="uk_gb")
        frb.fixed_rate = 3.0
        assert frb.fixed_rate == 3.0
        assert frb.kwargs.leg1["fixed_rate"] == 3.0


class TestIndexFixedRateBond:
    def test_fixed_rate_bond_price(self) -> None:
        # test pricing functions against Nominal Gilt Example prices from UK DMO
        # these prices should be equivalent for the REAL component of Index Bonds
        bond = IndexFixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
            index_base=100.0,
        )
        assert abs(bond.price(4.445, dt(1999, 5, 24), True) - 145.012268) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 5, 26), True) - 145.047301) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 5, 27), True) - 141.070132) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 6, 7), True) - 141.257676) < 1e-6

        bond = IndexFixedRateBond(
            dt(1997, 1, 1),
            dt(2004, 11, 26),
            "S",
            convention="ActActICMA",
            fixed_rate=6.75,
            ex_div=7,
            calendar="ldn",
            index_base=100.0,
        )
        assert abs(bond.price(4.634, dt(1999, 5, 10), True) - 113.315543) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 17), True) - 113.415969) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 18), True) - 110.058738) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 26), True) - 110.170218) < 1e-6

    @pytest.mark.skip(reason="Frequency of zero calculates but is wrong. Docs do not allow.")
    def test_fixed_rate_bond_zero_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="`frequency` must be provided"):
            IndexFixedRateBond(
                dt(1999, 5, 7),
                dt(2002, 12, 7),
                "Z",
                convention="ActActICMA",
                fixed_rate=1.0,
            )

    def test_fixed_rate_bond_no_amortization(self) -> None:
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'amortization"):
            IndexFixedRateBond(
                effective=dt(1998, 12, 7),
                termination=dt(2015, 12, 7),
                frequency="S",
                calendar="ldn",
                currency="gbp",
                convention="ActActICMA",
                ex_div=7,
                fixed_rate=8.0,
                notional=-100,
                amortization=100,
                index_base=100.0,
            )

    def test_fixed_rate_bond_rate_raises(self) -> None:
        gilt = IndexFixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            index_base=100.0,
        )
        curve = Curve({dt(1998, 12, 7): 1.0, dt(2015, 12, 7): 0.50})
        with pytest.raises(ValueError, match="`metric` must be in"):
            gilt.rate(
                curves=[
                    Curve({dt(1992, 1, 1): 1.0, dt(2070, 1, 1): 0.13}, index_base=100.0),
                    curve,
                ],
                metric="bad_metric",
            )

    def test_initialisation_rate_metric(self) -> None:
        gilt = IndexFixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            settle=0,
            index_base=100.0,
            index_lag=3,
            metric="ytm",
        )
        disc_curve = Curve(
            {dt(1998, 12, 9): 1.0, dt(2015, 12, 7): 0.50}, index_base=100.0, index_lag=3
        )
        curve = Curve({dt(1998, 12, 1): 1.0, dt(2015, 12, 7): 0.50}, index_base=100.0, index_lag=3)
        clean_price = gilt.rate(curves=[curve, disc_curve], metric="clean_price")
        expected = gilt.ytm(price=clean_price, settlement=dt(1998, 12, 9))
        result = gilt.rate(curves=[curve, disc_curve])  # default metric is "ytm"
        assert abs(result - expected) < 1e-8

    @pytest.mark.parametrize(
        ("i_fixings", "expected"),
        [
            (NoInput(0), 1.161227269),
            ("index_series", (90 + 14 / 30 * 200) / 95),
        ],
    )
    def test_index_ratio(self, i_fixings, expected) -> None:
        if isinstance(i_fixings, str):
            fixings.add("index_series", Series([90.0, 290], index=[dt(2022, 1, 1), dt(2022, 2, 1)]))
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=3,
            index_base=110.0,
            interpolation="linear_index",
        )
        bond = IndexFixedRateBond(
            dt(2022, 1, 1),
            "9m",
            "Q",
            convention="ActActICMA",
            fixed_rate=4,
            ex_div=0,
            calendar="ldn",
            index_base=95.0,
            index_fixings=i_fixings,
            index_method="daily",
            index_lag=3,
        )
        result = bond.index_ratio(settlement=dt(2022, 4, 15), index_curve=i_curve)
        if isinstance(i_fixings, str):
            fixings.pop("index_series")
        assert abs(result - expected) < 1e-5

    @pytest.mark.skip(
        reason="This will calculate from the curve but will not be aligned with the specific list "
        "fixings, but since list fixings are not recommended in the documentation and the"
        "advice is to use a `fixings` object then this is OK."
    )
    def test_index_ratio_raises_float_index_fixings(self) -> None:
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=3,
            index_base=110.0,
            interpolation="linear_index",
        )
        bond = IndexFixedRateBond(
            dt(2022, 1, 1),
            "9m",
            "Q",
            convention="ActActICMA",
            fixed_rate=4,
            ex_div=0,
            calendar="ldn",
            index_base=95.0,
            index_fixings=[100.0, 200.0],
            index_method="daily",
        )
        # with pytest.raises(TypeError, match="`index_fixings` must be of type: Str, Series, Dual"):
        bond.index_ratio(settlement=dt(2022, 4, 15), curve=i_curve)

    def test_fixed_rate_bond_npv_private(self) -> None:
        # this test shadows 'fixed_rate_bond_npv' but extends it for projection
        curve = Curve({dt(2004, 11, 25): 1.0, dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})
        index_curve = Curve(
            {dt(2004, 11, 25): 1.0, dt(2034, 1, 1): 1.0},
            index_base=100.0,
            interpolation="linear_index",
        )
        gilt = IndexFixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=0,
            index_base=50.0,
            index_lag=3,
            index_method="daily",
        )
        with pytest.warns(UserWarning):
            result = gilt.npv(
                curves=[index_curve, curve], settlement=dt(2010, 11, 27), forward=dt(2010, 11, 25)
            )
            expected = 109.229489312983 * 2.0  # npv should match associated test
            assert abs(result - expected) < 1e-6

    def test_index_base_forecast(self, curve) -> None:
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=3,
            index_base=95.0,
            interpolation="linear_index",
        )
        bond = IndexFixedRateBond(
            dt(2022, 1, 1),
            "9m",
            "Q",
            convention="ActActICMA",
            fixed_rate=4,
            ex_div=0,
            calendar=NoInput(0),
            index_method="daily",
            settle=0,
        )
        cashflows = bond.cashflows(curves=[i_curve, curve])
        for i in range(4):
            assert cashflows.iloc[i]["Index Base"] == 95.0

        result = bond.npv(curves=[i_curve, curve])
        expected = -1006875.3812
        assert abs(result - expected) < 1e-4

        result = bond.rate(curves=[i_curve, curve], metric="index_dirty_price")
        assert abs(result * -1e4 - expected) < 1e-4

    def test_fixed_rate_bond_fwd_rate(self) -> None:
        gilt = IndexFixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            settle=0,
            index_base=50.0,
            index_lag=3,
        )
        curve = Curve({dt(1998, 12, 9): 1.0, dt(2015, 12, 7): 0.50})
        i_curve = Curve(
            {dt(1998, 12, 1): 1.0, dt(2015, 12, 7): 1.0},
            index_base=100.0,
            interpolation="linear_index",
            index_lag=3,
        )
        clean_price = gilt.rate(curves=[i_curve, curve], metric="clean_price")
        index_clean_price = gilt.rate(curves=[i_curve, curve], metric="index_clean_price")
        assert abs(index_clean_price * 0.5 - clean_price) < 1e-3

        result = gilt.rate(
            curves=[i_curve, curve],
            metric="clean_price",
            settlement=dt(1998, 12, 9),
            # forward
        )
        assert abs(result - clean_price) < 1e-8
        result = gilt.rate(
            curves=[i_curve, curve],
            metric="index_clean_price",
            settlement=dt(1998, 12, 9),
            # forward
        )
        assert abs(result * 0.5 - clean_price) < 1e-8

        result = gilt.rate(curves=[i_curve, curve], metric="dirty_price")
        expected = clean_price + gilt.accrued(dt(1998, 12, 9))
        assert result == expected
        result = gilt.rate(
            curves=[i_curve, curve],
            metric="dirty_price",
            settlement=dt(1998, 12, 9),
        )
        assert abs(result - clean_price - gilt.accrued(dt(1998, 12, 9))) < 1e-8
        result = gilt.rate(
            curves=[i_curve, curve],
            metric="index_dirty_price",
            settlement=dt(1998, 12, 9),
        )
        assert abs(result * 0.5 - clean_price - gilt.accrued(dt(1998, 12, 9))) < 1e-8

        result = gilt.rate(curves=[i_curve, curve], metric="ytm")
        expected = gilt.ytm(clean_price, dt(1998, 12, 9), False)
        assert abs(result - expected) < 1e-8

    def test_base_setting_and_index_ratio(self):
        # GB00BMY62Z61
        name = str(hash(os.urandom(8)))
        fixings.add(
            name,
            Series(
                index=[
                    dt(2025, 3, 1),
                    dt(2025, 4, 1),
                    dt(2025, 5, 1),
                    dt(2025, 6, 1),
                    dt(2025, 7, 1),
                    dt(2025, 8, 1),
                    dt(2025, 9, 1),
                    dt(2025, 10, 1),
                ],
                data=[395.3, 402.2, 402.9, 404.5, 406.2, 407.7, 406.1, 407.4],
            ),
        )
        gilt = IndexFixedRateBond(
            effective=dt(2025, 6, 11),
            termination=dt(2038, 9, 22),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            modifier="None",
            ex_div=7,
            fixed_rate=1.75,
            settle=0,
            index_fixings=name,
            index_lag=3,
            index_method="daily",
        )

        # these index base and index ratio are calculated externally and verified here
        assert gilt.leg1.periods[0].index_params.index_base.value == 397.60
        index_ratio = gilt.index_ratio(settlement=dt(2025, 9, 12), index_curve=NoInput(0))
        fixings.pop(name)
        assert abs(index_ratio - 1.018920) < 1e-5

    def test_accrued_and_indexed_accrued(self):
        # GB00BMY62Z61
        name = str(hash(os.urandom(8)))
        fixings.add(
            name,
            Series(
                index=[
                    dt(2025, 3, 1),
                    dt(2025, 4, 1),
                    dt(2025, 5, 1),
                    dt(2025, 6, 1),
                    dt(2025, 7, 1),
                    dt(2025, 8, 1),
                    dt(2025, 9, 1),
                    dt(2025, 10, 1),
                ],
                data=[395.3, 402.2, 402.9, 404.5, 406.2, 407.7, 406.1, 407.4],
            ),
        )
        gilt = IndexFixedRateBond(
            effective=dt(2025, 6, 11),
            termination=dt(2038, 9, 22),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            modifier="None",
            ex_div=7,
            fixed_rate=1.75,
            settle=0,
            index_fixings=name,
            index_lag=3,
            index_method="daily",
        )

        accrued = gilt.accrued(settlement=dt(2025, 9, 12))
        index_ratio = gilt.index_ratio(settlement=dt(2025, 9, 12), index_curve=NoInput(0))
        indexed_accrued = accrued * index_ratio
        # this indexed accrued is calculated externally and verified here
        assert abs(indexed_accrued + 0.048454076) < 1e-7
        assert abs(gilt.accrued(settlement=dt(2025, 9, 12), indexed=True) - indexed_accrued) < 1e-7

        fixings.pop(name)

    @pytest.mark.parametrize(
        ("price", "indexed", "dirty", "expected"),
        [
            (99.423682, True, True, 100.2169930),
            (99.1924173, True, False, 99.9007374),
            (98.1322608, False, True, 98.0424122),
            (97.904000, False, False, 97.733020),
        ],
    )
    def test_fwd_from_repo(self, price, indexed, dirty, expected):
        # GB00BMY62Z61
        name = str(hash(os.urandom(8)))
        fixings.add(
            name,
            Series(
                index=[
                    dt(2025, 3, 1),
                    dt(2025, 4, 1),
                    dt(2025, 5, 1),
                    dt(2025, 6, 1),
                    dt(2025, 7, 1),
                    dt(2025, 8, 1),
                    dt(2025, 9, 1),
                    dt(2025, 10, 1),
                ],
                data=[395.3, 402.2, 402.9, 404.5, 406.2, 407.7, 406.1, 407.4],
            ),
        )
        gilt = IndexFixedRateBond(
            effective=dt(2025, 6, 11),
            termination=dt(2038, 9, 22),
            fixed_rate=1.75,
            spec="uk_gbi",
            index_fixings=name,
        )
        fwd = gilt.fwd_from_repo(
            price=price,
            settlement=dt(2025, 7, 29),
            forward_settlement=dt(2025, 11, 25),
            repo_rate=4.00,
            convention="act365F",
            dirty=dirty,
            indexed=indexed,
        )
        fixings.pop(name)
        assert abs(fwd - expected) < 5e-4

    @pytest.mark.parametrize(
        ("price", "indexed", "dirty", "fwd_price"),
        [
            (99.423682, True, True, 100.2169930),
            (99.1924173, True, False, 99.9007374),
            (98.1322608, False, True, 98.0424122),
            (97.904000, False, False, 97.733020),
        ],
    )
    def test_repo_from_fwd(self, price, indexed, dirty, fwd_price):
        # GB00BMY62Z61
        name = str(hash(os.urandom(8)))
        fixings.add(
            name,
            Series(
                index=[
                    dt(2025, 3, 1),
                    dt(2025, 4, 1),
                    dt(2025, 5, 1),
                    dt(2025, 6, 1),
                    dt(2025, 7, 1),
                    dt(2025, 8, 1),
                    dt(2025, 9, 1),
                    dt(2025, 10, 1),
                ],
                data=[395.3, 402.2, 402.9, 404.5, 406.2, 407.7, 406.1, 407.4],
            ),
        )
        gilt = IndexFixedRateBond(
            effective=dt(2025, 6, 11),
            termination=dt(2038, 9, 22),
            fixed_rate=1.75,
            spec="uk_gbi",
            index_fixings=name,
        )
        repo = gilt.repo_from_fwd(
            price=price,
            settlement=dt(2025, 7, 29),
            forward_settlement=dt(2025, 11, 25),
            forward_price=fwd_price,
            convention="act365F",
            dirty=dirty,
            indexed=indexed,
        )
        fixings.pop(name)
        assert abs(repo - 4.00) < 2e-3

    @pytest.mark.parametrize(
        ("indexed_price", "indexed_ytm"),
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_duration_index_linked_finite_diff(self, indexed_price, indexed_ytm):
        # GB00BMY62Z61
        name = str(hash(os.urandom(8)))
        fixings.add(
            name,
            Series(
                index=[
                    dt(2025, 3, 1),
                    dt(2025, 4, 1),
                    dt(2025, 5, 1),
                    dt(2025, 6, 1),
                    dt(2025, 7, 1),
                    dt(2025, 8, 1),
                    dt(2025, 9, 1),
                    dt(2025, 10, 1),
                ],
                data=[395.3, 402.2, 402.9, 404.5, 406.2, 407.7, 406.1, 407.4],
            ),
        )
        index_curve = Curve({dt(2025, 10, 1): 1.0, dt(2045, 10, 1): 1.0}, index_base=407.4).shift(
            100
        )
        gilt = IndexFixedRateBond(
            effective=dt(2025, 6, 11),
            termination=dt(2038, 9, 22),
            fixed_rate=1.75,
            spec="uk_gbi",
            index_fixings=name,
        )
        value = gilt.duration(
            ytm=2.00,
            settlement=dt(2025, 7, 29),
            metric="risk",
            indexed_price=indexed_price,
            indexed_ytm=indexed_ytm,
            index_curve=index_curve,
        )

        # finite diff test:
        original_price = gilt.price(
            ytm=2.00,
            settlement=dt(2025, 7, 29),
            indexed_price=indexed_price,
            indexed_ytm=indexed_ytm,
            index_curve=index_curve,
            dirty=True,
        )
        bumped_price = gilt.price(
            ytm=1.999,
            settlement=dt(2025, 7, 29),
            indexed_ytm=indexed_ytm,
            indexed_price=indexed_price,
            index_curve=index_curve,
            dirty=True,
        )
        expected = (bumped_price - original_price) * 1000.0
        assert abs(value - expected) < 1e-3

        ## Test modified
        modified = gilt.duration(
            ytm=2.00,
            settlement=dt(2025, 7, 29),
            metric="modified",
            indexed_price=indexed_price,
            indexed_ytm=indexed_ytm,
            index_curve=index_curve,
        )
        assert abs(value / original_price * 100.0 - modified) < 1e-6

        # Test macauley
        macauley = gilt.duration(
            ytm=2.00,
            settlement=dt(2025, 7, 29),
            metric="duration",
            indexed_price=indexed_price,
            indexed_ytm=indexed_ytm,
            index_curve=index_curve,
        )
        assert abs(modified * (1 + 0.02 / 2) - macauley) < 1e-6
        fixings.pop(name)

    # TODO: implement these tests
    #
    # def test_convexity(self):
    #     assert False

    def test_latest_fixing(self) -> None:
        # this is German government inflation bond with fixings given for a specific settlement
        # calculation

        ibnd = IndexFixedRateBond(
            effective=dt(2021, 2, 11),
            front_stub=dt(2022, 4, 15),
            termination=dt(2033, 4, 15),
            convention="ActActICMA",
            calendar="tgt",
            frequency="A",
            index_lag=3,
            index_base=124.17000 / 1.18851,  # implying from 1st Jan 2024 on webpage
            index_method="daily",
            payment_lag=0,
            currency="eur",
            fixed_rate=0.1,
            ex_div=1,
            settle=1,
            index_fixings=Series(data=[124.17, 123.46], index=[dt(2024, 1, 1), dt(2024, 2, 1)]),
        )
        result = ibnd.ytm(price=100.32, settlement=dt(2024, 1, 5))
        expected = 0.065
        assert (result - expected) < 1e-2

    def test_rate_with_fx_is_same(self) -> None:
        usd = Curve(nodes={dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.9, dt(2010, 1, 5): 0.8})
        gbp = Curve(nodes={dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.9, dt(2010, 1, 5): 0.8})
        gbpi = Curve(
            nodes={dt(2000, 1, 1): 1.0, dt(2010, 1, 1): 0.95},
            index_base=100.0,
            interpolation="linear_index",
            index_lag=3,
        )
        fxf = FXForwards(
            fx_rates=FXRates({"gbpusd": 1.25}, settlement=dt(2000, 1, 1)),
            fx_curves={"gbpgbp": gbp, "usdusd": usd, "gbpusd": gbp},
        )
        result = IndexFixedRateBond(
            dt(2000, 1, 1),
            "5y",
            index_base=100.5,
            spec="uk_gbi",
            fixed_rate=1.0,
        ).rate(curves=[gbpi, gbp], metric="clean_price")
        result2 = IndexFixedRateBond(
            dt(2000, 1, 1),
            "5y",
            index_base=100.5,
            spec="uk_gbi",
            fixed_rate=1.0,
        ).rate(curves=[gbpi, gbp], metric="clean_price", fx=fxf)
        assert result == result2

    def test_spec_kwargs(self) -> None:
        # GH346
        fixings = Series(data=[314.175, 314.54], index=[dt(2024, 9, 1), dt(2024, 10, 1)])
        tii_0728 = IndexFixedRateBond(
            effective=dt(2018, 7, 31),
            termination=dt(2028, 7, 15),
            spec="us_gb_tsy",
            fixed_rate=0.75,
            notional=-100e6,
            curves=["sofr", "sofr"],
            index_lag=3,
            index_method="monthly",
            index_base=251.01658,
            index_fixings=fixings,
        )
        result = tii_0728.ytm(100, dt(2024, 8, 26))
        assert (result - 0.749935) < 1e-5

    def test_custom_calc_mode(self):
        cm = BondCalcMode(
            settle_accrual="linear_days",
            ytm_accrual="linear_days",
            v1="compounding",
            v2="regular",
            v3="compounding",
            c1="cashflow",
            ci="cashflow",
            cn="cashflow",
        )
        bond = IndexFixedRateBond(
            effective=dt(2001, 1, 1),
            termination="10y",
            frequency="s",
            calendar="ldn",
            convention="ActActICMA",
            modifier="none",
            settle=1,
            calc_mode=cm,
            fixed_rate=1.0,
            index_base=100.0,
        )
        bond2 = IndexFixedRateBond(
            dt(2001, 1, 1), "10y", spec="uk_gb", fixed_rate=1.0, index_base=100.0
        )
        assert bond.price(3.0, dt(2002, 3, 4)) == bond2.price(3.0, dt(2002, 3, 4))
        assert bond.accrued(dt(2002, 3, 4)) == bond2.accrued(dt(2002, 3, 4))

    def test_fixed_rate_getter_and_setter(self):
        tii_0728 = IndexFixedRateBond(
            effective=dt(2018, 7, 31),
            termination=dt(2028, 7, 15),
            spec="us_gbi",
            fixed_rate=0.75,
        )
        assert tii_0728.fixed_rate == 0.75
        tii_0728.fixed_rate = 1.90
        assert tii_0728.fixed_rate == 1.90

    def test_no_fixed_rate_raises(self):
        with pytest.raises(ValueError, match="`fixed_rate` must be provided for IndexFixedRateBo"):
            IndexFixedRateBond(
                effective=dt(2018, 7, 31),
                termination=dt(2028, 7, 15),
                spec="us_gbi",
            )

    def test_parse_curves(self, curve):
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=3,
            index_base=95.0,
            interpolation="linear_index",
        )
        bond = IndexFixedRateBond(
            dt(2022, 1, 1),
            "9m",
            "Q",
            convention="ActActICMA",
            fixed_rate=4,
            ex_div=0,
            calendar=NoInput(0),
            index_method="daily",
            settle=0,
        )
        result1 = bond.npv(curves=[i_curve, curve])
        result2 = bond.npv(curves={"index_curve": i_curve, "disc_curve": curve})
        expected = -1006875.3812
        assert abs(result1 - expected) < 1e-5
        assert abs(result1 - result2) < 1e-5

    def test_rate_docs(self):
        disc_curve = Curve(
            nodes={dt(2025, 7, 28): 1.0, dt(2045, 7, 25): 1.0}, convention="act365f"
        ).shift(250)  # curve begins at 0% and gets shifted by 250 Act365F O/N basis points
        index_curve = Curve(
            nodes={dt(2025, 5, 1): 1.0, dt(2045, 5, 1): 1.0},
            convention="act365f",
            index_lag=0,
            index_base=402.9,
        ).shift(100)  # curves begins at 0% and gets shifted by 100 Ac6t365f O/N basis points
        fixings.add(
            "UK_RPI_987",
            Series(
                index=[dt(2025, 3, 1), dt(2025, 4, 1), dt(2025, 5, 1)], data=[395.3, 402.2, 402.9]
            ),
        )
        ukti = IndexFixedRateBond(  # ISIN: GB00BMY62Z61
            effective=dt(2025, 6, 11),
            termination=dt(2038, 9, 22),
            fixed_rate=1.75,
            spec="uk_gbi",
            index_fixings="UK_RPI_987",
        )
        a1 = ukti.rate(
            curves=[index_curve, disc_curve], metric="clean_price"
        )  # settles T+1 i.e. 29th July
        a2 = ukti.rate(curves=[index_curve, disc_curve], metric="dirty_price")
        a3 = ukti.rate(curves=[index_curve, disc_curve], metric="index_clean_price")
        a4 = ukti.rate(curves=[index_curve, disc_curve], metric="index_dirty_price")
        a5 = ukti.rate(curves=[index_curve, disc_curve], metric="ytm")
        a6 = ukti.accrued(settlement=dt(2025, 7, 29))
        a7 = ukti.accrued(settlement=dt(2025, 7, 29), indexed=True)
        a8 = ukti.rate(curves=[index_curve, disc_curve], metric="index_ytm")

        assert abs(a1 - 102.90237315163287) < 1e-5
        assert abs(a2 - 103.13063402119809) < 1e-5
        assert abs(a3 - 104.25652750721756) < 1e-5
        assert abs(a4 - 104.487792199156) < 1e-5
        assert abs(a5 - 1.5058915118424034) < 1e-5
        assert abs(a6 - 0.228260) < 1e-5
        assert abs(a7 - 0.231264) < 1e-5
        assert abs(a8 - 2.5174145913908443) < 1e-5

        fixings.pop("UK_RPI_987")

    def test_index_ytm(self):
        fixings.add(
            "UK_RPI_9843",
            Series(
                index=[
                    dt(2025, 1, 1),
                    dt(2026, 1, 1),
                    dt(2027, 1, 1),
                    dt(2028, 1, 1),
                    dt(2029, 1, 1),
                    dt(2030, 1, 1),
                ],
                data=[100.0, 102, 103, 104, 105, 106],
            ),
        )
        bond = IndexFixedRateBond(
            effective=dt(2025, 1, 6),
            termination=dt(2030, 1, 6),
            roll=6,
            calendar="bus",
            convention="actacticma",
            frequency="A",
            # index_base=100.0,
            index_lag=0,
            index_method="monthly",
            ex_div=1,
            fixed_rate=2.0,
            index_fixings="UK_RPI_9843",
        )
        assert bond.leg1.periods[0].index_params.index_base.value == 100.0
        result = bond.ytm(
            price=101.9456166,
            settlement=dt(2026, 1, 6),
            indexed_price=True,
            indexed_ytm=True,
            dirty=True,
        )
        expected = 3.00
        # 101.9456166 = 2 * 1.03/1.03 + 2 * 1.04/1.03**2 + 2 * 1.05/1.03**3 + 102 * 1.06/1.03**4
        fixings.pop("UK_RPI_9843")
        assert abs(result - expected) < 1e-6

        result = bond.ytm(
            price=101.9456166 / 1.02,
            settlement=dt(2026, 1, 6),
            indexed_price=False,
            indexed_ytm=False,
            dirty=True,
        )
        expected = 2.0140070859464996
        assert abs(result - expected) < 1e-6
        # clean yield is approximately 1% lower than indexed yield since inflation is approx 1%

    def test_index_ytm2(self):
        index_curve = Curve(
            nodes={dt(2025, 5, 1): 1.0, dt(2045, 5, 1): 1.0},
            convention="act365f",
            index_lag=0,
            index_base=402.9,
        ).shift(100)  # curves begins at 0% and gets shifted by 100 Ac6t365f O/N basis points
        ukti = IndexFixedRateBond(  # ISIN: GB00BMY62Z61
            effective=dt(2025, 6, 11),
            termination=dt(2038, 9, 22),
            fixed_rate=1.75,
            spec="uk_gbi",
            index_base=397.60,
        )

        prices = [104.62775438373183, 103.24009646398126, 104.3626899720302, 102.97854755093778]

        for i, (dirty, indexed_price) in enumerate(product([True, False], [True, False])):
            indexed_ytm = ukti.ytm(
                price=prices[i],
                settlement=dt(2025, 8, 5),
                indexed_price=indexed_price,
                indexed_ytm=True,
                dirty=dirty,
                index_curve=index_curve,
            )
            assert abs(indexed_ytm - 2.5100000) < 1e-8

        for i, (dirty, indexed_price) in enumerate(product([True, False], [True, False])):
            unindexed_ytm = ukti.ytm(
                price=prices[i],
                settlement=dt(2025, 8, 5),
                indexed_price=indexed_price,
                indexed_ytm=False,
                dirty=dirty,
                index_curve=index_curve,
            )
            assert abs(unindexed_ytm - 1.499260363) < 1e-8

    def test_index_price(self):
        index_curve = Curve(
            nodes={dt(2025, 5, 1): 1.0, dt(2045, 5, 1): 1.0},
            convention="act365f",
            index_lag=0,
            index_base=402.9,
        ).shift(100)  # curves begins at 0% and gets shifted by 100 Ac6t365f O/N basis points
        ukti = IndexFixedRateBond(  # ISIN: GB00BMY62Z61
            effective=dt(2025, 6, 11),
            termination=dt(2038, 9, 22),
            fixed_rate=1.75,
            spec="uk_gbi",
            index_base=397.60,
        )

        prices_from_indexed_ytm = []
        for dirty, indexed_price in product([True, False], [True, False]):
            prices_from_indexed_ytm.append(
                ukti.price(
                    ytm=2.5100000,
                    settlement=dt(2025, 8, 5),
                    indexed_price=indexed_price,
                    indexed_ytm=True,
                    dirty=dirty,
                    index_curve=index_curve,
                )
            )
        prices_from_unindexed_ytm = []
        for dirty, indexed_price in product([True, False], [True, False]):
            prices_from_unindexed_ytm.append(
                ukti.price(
                    ytm=1.499260363,
                    settlement=dt(2025, 8, 5),
                    indexed_price=indexed_price,
                    indexed_ytm=False,
                    dirty=dirty,
                    index_curve=index_curve,
                )
            )

        for p1, p2 in zip(prices_from_indexed_ytm, prices_from_unindexed_ytm):
            assert abs(p1 - p2) < 1e-8


class TestBill:
    def test_bill_discount_rate(self) -> None:
        # test pricing functions against Treasury Bill Example from US Treasury
        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
            calc_mode="ustb",
        )

        assert bill.discount_rate(99.93777, dt(2004, 1, 22)) == 0.8000999999999543
        assert bill.price(0.800, dt(2004, 1, 22)) == 99.93777777777778

    def test_bill_ytm(self) -> None:
        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
            calc_mode="ustb",
        )
        # this YTM is equivalent to the FixedRateBond ytm with coupon of 0.0
        result = bill.ytm(99.937778, dt(2004, 1, 22))

        # TODO this does not match US treasury example because the method is different
        assert abs(result - 0.814) < 1e-2

    def test_bill_ytm2(self) -> None:
        # this is a longer than 6m period
        bill = Bill(
            effective=dt(1990, 6, 7),
            termination=dt(1991, 6, 6),
            convention="act360",
            calc_mode="ustb",
        )
        price = bill.price(7.65, settlement=dt(1990, 6, 7))
        result = bill.ytm(price, settlement=dt(1990, 6, 7))
        assert abs(result - 8.237) < 1e-3

    def test_bill_simple_rate(self) -> None:
        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
            calc_mode="ustb",
        )
        d = dcf(dt(2004, 1, 22), dt(2004, 2, 19), "Act360")
        expected = 100 * (1 / (1 - 0.0080009999999 * d) - 1) / d  # floating point truncation
        expected = 100 * (100 / 99.93777777777778 - 1) / d
        result = bill.simple_rate(99.93777777777778, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

    def test_bill_initialised_rate_metric(self) -> None:
        curve = Curve({dt(2004, 1, 22): 1.00, dt(2005, 1, 22): 0.992})

        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
            settle=0,
            calc_mode="ustb",
            metric="simple_rate",
        )
        price = bill.rate(curves=curve, metric="price")
        expected = bill.simple_rate(price, dt(2004, 1, 22))
        result = bill.rate(curves=curve)
        assert abs(result - expected) < 1e-6

    def test_bill_rate(self) -> None:
        curve = Curve({dt(2004, 1, 22): 1.00, dt(2005, 1, 22): 0.992})

        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
            settle=0,
            calc_mode="ustb",
        )

        result = bill.rate(curves=curve, metric="price")
        expected = 99.9385705675
        assert abs(result - expected) < 1e-6

        result = bill.rate(curves=curve, metric="discount_rate")
        expected = bill.discount_rate(99.9385705675, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curves=curve, metric="simple_rate")
        expected = bill.simple_rate(99.9385705675, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curves=curve, metric="ytm")
        expected = bill.ytm(99.9385705675, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

        bill.kwargs.meta["settle"] = 2  # set the bill to T+2 settlement and re-run the calculations

        result = bill.rate(curves=curve, metric="price")
        expected = 99.94734388985547
        assert abs(result - expected) < 1e-6

        result = bill.rate(curves=curve, metric="discount_rate")
        expected = bill.discount_rate(99.94734388985547, dt(2004, 1, 26))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curves=curve, metric="simple_rate")
        expected = bill.simple_rate(99.94734388985547, dt(2004, 1, 26))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curves=curve, metric="ytm")
        expected = bill.ytm(99.94734388985547, dt(2004, 1, 26))
        assert abs(result - expected) < 1e-6

    def test_bill_default_calc_mode(self) -> None:
        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
            settle=0,
        )
        assert bill.kwargs.meta["calc_mode"] == US_GBB

    def test_bill_rate_raises(self) -> None:
        curve = Curve({dt(2004, 1, 22): 1.00, dt(2005, 1, 22): 0.992})

        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
        )

        with pytest.raises(ValueError, match="`metric` must be in"):
            bill.rate(curves=curve, metric="bad vibes")

    def test_sgbb(self) -> None:
        bill = Bill(
            effective=dt(2023, 3, 15),
            termination=dt(2024, 3, 20),
            spec="se_gbb",
        )
        result = bill.price(3.498, settlement=dt(2023, 3, 15))
        expected = 96.520547
        assert abs(result - expected) < 1e-6

        ytm = bill.ytm(price=96.520547, settlement=dt(2023, 3, 15))
        assert abs(ytm - 3.5546338) < 1e-5

    def test_text_example(self) -> None:
        bill = Bill(effective=dt(2023, 5, 17), termination=dt(2023, 9, 26), spec="us_gbb")
        result = bill.ytm(99.75, settlement=dt(2023, 9, 7))
        bond = FixedRateBond(
            effective=dt(2023, 3, 26),
            termination=dt(2023, 9, 26),
            fixed_rate=0.0,
            spec="us_gb",
        )
        expected = bond.ytm(99.75, settlement=dt(2023, 9, 7))
        assert abs(result - expected) < 1e-14
        assert abs(result - 4.90740754) < 1e-7

    @pytest.mark.parametrize(
        ("price", "tol"), [(96.0, 1e-6), (95.0, 1e-6), (93.0, 1e-5), (80.0, 1e-2)]
    )
    def test_oaspread(self, price, tol) -> None:
        bill = Bill(
            effective=dt(1998, 12, 7),
            termination=dt(1999, 10, 7),
            spec="us_gbb",
        )
        curve = Curve({dt(1998, 12, 7): 1.0, dt(2015, 12, 7): 0.75})
        # result = bill.rate(curve, metric="price") # = 98.605
        result = bill.oaspread(curves=curve, price=price)
        curve_z = curve.shift(result)
        result = bill.rate(curves=curve_z, metric="clean_price")
        assert abs(result - price) < tol

    def test_with_fx_supplied(self) -> None:
        usd = Curve(nodes={dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.9, dt(2010, 1, 5): 0.8})
        gbp = Curve(nodes={dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.9, dt(2010, 1, 5): 0.8})
        fxf = FXForwards(
            fx_rates=FXRates({"gbpusd": 1.25}, settlement=dt(2000, 1, 1)),
            fx_curves={"gbpgbp": gbp, "usdusd": usd, "gbpusd": gbp},
        )
        result = Bill(dt(2000, 1, 1), "3m", spec="us_gbb").rate(curves=gbp, metric="discount_rate")
        result2 = Bill(dt(2000, 1, 1), "3m", spec="us_gbb").rate(
            curves=gbp,
            metric="discount_rate",
            fx=fxf,
        )
        assert result == result2

    def test_duration(self) -> None:
        b = Bill(dt(2000, 1, 1), "6m", frequency="A", spec="us_gbb")
        result = b.duration(ytm=5.0, settlement=dt(2000, 1, 10), metric="duration")
        assert result == 0.5170058346378255

        b = Bill(dt(2000, 1, 1), "6m", spec="us_gbb")
        result = b.duration(ytm=5.0, settlement=dt(2000, 1, 10), metric="duration")
        assert result == 0.5046961719083534

        b = Bill(dt(2000, 1, 1), "6m", frequency="Q", spec="us_gbb")
        result = b.duration(ytm=5.0, settlement=dt(2000, 1, 10), metric="duration")
        assert result == 0.4985413405436174

    def test_custom_calc_mode(self):
        from rateslib.instruments.bonds import BillCalcMode

        cm = BillCalcMode(price_type="simple", ytm_clone_kwargs="uk_gb")
        bill = Bill(
            effective=dt(2001, 1, 1),
            termination="3m",
            calendar="ldn",
            convention="Act365f",
            modifier="none",
            settle=1,
            calc_mode=cm,
        )
        bill2 = Bill(dt(2001, 1, 1), "3m", spec="uk_gbb")
        assert bill.simple_rate(99.0, dt(2001, 2, 4)) == bill2.simple_rate(99.0, dt(2001, 2, 4))

    def test_us_gbb_eom(self):
        b = Bill(dt(2023, 2, 28), "3m", spec="us_gbb")
        assert b.leg1._regular_periods[0].period_params.end == dt(2023, 5, 31)

    def test_se_gbb_eom(self):
        b = Bill(dt(2023, 2, 28), "3m", spec="se_gbb")
        assert b.leg1._regular_periods[0].period_params.end == dt(2023, 5, 28)

    def test_act_act_icma(self):
        # gh 144
        with pytest.warns(
            UserWarning,
            match="`frequency` cannot be 'Zero' variant in combination with 'ActActICMA",
        ):
            bill_actacticma = Bill(
                effective=dt(2024, 2, 29),
                termination=dt(2024, 5, 29),  # 90 calendar days
                modifier="NONE",
                calendar="bus",
                payment_lag=0,
                notional=-1000000,
                currency="usd",
                convention="ACTACTICMA",
                settle=0,
                calc_mode="us_gbb",
            )
            assert bill_actacticma.leg1._regular_periods[0].period_params.dcf == 0.2465753424657534

        bill_act360 = Bill(
            effective=dt(2024, 2, 29),
            termination=dt(2024, 5, 29),  # 90 calendar days
            modifier="NONE",
            calendar="bus",
            payment_lag=0,
            notional=-1000000,
            currency="usd",
            convention="ACT360",
            settle=0,
            calc_mode="us_gbb",
        )
        assert bill_act360.leg1._regular_periods[0].period_params.dcf == 0.25

    def test_ex_div(self):
        b1 = Bill(dt(2000, 1, 3), "3m", spec="us_gbb")
        assert b1.ex_div(dt(200, 4, 3)) is False

        b2 = Bill(dt(2000, 1, 3), "3m", ex_div=2, spec="us_gbb")
        assert b2.ex_div(dt(2000, 4, 3)) is True
        assert b2.ex_div(dt(2000, 3, 31)) is True
        assert b2.ex_div(dt(2000, 3, 30)) is False

    def test_bill_roll(self):
        b1 = Bill(dt(2026, 1, 30), "6m", spec="us_gbb", roll=30)
        b2 = Bill(dt(2026, 1, 30), "6m", spec="us_gbb", roll=31)
        assert b1.leg1.schedule.termination == dt(2026, 7, 30)
        assert b2.leg1.schedule.termination == dt(2026, 7, 31)

    def test_bill_eom(self):
        b1 = Bill(dt(2026, 1, 30), "6m", spec="us_gbb", eom=False)
        b2 = Bill(dt(2026, 1, 30), "6m", spec="us_gbb", eom=True)
        assert b1.leg1.schedule.termination == dt(2026, 7, 30)
        assert b2.leg1.schedule.termination == dt(2026, 7, 31)


class TestFloatRateNote:
    @pytest.mark.parametrize(
        ("curve_spd", "method", "float_spd", "expected"),
        [
            (10, NoInput(0), 0, 10.055032859883),
            (500, NoInput(0), 0, 508.93107035125325),
            (-200, NoInput(0), 0, -200.053341848676),
            (10, "isda_compounding", 0, 10.00000120),
            (500, "isda_compounding", 0, 499.9999999997),
            (-200, "isda_compounding", 0, -199.99999999),
            (10, NoInput(0), 25, 10.055032859883),
            (500, NoInput(0), 250, 508.93107035125325),
            (10, "isda_compounding", 25, 10.00000120),
            (500, "isda_compounding", 250, 499.99999999975523),
            (10, NoInput(0), -25, 10.055032859883),
            (500, NoInput(0), -250, 508.93107035125325),
            (10, "isda_compounding", -25, 10.00000120),
            (500, "isda_compounding", -250, 499.9999999997),
        ],
    )
    def test_float_rate_bond_rate_spread(self, curve_spd, method, float_spd, expected) -> None:
        """
        When a DF curve is shifted it bumps daily rates.
        But under the "none_simple" compounding method this does not compound daily
        therefore the `float_spread` should be slightly higher than the bumped curve.
        When the method is "isda_compounding" this closely matches the bumping method
        of the curve.
        """

        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=0,
            settle=0,
            float_spread=float_spd,
            spread_compound_method=method,
        )
        curve = Curve({dt(2007, 1, 1): 1.0, dt(2017, 1, 1): 0.9}, convention="Act365f")
        disc_curve = curve.shift(curve_spd)
        result = bond.rate(curves=[curve, disc_curve], metric="spread")
        assert abs(result - expected) < 1e-4

        bond.float_spread = result
        validate = bond.npv(curves=[curve, disc_curve])
        assert abs(validate + bond.leg1.settlement_params.notional) < 0.30 * abs(curve_spd)

    @pytest.mark.parametrize(
        ("curve_spd", "method", "float_spd", "expected"),
        [
            (10, "isda_compounding", 0, 10.00000120),
        ],
    )
    def test_float_rate_bond_rate_spread_fx(self, curve_spd, method, float_spd, expected) -> None:
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=0,
            settle=0,
            float_spread=float_spd,
            spread_compound_method=method,
        )
        curve = Curve({dt(2007, 1, 1): 1.0, dt(2017, 1, 1): 0.9}, convention="Act365f")
        disc_curve = curve.shift(curve_spd)
        fxr = FXRates({"usdnok": 10.0}, settlement=dt(2007, 1, 1))
        result = bond.rate(
            curves=[curve, disc_curve],
            metric="spread",
            fx=fxr,
        )
        assert abs(result - expected) < 1e-4

        bond.float_spread = result
        validate = bond.npv(curves=[curve, disc_curve], fx=fxr)
        assert abs(validate + bond.leg1.settlement_params.notional) < 0.30 * abs(curve_spd)

    def test_float_rate_bond_accrued(self) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_1B", Series(2.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 1))))
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=100,
            fixing_method="rfr_observation_shift",
            rate_fixings=name,
            method_param=5,
            spread_compound_method="none_simple",
        )
        result = bond.accrued(dt(2010, 3, 3))
        expected = 0.5019199020076  # 3% * 2 / 12
        fixings.pop(name + "_1B")
        assert abs(result - expected) < 1e-8

    @pytest.mark.parametrize(
        ("metric", "spd", "exp"),
        [
            ("clean_price", 0.0, 100.0),
            ("dirty_price", 0.0, 100.0),
            ("clean_price", 10.0, 99.99982764447981),  # compounding diff between shift
            ("dirty_price", 10.0, 100.0165399732469),
        ],
    )
    def test_float_rate_bond_rate_metric(self, metric, spd, exp) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_1B", Series(0.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 1))))
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=spd,
            fixing_method="rfr_observation_shift",
            rate_fixings=name,
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )
        curve = Curve({dt(2010, 3, 1): 1.0, dt(2017, 1, 1): 1.0}, convention="act365f")
        disc_curve = curve.shift(spd)

        result = bond.rate(curves=[curve, disc_curve], metric=metric)
        fixings.pop(name + "_1B")
        assert abs(result - exp) < 1e-8

    @pytest.mark.parametrize(
        ("metric", "spd", "exp"),
        [
            ("clean_price", 10.0, 99.99982764447981),  # compounding diff between shift
            ("dirty_price", 10.0, 100.0165399732469),
        ],
    )
    def test_initialised_rate_metric(self, metric, spd, exp) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_1B", Series(0.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 1))))
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=spd,
            fixing_method="rfr_observation_shift",
            rate_fixings=name,
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
            metric=metric,
        )
        curve = Curve({dt(2010, 3, 1): 1.0, dt(2017, 1, 1): 1.0}, convention="act365f")
        disc_curve = curve.shift(spd)

        result = bond.rate(curves=[curve, disc_curve])
        fixings.pop(name + "_1B")
        assert abs(result - exp) < 1e-8

    @pytest.mark.parametrize(
        ("settlement", "expected"),
        [
            (dt(2010, 3, 3), 0.501369863013698),
            (dt(2010, 6, 30), -0.008219178082191761),  # ex div with fixed IBOR
        ],
    )
    def test_float_rate_bond_accrued_ibor(self, settlement, expected) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_6M", Series(2.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 1))))
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=100,
            fixing_method="ibor",
            rate_fixings=name,
            method_param=2,
            spread_compound_method="none_simple",
        )
        result = bond.accrued(settlement)
        fixings.pop(name + "_6M")
        assert abs(result - expected) < 1e-8

    def test_float_rate_bond_raise_frequency(self) -> None:
        with pytest.raises(ValueError, match="A `FloatRateNote` cannot have a 'zero' freq"):
            FloatRateNote(
                effective=dt(2007, 1, 1),
                termination=dt(2017, 1, 1),
                frequency="Z",
                convention="Act365f",
                ex_div=3,
                float_spread=100,
                fixing_method="rfr_observation_shift",
                rate_fixings=NoInput(0),
                method_param=5,
                spread_compound_method="none_simple",
            )

    def test_negative_accrued_needs_forecasting(self) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_1B", Series(2.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 8))))
        bond = FloatRateNote(
            effective=dt(2009, 9, 16),
            termination=dt(2017, 3, 16),
            frequency="Q",
            convention="Act365f",
            ex_div=6,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            rate_fixings=name,
            method_param=5,
            spread_compound_method="none_simple",
            calendar=NoInput(0),
        )
        from rateslib.data.fixings import FixingMissingForecasterError

        with pytest.raises(  # noqa: PT012
            FixingMissingForecasterError,
            match="A `rate_curve` is required to forecast missing RFR rates",
        ):
            bond.accrued(dt(2010, 3, 11))
            fixings.pop(name + "_1B")

        # # approximate calculation 5 days of negative accrued at 2% = -0.027397
        # assert abs(result + 2 * 5 / 365) < 1e-3

    @pytest.mark.parametrize(
        "rate_fixings",
        [
            NoInput(0),
        ],
    )
    def test_negative_accrued_raises(self, rate_fixings) -> None:
        bond = FloatRateNote(
            effective=dt(2009, 9, 16),
            termination=dt(2017, 3, 16),
            frequency="Q",
            convention="Act365f",
            ex_div=5,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            rate_fixings=rate_fixings,
            method_param=5,
            spread_compound_method="none_simple",
            calendar=NoInput(0),
        )
        from rateslib.data.fixings import FixingMissingForecasterError

        with pytest.raises(
            FixingMissingForecasterError,
            match="A `rate_curve` is required to forecast missing RFR rate",
        ):
            bond.accrued(dt(2010, 3, 11))

    @pytest.mark.skip(reason="v2.5 removed these validations")
    def test_bad_accrued_parameter_combo_raises(self, rate_fixings) -> None:
        with pytest.raises(ValueError, match="For RFR FRNs `ex_div` must be less than"):
            FloatRateNote(
                effective=dt(2009, 9, 16),
                termination=dt(2017, 3, 16),
                frequency="Q",
                ex_div=5,
                fixing_method="rfr_observation_shift",
                method_param=3,
            )

    def test_accrued_no_fixings_in_period(self) -> None:
        bond = FloatRateNote(
            effective=dt(2010, 3, 16),
            termination=dt(2017, 3, 16),
            frequency="Q",
            convention="Act365f",
            ex_div=0,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            rate_fixings=NoInput(0),
            method_param=0,
            spread_compound_method="none_simple",
            calendar=NoInput(0),
        )
        result = bond.accrued(dt(2010, 3, 16))
        assert result == 0.0

    def test_float_rate_bond_analytic_delta(self) -> None:
        frn = FloatRateNote(
            effective=dt(2010, 6, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            float_spread=100,
            notional=-1000000,
            settle=0,
            fixing_method="ibor",
            rate_fixings=2.0,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 1.0})
        result = frn.analytic_delta(curves=curve)
        expected = -550.0
        assert abs(result - expected) < 1e-6

        frn.kwargs.meta["settle"] = 2
        result = frn.analytic_delta(curves=curve)  # bond is ex div on settle 27th Nov 2010
        expected = -500.0  # bond has dropped a 6m coupon payment
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize(
        ("metric", "spd", "exp"),
        [
            ("clean_price", 0.0, 100),
            ("dirty_price", 0.0, 100),
            ("clean_price", 50.0, 99.99601798513253),
            ("dirty_price", 50.0, 100.03848373855718),
        ],
    )
    def test_float_rate_bond_forward_prices(self, metric, spd, exp) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(
            name + "_1B",
            Series(
                data=2.0,
                index=get_calendar("bus").bus_date_range(start=dt(2007, 1, 1), end=dt(2010, 2, 26)),
            ),
        )
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=spd,
            fixing_method="rfr_observation_shift",
            calendar="bus",
            rate_fixings=name,
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )
        curve = Curve(
            {dt(2010, 3, 1): 1.0, dt(2017, 1, 1): 1.0},
            convention="act365f",
            calendar="bus",
        )
        disc_curve = curve.shift(spd)

        result = bond.rate(
            curves=[curve, disc_curve],
            metric=metric,
            settlement=dt(2010, 8, 1),
        )
        fixings.pop(name + "_1B")
        assert abs(result - exp) < 1e-8

    def test_float_rate_bond_forward_accrued(self) -> None:
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )
        curve = Curve({dt(2010, 3, 1): 1.0, dt(2017, 1, 1): 0.9}, convention="act365f")
        # disc_curve = curve.shift(0)
        result = bond.accrued(dt(2010, 8, 1), rate_curve=curve)
        expected = 0.13083715795372267
        assert abs(result - expected) < 1e-8

    def test_rate_raises(self) -> None:
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=0.0,
            fixing_method="rfr_observation_shift",
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )

        with pytest.raises(ValueError, match="`metric` must be in"):
            bond.rate(metric="BAD")

    def test_forecast_ibor(self, curve) -> None:
        f_curve = LineCurve({dt(2022, 1, 1): 3.0, dt(2022, 2, 1): 4.0})
        frn = FloatRateNote(
            effective=dt(2022, 2, 1),
            termination="3m",
            frequency="Q",
            fixing_method="ibor",
            method_param=0,
        )
        result = frn.accrued(dt(2022, 2, 5), rate_curve=f_curve)
        expected = 0.044444444
        assert abs(result - expected) < 1e-4

    @pytest.mark.parametrize(
        ("price", "tol"), [(98.0, 1e-7), (95.0, 1e-5), (90.0, 1e-3), (80.0, 1e-2)]
    )
    def test_oaspread(self, price, tol) -> None:
        bond = FloatRateNote(
            effective=dt(1998, 12, 7),
            termination=dt(2008, 12, 7),
            frequency="q",
            fixing_method="rfr_payment_delay",
            rate_fixings=[4.0],
        )
        curve = Curve({dt(1998, 12, 7): 1.0, dt(2015, 12, 7): 0.75})
        # result = bond.rate(curve, metric="clean_price") = 99.999999999999953
        result = bond.oaspread(curves=curve, price=price)
        curve_z = curve.shift(result)
        result = bond.rate(curves=[curve, curve_z], metric="clean_price")
        assert abs(result - price) < tol

    def test_settle_method_param_combinations(self) -> None:
        # for RFR when method_param is less than settle curve based pricing methods will
        # require forecasting from RFR curve to correctly calculate the accrued.
        name = str(hash(os.urandom(8)))
        fixings.add(
            name + "_1B",
            Series(
                [2.0, 3.0, 4.0, 5.0, 6.0],
                index=[
                    dt(2022, 1, 2),
                    dt(2022, 1, 3),
                    dt(2022, 1, 4),
                    dt(2022, 1, 5),
                    dt(2022, 1, 6),
                ],
            ),
        )
        frn = FloatRateNote(
            effective=dt(2022, 1, 5),
            termination="1Y",
            frequency="Q",
            settle=3,
            method_param=2,
            fixing_method="rfr_observation_shift",
            rate_fixings=name,
            convention="Act365F",
            ex_div=1,
        )
        curve = Curve(
            nodes={dt(2022, 1, 7): 1.0, dt(2023, 1, 7): 0.95},
            convention="act365f",
        )

        # Case1: All fixings are known and are published
        # in this case a Curve is not required and is not given
        result = frn.accrued(settlement=dt(2022, 1, 9))
        assert abs(result - 0.04932400) < 1e-6

        # Case2: Some fixings are unknown and must be forecast by a curve.
        # If a curve is not supplied this will error
        from rateslib.data.fixings import FixingMissingForecasterError

        with pytest.raises(
            FixingMissingForecasterError, match="A `rate_curve` is required to forecast missing RFR"
        ):
            frn.accrued(settlement=dt(2022, 1, 10))

        # Case3: Some fixings are unknown and must be forecast by a curve.
        # A curve is given so this is used to forecast the values.
        result = frn.accrued(settlement=dt(2022, 1, 10), rate_curve=curve)
        assert abs(result - 0.06338487826265116) < 1e-6

        # Case4: The bond settles on Issue date and there is no accrued if curve supplied or not
        result1 = frn.accrued(settlement=dt(2022, 1, 5))
        result2 = frn.accrued(settlement=dt(2022, 1, 5), rate_curve=curve)
        assert abs(result1) < 1e-6
        assert abs(result2) < 1e-6

        # Case5: The bond settles on a coupon date and there is no accrued if curve supplied or not
        result1 = frn.accrued(settlement=dt(2022, 4, 5))
        result2 = frn.accrued(settlement=dt(2022, 4, 5), rate_curve=curve)
        assert abs(result1) < 1e-6
        assert abs(result2) < 1e-6

        # Case6: Bond settles on issue date and there is no accrued. No fixings are input
        frn_no_fixings = FloatRateNote(
            effective=dt(2022, 1, 5),
            termination="1Y",
            frequency="Q",
            settle=3,
            method_param=2,
            fixing_method="rfr_observation_shift",
            convention="Act365F",
            ex_div=1,
        )
        result1 = frn_no_fixings.accrued(settlement=dt(2022, 1, 5))
        result2 = frn_no_fixings.accrued(settlement=dt(2022, 1, 5), rate_curve=curve)
        assert abs(result1) < 1e-6
        assert abs(result2) < 1e-6

        # Case7: Bond settles a few days forward(settle) no previous fixings are given, all
        # can be forecast from curve
        frn_no_fixings = FloatRateNote(
            effective=dt(2022, 1, 7),
            termination="1Y",
            frequency="Q",
            settle=3,
            method_param=0,
            fixing_method="rfr_observation_shift",
            convention="Act365F",
            ex_div=1,
        )
        result = frn_no_fixings.accrued(settlement=dt(2022, 1, 10), rate_curve=curve)
        assert abs(result - 0.04216776020085078) < 1e-6

        # Case8: bond settles a few days forward, no fixings are given and no curve. Must error.
        with pytest.raises(
            FixingMissingForecasterError,
            match="A `rate_curve` is required to forecast missing RFR rates",
        ):
            frn_no_fixings.accrued(settlement=dt(2022, 1, 10))
        fixings.pop(name + "_1B")

    def test_ibor_fixings_table_historical_before_curve(self, curve):
        # see test FloatPeriod.test_ibor_fixings_table_historical_before_curve
        bond = FloatRateNote(
            effective=dt(2001, 11, 7),
            termination=dt(2002, 8, 7),
            frequency="q",
            fixing_method="ibor",
            rate_fixings=[4.0],
            curves=[curve],
        )
        result = bond.local_analytic_rate_fixings()
        assert isinstance(result, DataFrame)

    def test_ibor_fixings_table_with_fixing(self, curve):
        # see test FloatPeriod.test_ibor_fixings_table_historical_before_curve
        bond = FloatRateNote(
            effective=dt(2021, 11, 7),
            termination=dt(2022, 8, 7),
            frequency="q",
            fixing_method="ibor",
            rate_fixings=[4.0],
            curves=[curve],
        )
        result = bond.local_analytic_rate_fixings()
        assert isinstance(result, DataFrame)
        assert result.iloc[0, 0] == 0.0
        assert abs(result.iloc[1, 0] + 24.376897) < 1e-6
        assert abs(result.iloc[2, 0] + 24.941351) < 1e-6

    def test_ibor_ytm_rate(self, curve):
        # test a FixedRateBond and FloatRateNote with same conventions and cashflows have same ytm
        ibor_curve = LineCurve({dt(2021, 12, 1): 4.0, dt(2027, 12, 2): 4.0})
        disc_curve = Curve({dt(2021, 12, 1): 1.0, dt(2027, 12, 2): 0.92})
        frn = FloatRateNote(
            effective=dt(2021, 11, 7),
            termination=dt(2022, 8, 7),
            frequency="q",
            fixing_method="ibor",
            convention="actacticma",
            calendar="nyc",
            modifier="none",
            rate_fixings=[4.0],
            curves=[ibor_curve, disc_curve],
            calc_mode="us_gb",
            fixing_series="eur_ibor",
            settle=1,
        )
        frb = FixedRateBond(
            effective=dt(2021, 11, 7),
            termination=dt(2022, 8, 7),
            spec="us_gb",
            frequency="q",
            fixed_rate=4.0,
            curves=[disc_curve],
        )
        dp1 = frn.rate(metric="dirty_price")
        dp2 = frb.rate(metric="dirty_price")
        assert abs(dp1 - dp2) < 1e-12  # FRN and equivalent FRB have the same dirty price.

        y2 = frb.rate(metric="ytm")
        y1 = frn.rate(metric="ytm")
        assert abs(y1 - y2) < 1e-12  # FRN and equivalent FRB have the same yield-to-maturity.

    def test_ytm_rate_fixings_provided(self, curve):
        # test a FixedRateBond and FloatRateNote with same conventions and cashflows have same ytm
        disc_curve = Curve({dt(2021, 12, 1): 1.0, dt(2027, 12, 2): 0.92})
        frn = FloatRateNote(
            effective=dt(2021, 11, 7),
            termination=dt(2022, 8, 7),
            frequency="q",
            fixing_method="ibor",
            convention="actacticma",
            calendar="nyc",
            modifier="none",
            rate_fixings=[4.0, 4.0, 4.0],
            calc_mode="us_gb",
            settle=1,
        )
        frb = FixedRateBond(
            effective=dt(2021, 11, 7),
            termination=dt(2022, 8, 7),
            spec="us_gb",
            frequency="q",
            fixed_rate=4.0,
            curves=[disc_curve],
        )
        dp1 = frn.rate(metric="dirty_price", curves=[None, disc_curve])
        dp2 = frb.rate(metric="dirty_price")
        assert abs(dp1 - dp2) < 1e-12  # FRN and equivalent FRB have the same dirty price.

        y2 = frb.ytm(price=dp2, dirty=True, settlement=dt(2022, 12, 1))
        y1 = frn.ytm(price=dp1, dirty=True, settlement=dt(2022, 12, 1))
        assert abs(y1 - y2) < 1e-12  # FRN and equivalent FRB have the same yield-to-maturity.

    def test_cashflows_known_fixings(self):
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_1B", Series(2.0, index=date_range(dt(1999, 12, 1), dt(2004, 6, 2))))

        frn = FloatRateNote(
            effective=dt(2000, 12, 7),
            termination=dt(2001, 12, 7),
            frequency="S",
            currency="gbp",
            convention="Act365F",
            ex_div=3,
            rate_fixings=name,
            fixing_method="rfr_observation_shift_avg",
            method_param=5,
        )
        result = frn.cashflows()
        fixings.pop(name + "_1B")
        assert isinstance(result, DataFrame)
        assert abs(result["Cashflow"].iloc[0] + 10000) < 50.0
        assert abs(result["Cashflow"].iloc[1] + 10000) < 50.0


class TestBondFuture:
    def test_repr(self):
        kwargs = dict(
            effective=dt(2020, 1, 1),
            stub="ShortFront",
            frequency="A",
            calendar="tgt",
            currency="eur",
            convention="ActActICMA",
        )
        bond1 = FixedRateBond(termination=dt(2022, 3, 1), fixed_rate=1.5, **kwargs)
        fut = BondFuture(delivery=dt(2021, 3, 1), coupon=6.0, basket=[bond1])
        expected = f"<rl.BondFuture at {hex(id(fut))}>"
        assert expected == fut.__repr__()

    @pytest.mark.parametrize(
        ("delivery", "mat", "coupon", "exp"),
        [
            (dt(2023, 6, 12), dt(2032, 2, 15), 0.0, 0.603058),
            (dt(2023, 6, 12), dt(2032, 8, 15), 1.7, 0.703125),
            (dt(2023, 6, 12), dt(2033, 2, 15), 2.3, 0.733943),
            (dt(2023, 9, 11), dt(2032, 8, 15), 1.7, 0.709321),
            (dt(2023, 9, 11), dt(2033, 2, 15), 2.3, 0.739087),
            (dt(2023, 12, 11), dt(2032, 8, 15), 1.7, 0.715464),
            (dt(2023, 12, 11), dt(2033, 2, 15), 2.3, 0.744390),
        ],
    )
    def test_conversion_factors_eurex_bund_ytm(self, delivery, mat, coupon, exp) -> None:
        # The expected results are downloaded from the EUREX website
        # regarding precalculated conversion factors.
        # this test allows for an error in the cf < 1e-4, due to YTM method
        kwargs = dict(
            effective=dt(2020, 1, 1),
            stub="ShortFront",
            frequency="A",
            calendar="tgt",
            currency="eur",
            convention="ActActICMA",
        )
        bond1 = FixedRateBond(termination=mat, fixed_rate=coupon, **kwargs)

        fut = BondFuture(delivery=delivery, coupon=6.0, basket=[bond1])
        result = fut.cfs
        assert abs(result[0] - exp) < 1e-4

    @pytest.mark.parametrize(
        ("delivery", "issue", "mat", "coupon", "exp"),
        [
            (dt(2023, 6, 12), dt(2022, 7, 1), dt(2032, 2, 15), 0.0, 0.603058),
            (dt(2023, 6, 12), dt(2022, 7, 8), dt(2032, 8, 15), 1.7, 0.703125),
            (dt(2023, 6, 12), dt(2023, 1, 13), dt(2033, 2, 15), 2.3, 0.733943),
            (dt(2023, 9, 11), dt(2022, 7, 8), dt(2032, 8, 15), 1.7, 0.709321),
            (dt(2023, 9, 11), dt(2023, 1, 13), dt(2033, 2, 15), 2.3, 0.739087),
            (dt(2023, 12, 11), dt(2022, 7, 8), dt(2032, 8, 15), 1.7, 0.715464),
            (dt(2023, 12, 11), dt(2023, 1, 13), dt(2033, 2, 15), 2.3, 0.744390),
        ],
    )
    def test_conversion_factors_eurex_bund_method(self, delivery, issue, mat, coupon, exp) -> None:
        # The expected results are downloaded from the EUREX website
        # regarding precalculated conversion factors.
        # these should be exact due to specifically coded methods
        kwargs = dict(
            effective=issue,
            stub="LongFront",
            frequency="A",
            calendar="tgt",
            currency="eur",
            convention="ActActICMA",
            modifier="none",
        )
        bond1 = FixedRateBond(termination=mat, fixed_rate=coupon, **kwargs)

        fut = BondFuture(delivery=delivery, coupon=6.0, basket=[bond1], calc_mode="eurex_eur")
        result = fut.cfs
        assert result[0] == exp

    @pytest.mark.parametrize(
        ("effective", "maturity", "delivery", "coupon", "exp"),
        [
            # (dt(2019, 6, 26), dt(2034, 6, 26), dt(2025, 6, 10), 0.0, 0.591898),
            # (dt(2006, 3, 8), dt(2036, 3, 8), dt(2025, 6, 10), 2.5, 0.729825),
            # (dt(2021, 6, 23), dt(2035, 6, 23), dt(2025, 6, 10), 0.25, 0.576795),
            (dt(2012, 6, 27), dt(2037, 6, 27), dt(2025, 6, 10), 1.25, 0.601767),
        ],
    )
    def test_conversion_factors_eurex_chf_method_jun25(
        self, effective, maturity, delivery, coupon, exp
    ) -> None:
        # The expected results are downloaded from the EUREX website
        # regarding precalculated conversion factors.
        # these should be exact due to specifically coded methods
        bond1 = FixedRateBond(effective, maturity, fixed_rate=coupon, spec="ch_gb")
        fut = BondFuture(basket=[bond1], delivery=delivery, spec="ch_gb_10y")
        result = fut.cfs
        assert result[0] == exp

    @pytest.mark.parametrize(
        ("effective", "maturity", "delivery", "coupon", "exp"),
        [
            (dt(2019, 1, 1), dt(2033, 4, 8), dt(2025, 3, 10), 3.5, 0.844755),
            (dt(2019, 1, 1), dt(2034, 6, 26), dt(2025, 3, 10), 0.0, 0.583339),
            (dt(2006, 1, 1), dt(2036, 3, 8), dt(2025, 3, 10), 2.5, 0.725400),
            (dt(2021, 1, 1), dt(2035, 6, 23), dt(2025, 3, 10), 0.25, 0.569042),
            (dt(2012, 1, 1), dt(2037, 6, 27), dt(2025, 3, 10), 1.25, 0.596009),
        ],
    )
    def test_conversion_factors_eurex_chf_method_mar25(
        self, effective, maturity, delivery, coupon, exp
    ) -> None:
        # The expected results are downloaded from the EUREX website
        # regarding precalculated conversion factors.
        # these should be exact due to specifically coded methods
        bond1 = FixedRateBond(effective, maturity, fixed_rate=coupon, spec="ch_gb")
        fut = BondFuture(basket=[bond1], delivery=delivery, spec="ch_gb_10y")
        result = fut.cfs
        assert result[0] == exp

    @pytest.mark.parametrize(
        ("mat", "coupon", "exp"),
        [
            (dt(2032, 6, 7), 4.25, 1.0187757),
            (dt(2033, 7, 31), 0.875, 0.7410593),
            (dt(2034, 9, 7), 4.5, 1.0449380),
            (dt(2035, 7, 31), 0.625, 0.6773884),
            (dt(2036, 3, 7), 4.25, 1.0247516),
        ],
    )
    def test_conversion_factors_ice_gilt(self, mat, coupon, exp) -> None:
        # The expected results are downloaded from the ICE LIFFE website
        # regarding precalculated conversion factors.
        # this test allows for an error in the cf < 1e-6.
        kwargs = dict(
            effective=dt(2020, 1, 1),
            stub="ShortFront",
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
        )
        bond1 = FixedRateBond(termination=mat, fixed_rate=coupon, **kwargs)

        fut = BondFuture(delivery=(dt(2023, 6, 1), dt(2023, 6, 30)), coupon=4.0, basket=[bond1])
        result = fut.cfs
        assert abs(result[0] - exp) < 1e-6

    def test_conversion_factors_ice_gilt_default_spec(self) -> None:
        # this uses the v2.5 implementation that rounds exactly to the exchange quantity
        # this tests data directly from ice rounded to 7 dp.
        # note this requires 'linear_days_long_front_split' on GB00BTXS1K06 which is the last bond.
        bf = BondFuture(
            basket=[
                FixedRateBond(dt(1999, 1, 1), dt(2035, 7, 31), fixed_rate=0.625, spec="uk_gb"),
                FixedRateBond(dt(1999, 1, 1), dt(2038, 1, 29), fixed_rate=3.75, spec="uk_gb"),
                FixedRateBond(dt(1999, 1, 1), dt(2034, 9, 7), fixed_rate=4.5, spec="uk_gb"),
                FixedRateBond(dt(1999, 1, 1), dt(2035, 3, 7), fixed_rate=4.5, spec="uk_gb"),
                FixedRateBond(dt(1999, 1, 1), dt(2036, 3, 7), fixed_rate=4.25, spec="uk_gb"),
                FixedRateBond(dt(1999, 1, 1), dt(2037, 9, 7), fixed_rate=1.75, spec="uk_gb"),
                FixedRateBond(dt(2025, 9, 3), dt(2035, 10, 22), fixed_rate=4.75, spec="uk_gb"),
            ],
            delivery=(dt(2025, 12, 1), dt(2025, 12, 31)),
            spec="uk_gb_10y",
        )
        expected = (
            0.7316293,
            0.9760712,
            1.0366069,
            1.0383390,
            1.0208264,
            0.7904642,
            1.0606298,
        )
        assert bf.cfs == expected

    def test_conversion_factors_ice_gilt_default_spec2(self) -> None:
        # this test has the first calendar day of the month as a holiday
        # this uses the v2.5 implementation that rounds exactly to the exchange quantity
        # this tests data directly from ice rounded to 7 dp.
        bf = BondFuture(
            basket=[
                FixedRateBond(dt(1999, 1, 1), dt(2034, 9, 7), fixed_rate=4.5, spec="uk_gb"),
                FixedRateBond(dt(1999, 1, 1), dt(2038, 1, 29), fixed_rate=3.75, spec="uk_gb"),
                FixedRateBond(dt(1999, 1, 1), dt(2034, 7, 31), fixed_rate=4.25, spec="uk_gb"),
                FixedRateBond(dt(2025, 2, 12), dt(2035, 3, 7), fixed_rate=4.5, spec="uk_gb"),
                FixedRateBond(dt(1999, 1, 1), dt(2037, 9, 7), fixed_rate=1.75, spec="uk_gb"),
                FixedRateBond(dt(1999, 1, 1), dt(2036, 3, 7), fixed_rate=4.25, spec="uk_gb"),
                FixedRateBond(dt(2020, 9, 11), dt(2035, 7, 31), fixed_rate=0.625, spec="uk_gb"),
            ],
            delivery=(dt(2025, 6, 2), dt(2025, 6, 30)),
            spec="uk_gb_10y",
        )
        expected = (1.0383429, 0.9753142, 1.0189797, 1.0400109, 0.7835277, 1.0216443, 0.7203475)
        assert bf.cfs == expected

    @pytest.mark.parametrize(
        ("mat", "coupon", "calc_mode", "exp"),
        [
            (dt(2010, 10, 31), 1.5, "ust_short", 0.9229),
            (dt(2013, 10, 31), 2.75, "ust_short", 0.8653),
            (dt(2018, 11, 15), 3.75, "ust_long", 0.8357),
            (dt(2038, 5, 15), 4.5, "ust_long", 0.7943),
        ],
    )
    def test_conversion_factors_cme_treasury(self, mat, coupon, calc_mode, exp) -> None:
        # The expected results are downloaded from the CME website
        # regarding precalculated conversion factors.
        # this test allows for an error in the cf < 1e-6.
        kwargs = dict(
            effective=dt(2005, 1, 1),
            spec="us_gb",
        )
        bond1 = FixedRateBond(termination=mat, fixed_rate=coupon, **kwargs)

        fut = BondFuture(
            delivery=(dt(2008, 12, 1), dt(2008, 12, 29)),
            coupon=6.0,
            basket=[bond1],
            calc_mode=calc_mode,
        )
        result = fut.cfs
        assert abs(result[0] - exp) < 1e-6

    def test_dlv_screen_print(self) -> None:
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=NoInput(0))
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds)
        result = future.dlv(
            future_price=112.98,
            prices=[102.732, 131.461, 107.877, 134.455],
            repo_rate=6.24,
            settlement=dt(2000, 3, 16),
            convention="Act365f",
        )
        expected = DataFrame(
            {
                "Bond": [
                    "5.750% 07-12-2009",
                    "9.000% 12-07-2011",
                    "6.250% 25-11-2010",
                    "9.000% 06-08-2012",
                ],
                "Price": [102.732, 131.461, 107.877, 134.455],
                "YTM": [5.384243, 5.273217, 5.275481, 5.193851],
                "C.Factor": [0.914225, 1.152571, 0.944931, 1.161956],
                "Gross Basis": [-0.557192, 1.243582, 1.118677, 3.177230],
                "Implied Repo": [7.381345, 3.564685, 2.199755, -1.414670],
                "Actual Repo": [6.24, 6.24, 6.24, 6.24],
                "Net Basis": [-0.343654, 1.033668, 1.275866, 3.010371],
            },
        )
        assert_frame_equal(result, expected)

        result2 = future.dlv(
            future_price=112.98,
            prices=[102.732, 131.461, 107.877, 134.455],
            repo_rate=[6.24, 6.24, 6.24, 6.24],  # test individual repo input
            settlement=dt(2000, 3, 16),
            convention="Act365f",
        )
        assert_frame_equal(result2, expected)

    def test_notional(self) -> None:
        future = BondFuture(
            coupon=0,
            delivery=dt(2000, 6, 1),
            basket=[],
            nominal=100000,
            contracts=10,
        )
        assert future.notional == -1e6

    def test_dirty_in_methods(self) -> None:
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=NoInput(0))
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds)
        prices = [102.732, 131.461, 107.877, 134.455]
        basket = future.kwargs.meta["basket"]
        dirty_prices = [
            price + basket[i].accrued(dt(2000, 3, 16)) for i, price in enumerate(prices)
        ]
        result = future.gross_basis(112.98, dirty_prices, dt(2000, 3, 16), True)
        expected = future.gross_basis(112.98, prices, dt(2000, 3, 16), False)
        assert result == expected

    def test_delivery_in_methods(self) -> None:
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=NoInput(0))
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds)
        prices = [102.732, 131.461, 107.877, 134.455]
        expected = future.net_basis(112.98, prices, 6.24, dt(2000, 3, 16))
        result = future.net_basis(112.98, prices, 6.24, dt(2000, 3, 16), delivery=dt(2000, 6, 30))
        assert result == expected

        expected = future.implied_repo(112.98, prices, dt(2000, 3, 16))
        result = future.implied_repo(112.98, prices, dt(2000, 3, 16), delivery=dt(2000, 6, 30))
        assert result == expected

        expected = future.ytm(112.98)
        result = future.ytm(112.98, delivery=dt(2000, 6, 30))
        assert result == expected

        expected = future.duration(112.98)
        result = future.duration(112.98, delivery=dt(2000, 6, 30))
        assert result == expected

    def test_ctd_index(self) -> None:
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=NoInput(0))
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds)
        prices = [102.732, 131.461, 107.877, 134.455]
        assert future.ctd_index(112.98, prices, dt(2000, 3, 16)) == 0

    @pytest.mark.parametrize(("metric", "expected"), [("future_price", 112.98), ("ytm", 5.301975)])
    @pytest.mark.parametrize("delivery", [NoInput(0), dt(2000, 6, 30)])
    def test_futures_rates(self, metric, expected, delivery) -> None:
        curve = Curve(
            nodes={
                dt(2000, 3, 15): 1.0,
                dt(2000, 6, 30): 1.0,
                dt(2009, 12, 7): 1.0,
                dt(2010, 11, 25): 1.0,
                dt(2011, 7, 12): 1.0,
                dt(2012, 8, 6): 1.0,
            },
            id="gilt_curve",
            convention="act365f",
        )
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=NoInput(0),
            settle=1,
            curves="gilt_curve",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        solver = Solver(
            curves=[curve],
            instruments=[
                IRS(
                    dt(2000, 3, 15),
                    dt(2000, 6, 30),
                    "A",
                    convention="act365f",
                    curves="gilt_curve",
                ),
            ]
            + bonds,
            s=[7.381345, 102.732, 131.461, 107.877, 134.455],
        )  # note the repo rate as defined by 'gilt_curve' is set to analogue implied
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
        )
        result = future.rate(
            solver=solver,
            metric=metric,
            settlement=delivery,
        )
        assert abs(result - expected) < 1e-3

    def test_future_rate_raises(self) -> None:
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=NoInput(0),
            settle=1,
            curves="gilt_curve",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
        )
        with pytest.raises(ValueError, match="`metric`"):
            future.rate(metric="badstr")

    def test_futures_npv(self) -> None:
        curve = Curve(
            nodes={
                dt(2000, 3, 15): 1.0,
                dt(2000, 6, 30): 1.0,
                dt(2009, 12, 7): 1.0,
                dt(2010, 11, 25): 1.0,
                dt(2011, 7, 12): 1.0,
                dt(2012, 8, 6): 1.0,
            },
            id="gilt_curve",
            convention="act365f",
        )
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=NoInput(0),
            settle=1,
            curves="gilt_curve",
            currency="gbp",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        solver = Solver(
            curves=[curve],
            instruments=[
                IRS(
                    dt(2000, 3, 15),
                    dt(2000, 6, 30),
                    "A",
                    convention="act365f",
                    curves="gilt_curve",
                ),
            ]
            + bonds,
            s=[7.381345, 102.732, 131.461, 107.877, 134.455],
            algorithm="gauss_newton",
        )  # note the repo rate as defined by 'gilt_curve' is set to analogue implied
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
            nominal=100000,
            contracts=10,
            currency="gbp",
        )
        result = future.npv(solver=solver, local=False)
        expected = 1129798.770872
        assert abs(result - expected) < 1e-5

        result2 = future.npv(solver=solver, local=True)
        assert abs(result2["gbp"] - expected) < 1e-5

    @pytest.mark.parametrize("delivery", [NoInput(0), dt(2000, 6, 30)])
    def test_futures_duration_and_convexity(self, delivery) -> None:
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=NoInput(0),
            settle=1,
            curves="gilt_curve",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
        )
        result = future.duration(112.98, delivery=delivery)[0]
        expected = 8.20178546111
        assert abs(result - expected) < 1e-3

        expected = (
            future.duration(112.98, delivery=delivery)[0]
            - future.duration(112.98 - result / 100, delivery=delivery)[0]
        )
        result2 = future.convexity(112.98, delivery=delivery)[0]
        assert abs(result2 - expected * 100) < 1e-3

        # Bond future duration which is not risk is not adjusted by CFs
        result = future.duration(112.98, delivery=delivery, metric="modified")[0]
        expected = 7.23419455163
        assert abs(result - expected) < 1e-3

    def test_cms(self):
        data = DataFrame(
            data=[
                [
                    FixedRateBond(
                        dt(2022, 1, 1),
                        dt(2039, 8, 15),
                        fixed_rate=4.5,
                        spec="us_gb",
                        curves="bcurve",
                    ),
                    98.6641,
                ],
                [
                    FixedRateBond(
                        dt(2022, 1, 1),
                        dt(2040, 2, 15),
                        fixed_rate=4.625,
                        spec="us_gb",
                        curves="bcurve",
                    ),
                    99.8203,
                ],
                [
                    FixedRateBond(
                        dt(2022, 1, 1),
                        dt(2041, 2, 15),
                        fixed_rate=4.75,
                        spec="us_gb",
                        curves="bcurve",
                    ),
                    100.7734,
                ],
                [
                    FixedRateBond(
                        dt(2022, 1, 1),
                        dt(2040, 5, 15),
                        fixed_rate=4.375,
                        spec="us_gb",
                        curves="bcurve",
                    ),
                    96.6953,
                ],
                [
                    FixedRateBond(
                        dt(2022, 1, 1),
                        dt(2042, 11, 15),
                        fixed_rate=4.00,
                        spec="us_gb",
                        curves="bcurve",
                    ),
                    90.4766,
                ],
            ],
            columns=["bonds", "prices"],
        )
        usz3 = BondFuture(  # Construct the BondFuture Instrument
            coupon=6.0,
            delivery=(dt(2023, 12, 1), dt(2023, 12, 29)),
            basket=data["bonds"],
            nominal=100e3,
            calendar="nyc",
            currency="usd",
            calc_mode="ust_long",
        )
        result = usz3.cms(prices=data["prices"], settlement=dt(2023, 11, 22), shifts=[-50, 0, 50])
        expected = DataFrame(
            data={
                "Bond": [
                    "4.500% 15-08-2039",
                    "4.625% 15-02-2040",
                    "4.750% 15-02-2041",
                    "4.375% 15-05-2040",
                    "4.000% 15-11-2042",
                ],
                -50: [
                    0.0,
                    0.10938764224876252,
                    0.32693578691382186,
                    0.24721845093496597,
                    1.1960030963801813,
                ],
                0: [
                    0.0,
                    0.01148721023514554,
                    0.016282194434154462,
                    0.032902987886402,
                    0.33598669301149187,
                ],
                50: [
                    0.43066112621522734,
                    0.3653207547713322,
                    0.19632745772335625,
                    0.27120849999053576,
                    0.0,
                ],
            }
        )
        assert_frame_equal(result, expected)

    def test_curves_on_individual_bonds(self):
        # no curves are supplied to BondFuture meta or method an local instrument meta are used
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2042, 1, 1): 1.0})
        c2 = Curve({dt(2022, 1, 2): 1.0, dt(2042, 1, 1): 0.5})
        usz3 = BondFuture(  # Construct the BondFuture Instrument
            coupon=6.0,
            delivery=(dt(2023, 12, 1), dt(2023, 12, 29)),
            basket=[
                FixedRateBond(dt(2022, 1, 1), "10y", fixed_rate=4.5, spec="us_gb", curves=c1),
                FixedRateBond(dt(2022, 1, 1), "10Y", fixed_rate=4.5, spec="us_gb", curves=c2),
            ],
            nominal=100e3,
            calendar="nyc",
            currency="usd",
            calc_mode="ust_long",
        )
        cfs = usz3.cfs
        assert cfs[0] == cfs[1]
        result = usz3.rate()
        assert abs(result - 118.06972328) < 1e-7

    def test_curves_supplied_to_rate_method(self):
        # the curves supplied to the method overrides the local instruments meta
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2042, 1, 1): 1.0})
        c2 = Curve({dt(2022, 1, 2): 1.0, dt(2042, 1, 1): 0.5})
        usz3 = BondFuture(  # Construct the BondFuture Instrument
            coupon=6.0,
            delivery=(dt(2023, 12, 1), dt(2023, 12, 29)),
            basket=[
                FixedRateBond(dt(2022, 1, 1), "10y", fixed_rate=4.5, spec="us_gb", curves=c1),
                FixedRateBond(dt(2022, 1, 1), "10Y", fixed_rate=4.5, spec="us_gb", curves=c2),
            ],
            nominal=100e3,
            calendar="nyc",
            currency="usd",
            calc_mode="ust_long",
        )
        cfs = usz3.cfs
        assert cfs[0] == cfs[1]
        # in this c1 is used as an override so both bonds are priced expensively - price is higher
        result = usz3.rate(curves=c1)
        assert abs(result - 150.184019411) < 1e-7

    def test_curves_supplied_as_future_meta(self):
        # the curves supplied to the BondFuture.curves meta override local instruments.
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2042, 1, 1): 1.0})
        c2 = Curve({dt(2022, 1, 2): 1.0, dt(2042, 1, 1): 0.5})
        usz3 = BondFuture(  # Construct the BondFuture Instrument
            coupon=6.0,
            delivery=(dt(2023, 12, 1), dt(2023, 12, 29)),
            basket=[
                FixedRateBond(dt(2022, 1, 1), "10y", fixed_rate=4.5, spec="us_gb", curves=c1),
                FixedRateBond(dt(2022, 1, 1), "10Y", fixed_rate=4.5, spec="us_gb", curves=c2),
            ],
            nominal=100e3,
            calendar="nyc",
            currency="usd",
            calc_mode="ust_long",
            curves=c1,
        )
        cfs = usz3.cfs
        assert cfs[0] == cfs[1]
        # in this c1 is used as an override via the bond future metric so both bonds are priced
        # expensively - price is higher
        result = usz3.rate()
        assert abs(result - 150.184019411) < 1e-7
