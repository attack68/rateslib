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

import numpy as np
import pytest
import rateslib.errors as err
from pandas import DataFrame, Index, MultiIndex, Series, date_range, isna
from pandas.testing import assert_frame_equal, assert_series_equal
from rateslib import default_context, defaults, fixings
from rateslib.curves import Curve
from rateslib.data.fixings import FloatRateSeries, FXIndex
from rateslib.default import NoInput
from rateslib.dual import Dual
from rateslib.enums.generics import _drb
from rateslib.enums.parameters import LegMtm
from rateslib.fx import FXForwards, FXRates
from rateslib.legs import (
    Amortization,
    CreditPremiumLeg,
    CreditProtectionLeg,
    CustomLeg,
    FixedLeg,
    FloatLeg,
    ZeroFixedLeg,
    ZeroFloatLeg,
    ZeroIndexLeg,
)
from rateslib.legs.amortization import _AmortizationType
from rateslib.periods import (
    Cashflow,
    CreditPremiumPeriod,
    CreditProtectionPeriod,
    FixedPeriod,
    FloatPeriod,
)
from rateslib.scheduling import Frequency, Schedule, get_calendar


@pytest.fixture
def curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.99,
        dt(2022, 7, 1): 0.98,
        dt(2022, 10, 1): 0.97,
    }
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture
def hazard_curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.999,
        dt(2022, 7, 1): 0.997,
        dt(2022, 10, 1): 0.991,
    }
    return Curve(nodes=nodes, interpolation="log_linear", id="hazard_fixture")


@pytest.mark.parametrize(
    "Leg",
    [
        FloatLeg,
        FixedLeg,
        ZeroFixedLeg,
        ZeroFloatLeg,
    ],
)
def test_repr(Leg):
    leg = Leg(schedule=Schedule(dt(2022, 1, 1), "1y", "Q"))
    result = leg.__repr__()
    expected = f"<rl.{type(leg).__name__} at {hex(id(leg))}>"
    assert result == expected


@pytest.mark.parametrize("Leg", [FixedLeg, FloatLeg])
def test_repr_mtm(Leg):
    leg = Leg(
        schedule=Schedule(dt(2022, 1, 1), "1y", "Q"),
        currency="usd",
        pair="eurusd",
        mtm="xcs",
        initial_exchange=True,
    )
    result = leg.__repr__()
    expected = f"<rl.{type(leg).__name__} at {hex(id(leg))}>"
    assert result == expected


def test_repr_custom():
    period = FixedPeriod(
        start=dt(2000, 1, 1),
        end=dt(2000, 2, 1),
        payment=dt(2000, 2, 1),
        frequency=Frequency.Months(1, None),
    )
    leg = CustomLeg([period])
    assert leg.__repr__() == f"<rl.CustomLeg at {hex(id(leg))}>"


class TestFloatLeg:
    @pytest.mark.parametrize(
        "obj",
        [
            (
                FloatLeg(
                    schedule=Schedule(
                        effective=dt(2022, 1, 1),
                        termination=dt(2022, 6, 1),
                        payment_lag=0,
                        frequency="Q",
                    ),
                    notional=1e9,
                    convention="Act360",
                    fixing_method="rfr_payment_delay",
                    spread_compound_method="none_simple",
                    currency="nok",
                )
            ),
            (
                FloatLeg(
                    schedule=Schedule(
                        effective=dt(2022, 1, 1),
                        termination=dt(2022, 6, 1),
                        payment_lag=0,
                        payment_lag_exchange=0,
                        frequency="Q",
                    ),
                    notional=1e9,
                    convention="Act360",
                    fixing_method="rfr_payment_delay",
                    spread_compound_method="none_simple",
                    currency="nok",
                    initial_exchange=True,
                    final_exchange=True,
                )
            ),
        ],
    )
    def test_float_leg_analytic_delta_with_npv(self, curve, obj) -> None:
        result = 5 * obj.analytic_delta(rate_curve=curve, disc_curve=curve)
        before_npv = -obj.npv(rate_curve=curve, disc_curve=curve)
        obj.float_spread = 5
        after_npv = -obj.npv(rate_curve=curve, disc_curve=curve)
        expected = after_npv - before_npv
        assert abs(result - expected) < 1e-7

    def test_float_leg_analytic_delta_with_npv_mtm_exchange(self, curve) -> None:
        obj = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=0,
                payment_lag_exchange=0,
                frequency="Q",
            ),
            convention="Act360",
            fixing_method="rfr_payment_delay",
            spread_compound_method="none_simple",
            currency="nok",
            pair=FXIndex("usdnok", "osl|fed", 2, "osl", -2),
            notional=1e8,
            mtm="xcs",
            initial_exchange=True,
        )
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0})
        fxf = FXForwards(
            fx_curves={"usdusd": curve, "usdnok": curve, "noknok": curve},
            fx_rates=FXRates({"usdnok": 1.0}, settlement=dt(2022, 1, 1)),
        )
        result = 5 * obj.analytic_delta(rate_curve=curve, disc_curve=curve, fx=fxf)
        before_npv = -obj.npv(rate_curve=curve, disc_curve=curve, fx=fxf)
        obj.float_spread = 5
        after_npv = -obj.npv(rate_curve=curve, disc_curve=curve, fx=fxf)
        expected = after_npv - before_npv
        assert abs(result - expected) < 1e-7

    def test_float_leg_analytic_delta(self, curve) -> None:
        float_leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=1e9,
            convention="Act360",
        )
        result = float_leg.analytic_delta(rate_curve=curve)
        assert abs(result - 41400.42965267) < 1e-7

    def test_float_leg_cashflows(self, curve) -> None:
        float_leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            float_spread=NoInput(0),
            notional=1e9,
            convention="Act360",
        )
        result = float_leg.cashflows(rate_curve=curve)
        # test a couple of return elements
        assert abs(result.loc[0, defaults.headers["cashflow"]] + 6610305.76834) < 1e-4
        assert abs(result.loc[1, defaults.headers["df"]] - 0.98307) < 1e-4
        assert abs(result.loc[1, defaults.headers["notional"]] - 1e9) < 1e-7

    def test_float_leg_npv(self, curve) -> None:
        float_leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            float_spread=NoInput(0),
            notional=1e9,
            convention="Act360",
        )
        result = float_leg.npv(rate_curve=curve)
        assert abs(result + 16710777.50089434) < 1e-7

    def test_float_leg_fixings(self, curve) -> None:
        float_leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 2, 1),
                termination="9M",
                frequency="Q",
                payment_lag=0,
            ),
            rate_fixings=[10.0, 20.0],
        )
        assert float_leg.periods[0].rate_params.rate_fixing.value == 10
        assert float_leg.periods[1].rate_params.rate_fixing.value == 20
        assert float_leg.periods[2].rate_params.rate_fixing.value is NoInput(0)

    def test_float_leg_fixings2(self, curve) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_3M", Series(index=[dt(2022, 2, 1), dt(2022, 5, 1)], data=[10.0, 20.0]))
        float_leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 2, 1),
                termination="9M",
                frequency="Q",
                payment_lag=0,
            ),
            rate_fixings=name,
            fixing_method="IBOR",
            method_param=0,
        )
        assert float_leg.periods[0].rate_params.rate_fixing.value == 10
        assert float_leg.periods[1].rate_params.rate_fixing.value == 20
        assert float_leg.periods[2].rate_params.rate_fixing.value is NoInput(0)

    def test_float_leg_fixings_series(self, curve) -> None:
        fixings = Series(0.5, index=date_range(dt(2021, 11, 1), dt(2022, 2, 15)))
        float_leg = FloatLeg(
            schedule=Schedule(dt(2021, 12, 1), "9M", "M", payment_lag=0), rate_fixings=fixings
        )
        assert float_leg.periods[0].rate_params.rate_fixing.value != NoInput(0)  # december fixings
        assert float_leg.periods[1].rate_params.rate_fixing.value != NoInput(0)  # january fixings
        assert float_leg.periods[2].rate_params.rate_fixing.value == NoInput(0)  # february fixings
        assert float_leg.periods[4].rate_params.rate_fixing.value == NoInput(0)  # no march fixings

    def test_float_leg_fixings_scalar(self, curve) -> None:
        float_leg = FloatLeg(
            schedule=Schedule(dt(2022, 2, 1), "9M", "Q", payment_lag=0), rate_fixings=5.0
        )
        assert float_leg.periods[0].rate_params.rate_fixing.value == 5.0
        assert float_leg.periods[1].rate_params.rate_fixing.value is NoInput(0)
        assert float_leg.periods[2].rate_params.rate_fixing.value is NoInput(0)

    @pytest.mark.parametrize(
        ("method", "param"),
        [
            ("rfr_payment_delay", NoInput(0)),
            ("rfr_lockout", 1),
            ("rfr_observation_shift", 0),
        ],
    )
    def test_float_leg_rfr_fixings_table(self, method, param, curve) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(
            f"{name}_1B",
            Series(
                [1.19, 1.19, -8.81],
                index=[dt(2022, 12, 28), dt(2022, 12, 29), dt(2022, 12, 30)],
            ),
        )

        curve._set_ad_order(order=1)
        float_leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 12, 28),
                termination="2M",
                frequency="M",
                payment_lag=0,
            ),
            rate_fixings=name,
            currency="SEK",
            fixing_method=method,
            method_param=param,
        )
        result = float_leg.local_analytic_rate_fixings(rate_curve=curve)
        result = result[dt(2022, 12, 28) : dt(2023, 1, 1)]
        assert isinstance(result.iloc[0, 0], Dual)
        data = [_.real for _ in result.iloc[0:5, 0]]
        expected = [0, 0, 0, -0.266647, -0.266647]
        for x, y in zip(data, expected):
            assert abs(x - y) < 1e-6
        fixings.pop(f"{name}_1B")

    @pytest.mark.skip(reason="Unclear what this does: maybe tests an IRS fixing table?")
    def test_rfr_with_fixings_fixings_table_issue(self) -> None:
        from rateslib import IRS

        instruments = [
            IRS(dt(2024, 1, 15), dt(2024, 3, 20), spec="eur_irs", curves="estr"),
            IRS(dt(2024, 3, 20), dt(2024, 6, 19), spec="eur_irs", curves="estr"),
            IRS(dt(2024, 6, 19), dt(2024, 9, 18), spec="eur_irs", curves="estr"),
        ]
        curve = Curve(
            nodes={
                dt(2024, 1, 11): 1.0,
                dt(2024, 3, 20): 1.0,
                dt(2024, 6, 19): 1.0,
                dt(2024, 9, 18): 1.0,
            },
            calendar="tgt",
            convention="act360",
            id="estr",
        )
        from rateslib import Solver

        Solver(
            curves=[curve],
            instruments=instruments,
            s=[
                3.89800324,
                3.63414284,
                3.16864932,
            ],
            id="eur",
        )
        fixings = Series(
            data=[
                3.904,
                3.904,
                3.904,
                3.905,
                3.902,
                3.904,
                3.906,
                3.882,
                3.9,
                3.9,
                3.899,
                3.899,
                3.901,
                3.901,
            ],
            index=[
                dt(2024, 1, 10),
                dt(2024, 1, 9),
                dt(2024, 1, 8),
                dt(2024, 1, 5),
                dt(2024, 1, 4),
                dt(2024, 1, 3),
                dt(2024, 1, 2),
                dt(2023, 12, 29),
                dt(2023, 12, 28),
                dt(2023, 12, 27),
                dt(2023, 12, 22),
                dt(2023, 12, 21),
                dt(2023, 12, 20),
                dt(2023, 12, 19),
            ],
        )

        swap = IRS(
            dt(2023, 12, 20),
            dt(2024, 1, 31),
            spec="eur_irs",
            curves="estr",
            leg2_fixings=fixings,
            notional=3e9,
            fixed_rate=3.922,
        )
        result = swap.leg2.local_rate_fixings(rate_curve=curve)
        assert result.loc[dt(2024, 1, 10), (curve.id, "notional")] == 0.0
        assert abs(result.loc[dt(2024, 1, 11), (curve.id, "notional")] - 3006829846) < 1.0
        assert abs(result.loc[dt(2023, 12, 20), (curve.id, "rates")] - 3.901) < 0.001

    def test_float_leg_set_float_spread(self, curve) -> None:
        float_leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
        )
        assert float_leg.float_spread == 0.0
        assert float_leg.periods[0].rate_params.float_spread == 0

        float_leg.float_spread = 2.0
        assert float_leg.float_spread == 2.0
        assert float_leg.periods[0].rate_params.float_spread == 2.0

    @pytest.mark.parametrize(
        ("method", "spread_method", "expected"),
        [
            ("ibor", NoInput(0), True),
            ("rfr_payment_delay", "none_simple", True),
            ("rfr_payment_delay", "isda_compounding", False),
            ("rfr_payment_delay", "isda_flat_compounding", False),
        ],
    )
    def test_is_linear(self, method, spread_method, expected) -> None:
        float_leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            fixing_method=method,
            spread_compound_method=spread_method,
        )
        assert float_leg._is_linear is expected

    @pytest.mark.parametrize(
        ("method", "settlement", "forward", "expected"),
        [
            ("ISDA_compounding", NoInput(0), NoInput(0), 357.7019143401966),
            ("ISDA_compounding", dt(2022, 4, 6), dt(2022, 4, 6), 580.3895480501503),
            ("ISDA_flat_compounding", NoInput(0), NoInput(0), 360.65913016465225),
            ("ISDA_flat_compounding", dt(2022, 4, 6), dt(2022, 4, 6), 587.64160672647),
            ("NONE_Simple", NoInput(0), NoInput(0), 362.2342162),
            ("NONE_Simple", NoInput(0), dt(2022, 2, 1), 360.98240826375957),
            ("NONE_Simple", dt(2022, 4, 6), dt(2022, 4, 6), 590.6350781908598),
        ],
    )
    def test_float_leg_spread_calculation(
        self, method, settlement, forward, expected, curve
    ) -> None:
        leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=0,
                frequency="Q",
            ),
            notional=1e9,
            convention="Act360",
            fixing_method="rfr_payment_delay",
            spread_compound_method=method,
            currency="nok",
            float_spread=0,
        )
        base_npv = leg.npv(
            rate_curve=curve, disc_curve=curve, forward=forward, settlement=settlement
        )
        result = leg.spread(
            target_npv=-15000000 + base_npv,
            rate_curve=curve,
            disc_curve=curve,
            settlement=settlement,
            forward=forward,
        )
        assert abs(result - expected) < 1e-3
        leg.float_spread = result
        assert (
            abs(
                leg.npv(rate_curve=curve, disc_curve=curve, forward=forward, settlement=settlement)
                - base_npv
                + 15000000
            )
            < 2e2
        )

    def test_fixing_method_raises(self) -> None:
        with pytest.raises(ValueError, match="`fixing_method`"):
            FloatLeg(schedule=Schedule(dt(2022, 2, 1), "9M", "Q"), fixing_method="bad")

    @pytest.mark.parametrize(
        ("eff", "term", "freq", "stub", "expected"),
        [
            (
                dt(2022, 1, 1),
                dt(2022, 6, 15),
                "Q",
                "ShortFront",
                [dt(2022, 1, 1), dt(2022, 3, 15), dt(2022, 6, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 6, 15),
                "Q",
                "ShortBack",
                [dt(2022, 1, 1), dt(2022, 4, 1), dt(2022, 6, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 9, 15),
                "Q",
                "LongFront",
                [dt(2022, 1, 1), dt(2022, 6, 15), dt(2022, 9, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 9, 15),
                "Q",
                "LongBack",
                [dt(2022, 1, 1), dt(2022, 4, 1), dt(2022, 9, 15)],
            ),
        ],
    )
    def test_leg_periods_unadj_dates(self, eff, term, freq, stub, expected) -> None:
        leg = FloatLeg(
            schedule=Schedule(effective=eff, termination=term, frequency=freq, stub=stub)
        )
        assert leg.schedule.uschedule == expected

    @pytest.mark.parametrize(
        ("eff", "term", "freq", "stub", "expected"),
        [
            (
                dt(2022, 1, 1),
                dt(2022, 6, 15),
                "Q",
                "ShortFront",
                [dt(2022, 1, 3), dt(2022, 3, 15), dt(2022, 6, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 6, 15),
                "Q",
                "ShortBack",
                [dt(2022, 1, 3), dt(2022, 4, 1), dt(2022, 6, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 9, 15),
                "Q",
                "LongFront",
                [dt(2022, 1, 3), dt(2022, 6, 15), dt(2022, 9, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 9, 15),
                "Q",
                "LongBack",
                [dt(2022, 1, 3), dt(2022, 4, 1), dt(2022, 9, 15)],
            ),
        ],
    )
    def test_leg_periods_adj_dates(self, eff, term, freq, stub, expected) -> None:
        leg = FloatLeg(
            schedule=Schedule(
                effective=eff, termination=term, frequency=freq, stub=stub, calendar="bus"
            )
        )
        assert leg.schedule.aschedule == expected

    @pytest.mark.parametrize(
        ("eff", "term", "freq", "stub", "expected"),
        [
            (
                dt(2022, 1, 1),
                dt(2022, 6, 15),
                "Q",
                "ShortFront",
                [
                    FloatPeriod(
                        start=dt(2022, 1, 3),
                        end=dt(2022, 3, 15),
                        payment=dt(2022, 3, 17),
                        frequency=Frequency.Months(3, None),
                        notional=defaults.notional,
                        convention=defaults.convention,
                        termination=dt(2022, 6, 15),
                    ),
                    FloatPeriod(
                        start=dt(2022, 3, 15),
                        end=dt(2022, 6, 15),
                        payment=dt(2022, 6, 17),
                        frequency=Frequency.Months(3, None),
                        notional=defaults.notional,
                        convention=defaults.convention,
                        termination=dt(2022, 6, 15),
                    ),
                ],
            ),
        ],
    )
    def test_leg_periods_adj_dates2(self, eff, term, freq, stub, expected) -> None:
        # as of v2.5 rateslib no longer puts details of the period into the str REPR.
        leg = FloatLeg(
            schedule=Schedule(
                effective=eff,
                termination=term,
                frequency=freq,
                stub=stub,
                payment_lag=2,
                calendar="bus",
            )
        )
        for i in range(2):
            assert leg.periods[i].__str__()[:19] == expected[i].__str__()[:19]

    def test_spread_compound_method_raises(self) -> None:
        with pytest.raises(ValueError, match="`spread_compound_method`"):
            FloatLeg(
                schedule=Schedule(
                    dt(2022, 2, 1),
                    "9M",
                    "Q",
                ),
                spread_compound_method="bad",
            )

    def test_leg_fixings_as_2_tuple(self) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(f"{name}_1M", Series([2.0, 3.0], index=[dt(2022, 6, 2), dt(2022, 7, 4)]))
        float_leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 5, 2),
                termination="4M",
                frequency="M",
                calendar="stk",
            ),
            rate_fixings=(1.5, name),
            currency="SEK",
            fixing_method="ibor",
            method_param=0,
        )
        assert float_leg.periods[0].rate_params.rate_fixing.value == 1.5
        assert float_leg.periods[1].rate_params.rate_fixing.value == 2.0
        assert float_leg.periods[2].rate_params.rate_fixing.value == 3.0
        assert float_leg.periods[3].rate_params.rate_fixing.value == NoInput.blank
        assert float_leg.periods[3].rate_params.rate_fixing.identifier == f"{name}_1M"

    def test_ex_div(self):
        leg = FloatLeg(schedule=Schedule(dt(2000, 1, 1), dt(2001, 1, 1), "Q", extra_lag=-3))
        assert not leg.ex_div(dt(2000, 3, 29))
        assert leg.ex_div(dt(2000, 3, 30))
        assert leg.ex_div(dt(2000, 4, 1))

    def test_mtm_xcs_type_type_sets_fx_fixing_start_initially(self):
        fixings.add(
            "EURUSD_1600",
            Series(
                index=[dt(2000, 4, 1), dt(2000, 4, 2), dt(2000, 7, 2)], data=[1.268, 1.27, 1.29]
            ),
        )
        leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2000, 7, 1),
                frequency="Q",
                payment_lag=1,
                payment_lag_exchange=0,
            ),
            currency="usd",
            pair="eurusd",
            initial_exchange=True,
            mtm="xcs",
            notional=5e6,
            fx_fixings=(1.25, "EURUSD_1600"),
        )
        assert leg.periods[2].mtm_params.fx_fixing_start.value == 1.25
        fixings.pop("EURUSD_1600")

    ## 4 types of non-deliverability

    @pytest.mark.parametrize(
        ("fx_fixings", "expected"),
        [
            ("ABCD", 1.10),
            (1.5, 1.5),
            ((1.2, "ABCD"), 1.2),
        ],
    )
    def test_non_mtm_xcs_type(self, fx_fixings, expected):
        fixings.add("ABCD_EURUSD", Series(index=[dt(1999, 12, 30)], data=[1.10]))
        fl = FloatLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2000, 3, 1),
                frequency="M",
                payment_lag=2,
                payment_lag_exchange=1,
                calendar="all",
            ),
            currency="usd",
            pair="eurusd",
            mtm="initial",
            initial_exchange=True,
            final_exchange=True,
            fx_fixings=fx_fixings,
        )
        # this leg has 4 periods with only one initial fixing date
        assert fl.periods[0].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
        assert fl.periods[1].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
        assert fl.periods[2].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
        assert fl.periods[3].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == expected
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == expected
        assert fl.periods[2].non_deliverable_params.fx_fixing.value == expected
        assert fl.periods[3].non_deliverable_params.fx_fixing.value == expected
        fixings.pop("ABCD_EURUSD")

    @pytest.mark.parametrize(
        ("fx_fixings", "expected"),
        [
            ("ABCDE", [1.21, 1.31]),
            (1.5, [1.5, NoInput(0)]),  # this is bad practice: should just supply str ID
            ((1.5, "ABCDE"), [1.5, 1.31]),  # this is bad practice: should just supply str ID
        ],
    )
    def test_irs_nd_type(self, fx_fixings, expected):
        fixings.add(
            "ABCDE_EURUSD",
            Series(
                index=[
                    dt(1999, 12, 30),
                    dt(2000, 1, 31),
                    dt(2000, 2, 1),
                    dt(2000, 2, 29),
                    dt(2000, 3, 1),
                ],
                data=[1.10, 1.20, 1.21, 1.30, 1.31],
            ),
        )
        fl = FloatLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2000, 3, 1),
                frequency="M",
                payment_lag=2,
                payment_lag_exchange=1,
                calendar="all",
            ),
            currency="usd",
            pair="eurusd",
            mtm="payment",
            initial_exchange=False,
            final_exchange=False,
            fx_fixings=fx_fixings,
        )
        # this leg has 2 periods and only 2 relevant fixings dates
        assert fl.periods[0].non_deliverable_params.fx_fixing.date == dt(2000, 2, 1)
        assert fl.periods[1].non_deliverable_params.fx_fixing.date == dt(2000, 3, 1)
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == expected[0]
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == expected[1]
        fixings.pop("ABCDE_EURUSD")

    @pytest.mark.parametrize(
        ("fx_fixings", "expected"),
        [
            ("ADE", [1.10, 1.10, 1.20, 1.20, 1.20]),
            (
                1.5,
                [1.5, 1.5, NoInput(0), NoInput(0), NoInput(0)],
            ),  # this is bad practice: should just supply str ID
            (
                (1.5, "ADE"),
                [1.5, 1.5, 1.20, 1.20, 1.20],
            ),  # this is bad practice: should just supply str ID
        ],
    )
    def test_mtm_xcs_nd_type(self, fx_fixings, expected):
        fixings.add(
            "ADE_EURUSD",
            Series(
                index=[
                    dt(1999, 12, 30),
                    dt(2000, 1, 31),
                    dt(2000, 2, 1),
                    dt(2000, 2, 29),
                    dt(2000, 3, 1),
                ],
                data=[1.10, 1.20, 1.21, 1.30, 1.31],
            ),
        )
        fl = FloatLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2000, 3, 1),
                frequency="M",
                payment_lag=2,
                payment_lag_exchange=1,
                calendar="all",
            ),
            currency="usd",
            pair="eurusd",
            mtm=LegMtm.XCS,
            initial_exchange=True,
            final_exchange=True,
            fx_fixings=fx_fixings,
        )
        # this leg has 5 periods with only two relevant fixing dates
        assert fl.periods[0].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
        assert fl.periods[1].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
        assert fl.periods[2].mtm_params.fx_fixing_end.date == dt(2000, 1, 31)
        assert fl.periods[3].non_deliverable_params.fx_fixing.date == dt(2000, 1, 31)
        assert fl.periods[4].non_deliverable_params.fx_fixing.date == dt(2000, 1, 31)
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == expected[0]
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == expected[1]
        assert fl.periods[2].mtm_params.fx_fixing_end.value == expected[2]
        assert fl.periods[3].non_deliverable_params.fx_fixing.value == expected[3]
        assert fl.periods[4].non_deliverable_params.fx_fixing.value == expected[4]
        fixings.pop("ADE_EURUSD")

    @pytest.mark.parametrize(
        ("fx_fixings", "expected"),
        [
            ("AXDE", [1.10, 1.21, 1.31, 1.30]),
            (
                1.5,
                [1.5, NoInput(0), NoInput(0), NoInput(0)],
            ),  # this is bad practice: should just supply str ID
            (
                (1.5, "AXDE"),
                [1.5, 1.21, 1.31, 1.30],
            ),  # this is bad practice: should just supply str ID
        ],
    )
    def test_non_mtm_xcs_nd_type(self, fx_fixings, expected):
        fixings.add(
            "AXDE_EURUSD",
            Series(
                index=[
                    dt(1999, 12, 30),
                    dt(2000, 1, 31),
                    dt(2000, 2, 1),
                    dt(2000, 2, 29),
                    dt(2000, 3, 1),
                ],
                data=[1.10, 1.20, 1.21, 1.30, 1.31],
            ),
        )
        fl = FloatLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2000, 3, 1),
                frequency="M",
                payment_lag=2,
                payment_lag_exchange=1,
                calendar="all",
            ),
            currency="usd",
            pair="eurusd",
            mtm="payment",
            initial_exchange=True,
            final_exchange=True,
            fx_fixings=fx_fixings,
        )
        # this leg has 4 periods with 3 or 4 (if lag exchange is different) relevant fixing dates.
        assert fl.periods[0].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
        assert fl.periods[1].non_deliverable_params.fx_fixing.date == dt(2000, 2, 1)
        assert fl.periods[2].non_deliverable_params.fx_fixing.date == dt(2000, 3, 1)
        assert fl.periods[3].non_deliverable_params.fx_fixing.date == dt(2000, 2, 29)
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == expected[0]
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == expected[1]
        assert fl.periods[2].non_deliverable_params.fx_fixing.value == expected[2]
        assert fl.periods[3].non_deliverable_params.fx_fixing.value == expected[3]
        fixings.pop("AXDE_EURUSD")


class TestZeroFloatLeg:
    def test_zero_float_leg_set_float_spread(self, curve) -> None:
        float_leg = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
        )
        assert float_leg.float_spread == 0.0
        assert float_leg.periods[0].float_spread == 0.0

        float_leg.float_spread = 2.0
        assert float_leg.float_spread == 2.0
        assert float_leg.periods[0].float_spread == 2.0

    def test_with_fixings(self):
        name = str(hash(os.urandom(8)))
        fixings.add(
            f"{name}_3m",
            Series(
                index=[dt(2022, 1, 1), dt(2022, 2, 1), dt(2022, 5, 1)],
                data=[1.0, 2.0, 3.0],
            ),
        )
        fixings.add(
            f"{name}_1m",
            Series(
                index=[dt(2022, 1, 1), dt(2022, 2, 1), dt(2022, 5, 1)],
                data=[5.0, 0.0, 0.0],
            ),
        )
        leg = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 8, 1),
                front_stub=dt(2022, 2, 1),
                frequency="Q",
                calendar="all",
            ),
            method_param=0,
            fixing_method="ibor",
            rate_fixings=name,
        )
        expected = [5.0, 2.0, 3.0]
        for i, period in enumerate(leg.periods[0]._float_periods):
            assert period.rate_params.rate_fixing.value == expected[i]

        result = leg.periods[0].rate()
        assert abs(result - 2.8743158337825925) < 1e-8

    def test_zero_float_leg_dcf(self) -> None:
        ftl = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
        )
        p = ftl.periods[0]
        result = p.dcf
        expected = p._float_periods[0].period_params.dcf + p._float_periods[1].period_params.dcf
        assert result == expected

    def test_zero_float_leg_cashflow(self, curve) -> None:
        ftl = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            float_spread=500,
        )
        p = ftl.periods[0]
        result = p.try_unindexed_reference_cashflow(rate_curve=curve).unwrap()
        expected = (
            1
            + p._float_periods[0].period_params.dcf
            * p._float_periods[0].rate(rate_curve=curve)
            / 100
        )
        expected *= (
            1
            + p._float_periods[1].period_params.dcf
            * p._float_periods[1].rate(rate_curve=curve)
            / 100
        )
        expected = (expected - 1) * 1e9
        assert abs(result - expected) < 1e-9

    def test_zero_float_leg_cashflows(self, curve) -> None:
        ftl = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            float_spread=500,
        )
        result = ftl.cashflows(rate_curve=curve)
        expected = DataFrame(
            {
                "Type": ["ZeroFloatPeriod"],
                "Acc Start": [dt(2022, 1, 1)],
                "Acc End": [dt(2022, 6, 1)],
                "DCF": [0.419444444444444],
                "Spread": [500.0],
            },
        )
        assert_frame_equal(result[["Type", "Acc Start", "Acc End", "DCF", "Spread"]], expected)

    def test_zero_float_leg_npv(self, curve) -> None:
        ftl = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
        )
        result = ftl.npv(rate_curve=curve)
        expected = 16710778.891147703
        assert abs(result - expected) < 1e-2
        result2 = ftl.npv(rate_curve=curve, local=True)
        assert abs(result2["usd"] - expected) < 1e-2

    def test_cashflows_none(self) -> None:
        ftl = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
        )
        result = ftl.cashflows()
        assert result.iloc[0].to_dict()[defaults.headers["npv"]] is None
        assert result.iloc[0].to_dict()[defaults.headers["npv_fx"]] is None

    def test_amortization_raises(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            ZeroFloatLeg(
                schedule=Schedule(
                    effective=dt(2022, 1, 1),
                    termination=dt(2022, 6, 1),
                    payment_lag=2,
                    frequency="Q",
                ),
                notional=-1e9,
                convention="Act360",
                amortization=1.0,
            )

    def test_rfr_fixings_table(self, curve) -> None:
        zfl = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 10, 1),
                payment_lag=0,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
        )
        # fl = FloatLeg(
        #     effective=dt(2022, 1, 1),
        #     termination=dt(2022, 10, 1),
        #     payment_lag=0,
        #     notional=-1e9,
        #     convention="Act360",
        #     frequency="Q",
        # )
        result = zfl.local_analytic_rate_fixings(rate_curve=curve)
        # compare = fl.fixings_table(curve)
        for i in range(len(result.index)):
            # consistent risk throught the compounded leg
            assert abs(result.iloc[i, 0] - 277.75) < 1e-1

    def test_ibor_fixings_table(self, curve) -> None:
        zfl = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 10, 1),
                payment_lag=0,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            fixing_method="ibor",
            method_param=0,
        )
        result = zfl.local_analytic_rate_fixings(rate_curve=curve)
        assert abs(result.iloc[0, 0] - 24750) < 1e-3
        assert abs(result.iloc[1, 0] - 25022.4466) < 1e-2
        assert abs(result.iloc[2, 0] - 25294.7845) < 1e-2

    def test_ibor_stub_fixings_table(self, curve) -> None:
        curve2 = curve.copy()
        curve2._id = "3mIBOR"
        curve._id = "1mIBOR"
        zfl = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 9, 1),
                payment_lag=0,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            fixing_method="ibor",
            method_param=0,
        )
        result = zfl.local_analytic_rate_fixings(
            rate_curve={"1m": curve, "3m": curve2}, disc_curve=curve
        )
        assert abs(result.iloc[0, 0] - 8554.562) < 1e-2
        assert abs(result.iloc[0, 1] - 7726.701) < 1e-2
        assert isna(result.iloc[1, 0])
        assert abs(result.iloc[2, 1] - 25294.7235) < 1e-3

    @pytest.mark.parametrize(
        "fixings", [[2.0, 2.5], Series([2.0, 2.5], index=[dt(2021, 7, 1), dt(2021, 10, 1)])]
    )
    def test_ibor_fixings_table_after_known_fixings(self, curve, fixings) -> None:
        curve2 = curve.copy()
        curve2._id = "3mIBOR"
        curve._id = "1mIBOR"
        zfl = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2021, 7, 1),
                termination=dt(2022, 9, 1),
                payment_lag=0,
                frequency="Q",
                stub="shortBack",
            ),
            notional=-1e9,
            convention="Act360",
            fixing_method="ibor",
            method_param=0,
            rate_fixings=fixings,
        )
        result = zfl.local_analytic_rate_fixings(
            rate_curve={"1m": curve, "3m": curve2}, disc_curve=curve
        )
        assert abs(result.iloc[0, 0] - 0) < 1e-2
        assert abs(result.iloc[1, 0] - 0) < 1e-2
        assert isna(result.iloc[0, 1])
        assert abs(result.iloc[4, 0] - 8792.231) < 1e-2
        assert abs(result.iloc[4, 1] - 8508.6111) < 1e-3

    def test_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="`frequency` for a ZeroFloatLeg should not be 'Z'"):
            ZeroFloatLeg(
                schedule=Schedule(
                    effective=dt(2022, 1, 1),
                    termination="5y",
                    payment_lag=0,
                    frequency="Z",
                ),
                notional=-1e8,
                convention="ActAct",
            )

    def test_zero_float_leg_analytic_delta(self, curve) -> None:
        zfl = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                frequency="A",
            ),
            notional=-1e8,
            convention="ActAct",
            float_spread=1.0,
            fixing_series=FloatRateSeries(
                lag=0,
                calendar="all",
                modifier="f",
                convention="act360",
                eom=False,
            ),
        )
        result = zfl.analytic_delta(rate_curve=curve)
        expected = -47914.3660

        assert abs(result - expected) < 1e-3

    @pytest.mark.parametrize(
        ("settlement", "forward", "exp"),
        [
            (NoInput(0), NoInput(0), 0.71008),
            (NoInput(0), dt(2023, 1, 1), -0.11739),
            (dt(2026, 1, 1), dt(2026, 1, 1), -2.40765),
        ],
    )
    def test_zero_float_spread_calc(self, settlement, forward, exp, curve) -> None:
        rate_curve = curve.shift(25)
        zfl = ZeroFloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                frequency="A",
            ),
            notional=-1e8,
            convention="Act360",
            fixing_method="ibor",
        )
        tgt_npv = 25000000 * curve[dt(2027, 1, 1)]
        result = zfl.spread(
            target_npv=tgt_npv,
            rate_curve=rate_curve,
            disc_curve=curve,
            settlement=settlement,
            forward=forward,
        )

        zfl.float_spread = result
        tested = zfl.local_npv(
            rate_curve=rate_curve,
            disc_curve=curve,
            settlement=settlement,
            forward=forward,
        )
        assert abs(result / 100 - exp) < 1e-3
        assert abs(tgt_npv - tested) < 1e-3


class TestZeroFixedLeg:
    @pytest.mark.parametrize(
        ("freq", "cash", "rate"),
        [
            ("A", 13140821.29, 2.50),
            ("S", 13227083.80, 2.50),
            ("A", None, NoInput(0)),
        ],
    )
    def test_zero_fixed_leg_cashflows(self, freq, cash, rate, curve) -> None:
        zfl = ZeroFixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                frequency=freq,
            ),
            notional=-1e8,
            convention="ActAct",
            fixed_rate=rate,
        )
        result = zfl.cashflows(disc_curve=curve)
        expected = DataFrame(
            {
                "Type": ["ZeroFixedPeriod"],
                "Acc Start": [dt(2022, 1, 1)],
                "Acc End": [dt(2027, 1, 1)],
                "DCF": [5.0],
                "Rate": [_drb(None, rate)],
                "Cashflow": [cash],
            },
        )
        assert_frame_equal(
            result[["Type", "Acc Start", "Acc End", "DCF", "Rate", "Cashflow"]],
            expected,
            rtol=1e-3,
        )

    def test_zero_fixed_leg_cashflows_cal(self, curve) -> None:
        # assert stated cashflows accrual dates are adjusted according to calendar
        # GH561/562
        zfl = ZeroFixedLeg(
            schedule=Schedule(
                effective=dt(2024, 12, 15),
                termination="5y",
                payment_lag=0,
                calendar="tgt",
                modifier="mf",
                frequency="A",
            ),
            notional=-1e8,
            convention="ActAct",
            fixed_rate=2.0,
        )
        result = zfl.cashflows(disc_curve=curve)
        expected = DataFrame(
            {
                "Type": ["ZeroFixedPeriod"],
                "Acc Start": [dt(2024, 12, 16)],
                "Acc End": [dt(2029, 12, 17)],
                "DCF": [5.0],
                "Rate": [2.0],
            },
        )
        assert_frame_equal(
            result[["Type", "Acc Start", "Acc End", "DCF", "Rate"]],
            expected,
            rtol=1e-3,
        )

    def test_zero_fixed_leg_npv(self, curve) -> None:
        zfl = ZeroFixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                frequency="A",
            ),
            notional=-1e8,
            convention="ActAct",
            fixed_rate=2.5,
        )
        result = zfl.npv(disc_curve=curve)
        expected = 13140821.29 * curve[dt(2027, 1, 1)]
        assert abs(result - expected) < 1e-2
        result2 = zfl.npv(disc_curve=curve, local=True)
        assert abs(result2["usd"] - expected) < 1e-2

    def test_zero_fixed_leg_analytic_delta(self, curve) -> None:
        zfl = ZeroFixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                frequency="A",
            ),
            notional=-1e8,
            convention="ActAct",
            fixed_rate=2.5,
        )
        result2 = zfl.analytic_delta(disc_curve=curve)
        assert abs(result2 + 45024.1974) < 1e-3

    @pytest.mark.parametrize(
        ("settlement", "forward", "exp"),
        [
            (NoInput(0), NoInput(0), 2.50),
            (NoInput(0), dt(2023, 1, 1), 2.404826),
            (dt(2026, 1, 1), NoInput(0), 2.139550),
        ],
    )
    def test_zero_fixed_spread(self, settlement, forward, exp, curve) -> None:
        zfl = ZeroFixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                frequency="A",
            ),
            notional=-1e8,
            convention="ActAct",
            fixed_rate=NoInput(0),
        )
        result = zfl.spread(
            target_npv=13140821.29 * curve[dt(2027, 1, 1)],
            rate_curve=NoInput(0),
            disc_curve=curve,
            settlement=settlement,
            forward=forward,
        )
        assert abs(result / 100 - exp) < 1e-3

    def test_zero_fixed_spread_raises_settlement(self, curve) -> None:
        zfl = ZeroFixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                frequency="A",
            ),
            notional=-1e8,
            convention="ActAct",
            fixed_rate=NoInput(0),
        )
        with pytest.raises(ZeroDivisionError):
            zfl.spread(
                target_npv=13140821.29 * curve[dt(2027, 1, 1)],
                rate_curve=NoInput(0),
                disc_curve=curve,
                settlement=dt(2029, 1, 1),
                forward=NoInput(0),
            )

    def test_zero_fixed_spread_indexed(self, curve) -> None:
        zfl = ZeroFixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                frequency="A",
            ),
            notional=-1e8,
            convention="ActAct",
            fixed_rate=NoInput(0),
            index_base=100.0,
            index_fixings=110.0,
        )
        result = zfl.spread(
            target_npv=13140821.29 * curve[dt(2027, 1, 1)],
            rate_curve=NoInput(0),
            disc_curve=curve,
        )
        assert abs(result / 100 - 2.2826266057484057) < 1e-3

    def test_zero_fixed_spread_non_deliverable(self, curve) -> None:
        zfl = ZeroFixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                frequency="A",
            ),
            notional=-1e8,
            convention="ActAct",
            fixed_rate=NoInput(0),
            currency="usd",
            pair="eurusd",
            fx_fixings=2.0,
        )
        result = zfl.spread(
            target_npv=13140821.29 * curve[dt(2027, 1, 1)],
            rate_curve=NoInput(0),
            disc_curve=curve,
        )
        assert abs(result / 100 - 1.2808477472765924) < 1e-3

    def test_amortization_raises(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword argument 'amortization'"):
            ZeroFixedLeg(
                schedule=Schedule(
                    effective=dt(2022, 1, 1),
                    termination="5y",
                    payment_lag=0,
                    frequency="A",
                ),
                notional=-1e8,
                convention="ActAct",
                fixed_rate=NoInput(0),
                amortization=1.0,
            )

    def test_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="`frequency` for a ZeroFixedLeg should not be 'Z'"):
            ZeroFixedLeg(
                schedule=Schedule(
                    effective=dt(2022, 1, 1),
                    termination="5y",
                    payment_lag=0,
                    frequency="Z",
                ),
                notional=-1e8,
                convention="ActAct",
                fixed_rate=NoInput(0),
            )

    def test_analytic_delta_no_fixed_rate(self, curve) -> None:
        zfl = ZeroFixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                frequency="A",
            ),
            notional=-1e8,
            convention="ActAct",
            fixed_rate=NoInput(0),
        )
        with pytest.raises(ValueError, match="A `fixed_rate` must be set for a "):
            zfl.analytic_delta(disc_curve=curve)


class TestZeroIndexLeg:
    @pytest.mark.parametrize(
        ("index_base", "index_fixings", "meth", "exp"),
        [
            (NoInput(0), NoInput(0), "monthly", -61855.670),
            (NoInput(0), NoInput(0), "daily", -61782.379),
            (100.0, NoInput(0), "monthly", -61855.670),
            (NoInput(0), 110.0, "monthly", -100000.0),
            (NoInput(0), 110.0, "daily", -98696.645),
            (100.0, 110.0, "monthly", -100000.0),
            (100.0, 110.0, "daily", -100000.0),
        ],
    )
    def test_zero_index_cashflow(self, index_base, index_fixings, meth, exp) -> None:
        index_curve = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.97,
            },
            index_base=100.0,
            index_lag=3,
            interpolation="linear_index",
        )
        zil = ZeroIndexLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 15),
                termination="2Y",
                frequency="A",
            ),
            convention="1+",
            index_base=index_base,
            index_fixings=index_fixings,
            index_method=meth,
            final_exchange=False,
        )
        result = zil.cashflows(index_curve=index_curve).loc[0, "Cashflow"]
        assert abs(result - exp) < 1e-3

    @pytest.mark.skip(reason="v2.2 no longer permits fixing setting")
    def test_set_index_leg_after_init(self) -> None:
        leg = ZeroFixedLeg(
            schedule=Schedule(
                effective=dt(2022, 3, 15),
                termination="9M",
                frequency="Q",
                payment_lag=0,
            ),
            convention="1+",
            notional=40e6,
            index_base=None,
        )
        for period in leg.periods[:1]:
            assert period.index_base is None
        leg.index_base = 205.0
        for period in leg.periods[:1]:
            assert period.index_base == 205.0

    def test_zero_analytic_delta(self, curve) -> None:
        zil = ZeroIndexLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 15),
                termination="2Y",
                frequency="A",
            ),
            convention="1+",
            index_lag=0,
            index_base=100.0,
            index_fixings=110.0,
        )
        assert zil.analytic_delta(disc_curve=curve) == 0.0

    def test_cashflows(self) -> None:
        index_curve = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.97,
            },
            index_base=100.0,
            index_lag=3,
            interpolation="linear_index",
        )
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.97})
        zil = ZeroIndexLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 15),
                termination="2Y",
                frequency="A",
                payment_lag=0,
                payment_lag_exchange=0,
            ),
            convention="1+",
            index_lag=3,
            index_method="curve",
        )
        result = zil.cashflows(index_curve=index_curve, disc_curve=curve)
        expected = DataFrame(
            {
                "Type": ["Cashflow"],  #  ["ZeroIndexLeg"],
                "Notional": [1000000.0],
                "Unindexed Cashflow": [-1000000.0],
                "Index Base": [100.11863],
                "Index Ratio": [1.06178],
                "Cashflow": [-61782.379],
                "NPV": [-58063.1659],  # [-58053.47605],
            },
        )
        assert_frame_equal(
            result[
                [
                    "Type",
                    "Notional",
                    "Unindexed Cashflow",
                    "Index Base",
                    "Index Ratio",
                    "Cashflow",
                    "NPV",
                ]
            ],
            expected,
            rtol=1e-3,
        )

    @pytest.mark.parametrize("only", [True, False])
    def test_four_ways(self, only):
        # A Zero Index Legs can also be created in four ways.
        one = ZeroFixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="2Y",
                frequency="A",
                payment_lag=0,
                payment_lag_exchange=0,
            ),
            fixed_rate=0.0,
            index_base=100.0,
            index_fixings=110.0,
            index_only=only,
            final_exchange=True,
        )
        result1 = one.cashflows().loc[1, "Cashflow"]

        two = Cashflow(
            payment=dt(2024, 1, 1),
            notional=1e6,
            index_base=100.0,
            index_fixings=110.0,
            index_only=only,
        )
        result2 = two.cashflows()["Cashflow"]

        three = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="2Y",
                frequency="Z",
                payment_lag=0,
                payment_lag_exchange=0,
            ),
            fixed_rate=0.0,
            index_base=100.0,
            index_fixings=110.0,
            index_only=only,
            final_exchange=True,
        )
        result3 = three.cashflows().loc[1, "Cashflow"]

        four = ZeroIndexLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination="2Y",
                frequency="Z",
                payment_lag=0,
                payment_lag_exchange=0,
            ),
            index_base=100.0,
            index_fixings=110.0,
            final_exchange=not only,
        )
        result4 = four.cashflows().loc[0, "Cashflow"]

        assert abs(result1 - result2) < 1e-8
        assert abs(result1 - result3) < 1e-8
        assert abs(result1 - result4) < 1e-8

    @pytest.mark.parametrize(
        ("ini", "final", "mtm", "lenn", "nd_dt", "cf"),
        [
            (False, False, False, 1, dt(2000, 1, 1), 500e3 * 2.0),
            (False, False, True, 1, dt(2001, 1, 1), 500e3 * 3.0),
            (False, True, False, 1, dt(2000, 1, 1), 1.5e6 * 2.0),
            (False, True, True, 1, dt(2000, 1, 1), 1.5e6 * 2.0),
            # (True, False, False, 2, dt(2000, 1, 1)), # final exch True by default
            # (True, False, True, 2, dt(2000, 1, 1)),  # final exch True by default
            (True, True, False, 2, dt(2000, 1, 1), 1.5e6 * 2.0),
            (True, True, True, 2, dt(2000, 1, 1), 1.5e6 * 2.0),
        ],
    )
    def test_attributes(self, ini, final, mtm, lenn, nd_dt, cf) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(name, Series(index=[dt(2000, 1, 1), dt(2001, 1, 1)], data=[10.0, 15.0]))
        fixings.add(
            name + "fx_EURUSD", Series(index=[dt(1999, 12, 30), dt(2000, 12, 28)], data=[2.0, 3.0])
        )
        leg = ZeroIndexLeg(
            schedule=Schedule(effective=dt(2000, 1, 1), termination=dt(2001, 1, 1), frequency="Z"),
            currency="usd",
            initial_exchange=ini,
            final_exchange=final,
            pair="eurusd",
            mtm=mtm,
            fx_fixings=name + "fx",
            index_lag=0,
            index_fixings=name,
            notional=-1e6,
        )
        assert len(leg.periods) == lenn
        assert leg.periods[-1].non_deliverable_params.delivery == nd_dt
        assert leg.periods[-1].non_deliverable_params.publication == get_calendar(
            "ldn"
        ).lag_bus_days(nd_dt, -2, True)
        assert leg.periods[-1].cashflow() == cf
        fixings.pop(name)
        fixings.pop(name + "fx_EURUSD")


class TestFloatLegExchange:
    @pytest.mark.skip(reason="v 2.2 removed ability to mutate notional")
    def test_float_leg_exchange_notional_setter(self) -> None:
        float_leg_exc = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            initial_exchange=True,
            final_exchange=True,
        )
        float_leg_exc.notional = 200
        assert float_leg_exc.notional == 200

    @pytest.mark.skip(reason="v 2.2 removed ability to mutate amortisation.")
    def test_float_leg_exchange_amortization_setter(self) -> None:
        float_leg_exc = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 10, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1000,
            convention="Act360",
            initial_exchange=True,
            final_exchange=True,
        )
        float_leg_exc.amortization = -200

        cashflows = [2, 4, 6]
        cash_notionals = [None, -200, None, -200, None, -600]
        fixed_notionals = [None, -1000, None, -800, None, -600]
        for i in cashflows:
            assert isinstance(float_leg_exc.periods[i], Cashflow)
            assert float_leg_exc.periods[i].notional == cash_notionals[i - 1]

            assert isinstance(float_leg_exc.periods[i - 1], FloatPeriod)
            assert float_leg_exc.periods[i - 1].notional == fixed_notionals[i - 1]

    def test_float_leg_exchange_set_float_spread(self) -> None:
        float_leg_exc = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 10, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1000,
            convention="Act360",
            initial_exchange=True,
            final_exchange=True,
        )
        assert float_leg_exc.float_spread == 0.0
        float_leg_exc.float_spread = 2.0
        assert float_leg_exc.float_spread == 2.0
        for period in float_leg_exc.periods:
            if isinstance(period, FloatPeriod):
                period.rate_params.float_spread == 2.0

    def test_float_leg_exchange_amortization(self, curve) -> None:
        leg = FloatLeg(
            schedule=Schedule(
                dt(2022, 1, 1),
                dt(2023, 1, 1),
                "Q",
                payment_lag=0,
            ),
            notional=5e6,
            amortization=1e6,
            initial_exchange=True,
            final_exchange=True,
        )
        assert len(leg.periods) == 9
        for i in [0, 2, 4, 6, 8]:
            assert type(leg.periods[i]) is Cashflow
        for i in [1, 3, 5, 7]:
            assert type(leg.periods[i]) is FloatPeriod
        assert leg.periods[1].settlement_params.notional == 5e6
        assert leg.periods[7].settlement_params.notional == 2e6
        assert leg.periods[8].settlement_params.notional == 2e6
        assert abs(leg.npv(rate_curve=curve).real) < 1e-9

    def test_float_leg_exchange_npv(self, curve) -> None:
        fle = FloatLeg(
            schedule=Schedule(
                dt(2022, 2, 1),
                "6M",
                "Q",
                payment_lag=0,
            ),
            initial_exchange=True,
            final_exchange=True,
        )
        result = fle.npv(rate_curve=curve)
        assert abs(result) < 1e-9

    def test_float_leg_exchange_fixings_table(self, curve) -> None:
        fle = FloatLeg(
            schedule=Schedule(
                dt(2022, 2, 1),
                "6M",
                "Q",
                payment_lag=0,
            ),
            initial_exchange=True,
            final_exchange=True,
        )
        result = fle.local_analytic_rate_fixings(rate_curve=curve)
        expected = DataFrame(
            data=[-0.2767869527597316, -0.27405055522733884],
            index=Index([dt(2022, 4, 30), dt(2022, 5, 1)], name="obs_dates"),
            columns=MultiIndex.from_tuples(
                [(curve.id, "usd", "usd", "1B")],
                names=["identifier", "local_ccy", "display_ccy", "frequency"],
            ),
        )
        assert_frame_equal(result[dt(2022, 4, 30) : dt(2022, 5, 1)], expected)


class TestFixedLeg:
    def test_fixed_leg_analytic_delta(self, curve) -> None:
        fixed_leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=1e9,
            convention="Act360",
        )
        result = fixed_leg.analytic_delta(rate_curve=curve)
        assert abs(result - 41400.42965267) < 1e-7

    def test_fixed_leg_npv(self, curve) -> None:
        fixed_leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=1e9,
            convention="Act360",
            fixed_rate=4.00,
        )
        result = fixed_leg.npv(disc_curve=curve)
        assert abs(result + 400 * fixed_leg.analytic_delta(disc_curve=curve)) < 1e-7

    def test_fixed_leg_cashflows(self, curve) -> None:
        fixed_leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            fixed_rate=4.00,
            notional=-1e9,
            convention="Act360",
        )
        result = fixed_leg.cashflows(disc_curve=curve)
        # test a couple of return elements
        assert abs(result.loc[0, defaults.headers["cashflow"]] - 6555555.55555) < 1e-4
        assert abs(result.loc[1, defaults.headers["df"]] - 0.98307) < 1e-4
        assert abs(result.loc[1, defaults.headers["notional"]] + 1e9) < 1e-7

    def test_fixed_leg_set_fixed(self, curve) -> None:
        fixed_leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
        )
        assert fixed_leg.fixed_rate is NoInput(0)
        assert fixed_leg.periods[0].rate_params.fixed_rate is NoInput(0)

        fixed_leg.fixed_rate = 2.0
        assert fixed_leg.fixed_rate == 2.0
        assert fixed_leg.periods[0].rate_params.fixed_rate == 2.0

    def test_fixed_leg_final_exchange_custom_amort(self):
        leg = FixedLeg(
            schedule=Schedule(dt(2000, 1, 1), dt(2000, 5, 1), "M"),
            notional=100,
            amortization=Amortization(4, 100, [0, 50.0, 0]),
            final_exchange=True,
        )
        result = leg.cashflows()
        assert result["Notional"].tolist() == [100.0, 0.0, 100.0, 50.0, 50.0, 0.0, 50.0, 50.0]

    def test_non_deliverable(self, curve):
        fxf = FXForwards(
            fx_curves={"usdusd": curve, "brlusd": curve, "brlbrl": curve},
            fx_rates=FXRates({"usdbrl": 25.0}, settlement=dt(2022, 1, 3)),
        )
        fixed_leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                payment_lag_exchange=1,
                frequency="Q",
            ),
            notional=1e9,
            convention="Act360",
            fixed_rate=4.00,
            currency="usd",
            pair=FXIndex("usdbrl", "all", 0),
        )
        cf = fixed_leg.cashflows(disc_curve=curve, fx=fxf)

        assert fixed_leg.periods[0].non_deliverable_params.fx_fixing.date == dt(2022, 1, 2)
        assert fixed_leg.periods[1].non_deliverable_params.fx_fixing.date == dt(2022, 1, 2)

        assert abs(cf.loc[1, "Cashflow"] + 408888.8888) < 1e-4
        assert cf.loc[0, "Reference Ccy"] == "BRL"

    # v2.5

    @pytest.mark.parametrize(
        ("settlement", "forward", "exp"),
        [
            (NoInput(0), NoInput(0), 403.9491881327746),
            (dt(2022, 3, 30), dt(2022, 3, 30), 399.9990223763462),
            (dt(2022, 4, 6), dt(2022, 4, 6), 799.0147512470912),
        ],
    )
    def test_fixed_leg_spread(self, settlement, forward, exp, curve) -> None:
        fixed_leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 7, 1),
                payment_lag=2,
                payment_lag_exchange=1,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            fixed_rate=4.00,
            currency="usd",
        )
        result = fixed_leg.spread(
            target_npv=20000000,
            disc_curve=curve,
            rate_curve=curve,
            index_curve=curve,
            settlement=settlement,
            forward=forward,
        )
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize("initial", [True, False])
    @pytest.mark.parametrize("final", [True, False])
    @pytest.mark.parametrize("amortization", [True, False])
    def test_construction_of_relevant_periods(self, initial, final, amortization):
        # test construction cases:
        #
        #  - Regular periods only; no amortization, no exchanges
        #  - Regular with different exchanges: final and initial
        #  - Regular with Amortization, but no exchanges.
        #  - Regular with Amortization and with exchanges.
        #
        fl = FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2000, 7, 1),
                frequency="Q",
            ),
            initial_exchange=initial,
            final_exchange=final,
            amortization=250000.0 if amortization else NoInput(0),
        )
        assert len(fl._regular_periods) == 2
        assert (fl._exchange_periods[0] is None) is not initial
        assert (fl._exchange_periods[1] is None) is not (final or initial)
        if not amortization:
            assert fl.amortization._type == _AmortizationType.NoAmortization
            assert fl._amortization_exchange_periods is None
        else:
            assert fl.amortization._type == _AmortizationType.ConstantPeriod
            if not (final or initial):  # initial sets final to True
                assert fl._amortization_exchange_periods is None
            else:
                assert len(fl._amortization_exchange_periods) == 1

    @pytest.mark.parametrize("initial", [True, False])
    @pytest.mark.parametrize("final", [True, False])
    @pytest.mark.parametrize("amortization", [True, False])
    def test_construction_of_relevant_periods_non_deliverable(self, initial, final, amortization):
        # when the leg is ND but not MTM the same construction as in the regular deliverable
        # case should be permitted. All FXFixings should beb determined by a single rate of
        # exchange. This test builds on the above test for non-deliverability.
        fl = FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2000, 7, 1),
                frequency="Q",
                payment_lag_exchange=1,
            ),
            fixed_rate=10.0,
            currency="usd",
            pair="eurusd",  # the notional of this leg is expressed in BRL but payments made in USD
            initial_exchange=initial,
            final_exchange=final,
            amortization=250000.0 if amortization else NoInput(0),
            fx_fixings=2.0,  # this should not impact the reference currency notional and amortiz
        )
        for rp in fl._regular_periods:
            assert rp.non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
            assert rp.non_deliverable_params.fx_fixing.value == 2.0

        if initial:
            assert fl._exchange_periods[0].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
            assert fl._exchange_periods[0].non_deliverable_params.fx_fixing.value == 2.0

        if final:
            assert fl._exchange_periods[1].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
            assert fl._exchange_periods[1].non_deliverable_params.fx_fixing.value == 2.0

        if amortization and final:
            assert fl._amortization_exchange_periods[0].non_deliverable_params.fx_fixing.date == dt(
                1999, 12, 30
            )
            assert (
                fl._amortization_exchange_periods[0].non_deliverable_params.fx_fixing.value == 2.0
            )
            assert fl.amortization.amortization == (250000.0,)

            cf = fl.cashflows()

            if initial:
                assert abs(cf.loc[0, "Cashflow"] - 2000000.0) < 1e-4  # ini exchange
                assert abs(cf.loc[1, "Cashflow"] + 50555.55555) < 1e-4  # fixed rate
                assert abs(cf.loc[2, "Cashflow"] + 500000.0) < 1e-4  # amort exchange
                assert abs(cf.loc[3, "Cashflow"] + 37916.66666) < 1e-4  # fixed rate
                assert abs(cf.loc[4, "Cashflow"] + 1500000.0) < 1e-4  # final exchange

    def test_construction_index_fixings(self):
        # test that amortization index_value date is correctly applied to each period.

        name = str(hash(os.urandom(8)))
        fixings.add(name, Series(index=[dt(2000, 1, 1)], data=[101.0]))
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2000, 7, 1),
                frequency="Q",
                payment_lag_exchange=1,
                payment_lag=2,
            ),
            fixed_rate=2.0,
            convention="Act360",
            notional=5000000,
            amortization=1000000,
            final_exchange=True,
            index_fixings=name,
            index_lag=0,
            index_method="monthly",
        )
        assert leg._regular_periods[0].index_params.index_fixing.date == leg.schedule.aschedule[1]
        assert leg._regular_periods[1].index_params.index_fixing.date == leg.schedule.aschedule[2]
        assert (
            leg._amortization_exchange_periods[0].index_params.index_fixing.date
            == leg.schedule.aschedule[1]
        )
        assert leg._exchange_periods[1].index_params.index_fixing.date == leg.schedule.aschedule[2]

        assert leg._regular_periods[0].index_params.index_base.value == 101.0
        assert leg._regular_periods[1].index_params.index_base.value == 101.0
        assert leg._amortization_exchange_periods[0].index_params.index_base.value == 101.0
        assert leg._exchange_periods[1].index_params.index_base.value == 101.0

        fixings.pop(name)

    @pytest.mark.parametrize("amortization", [True, False])
    def test_construction_of_relevant_periods_non_deliverable_mtm(self, amortization):
        # when the leg is ND and MTM the FXFixings should be determined by their appropriate
        # payment dates deriving fixing date. This test excludes notional exchanges,
        # designed for ND-IRS
        name = str(hash(os.urandom(8)))
        fixings.add(
            name + "_EURUSD",
            Series(
                index=[
                    dt(1999, 12, 24),
                    dt(1999, 12, 29),
                    dt(2000, 3, 29),
                    dt(2000, 3, 30),
                    dt(2000, 6, 28),
                    dt(2000, 6, 29),
                ],
                data=[1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            ),
        )

        fl = FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 5),
                termination=dt(2000, 7, 5),
                frequency="Q",
                payment_lag_exchange=1,
                payment_lag=0,
            ),
            fixed_rate=10.0,
            currency="usd",
            pair=FXIndex("eurusd", "tgt|fed", 2, "ldn", -5),
            mtm="payment",
            initial_exchange=False,
            final_exchange=False,
            amortization=250000.0 if amortization else NoInput(0),
            fx_fixings=name,  # this should not impact the reference currency notional and amortiz
        )
        expected = [3.3, 5.5]
        for i, rp in enumerate(fl._regular_periods):
            # every regular period in a typical leg has an FX fixing date equal to coupon payment dt
            assert rp.non_deliverable_params.fx_fixing.date == (
                get_calendar("ldn").lag_bus_days(fl.schedule.pschedule[i + 1], -5, True)
            )
            assert rp.non_deliverable_params.fx_fixing.value == expected[i]

        fixings.pop(name + "_EURUSD")

    def test_construction_of_relevant_periods_non_deliverable_mtm_exchange(self):
        # when the leg is ND and MTM the FXFixings should be determined at the start of a period.
        # MTM cashflows are generated with notional exchanges between FX fixings at start and end.
        name = str(hash(os.urandom(8)))
        fixings.add(
            name + "_EURUSD",
            Series(
                index=[
                    dt(1999, 12, 24),
                    dt(1999, 12, 29),
                    dt(2000, 3, 29),
                    dt(2000, 3, 30),
                    dt(2000, 6, 28),
                    dt(2000, 6, 29),
                ],
                data=[1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            ),
        )

        fl = FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 5),
                termination=dt(2000, 7, 5),
                frequency="Q",
                payment_lag_exchange=1,
                payment_lag=0,
            ),
            fixed_rate=10.0,
            currency="usd",
            pair=FXIndex("eurusd", "tgt|fed", 2, "ldn", -5),
            mtm=LegMtm.XCS,
            initial_exchange=True,
            final_exchange=True,
            amortization=NoInput(0),
            fx_fixings=name,  # this should not impact the reference currency notional and amortiz
        )
        expected = [2.2, 4.4]
        for i, rp in enumerate(fl._regular_periods):
            assert rp.non_deliverable_params.fx_fixing.date == (
                get_calendar("ldn").lag_bus_days(fl.schedule.pschedule2[i], -5, True)
            )
            assert rp.non_deliverable_params.fx_fixing.value == expected[i]

        # there should be 1 MTM cashflow exchanges:
        assert len(fl._mtm_exchange_periods) == 1
        assert fl._mtm_exchange_periods[0].mtm_params.fx_fixing_start.date == (
            get_calendar("ldn").lag_bus_days(dt(2000, 1, 6), -5, True)
        )
        assert fl._mtm_exchange_periods[0].mtm_params.fx_fixing_end.date == (
            get_calendar("ldn").lag_bus_days(dt(2000, 4, 6), -5, True)
        )

        fixings.pop(name + "_EURUSD")

    def test_construction_of_relevant_periods_non_deliverable_mtm_exchange_amortization(self):
        # when the leg is ND and MTM the FXFixings should be determined at the start of a period.
        # MTM cashflows are generated with notional exchanges between FX fixings at start and end.
        # Amortization has interim cashflows.
        usd = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.95})
        eur = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.075})
        fxf = FXForwards(
            fx_curves={"eureur": eur, "usdusd": usd, "eurusd": eur},
            fx_rates=FXRates({"eurusd": 1.1}, settlement=dt(2000, 1, 1)),
        )
        fl = FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 5),
                termination=dt(2000, 10, 5),
                frequency="Q",
                payment_lag=1,
                payment_lag_exchange=0,
            ),
            convention="actacticma",
            fixed_rate=1.0,
            currency="usd",
            pair=FXIndex("eurusd", "tgt|fed", 2, "ldn", -5),
            initial_exchange=True,
            mtm=LegMtm.XCS,
            notional=-1e6,
            amortization=-2e5,
            fx_fixings=Series(
                index=[
                    dt(1999, 12, 24),
                    dt(1999, 12, 29),
                    dt(2000, 3, 29),
                    dt(2000, 3, 30),
                    dt(2000, 6, 28),
                    dt(2000, 6, 29),
                    dt(2000, 9, 28),
                ],
                data=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            ),
        )
        d1, d2, d3 = dt(1999, 12, 24), dt(2000, 3, 29), dt(2000, 6, 28)
        expected = DataFrame(
            {
                "Type": [
                    "Cashflow",
                    "FixedPeriod",
                    "MtmCashflow",
                    "Cashflow",
                    "FixedPeriod",
                    "MtmCashflow",
                    "Cashflow",
                    "FixedPeriod",
                    "Cashflow",
                ],
                "Notional": [1e6, -1e6, 1e6, -2e5, -8e5, 8e5, -2e5, -6e5, -6e5],
                "Cashflow": [-1.1e6, 2750, -2e5, 2.6e5, 2600, -1.6e5, 3e5, 2250, 9e5],
                "FX Fix Date": [d1, d1, d2, d2, d2, d3, d3, d3, d3],
            }
        )
        result = fl.cashflows(fx=fxf)[["Type", "Notional", "Cashflow", "FX Fix Date"]]
        assert_frame_equal(result, expected)

    def test_ex_div(self):
        leg = FixedLeg(schedule=Schedule(dt(2000, 1, 1), dt(2001, 1, 1), "Q", extra_lag=-3))
        assert not leg.ex_div(dt(2000, 3, 29))
        assert leg.ex_div(dt(2000, 3, 30))
        assert leg.ex_div(dt(2000, 4, 1))

    def test_mtm_xcs_type_type_sets_fx_fixing_start_initially(self):
        fixings.add(
            "EURUSD_1600",
            Series(
                index=[dt(2000, 4, 1), dt(2000, 4, 2), dt(2000, 7, 2)], data=[1.268, 1.27, 1.29]
            ),
        )
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2000, 7, 1),
                frequency="Q",
                payment_lag=1,
                payment_lag_exchange=0,
            ),
            fixed_rate=1.0,
            currency="usd",
            pair="eurusd",
            initial_exchange=True,
            mtm="xcs",
            notional=5e6,
            fx_fixings=(1.25, "EURUSD_1600"),
        )
        assert leg.periods[2].mtm_params.fx_fixing_start.value == 1.25
        fixings.pop("EURUSD_1600")

    ## 4 types of non-deliverability

    @pytest.mark.parametrize(
        ("fx_fixings", "expected"),
        [
            ("ABCD", 1.10),
            (1.5, 1.5),
            ((1.2, "ABCD"), 1.2),
        ],
    )
    def test_non_mtm_xcs_type(self, fx_fixings, expected):
        fixings.add("ABCD_EURUSD", Series(index=[dt(1999, 12, 30)], data=[1.10]))
        fl = FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2000, 3, 1),
                frequency="M",
                payment_lag=2,
                payment_lag_exchange=1,
                calendar="all",
            ),
            currency="usd",
            pair="eurusd",
            mtm="initial",
            initial_exchange=True,
            final_exchange=True,
            fx_fixings=fx_fixings,
        )
        # this leg has 4 periods with only one initial fixing date
        assert fl.periods[0].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
        assert fl.periods[1].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
        assert fl.periods[2].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
        assert fl.periods[3].non_deliverable_params.fx_fixing.date == dt(1999, 12, 30)
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == expected
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == expected
        assert fl.periods[2].non_deliverable_params.fx_fixing.value == expected
        assert fl.periods[3].non_deliverable_params.fx_fixing.value == expected
        fixings.pop("ABCD_EURUSD")

    @pytest.mark.parametrize(
        ("fx_fixings", "expected"),
        [
            ("ABCDE", [1.20, 1.30]),
            (1.5, [1.5, NoInput(0)]),  # this is bad practice: should just supply str ID
            ((1.5, "ABCDE"), [1.5, 1.30]),  # this is bad practice: should just supply str ID
        ],
    )
    def test_irs_nd_type(self, fx_fixings, expected):
        fixings.add(
            "ABCDE_EURUSD",
            Series(
                index=[
                    dt(2000, 1, 5),
                    dt(2000, 2, 3),
                    dt(2000, 2, 4),
                    dt(2000, 3, 3),
                    dt(2000, 3, 6),
                ],
                data=[1.10, 1.20, 1.21, 1.30, 1.31],
            ),
        )
        fl = FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 7),
                termination=dt(2000, 3, 7),
                frequency="M",
                payment_lag=0,
                payment_lag_exchange=1,
                calendar="all",
            ),
            currency="usd",
            pair="eurusd",
            mtm="payment",
            initial_exchange=False,
            final_exchange=False,
            fx_fixings=fx_fixings,
        )
        # this leg has 2 periods and only 2 relevant fixings dates
        assert fl.periods[0].non_deliverable_params.fx_fixing.date == dt(2000, 2, 3)
        assert fl.periods[1].non_deliverable_params.fx_fixing.date == dt(2000, 3, 3)
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == expected[0]
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == expected[1]
        fixings.pop("ABCDE_EURUSD")

    @pytest.mark.parametrize(
        ("fx_fixings", "expected"),
        [
            ("ADE", [1.10, 1.10, 1.20, 1.20, 1.20]),
            (
                1.5,
                [1.5, 1.5, NoInput(0), NoInput(0), NoInput(0)],
            ),  # this is bad practice: should just supply str ID
            (
                (1.5, "ADE"),
                [1.5, 1.5, 1.20, 1.20, 1.20],
            ),  # this is bad practice: should just supply str ID
        ],
    )
    def test_mtm_xcs_nd_type(self, fx_fixings, expected):
        fixings.add(
            "ADE_EURUSD",
            Series(
                index=[
                    dt(2000, 1, 6),
                    dt(2000, 2, 4),
                    dt(2000, 2, 8),
                    dt(2000, 3, 7),
                    dt(2000, 3, 8),
                ],
                data=[1.10, 1.20, 1.21, 1.30, 1.31],
            ),
        )
        fl = FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 7),
                termination=dt(2000, 3, 7),
                frequency="M",
                payment_lag=2,
                payment_lag_exchange=1,
                calendar="all",
            ),
            currency="usd",
            pair="eurusd",
            mtm=LegMtm.XCS,
            initial_exchange=True,
            final_exchange=True,
            fx_fixings=fx_fixings,
        )
        # this leg has 5 periods with only two relevant fixing dates
        assert fl.periods[0].non_deliverable_params.fx_fixing.date == dt(2000, 1, 6)
        assert fl.periods[1].non_deliverable_params.fx_fixing.date == dt(2000, 1, 6)
        assert fl.periods[2].mtm_params.fx_fixing_end.date == dt(2000, 2, 4)
        assert fl.periods[3].non_deliverable_params.fx_fixing.date == dt(2000, 2, 4)
        assert fl.periods[4].non_deliverable_params.fx_fixing.date == dt(2000, 2, 4)
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == expected[0]
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == expected[1]
        assert fl.periods[2].mtm_params.fx_fixing_end.value == expected[2]
        assert fl.periods[3].non_deliverable_params.fx_fixing.value == expected[3]
        assert fl.periods[4].non_deliverable_params.fx_fixing.value == expected[4]
        fixings.pop("ADE_EURUSD")

    @pytest.mark.parametrize(
        ("fx_fixings", "expected"),
        [
            ("AXDE", [1.10, 1.21, 1.31, 1.30]),
            (
                1.5,
                [1.5, NoInput(0), NoInput(0), NoInput(0)],
            ),  # this is bad practice: should just supply str ID
            (
                (1.5, "AXDE"),
                [1.5, 1.21, 1.31, 1.30],
            ),  # this is bad practice: should just supply str ID
        ],
    )
    def test_non_mtm_xcs_nd_type(self, fx_fixings, expected):
        fixings.add(
            "AXDE_EURUSD",
            Series(
                index=[
                    dt(2000, 1, 5),
                    dt(2000, 2, 3),
                    dt(2000, 2, 4),
                    dt(2000, 3, 3),
                    dt(2000, 3, 6),
                ],
                data=[1.10, 1.20, 1.21, 1.30, 1.31],
            ),
        )
        fl = FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 7),
                termination=dt(2000, 3, 7),
                frequency="M",
                payment_lag=1,
                payment_lag_exchange=0,
                calendar="all",
            ),
            currency="usd",
            pair="eurusd",
            mtm="payment",
            initial_exchange=True,
            final_exchange=True,
            fx_fixings=fx_fixings,
        )
        # this leg has 4 periods with 3 or 4 (if lag exchange is different) relevant fixing dates.
        assert fl.periods[0].non_deliverable_params.fx_fixing.date == dt(2000, 1, 5)
        assert fl.periods[1].non_deliverable_params.fx_fixing.date == dt(2000, 2, 4)
        assert fl.periods[2].non_deliverable_params.fx_fixing.date == dt(2000, 3, 6)
        assert fl.periods[3].non_deliverable_params.fx_fixing.date == dt(2000, 3, 3)
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == expected[0]
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == expected[1]
        assert fl.periods[2].non_deliverable_params.fx_fixing.value == expected[2]
        assert fl.periods[3].non_deliverable_params.fx_fixing.value == expected[3]
        fixings.pop("AXDE_EURUSD")


class TestCreditPremiumLeg:
    @pytest.mark.parametrize(
        ("premium_accrued", "exp"), [(True, 41357.455568685626), (False, 41330.94188109829)]
    )
    def test_premium_leg_analytic_delta(self, hazard_curve, curve, premium_accrued, exp) -> None:
        leg = CreditPremiumLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=1e9,
            convention="Act360",
            premium_accrued=premium_accrued,
        )
        result = leg.analytic_delta(rate_curve=hazard_curve, disc_curve=curve)
        assert abs(result - exp) < 1e-7

    @pytest.mark.parametrize(("premium_accrued"), [True, False])
    def test_premium_leg_npv(self, hazard_curve, curve, premium_accrued) -> None:
        leg = CreditPremiumLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=1e9,
            convention="Act360",
            premium_accrued=premium_accrued,
            fixed_rate=4.00,
        )
        result = leg.npv(rate_curve=hazard_curve, disc_curve=curve)
        assert (
            abs(result + 400 * leg.analytic_delta(rate_curve=hazard_curve, disc_curve=curve)) < 1e-7
        )

    def test_premium_leg_cashflows(self, hazard_curve, curve) -> None:
        leg = CreditPremiumLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            fixed_rate=4.00,
        )
        result = leg.cashflows(rate_curve=hazard_curve, disc_curve=curve)
        # test a couple of return elements
        assert abs(result.loc[0, defaults.headers["cashflow"]] - 6555555.55555) < 1e-4
        assert abs(result.loc[1, defaults.headers["df"]] - 0.98307) < 1e-4
        assert abs(result.loc[1, defaults.headers["notional"]] + 1e9) < 1e-7

    def test_premium_leg_set_fixed_rate(self, curve) -> None:
        leg = CreditPremiumLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
        )
        assert leg.fixed_rate is NoInput(0)
        assert leg.periods[0].rate_params.fixed_rate is NoInput(0)

        leg.fixed_rate = 2.0
        assert leg.fixed_rate == 2.0
        assert leg.periods[0].rate_params.fixed_rate == 2.0

    @pytest.mark.parametrize(
        ("date", "exp"),
        [
            (dt(2022, 2, 1), 1e9 * 0.02 * 0.25 * 31 / 90),
            (dt(2022, 3, 1), 0.0),
            (dt(2022, 6, 1), 0.0),
        ],
    )
    def test_premium_leg_accrued(self, date, exp):
        leg = CreditPremiumLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="ActActICMA",
            fixed_rate=2.0,
        )
        result = leg.accrued(date)
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize("final", [True, False])
    def test_exchanges_raises(self, final):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            CreditPremiumLeg(
                schedule=Schedule(
                    effective=dt(2022, 1, 1),
                    termination=dt(2022, 6, 1),
                    payment_lag=2,
                    frequency="Q",
                ),
                notional=-1e9,
                convention="ActActICMA",
                fixed_rate=2.0,
                initial_exchange=final,
                final_exchange=not final,
            )

    @pytest.mark.parametrize(
        ("settlement", "forward", "exp"),
        [
            (NoInput(0), NoInput(0), 408.02994815795125),
            (dt(2022, 3, 30), dt(2022, 3, 30), 404.03987718823055),
            (dt(2022, 4, 6), dt(2022, 4, 6), 811.1815703665554),
        ],
    )
    def test_fixed_leg_spread(self, settlement, forward, exp, curve) -> None:
        fixed_leg = CreditPremiumLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 7, 1),
                payment_lag=2,
                payment_lag_exchange=1,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            fixed_rate=4.00,
            currency="usd",
        )
        result = fixed_leg.spread(
            target_npv=20000000,
            disc_curve=curve,
            rate_curve=curve,
            index_curve=curve,
            settlement=settlement,
            forward=forward,
        )
        assert abs(result - exp) < 1e-6

    def test_ex_div(self):
        leg = CreditPremiumLeg(schedule=Schedule(dt(2000, 1, 1), dt(2001, 1, 1), "Q", extra_lag=-3))
        assert not leg.ex_div(dt(2000, 3, 29))
        assert leg.ex_div(dt(2000, 3, 30))
        assert leg.ex_div(dt(2000, 4, 1))


class TestCreditProtectionLeg:
    def test_leg_analytic_delta(self, hazard_curve, curve) -> None:
        leg = CreditProtectionLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=1e9,
        )
        result = leg.analytic_delta(rate_curve=hazard_curve, disc_curve=curve)
        assert abs(result) < 1e-7

    def test_leg_analytic_rec_risk(self, hazard_curve, curve) -> None:
        leg = CreditProtectionLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2027, 1, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=1e7,
        )
        result = leg.analytic_rec_risk(rate_curve=hazard_curve, disc_curve=curve)

        pv0 = leg.npv(rate_curve=hazard_curve, disc_curve=curve)
        hazard_curve.update_meta("credit_recovery_rate", 0.41)
        pv1 = leg.npv(rate_curve=hazard_curve, disc_curve=curve)
        expected = pv1 - pv0
        assert abs(result - expected) < 1e-7

    @pytest.mark.parametrize(("premium_accrued"), [True, False])
    def test_leg_npv(self, hazard_curve, curve, premium_accrued) -> None:
        leg = CreditProtectionLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Z",
            ),
            notional=1e9,
        )
        result = leg.npv(rate_curve=hazard_curve, disc_curve=curve)
        expected = -1390922.0390295777  # with 1 cds_discretization this is -1390906.242843
        assert abs(result - expected) < 1e-7

    def test_leg_cashflows(self, hazard_curve, curve) -> None:
        leg = CreditProtectionLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
        )
        result = leg.cashflows(rate_curve=hazard_curve, disc_curve=curve)
        # test a couple of return elements
        assert abs(result.loc[0, defaults.headers["cashflow"]] - 600e6) < 1e-4
        assert abs(result.loc[1, defaults.headers["df"]] - 0.98307) < 1e-4
        assert abs(result.loc[1, defaults.headers["notional"]] + 1e9) < 1e-7

    def test_leg_zero_sched(self):
        leg = CreditProtectionLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2024, 6, 1),
                frequency="Z",
            ),
            notional=-1e9,
            convention="Act360",
        )
        assert len(leg.periods) == 1
        assert leg.periods[0].period_params.end == dt(2024, 6, 1)


class TestIndexFixedLegExchange:
    @pytest.mark.parametrize(
        "i_fixings",
        [
            NoInput(0),
            # [210, 220, 230], # list not supported in v2.0
            # 210, # dualtypes is not supported as of v2.2
            Series(
                [210.0, 220.0, 230.0],
                index=[dt(2022, 6, 15), dt(2022, 9, 15), dt(2022, 12, 15)],
            ),
        ],
    )
    def test_idx_leg_cashflows(self, i_fixings) -> None:
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 3, 15),
                termination="9M",
                frequency="Q",
                payment_lag=0,
            ),
            convention="ActActICMA",
            notional=40e6,
            fixed_rate=5.0,
            index_base=200.0,
            index_lag=0,
            index_fixings=i_fixings,
            initial_exchange=False,
            final_exchange=True,
            index_method="curve",
        )
        index_curve = Curve(
            nodes={
                dt(2022, 3, 15): 1.0,
                dt(2022, 6, 15): 1.0 / 1.05,
                dt(2022, 9, 15): 1.0 / 1.10,
                dt(2022, 12, 15): 1.0 / 1.15,
            },
            index_base=200.0,
            interpolation="linear_index",
            index_lag=0,
        )
        disc_curve = Curve({dt(2022, 3, 15): 1.0, dt(2022, 12, 15): 1.0})
        flows = leg.cashflows(index_curve=index_curve, disc_curve=disc_curve)

        def equals_with_tol(a, b):
            if isinstance(a, str):
                return a == b
            else:
                return abs(a - b) < 1e-7

        expected = {
            "Type": "FixedPeriod",
            "DCF": 0.250,
            "Notional": 40e6,
            "Rate": 5.0,
            "Unindexed Cashflow": -500e3,
            "Index Val": 210.0,
            "Index Ratio": 1.05,
            "Cashflow": -525000,
        }
        flow = flows.iloc[0].to_dict()
        for key in set(expected.keys()) & set(flow.keys()):
            assert equals_with_tol(expected[key], flow[key])

        final_flow = flows.iloc[3].to_dict()
        expected = {
            "Type": "Cashflow",
            "Notional": 40e6,
            "Unindexed Cashflow": -40e6,
            "Index Val": 230.0,
            "Index Ratio": 1.15,
            "Cashflow": -46e6,
        }
        for key in set(expected.keys()) & set(final_flow.keys()):
            assert equals_with_tol(expected[key], final_flow[key])

    def test_args_raises(self) -> None:
        with pytest.raises(ValueError, match="`index_method` as string: 'BAD' is not "):
            FixedLeg(
                schedule=Schedule(
                    effective=dt(2022, 3, 15),
                    termination="9M",
                    frequency="Q",
                ),
                index_base=200.0,
                index_method="BAD",
                initial_exchange=True,
                final_exchange=True,
            )

    @pytest.mark.skip(reason="v2.2 removed the ability to mutate `index_base` at period level.")
    def test_set_index_leg_after_init(self) -> None:
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 3, 15),
                termination="9M",
                frequency="Q",
                payment_lag=0,
            ),
            convention="ActActICMA",
            notional=40e6,
            fixed_rate=5.0,
            index_base=None,
            initial_exchange=False,
            final_exchange=True,
        )
        for period in leg.periods:
            assert period.index_base is None
        leg.index_base = 205.0
        for period in leg.periods:
            assert period.index_base == 205.0

    def test_npv(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
        index_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=100.0,
            interpolation="linear_index",
            index_lag=3,
        )
        index_leg_exch = FixedLeg(
            schedule=Schedule(
                dt(2022, 1, 1),
                "9M",
                "Q",
                payment_lag=2,
                payment_lag_exchange=0,
            ),
            notional=1000000,
            amortization=200000,
            index_base=100.0,
            initial_exchange=False,
            fixed_rate=1.0,
            final_exchange=True,
            index_lag=3,
        )
        result = index_leg_exch.npv(index_curve=index_curve, disc_curve=curve)
        expected = -999993.7970219046
        assert abs(result - expected) < 1e-4

    def test_index_lag_on_periods(self):
        index_leg_exch = FixedLeg(
            schedule=Schedule(
                dt(2022, 1, 1),
                "6M",
                "Q",
            ),
            notional=1000000,
            amortization=200000,
            index_base=100.0,
            fixed_rate=1.0,
            final_exchange=True,
            index_lag=4,
        )
        for period in index_leg_exch.periods:
            assert period.index_params.index_lag == 4


class TestIndexFixedLeg:
    @pytest.mark.parametrize(
        ("i_fixings", "meth"),
        [
            (NoInput(0), "daily"),
            # ([210, 220, 230], "daily"), # list unsupported in v2.0
            # (210, "daily"),  # dualtypes unsupported as of v2.2
            (
                Series(
                    [210.0, 210, 220, 220, 230, 230],
                    index=[
                        dt(2022, 6, 1),
                        dt(2022, 7, 1),
                        dt(2022, 9, 1),
                        dt(2022, 10, 1),
                        dt(2022, 12, 1),
                        dt(2023, 1, 1),
                    ],
                ),
                "daily",
            ),
            (
                Series(
                    [210.0, 220, 230],
                    index=[dt(2022, 6, 1), dt(2022, 9, 1), dt(2022, 12, 1)],
                ),
                "monthly",
            ),
        ],
    )
    def test_idx_leg_cashflows(self, i_fixings, meth) -> None:
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 3, 15),
                termination="9M",
                frequency="Q",
                payment_lag=0,
            ),
            convention="ActActICMA",
            notional=40e6,
            fixed_rate=5.0,
            index_base=200.0,
            index_fixings=i_fixings,
            index_method=meth,
            index_lag=0,
        )
        index_curve = Curve(
            nodes={
                dt(2022, 3, 15): 1.0,
                dt(2022, 6, 15): 1.0 / 1.05,
                dt(2022, 9, 15): 1.0 / 1.10,
                dt(2022, 12, 15): 1.0 / 1.15,
            },
            index_base=200.0,
            interpolation="linear_index",
            index_lag=0,
        )
        disc_curve = Curve({dt(2022, 3, 15): 1.0, dt(2022, 12, 15): 1.0})
        flows = leg.cashflows(index_curve=index_curve, disc_curve=disc_curve)

        def equals_with_tol(a, b):
            if isinstance(a, str):
                return a == b
            else:
                return abs(a - b) < 1e-7

        expected = {
            "Type": "FixedPeriod",
            "DCF": 0.250,
            "Notional": 40e6,
            "Rate": 5.0,
            "Unindexed Cashflow": -500e3,
            "Index Val": 210.0,
            "Index Ratio": 1.05,
            "Cashflow": -525000,
        }
        flow = flows.iloc[0].to_dict()
        for key in set(expected.keys()) & set(flow.keys()):
            assert equals_with_tol(expected[key], flow[key])

    @pytest.mark.parametrize(("meth", "exp"), [("daily", 230.0), ("monthly", 227.91208)])
    def test_missing_fixings(self, meth, exp) -> None:
        i_fixings = Series(
            [210.0, 210, 220, 220],
            index=[dt(2022, 6, 1), dt(2022, 7, 1), dt(2022, 9, 1), dt(2022, 10, 1)],
        )
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 3, 20),
                termination="9M",
                frequency="Q",
                payment_lag=0,
            ),
            convention="ActActICMA",
            notional=40e6,
            fixed_rate=5.0,
            index_base=200.0,
            index_fixings=i_fixings,
            index_method=meth,
            index_lag=0,
        )
        index_curve = Curve(
            nodes={
                dt(2022, 3, 20): 1.0,
                dt(2022, 6, 20): 1.0 / 1.05,
                dt(2022, 9, 20): 1.0 / 1.10,
                dt(2022, 12, 20): 1.0 / 1.15,
            },
            index_base=200.0,
            interpolation="linear_index",
            index_lag=0,
        )
        cashflows = leg.cashflows(index_curve=index_curve)
        result = cashflows.iloc[2]["Index Val"]
        assert abs(result - exp) < 1e-3

    @pytest.mark.skip(reason="v2.2 removed the ability to mutate `index_base` at period level.")
    def test_set_index_leg_after_init(self) -> None:
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 3, 15),
                termination="9M",
                frequency="Q",
                payment_lag=0,
            ),
            convention="ActActICMA",
            notional=40e6,
            fixed_rate=5.0,
            index_base=None,
        )
        for period in leg.periods:
            assert period.index_params.index_base is None
        leg.index_base = 205.0
        for period in leg.periods:
            assert period.index_params.index_base == 205.0

    @pytest.mark.skip(reason="v2.2 removed the ability to mutate `index_base` at period level.")
    @pytest.mark.parametrize(
        "i_base",
        [
            200.0,
            Series([199.0, 201.0], index=[dt(2022, 4, 1), dt(2022, 5, 1)]),
        ],
    )
    def test_set_index_base(self, curve, i_base) -> None:
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 4, 16),
                termination=dt(2022, 5, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            index_method="daily",
            index_lag=0,
        )
        assert leg.periods[0].index_params.index_base == NoInput(0)

        leg.index_base = i_base
        assert leg.periods[0].index_base == 200.0

    @pytest.mark.parametrize(
        ("i_base", "exp"),
        [
            (NoInput(0), NoInput(0)),
            (110.0, 110.0),
        ],
    )
    def test_initialise_index_base(self, i_base, exp) -> None:
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            index_base=i_base,
            index_lag=0,
        )
        assert leg.periods[-1].index_params.index_base.value == exp

    @pytest.mark.parametrize(
        ("i_base", "exp"),
        [
            (Series([199.0, 200.0], index=[dt(2021, 12, 31), dt(2022, 1, 1)]), 200.0),
            (Series([1.0, 2.0], index=[dt(2000, 1, 1), dt(2000, 12, 1)]), NoInput(0)),
        ],
    )
    def test_initialise_index_base2(self, i_base, exp) -> None:
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                frequency="Q",
            ),
            notional=-1e9,
            convention="Act360",
            index_fixings=i_base,
            index_lag=0,
        )
        assert leg.periods[-1].index_params.index_base.value == exp

    @pytest.mark.skip(reason="fixings as list removed in v2.0")
    def test_index_fixings_as_list(self) -> None:
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 10, 1),
                payment_lag=2,
                frequency="Q",
            ),
            convention="Act360",
            notional=1e6,
            amortization=250e3,
            index_base=NoInput(0),
            index_fixings=[100.0, 200.0],
        )
        assert leg.periods[0].index_fixings == 100.0
        assert leg.periods[1].index_fixings == 200.0
        assert leg.periods[2].index_fixings == NoInput(0)

    @pytest.mark.skip(reason="fixings as list removed in v2.0")
    def test_index_fixings_as_list_final_exchange(self) -> None:
        leg = FixedLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 10, 1),
            payment_lag=2,
            convention="Act360",
            frequency="Q",
            notional=1e6,
            amortization=250e3,
            index_base=NoInput(0),
            index_fixings=[100.0, 100.0, 200.0, 199.0],
            final_exchange=True,
        )
        assert leg.periods[0].index_fixings == 100.0
        assert leg.periods[1].index_fixings == 100.0
        assert leg.periods[2].index_fixings == 200.0
        assert leg.periods[3].index_fixings == 199.0
        assert leg.periods[4].index_fixings == NoInput(0)
        assert leg.periods[5].index_fixings == NoInput(0)

    @pytest.mark.skip(reason="v2.2 refactor fixings, + input as Series was stated as bad practice")
    @pytest.mark.parametrize(
        "index_fixings",
        [
            Series([1, 2, 3], index=[dt(2000, 1, 1), dt(1999, 1, 1), dt(2001, 1, 1)]),
            Series([1, 2, 3], index=[dt(2000, 1, 1), dt(2000, 1, 1), dt(2001, 1, 1)]),
        ],
    )
    def test_index_as_series_invalid(self, index_fixings):
        with pytest.raises(ValueError, match="`index_fixings` as Series must be"):
            FixedLeg(
                schedule=Schedule(
                    effective=dt(2022, 1, 1),
                    termination=dt(2022, 10, 1),
                    frequency="Q",
                ),
                index_base=NoInput(0),
                index_fixings=index_fixings,
            )

    @pytest.mark.skip(reason="v2.2 refactor fixings, + input as Series was stated as bad practice")
    def test_index_reverse_monotonic_decreasing_series(self):
        s = Series([1, 2, 3], index=[dt(2000, 1, 1), dt(1999, 1, 1), dt(1998, 1, 1)])
        assert s.index.is_monotonic_decreasing
        leg = FixedLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 10, 1),
                frequency="Q",
            ),
            index_base=NoInput(0),
            index_fixings=s,
        )
        assert leg.index_fixings.index.is_monotonic_increasing


class TestFloatLegExchangeMtm:
    @pytest.mark.parametrize(
        ("fx_fixings", "exp"),
        [
            (NoInput(0), [NoInput(0), NoInput(0), NoInput(0)]),
            ([1.5], [1.5, NoInput(0), NoInput(0)]),
            (1.25, [1.25, NoInput(0), NoInput(0)]),
            ([1.25, 1.35], [1.25, 1.35, NoInput(0)]),
            (Series([1.25, 1.3], index=[dt(2022, 1, 4), dt(2022, 4, 4)]), [1.25, 1.3, NoInput(0)]),
            (Series([1.25], index=[dt(2022, 1, 4)]), [1.25, NoInput(0), NoInput(0)]),
        ],
    )
    def test_float_leg_exchange_mtm(self, fx_fixings, exp) -> None:
        float_leg_exch = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 3),
                termination=dt(2022, 7, 3),
                frequency="Q",
                payment_lag_exchange=3,
            ),
            float_spread=5.0,
            currency="usd",
            pair="eurusd",
            notional=10e6,
            fx_fixings=fx_fixings,
            mtm="xcs",
            initial_exchange=True,
        )
        fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
        fxf = FXForwards(
            fxr,
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
            },
        )

        d = [
            dt(2022, 1, 6),
            dt(2022, 4, 6),
            dt(2022, 7, 6),
        ]  # payment_lag_exchange is 3 days.
        rate = [_ if _ is not NoInput(0) else fxf.rate("eurusd", d[i]) for i, _ in enumerate(exp)]

        float_leg_exch.cashflows(
            rate_curve=fxf.curve("usd", "usd"), disc_curve=fxf.curve("usd", "usd"), fx=fxf
        )
        assert (
            float(float_leg_exch.periods[0].try_cashflow(fx=fxf).unwrap() - 10e6 * rate[0]) < 1e-6
        )
        assert (
            float(
                float_leg_exch.periods[2].try_cashflow(fx=fxf).unwrap() - 10e6 * (rate[1] - rate[0])
            )
            < 1e-6
        )
        assert (
            float(
                float_leg_exch.periods[4].try_cashflow(fx=fxf).unwrap() - 10e6 * (rate[2] - rate[1])
            )
            < 1e-6
        )
        assert float_leg_exch.periods[4].settlement_params.payment == d[-1]

        assert float_leg_exch.periods[1].settlement_params.notional == 10e6
        assert float_leg_exch.periods[1].non_deliverable_params.fx_fixing.value == exp[0]
        assert float_leg_exch.periods[1].non_deliverable_params.fx_fixing.date == dt(2022, 1, 4)

        assert type(float_leg_exch.periods[1]) is FloatPeriod
        assert float_leg_exch.periods[3].settlement_params.notional == 10e6
        assert float_leg_exch.periods[3].non_deliverable_params.fx_fixing.value == exp[1]
        assert float_leg_exch.periods[3].non_deliverable_params.fx_fixing.date == dt(2022, 4, 4)
        assert type(float_leg_exch.periods[3]) is FloatPeriod

        assert float_leg_exch.periods[-1].settlement_params.notional == 10e6
        assert float_leg_exch.periods[-1].non_deliverable_params.fx_fixing.value == exp[1]
        assert float_leg_exch.periods[-1].non_deliverable_params.fx_fixing.date == dt(2022, 4, 4)

    def test_float_leg_exchange_fixings_table(self) -> None:
        float_leg_exch = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 3),
                termination=dt(2022, 7, 3),
                frequency="Q",
                payment_lag_exchange=3,
            ),
            float_spread=5.0,
            currency="usd",
            pair="eurusd",
            notional=10e6,
            fixing_method="ibor",
            method_param=0,
            mtm="xcs",
            initial_exchange=True,
        )
        fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
        fxf = FXForwards(
            fxr,
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
            },
        )

        result = float_leg_exch.local_analytic_rate_fixings(
            rate_curve=fxf.curve("usd", "usd"), fx=fxf
        )
        assert isinstance(result, DataFrame)
        assert isinstance(result.iloc[0, 0], Dual)
        assert abs(result.iloc[0, 0] + 260.1507) < 1e-3
        assert abs(result.iloc[1, 0] + 262.1683) < 1

    def test_float_leg_exchange_fixings_table_rfr(self) -> None:
        float_leg_exch = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 3),
                termination=dt(2022, 7, 3),
                frequency="Q",
                payment_lag_exchange=0,
            ),
            float_spread=5.0,
            currency="usd",
            pair="eurusd",
            notional=10e6,
            mtm="xcs",
            initial_exchange=True,
        )
        fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
        fxf = FXForwards(
            fxr,
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
            },
        )

        result = float_leg_exch.local_analytic_rate_fixings(
            rate_curve=fxf.curve("usd", "usd"), disc_curve=fxf.curve("usd", "usd"), fx=fxf
        )
        assert isinstance(result, DataFrame)
        assert isinstance(result.iloc[0, 0], Dual)  # Dual is converted to float for fixings table
        assert result.columns.values[0] == (fxf.curve("usd", "usd").id, "usd", "usd", "1B")

    def test_mtm_leg_exchange_spread(self) -> None:
        leg = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 3),
                termination=dt(2022, 7, 3),
                frequency="Q",
                payment_lag=0,
                payment_lag_exchange=0,
            ),
            currency="usd",
            pair="eurusd",
            notional=1e9,
            fixing_method="rfr_payment_delay",
            spread_compound_method="isda_compounding",
            float_spread=0.0,
            mtm="xcs",
            initial_exchange=True,
        )
        fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
        fxf = FXForwards(
            fxr,
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
            },
        )

        npv = leg.npv(
            rate_curve=fxf.curve("usd", "usd"), disc_curve=fxf.curve("usd", "usd"), fx=fxf
        )
        # a_delta = leg.analytic_delta(fxf.curve("usd", "usd"), fxf.curve("usd", "usd"), fxf)
        result = leg.spread(
            target_npv=100,
            rate_curve=fxf.curve("usd", "usd"),
            disc_curve=fxf.curve("usd", "usd"),
            fx=fxf,
        )
        leg.float_spread = result
        npv2 = leg.npv(
            rate_curve=fxf.curve("usd", "usd"), disc_curve=fxf.curve("usd", "usd"), fx=fxf
        )
        assert abs(npv2 - npv - 100) < 0.01

    @pytest.mark.parametrize(
        ("fx_fixings", "exp"),
        [
            (NoInput(0), [NoInput(0), NoInput(0), NoInput(0)]),
            ([1.5], [1.5, NoInput(0), NoInput(0)]),
            (1.25, [1.25, NoInput(0), NoInput(0)]),
        ],
    )
    def test_mtm_leg_fx_fixings_warn_raise(self, curve, fx_fixings, exp) -> None:
        float_leg_exch = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 3),
                termination=dt(2022, 7, 3),
                frequency="Q",
                payment_lag_exchange=3,
            ),
            float_spread=5.0,
            currency="usd",
            pair="eurusd",
            notional=10e6,
            fx_fixings=fx_fixings,
            mtm="xcs",
            initial_exchange=True,
        )
        with pytest.raises(ValueError, match="Must provide `fx` argument to forecast FXFixing."):
            float_leg_exch.npv(rate_curve=curve)

    def test_mtm_leg_fx_fixings_series_raises(self, curve) -> None:
        fl = FloatLeg(
            schedule=Schedule(
                effective=dt(2022, 1, 3),
                termination=dt(2022, 7, 3),
                frequency="Q",
                payment_lag_exchange=3,
            ),
            float_spread=5.0,
            currency="usd",
            pair="eurusd",
            notional=10e6,
            fx_fixings=Series([1.25], index=[dt(2022, 2, 6)]),
            mtm="xcs",
            initial_exchange=True,
        )
        with pytest.raises(ValueError, match="Must provide `fx` argument to forecast FXFixing."):
            fl.npv(rate_curve=curve)
        # assert False  # TODO: this test should possibly fail if the FX is before the series range.
        # although a FixingsRangeError is detected and the ixing value accepted is NoInput

    def test_mtm_raises_alt(self) -> None:
        with pytest.raises(ValueError, match="A non-deliverable pair must contain the settlement "):
            FloatLeg(
                schedule=Schedule(
                    effective=dt(2022, 1, 3),
                    termination=dt(2022, 7, 3),
                    frequency="Q",
                    payment_lag_exchange=3,
                ),
                float_spread=5.0,
                currency="usd",
                pair=FXIndex("eursek", "tgt,stk|fed", 2),
                notional=10e6,
            )


class TestCustomLeg:
    @pytest.mark.parametrize(
        "period",
        [
            FixedPeriod(
                start=dt(2022, 1, 1),
                end=dt(2023, 1, 1),
                payment=dt(2023, 1, 9),
                frequency=Frequency.Months(12, None),
                fixed_rate=1.0,
            ),
            FloatPeriod(
                start=dt(2022, 1, 1),
                end=dt(2022, 4, 1),
                payment=dt(2022, 4, 3),
                notional=1e9,
                convention="Act360",
                termination=dt(2022, 4, 1),
                frequency=Frequency.Months(3, None),
                float_spread=10.0,
            ),
            CreditPremiumPeriod(
                start=dt(2022, 1, 1),
                end=dt(2022, 4, 1),
                payment=dt(2022, 4, 3),
                notional=1e9,
                convention="Act360",
                termination=dt(2022, 4, 1),
                frequency=Frequency.Months(3, None),
                fixed_rate=4.0,
                currency="usd",
            ),
            CreditProtectionPeriod(
                start=dt(2022, 1, 1),
                end=dt(2022, 4, 1),
                payment=dt(2022, 4, 3),
                notional=1e9,
                convention="Act360",
                termination=dt(2022, 4, 1),
                frequency=Frequency.Months(3, None),
                currency="usd",
            ),
            Cashflow(notional=1e9, payment=dt(2022, 4, 3)),
        ],
    )
    def test_init(self, curve, period) -> None:
        CustomLeg(periods=[period, period])

    def test_npv(self, curve) -> None:
        cl = CustomLeg(
            periods=[
                FixedPeriod(
                    start=dt(2022, 1, 1),
                    end=dt(2023, 1, 1),
                    payment=dt(2023, 1, 9),
                    frequency=Frequency.Months(12, None),
                    fixed_rate=1.0,
                ),
                FixedPeriod(
                    start=dt(2022, 2, 1),
                    end=dt(2023, 2, 1),
                    payment=dt(2023, 2, 9),
                    frequency=Frequency.Months(12, None),
                    fixed_rate=2.0,
                ),
            ],
        )
        result = cl.npv(rate_curve=curve)
        expected = -29109.962157023772
        assert abs(result - expected) < 1e-6

    def test_cashflows(self, curve) -> None:
        cl = CustomLeg(
            periods=[
                FixedPeriod(
                    start=dt(2022, 1, 1),
                    end=dt(2023, 1, 1),
                    payment=dt(2023, 1, 9),
                    frequency=Frequency.Months(12, None),
                    fixed_rate=1.0,
                ),
                FixedPeriod(
                    start=dt(2022, 2, 1),
                    end=dt(2023, 2, 1),
                    payment=dt(2023, 2, 9),
                    frequency=Frequency.Months(12, None),
                    fixed_rate=2.0,
                ),
            ],
        )
        result = cl.cashflows(rate_curve=curve)
        assert isinstance(result, DataFrame)
        assert len(result.index) == 2

    def test_analytic_delta(self, curve) -> None:
        cl = CustomLeg(
            periods=[
                FixedPeriod(
                    start=dt(2022, 1, 1),
                    end=dt(2023, 1, 1),
                    payment=dt(2023, 1, 9),
                    frequency=Frequency.Months(12, None),
                    fixed_rate=1.0,
                ),
                FixedPeriod(
                    start=dt(2022, 2, 1),
                    end=dt(2023, 2, 1),
                    payment=dt(2023, 2, 9),
                    frequency=Frequency.Months(12, None),
                    fixed_rate=2.0,
                ),
            ],
        )
        result = cl.analytic_delta(rate_curve=curve)
        expected = 194.1782607729773
        assert abs(result - expected) < 1e-6


class TestNonDeliverableFixedLeg:
    def test_set_periods(self):
        leg = FixedLeg(
            schedule=Schedule(dt(2000, 1, 1), dt(2000, 3, 1), "M"),
            fixed_rate=2.0,
            currency="usd",
            pair=FXIndex("brlusd", "all", 0),
        )
        assert len(leg.periods) == 2

    def test_npv(self):
        fxr = FXRates({"usdbrl": 9.50}, settlement=dt(2022, 1, 3))
        fxf = FXForwards(
            fxr,
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
                "brlbrl": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
                "brlusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
            },
        )
        leg = FixedLeg(
            schedule=Schedule(dt(2022, 1, 1), dt(2022, 3, 1), "M"),
            fixed_rate=2.0,
            currency="usd",
            pair=FXIndex("brlusd", "all", 0),
            notional=1e6,  # 1mm BRL
            mtm="payment",
        )
        result = leg.npv(disc_curve=fxf.curve("usd", "brl"), fx=fxf)
        expected = -344.326093  #  2.0% * 1mm * (2 / 12) / 9.5
        assert abs(result - expected) < 1e-6

        result = leg.npv(disc_curve=fxf.curve("usd", "brl"), fx=fxf, base="brl")
        expected = -344.326093 * fxf.rate("usdbrl")  # 2.0% * 1mm * (2 / 12) / 9.5
        assert abs(result - expected) < 1e-5

    @pytest.mark.parametrize("fixings", [[1.66], 1.66, Series(data=[1.66], index=[dt(2022, 2, 3)])])
    def test_set_fixings(self, fixings):
        leg = FixedLeg(
            schedule=Schedule(dt(2022, 1, 1), dt(2022, 3, 1), "M"),
            fixed_rate=2.0,
            currency="usd",
            pair=FXIndex("brlusd", "all", 0),
            notional=1e6,  # 1mm BRL
            fx_fixings=fixings,
            mtm="payment",
        )
        assert leg.periods[0].non_deliverable_params.fx_fixing.value == 1.66
        assert leg.periods[1].non_deliverable_params.fx_fixing.value == NoInput(0)


class TestAmortization:
    def test_percent(self):
        a = Amortization(4, 100.0, "20%")
        assert a.outstanding == (100.0, 80.0, 64.0, 51.2)
        assert a.amortization == (20.0, 16.0, 12.8)
        assert a._type == _AmortizationType.CustomSchedule

    def test_to_zero(self):
        a = Amortization(4, 100.0, "to_zero")
        assert a.outstanding == (100.0, 75.0, 50.0, 25.0)
        assert a.amortization == (25.0, 25.0, 25.0)
        assert a._type == _AmortizationType.ConstantPeriod

    def test_custom(self):
        a = Amortization(4, 100.0, [10.0, 20.0, 30.0])
        assert a.outstanding == (100.0, 90.0, 70.0, 40.0)
        assert a.amortization == (10.0, 20.0, 30.0)
        assert a._type == _AmortizationType.CustomSchedule


def test_leg_amortization() -> None:
    fixed_leg = FixedLeg(
        schedule=Schedule(
            dt(2022, 1, 1),
            dt(2022, 10, 1),
            frequency="Q",
        ),
        notional=1e6,
        amortization=250e3,
        fixed_rate=2.0,
    )
    for i, period in enumerate(fixed_leg.periods):
        assert period.settlement_params.notional == 1e6 - 250e3 * i

    float_leg = FloatLeg(
        schedule=Schedule(
            dt(2022, 1, 1),
            dt(2022, 10, 1),
            frequency="Q",
        ),
        notional=1e6,
        amortization=250e3,
        float_spread=2.0,
    )
    for i, period in enumerate(float_leg.periods):
        assert period.settlement_params.notional == 1e6 - 250e3 * i

    index_leg = FixedLeg(
        schedule=Schedule(
            dt(2022, 1, 1),
            dt(2022, 10, 1),
            frequency="Q",
        ),
        notional=1e6,
        amortization=250e3,
        fixed_rate=2.0,
        index_base=100.0,
    )
    for i, period in enumerate(index_leg.periods):
        assert period.settlement_params.notional == 1e6 - 250e3 * i

    index_leg_exchange = FixedLeg(
        schedule=Schedule(
            dt(2022, 1, 1),
            dt(2022, 10, 1),
            frequency="Q",
        ),
        notional=1e6,
        amortization=250e3,
        fixed_rate=2.0,
        index_base=100.0,
        initial_exchange=False,
        final_exchange=True,
    )
    for i, period in enumerate(index_leg_exchange.periods[0::2]):
        assert period.settlement_params.notional == 1e6 - 250e3 * i
    for i, period in enumerate(index_leg_exchange.periods[1:4:2]):
        assert period.settlement_params.notional == 250e3


def test_custom_leg_raises() -> None:
    with pytest.raises(ValueError):
        _ = CustomLeg(periods=["bad_period"])


def test_custom_leg() -> None:
    float_leg = FloatLeg(
        schedule=Schedule(effective=dt(2022, 1, 1), termination=dt(2023, 1, 1), frequency="S"),
    )
    custom_leg = CustomLeg(periods=float_leg.periods)
    for i, period in enumerate(custom_leg.periods):
        assert period == float_leg.periods[i]


@pytest.mark.parametrize(
    ("fx_fixings", "exp"),
    [
        (NoInput(0), [NoInput(0), NoInput(0), NoInput(0)]),
        ([1.5], [1.5, NoInput(0), NoInput(0)]),
        (1.25, [1.25, NoInput(0), NoInput(0)]),
        ((1.25, Series([1.5], index=[dt(2022, 4, 4)])), [1.25, 1.5, NoInput(0)]),
    ],
)
def test_fixed_leg_exchange_mtm(fx_fixings, exp) -> None:
    fixed_leg_exch = FixedLeg(
        schedule=Schedule(
            effective=dt(2022, 1, 3),
            termination=dt(2022, 7, 3),
            frequency="Q",
            payment_lag_exchange=3,
        ),
        fixed_rate=5.0,
        currency="usd",
        pair="eurusd",
        notional=10e6,
        fx_fixings=fx_fixings,
        mtm="xcs",
        initial_exchange=True,
    )
    fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxf = FXForwards(
        fxr,
        {
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
            "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
        },
    )

    d = [
        dt(2022, 1, 6),
        dt(2022, 4, 6),
        dt(2022, 7, 6),
    ]  # payment_lag_exchange is 3 days.
    rate = [_ if _ is not NoInput(0) else fxf.rate("eurusd", d[i]) for i, _ in enumerate(exp)]

    fixed_leg_exch.cashflows(
        rate_curve=fxf.curve("usd", "usd"), disc_curve=fxf.curve("usd", "usd"), fx=fxf
    )
    assert float(fixed_leg_exch.periods[0].try_cashflow(fx=fxf).unwrap() - 10e6 * rate[0]) < 1e-6
    assert (
        float(fixed_leg_exch.periods[2].try_cashflow(fx=fxf).unwrap() - 10e6 * (rate[1] - rate[0]))
        < 1e-6
    )
    assert (
        float(fixed_leg_exch.periods[4].try_cashflow(fx=fxf).unwrap() - 10e6 * (rate[2] - rate[1]))
        < 1e-6
    )
    assert fixed_leg_exch.periods[4].settlement_params.payment == dt(2022, 7, 6)

    assert fixed_leg_exch.periods[1].settlement_params.notional == 10e6
    assert fixed_leg_exch.periods[1].non_deliverable_params.fx_fixing.value == exp[0]
    assert fixed_leg_exch.periods[1].non_deliverable_params.fx_fixing.date == dt(2022, 1, 4)

    assert type(fixed_leg_exch.periods[1]) is FixedPeriod
    assert fixed_leg_exch.periods[3].settlement_params.notional == 10e6
    assert fixed_leg_exch.periods[3].non_deliverable_params.fx_fixing.value == exp[1]
    assert fixed_leg_exch.periods[3].non_deliverable_params.fx_fixing.date == dt(2022, 4, 4)
    assert type(fixed_leg_exch.periods[3]) is FixedPeriod

    assert fixed_leg_exch.periods[-1].settlement_params.notional == 10e6
    assert fixed_leg_exch.periods[-1].non_deliverable_params.fx_fixing.value == exp[1]
    assert fixed_leg_exch.periods[-1].non_deliverable_params.fx_fixing.date == dt(2022, 4, 4)


@pytest.mark.parametrize(
    ("type_", "expected", "kw"),
    [
        (FloatLeg, [522.324262, 522.324262], {"float_spread": 1.0}),
        (FixedLeg, [522.324262, 53772.226595], {"fixed_rate": 2.5}),
    ],
)
def test_mtm_leg_exchange_metrics(type_, expected, kw) -> None:
    leg = type_(
        schedule=Schedule(
            effective=dt(2022, 1, 3),
            termination=dt(2022, 7, 3),
            frequency="Q",
            payment_lag=0,
            payment_lag_exchange=0,
        ),
        currency="usd",
        pair="eurusd",
        notional=10e6,
        initial_exchange=True,
        mtm="xcs",
        **kw,
    )
    fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxf = FXForwards(
        fxr,
        {
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
            "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
        },
    )

    # d = [
    #     dt(2022, 1, 6),
    #     dt(2022, 4, 6),
    #     dt(2022, 7, 6),
    # ]  # payment_lag_exchange is 3 days.
    # rate = [fxf.rate("eurusd", d[i]) for i in range(3)]

    result = leg.analytic_delta(
        rate_curve=fxf.curve("usd", "usd"), disc_curve=fxf.curve("usd", "usd"), fx=fxf
    )
    assert float(result - expected[0]) < 1e-6

    result = leg.npv(rate_curve=fxf.curve("usd", "usd"), disc_curve=fxf.curve("usd", "usd"), fx=fxf)
    assert float(result - expected[1]) < 1e-6


@pytest.mark.parametrize(
    ("klass", "kwargs", "expected"),
    [
        (FixedLeg, {}, [200.0, 300.0, 400.0]),
        (
            FixedLeg,
            {"initial_exchange": False, "final_exchange": True},
            [200.0, 300.0, 400.0, 400.0],
        ),
        (ZeroFixedLeg, {}, [400.0]),
    ],
)
def test_set_index_fixings_series_leg_types(klass, kwargs, expected) -> None:
    index_fixings = Series(
        [100.0, 200.0, 300, 400.0, 500.0],
        index=[dt(2022, 1, 1), dt(2022, 2, 1), dt(2022, 5, 1), dt(2022, 8, 1), dt(2022, 11, 1)],
    )
    obj = klass(
        schedule=Schedule(
            effective=dt(2022, 2, 5),
            termination="9M",
            frequency="Q",
        ),
        index_fixings=index_fixings,
        index_base=100.0,
        index_lag=3,
        index_method="monthly",
        **kwargs,
    )
    for i, period in enumerate(obj.periods):
        if type(period) is Cashflow:
            continue
        assert period.index_params.index_fixing.value == expected[i]


@pytest.mark.skip(reason="fixings as a list removed in v2.0")
@pytest.mark.parametrize(
    ("klass", "kwargs", "expected"),
    [
        (FixedLeg, {"index_fixings": [200.0, 300.0, 400.0]}, [200.0, 300.0, 400.0]),
        (
            FixedLeg,
            {
                "initial_exchange": False,
                "final_exchange": True,
                "index_fixings": [200.0, 300.0, 400.0, 400.0],
            },
            [200.0, 300.0, 400.0, 400.0],
        ),
        (ZeroFixedLeg, {"index_fixings": [400.0]}, [400.0]),
    ],
)
def test_set_index_fixings_list_leg_types(klass, kwargs, expected) -> None:
    obj = klass(
        schedule=Schedule(
            effective=dt(2022, 2, 5),
            termination="9M",
            frequency="Q",
        ),
        index_base=100.0,
        index_lag=3,
        index_method="monthly",
        **kwargs,
    )
    for i, period in enumerate(obj.periods):
        if type(period) is Cashflow:
            continue
        assert period.index_fixings == expected[i]


@pytest.mark.skip(reason="v2.2 refactored fixings. Fixing as dualtype is not allowed.")
@pytest.mark.parametrize(
    ("klass", "kwargs", "expected"),
    [
        (FixedLeg, {"index_fixings": 200.0}, [200.0, NoInput(0), NoInput(0)]),
        (
            FixedLeg,
            {"initial_exchange": False, "final_exchange": True, "index_fixings": 200.0},
            [200.0, NoInput(0), NoInput(0), NoInput(0)],
        ),
        (ZeroFixedLeg, {"index_fixings": 400.0}, [400.0]),
    ],
)
def test_set_index_fixings_float_leg_types(klass, kwargs, expected) -> None:
    obj = klass(
        schedule=Schedule(
            effective=dt(2022, 2, 5),
            termination="9M",
            frequency="Q",
        ),
        index_base=100.0,
        index_lag=3,
        index_method="monthly",
        **kwargs,
    )
    for i, period in enumerate(obj.periods):
        if type(period) is Cashflow:
            continue
        assert period.index_fixings == expected[i]
