import os
from datetime import datetime as dt

import numpy as np
import pandas as pd
import pytest
from rateslib import fixings
from rateslib.curves import Curve
from rateslib.enums import FloatFixingMethod, SpreadCompoundMethod
from rateslib.instruments import IRS
from rateslib.periods import FixedPeriod, FloatPeriod
from rateslib.solver import Solver


@pytest.fixture
def curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.99,
        dt(2022, 7, 1): 0.98,
        dt(2022, 10, 1): 0.97,
    }
    return Curve(nodes=nodes, interpolation="log_linear", id="curve_fixture")


class TestFloatPeriod:
    @pytest.mark.parametrize(
        ("method", "param"),
        [
            (FloatFixingMethod.RFRPaymentDelay, 0),
            (FloatFixingMethod.RFRObservationShift, 3),
            (FloatFixingMethod.RFRLockout, 2),
            (FloatFixingMethod.RFRLookback, 3),
            (FloatFixingMethod.RFRLockoutAverage, 2),
            (FloatFixingMethod.RFRPaymentDelayAverage, 0),
            (FloatFixingMethod.RFRObservationShiftAverage, 3),
            (FloatFixingMethod.RFRLookbackAverage, 3),
        ],
    )
    @pytest.mark.parametrize(
        ("scm", "spread"),
        [
            (SpreadCompoundMethod.NoneSimple, 0.0),
            (SpreadCompoundMethod.NoneSimple, 500.0),
            (SpreadCompoundMethod.ISDACompounding, 0.0),
            (SpreadCompoundMethod.ISDACompounding, 500.0),
            (SpreadCompoundMethod.ISDAFlatCompounding, 0.0),
            (SpreadCompoundMethod.ISDAFlatCompounding, 500.0),
        ],
    )
    def test_baseline_versus_solver_fixings_sensitivity(self, method, param, scm, spread, curve):
        # the Solver can make fixings exposure calculations independently from analytical
        # calculations and approximations. This tests validates the analytical calculations
        # against the Solver
        if method in [
            FloatFixingMethod.RFRLockoutAverage,
            FloatFixingMethod.RFRPaymentDelayAverage,
            FloatFixingMethod.RFRObservationShiftAverage,
            FloatFixingMethod.RFRLookbackAverage,
        ] and scm in [
            SpreadCompoundMethod.ISDAFlatCompounding,
            SpreadCompoundMethod.ISDACompounding,
        ]:
            pytest.skip(reason="Impossible combination raises ValueError on initialisation.")

        # let us construct baseline instruments
        rate_curve = Curve(
            nodes={
                dt(2022, 1, 1): 1.00,
                dt(2022, 1, 31): 0.99,
                dt(2022, 2, 1): 0.99,
                dt(2022, 2, 2): 0.99,
                dt(2022, 2, 3): 0.99,
                dt(2022, 2, 4): 0.99,
                dt(2022, 2, 7): 0.99,
                dt(2022, 2, 8): 0.99,
                dt(2022, 2, 9): 0.99,
                dt(2022, 2, 10): 0.98,
                dt(2029, 2, 1): 0.97,
            },
            interpolation="log_linear",
            calendar="nyc",
            id="curve",
        )
        solver = Solver(
            curves=[rate_curve],
            instruments=[
                IRS(
                    dt(2022, 1, 4), "1b", spec="usd_irs", payment_lag=0, curves=[rate_curve, curve]
                ),
                IRS(
                    dt(2022, 1, 31), "1b", spec="usd_irs", payment_lag=0, curves=[rate_curve, curve]
                ),
                IRS(
                    dt(2022, 2, 1), "1b", spec="usd_irs", payment_lag=0, curves=[rate_curve, curve]
                ),
                IRS(
                    dt(2022, 2, 2), "1b", spec="usd_irs", payment_lag=0, curves=[rate_curve, curve]
                ),
                IRS(
                    dt(2022, 2, 3), "1b", spec="usd_irs", payment_lag=0, curves=[rate_curve, curve]
                ),
                IRS(
                    dt(2022, 2, 4), "1b", spec="usd_irs", payment_lag=0, curves=[rate_curve, curve]
                ),
                IRS(
                    dt(2022, 2, 7), "1b", spec="usd_irs", payment_lag=0, curves=[rate_curve, curve]
                ),
                IRS(
                    dt(2022, 2, 8), "1b", spec="usd_irs", payment_lag=0, curves=[rate_curve, curve]
                ),
                IRS(
                    dt(2022, 2, 9), "1b", spec="usd_irs", payment_lag=0, curves=[rate_curve, curve]
                ),
                IRS(
                    dt(2022, 2, 10), "1b", spec="usd_irs", payment_lag=0, curves=[rate_curve, curve]
                ),
            ],
            s=[4.03] * 10,
        )
        p = FloatPeriod(
            notional=-10e6,
            fixing_series="usd_rfr",
            fixing_method=method,
            method_param=param,
            frequency="A",
            start=dt(2022, 2, 3),
            end=dt(2022, 2, 10),
            float_spread=spread,
            payment=dt(2022, 2, 10),
            convention="act360",
            spread_compound_method=scm,
        )
        risk = solver.delta(npv=p.npv(rate_curve=rate_curve, disc_curve=curve, local=True))
        fixings_ = p.local_analytic_rate_fixings(rate_curve=rate_curve, disc_curve=curve)
        fixings_ = fixings_.reindex(
            [
                dt(2022, 1, 30),
                dt(2022, 1, 31),
                dt(2022, 2, 1),
                dt(2022, 2, 2),
                dt(2022, 2, 3),
                dt(2022, 2, 4),
                dt(2022, 2, 7),
                dt(2022, 2, 8),
                dt(2022, 2, 9),
                dt(2022, 2, 10),
            ],
            fill_value=np.nan,
        )

        risk_compare = fixings_[("curve", "usd", "usd", "1B")].astype(float).fillna(0.0).to_numpy()
        risk_array = risk.to_numpy()[:, 0]

        _diff = np.max(np.abs(risk_compare - risk_array))
        if scm == SpreadCompoundMethod.ISDAFlatCompounding and spread > 100.0:
            atol = 1e-2
        else:
            atol = 1e-12
        assert np.all(np.isclose(risk_array, risk_compare, atol=atol))

        # now add some fixings
        name = str(hash(os.urandom(8)))
        fixings.add(
            f"{name}_1B",
            pd.Series(
                index=[dt(2022, 1, 31), dt(2022, 2, 1), dt(2022, 2, 2), dt(2022, 2, 3)],
                data=[4.03, 4.03, 4.03, 4.03],
            ),
        )
        p = FloatPeriod(
            notional=-10e6,
            fixing_series="usd_rfr",
            fixing_method=method,
            method_param=param,
            frequency="A",
            start=dt(2022, 2, 3),
            end=dt(2022, 2, 10),
            float_spread=spread,
            payment=dt(2022, 2, 10),
            convention="act360",
            spread_compound_method=scm,
            rate_fixings=name,
        )

        fixings_ = p.local_analytic_rate_fixings(rate_curve=rate_curve, disc_curve=curve)
        fixings_ = fixings_.reindex(
            [
                dt(2022, 1, 30),
                dt(2022, 1, 31),
                dt(2022, 2, 1),
                dt(2022, 2, 2),
                dt(2022, 2, 3),
                dt(2022, 2, 4),
                dt(2022, 2, 7),
                dt(2022, 2, 8),
                dt(2022, 2, 9),
                dt(2022, 2, 10),
            ],
            fill_value=np.nan,
        )

        risk_array[:5] = 0.0
        risk_compare = fixings_[("curve", "usd", "usd", "1B")].astype(float).fillna(0.0).to_numpy()

        assert np.all(np.isclose(risk_array, risk_compare, atol=atol))

    def test_ibor_curve_example_book(self, curve):
        p = FloatPeriod(
            notional=-10e6,
            fixing_series="eur_ibor",
            fixing_method="ibor",
            method_param=1,
            frequency="Q",
            start=dt(2025, 10, 8),
            end=dt(2026, 1, 8),
            float_spread=100.0,
            payment=dt(2026, 1, 8),
            convention="act360",
            calendar="tgt",
        )
        result = p.try_unindexed_reference_cashflow_analytic_rate_fixings(rate_curve=curve).unwrap()
        assert abs(result.iloc[0, 0] - 10e2 * 92 / 360) < 1e-12
        assert result.index[0] == dt(2025, 10, 6)

    def test_ibor_stub_curve_example_book(self, curve):
        p = FloatPeriod(
            notional=-10e6,
            fixing_method="ibor",
            method_param=2,
            frequency="Q",
            start=dt(2025, 10, 8),
            end=dt(2025, 12, 16),
            float_spread=100.0,
            payment=dt(2025, 12, 16),
            convention="act360",
            calendar="tgt",
            stub=True,
        )
        result = p.try_unindexed_reference_cashflow_analytic_rate_fixings(
            rate_curve={"2m": curve, "3m": curve, "6m": curve}
        ).unwrap()
        alpha = 23 / 31.0
        assert abs(result.iloc[0, 0] - 10e2 * 69 / 360 * alpha) < 1e-12
        assert abs(result.iloc[0, 1] - 10e2 * 69 / 360 * (1 - alpha)) < 1e-12
        assert result.index[0] == dt(2025, 10, 6)

    def test_ibor_fixing_set(self, curve):
        p = FloatPeriod(
            notional=-10e6,
            fixing_series="eur_ibor",
            fixing_method="ibor",
            method_param=1,
            rate_fixings=2.0,
            frequency="Q",
            start=dt(2025, 10, 8),
            end=dt(2026, 1, 8),
            float_spread=100.0,
            payment=dt(2026, 1, 8),
            convention="act360",
            calendar="tgt",
        )
        result = p.try_unindexed_reference_cashflow_analytic_rate_fixings(rate_curve=curve).unwrap()
        assert abs(result.iloc[0, 0]) < 1e-12
        assert result.index[0] == dt(2025, 10, 6)

    def test_ibor_stub_curve_fixings_set(self, curve):
        p = FloatPeriod(
            notional=-10e6,
            fixing_method="ibor",
            method_param=2,
            frequency="Q",
            start=dt(2025, 10, 8),
            end=dt(2025, 12, 16),
            float_spread=100.0,
            payment=dt(2025, 12, 16),
            convention="act360",
            calendar="tgt",
            stub=True,
            rate_fixings=2.0,
        )
        result = p.try_unindexed_reference_cashflow_analytic_rate_fixings(
            rate_curve={"2m": curve, "3m": curve, "6m": curve}
        ).unwrap()
        assert abs(result.iloc[0, 0]) < 1e-12
        assert abs(result.iloc[0, 1]) < 1e-12
        assert result.index[0] == dt(2025, 10, 6)

    @pytest.mark.parametrize(
        ("method", "param", "expected"),
        [
            (FloatFixingMethod.RFRPaymentDelay, 0, [0, 0, 0, 0, 277, 830, 277, 277, 277, 0]),
            (FloatFixingMethod.RFRLockout, 2, [0, 0, 0, 0, 277, 830, 830, 0, 0, 0]),
            (FloatFixingMethod.RFRLookback, 3, [0, 277, 830, 277, 277, 277, 0, 0, 0, 0]),
            (FloatFixingMethod.RFRObservationShift, 3, [0, 277, 277, 277, 277, 830, 0, 0, 0, 0]),
        ],
    )
    def test_rfr_curve_book(self, method, param, expected, curve):
        p = FloatPeriod(
            notional=-1e6,
            fixing_series="usd_rfr",
            fixing_method=method,
            method_param=param,
            frequency="Q",
            start=dt(2022, 2, 3),
            end=dt(2022, 2, 10),
            float_spread=0.0,
            payment=dt(2022, 2, 10),
        )
        result = p.local_analytic_rate_fixings(rate_curve=curve)
        result = result.reindex(
            pd.Index(
                data=[
                    dt(2022, 1, 30),
                    dt(2022, 1, 31),
                    dt(2022, 2, 1),
                    dt(2022, 2, 2),
                    dt(2022, 2, 3),
                    dt(2022, 2, 4),
                    dt(2022, 2, 7),
                    dt(2022, 2, 8),
                    dt(2022, 2, 9),
                    dt(2022, 2, 10),
                ]
            ),
            fill_value=0.0,
        )
        for i in range(10):
            assert abs(expected[i] - result.iloc[i, 0] * 1000) < 5e-1


class TestFixedPeriod:
    def test_immediate_fixing_sensitivity(self, curve):
        p = FixedPeriod(
            fixed_rate=2.0,
            start=dt(2022, 1, 1),
            end=dt(2022, 2, 1),
            payment=dt(2022, 2, 1),
            frequency="M",
            notional=2e6,
            currency="usd",
            convention="act360",
        )
        result = p.try_immediate_analytic_rate_fixings(disc_curve=curve).unwrap()
        assert isinstance(result, pd.DataFrame)
        assert result.empty
