import os
from datetime import datetime as dt

import numpy as np
import pandas as pd
import pytest
from rateslib import fixings
from rateslib.curves import Curve
from rateslib.enums import FloatFixingMethod, SpreadCompoundMethod
from rateslib.instruments import IRS
from rateslib.periods.components.float_period import FloatPeriod
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
    def test_baseline_versus_solver(self, method, param, scm, spread, curve):
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
        fixings_ = p.try_unindexed_reference_fixings_exposure(
            rate_curve=rate_curve, disc_curve=curve
        ).unwrap()
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

        risk_compare = fixings_[("curve", "risk")].astype(float).fillna(0.0).to_numpy()
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

        fixings_ = p.try_unindexed_reference_fixings_exposure(
            rate_curve=rate_curve, disc_curve=curve
        ).unwrap()
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
        risk_compare = fixings_[("curve", "risk")].astype(float).fillna(0.0).to_numpy()

        assert np.all(np.isclose(risk_array, risk_compare, atol=atol))
