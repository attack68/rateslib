from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from rateslib._spec_loader import INSTRUMENT_SPECS
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import FloatFixingMethod
from rateslib.rs import NamedCal

PlotOutput = tuple[plt.Figure, plt.Axes, list[plt.Line2D]]  # type: ignore[name-defined]

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CalTypes,
    )

DEFAULTS = dict(
    stub="SHORTFRONT",
    stub_length="SHORT",
    eval_mode="swaps_align",
    modifier="MF",
    calendars={
        "all": NamedCal("all"),
        "bus": NamedCal("bus"),
        "tgt": NamedCal("tgt"),
        "ldn": NamedCal("ldn"),
        "nyc": NamedCal("nyc"),
        "fed": NamedCal("fed"),
        "stk": NamedCal("stk"),
        "osl": NamedCal("osl"),
        "zur": NamedCal("zur"),
        "tro": NamedCal("tro"),
        "tyo": NamedCal("tyo"),
        "syd": NamedCal("syd"),
        "nsw": NamedCal("nsw"),
        "wlg": NamedCal("wlg"),
        "mum": NamedCal("mum"),
    },
    eom=False,
    eom_fx=True,
    # Instrument parameterisation
    metric={
        "SBS": "leg1",
    },
    convention="ACT360",
    notional=1.0e6,
    index_lag=3,
    index_lag_curve=0,
    index_method="daily",
    payment_lag=2,
    payment_lag_exchange=0,
    payment_lag_specific={
        "IRS": 2,
        "STIRFuture": 0,
        "IIRS": 2,
        "ZCS": 2,
        "ZCIS": 0,
        "FXSwap": 0,
        "SBS": 2,
        "Swap": 2,
        "XCS": 2,
        "NDXCS": 2,
        "FixedRateBond": 0,
        "IndexFixedRateBond": 0,
        "FloatRateNote": 0,
        "Bill": 0,
        "FRA": 0,
        "CDS": 0,
        "NDF": 2,
    },
    fixing_method="rfr_payment_delay",
    fixing_method_param={
        "rfr_payment_delay": 0,  # no observation shift - use payment_delay param
        "rfr_observation_shift": 2,
        "rfr_lockout": 2,
        "rfr_lookback": 2,
        "rfr_payment_delay_avg": 0,  # no observation shift - use payment_delay param
        "rfr_observation_shift_avg": 2,
        "rfr_lockout_avg": 2,
        "rfr_lookback_avg": 2,
        "ibor": 2,
        FloatFixingMethod.RFRPaymentDelayAverage: 0,
        FloatFixingMethod.RFRPaymentDelay: 0,
        FloatFixingMethod.IBOR: 2,
        FloatFixingMethod.RFRLockout: 2,
        FloatFixingMethod.RFRLockoutAverage: 2,
        FloatFixingMethod.RFRObservationShiftAverage: 2,
        FloatFixingMethod.RFRObservationShift: 2,
        FloatFixingMethod.RFRLookback: 2,
        FloatFixingMethod.RFRLookbackAverage: 2,
    },
    spread_compound_method="none_simple",
    base_currency="usd",
    fx_delivery_lag=2,
    fx_delta_type="spot",
    fx_option_metric="pips",
    cds_premium_accrued=True,
    cds_recovery_rate=0.40,
    cds_protection_discretization=23,
    # Curves
    interpolation={
        "dfs": "log_linear",
        "values": "linear",
    },
    endpoints="natural",
    multi_csa_steps=[
        2,
        5,
        10,
        20,
        30,
        50,
        77,
        81,
        86,
        91,
        96,
        103,
        110,
        119,
        128,
        140,
        153,
        169,
        188,
        212,
        242,
        281,
        332,
        401,
        498,
        636,
        835,
        1104,
        1407,
        1646,
        1766,
        1808,
        1821,
        1824,
        1825,
    ],
    multi_csa_min_step=1,
    multi_csa_max_step=1825,
    curve_caching=True,
    curve_caching_max=1000,
    # Solver
    tag="v",
    algorithm="levenberg_marquardt",
    curve_not_in_solver="ignore",  # or "warn" or "raise"
    ini_lambda=(1000.0, 0.25, 2.0),
    # bonds
    calc_mode={
        "FixedRateBond": "uk_gb",
        "FloatRateNote": "uk_gb",
        "Bill": "us_gbb",
        "IndexFixedRateBond": "uk_gb",
    },
    settle=1,
    ex_div=1,
    calc_mode_futures="ytm",
    # misc
    pool=1,
    no_fx_fixings_for_xcs="warn",  # or "raise" or "ignore"
    headers={
        "type": "Type",
        "stub_type": "Period",
        "u_acc_start": "Unadj Acc Start",
        "u_acc_end": "Unadj Acc End",
        "a_acc_start": "Acc Start",
        "a_acc_end": "Acc End",
        "payment": "Payment",
        "convention": "Convention",
        "dcf": "DCF",
        "df": "DF",
        "notional": "Notional",
        "reference_currency": "Reference Ccy",
        "currency": "Ccy",
        "fx_fixing": "FX Fixing",
        "fx_fixing_date": "FX Fix Date",
        "rate": "Rate",
        "spread": "Spread",
        "npv": "NPV",
        "cashflow": "Cashflow",
        "fx": "FX Rate",
        "npv_fx": "NPV Ccy",
        "base": "Base Ccy",
        "unindexed_cashflow": "Unindexed Cashflow",
        "index_fix_date": "Index Fix Date",
        "index_value": "Index Val",
        "index_ratio": "Index Ratio",
        "index_base": "Index Base",
        "collateral": "Collateral",
        # Options headers
        "pair": "Pair",
        "expiry": "Expiry",
        "t_e": "Time to Expiry",
        "delivery": "Delivery",
        "model": "Model",
        "vol": "Vol",
        "strike": "Strike",
        # CDS headers
        "survival": "Survival",
        "recovery": "Recovery",
    },
    _global_ad_order=1,
    oaspread_func_tol=1e-6,
    oaspread_conv_tol=1e-8,
    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.
    spec=INSTRUMENT_SPECS,
)


class Defaults:
    """
    The *defaults* object used by initialising objects. Values are printed below:

    .. ipython:: python

       from rateslib import defaults
       print(defaults.print())

    """

    _instance = None

    stub: str
    stub_length: str
    eval_mode: str
    modifier: str
    calendars: dict[str, CalTypes]
    eom: bool
    eom_fx: bool

    metric: dict[str, str]
    convention: str
    notional: float
    index_lag: int
    index_lag_curve: int
    index_method: str
    payment_lag: int
    payment_lag_exchange: int
    payment_lag_specific: dict[str, int]
    fixing_method: str
    fixing_method_param: dict[str | FloatFixingMethod, int]
    spread_compound_method: str
    base_currency: str
    fx_delivery_lag: int
    fx_delta_type: str
    fx_option_metric: str
    cds_premium_accrued: bool
    cds_recovery_rate: float
    cds_protection_discretization: int

    interpolation: dict[str, str]
    endpoints: str
    multi_csa_steps: list[int]
    multi_csa_min_step: int
    multi_csa_max_step: int
    curve_caching: bool
    curve_caching_max: int

    tag: str
    algorithm: str
    curve_not_in_solver: str
    ini_lambda: tuple[int, float, float]

    calc_mode: dict[str, str]
    settle: int
    ex_div: int
    calc_mode_futures: str

    pool: int
    no_fx_fixings_for_xcs: str
    headers: dict[str, str]
    _global_ad_order: int
    oaspread_func_tol: float
    oaspread_conv_tol: float
    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.
    spec: dict[str, dict[str, Any]]

    def __new__(cls) -> Defaults:
        if cls._instance is None:
            # Singleton pattern creates only one instance: TODO (low) might not be thread safe
            cls._instance = super(Defaults, cls).__new__(cls)  # noqa: UP008

            for k, v in DEFAULTS.items():
                setattr(cls._instance, k, deepcopy(v))

        return cls._instance

    def reset_defaults(self) -> None:
        """
        Revert defaults back to their initialisation status.

        Examples
        --------
        .. ipython:: python

           from rateslib import defaults
           defaults.reset_defaults()
        """
        attrs = [
            v
            for v in dir(self)
            if "__" not in v and not callable(getattr(self, v)) and v != "_instance"
        ]
        for attr in attrs:
            delattr(self, attr)

        for k, v in DEFAULTS.items():
            setattr(self, k, deepcopy(v))

    def print(self) -> str:
        """
        Return a string representation of the current values in the defaults object.
        """

        def _t_n(v: str) -> str:  # teb-newline
            return f"\t{v}\n"

        _: str = f"""\
Scheduling:\n
{
            "".join(
                [
                    _t_n(f"{attribute}: {getattr(self, attribute)}")
                    for attribute in [
                        "stub",
                        "stub_length",
                        "modifier",
                        "eom",
                        "eom_fx",
                        "eval_mode",
                    ]
                ]
            )
        }
Instruments:\n
{
            "".join(
                [
                    _t_n(f"{attribute}: {getattr(self, attribute)}")
                    for attribute in [
                        "convention",
                        "payment_lag",
                        "payment_lag_exchange",
                        "payment_lag_specific",
                        "notional",
                        "fixing_method",
                        "fixing_method_param",
                        "spread_compound_method",
                        "base_currency",
                        "fx_delivery_lag",
                        "fx_delta_type",
                        "fx_option_metric",
                        "cds_premium_accrued",
                        "cds_recovery_rate",
                        "cds_protection_discretization",
                    ]
                ]
            )
        }
Curves:\n
{
            "".join(
                [
                    _t_n(f"{attribute}: {getattr(self, attribute)}")
                    for attribute in [
                        "interpolation",
                        "endpoints",
                        "multi_csa_steps",
                        "curve_caching",
                    ]
                ]
            )
        }
Solver:\n
{
            "".join(
                [
                    _t_n(f"{attribute}: {getattr(self, attribute)}")
                    for attribute in [
                        "algorithm",
                        "tag",
                        "curve_not_in_solver",
                    ]
                ]
            )
        }
Miscellaneous:\n
{
            "".join(
                [
                    _t_n(f"{attribute}: {getattr(self, attribute)}")
                    for attribute in [
                        "headers",
                        "no_fx_fixings_for_xcs",
                        "pool",
                    ]
                ]
            )
        }
"""  # noqa: W291
        return _


def plot(
    x: list[list[Any]], y: list[list[Any]], labels: list[str] | NoInput = NoInput(0)
) -> PlotOutput:
    labels = _drb([], labels)
    fig, ax = plt.subplots(1, 1)
    lines = []
    for _x, _y in zip(x, y, strict=True):
        (line,) = ax.plot(_x, _y)
        lines.append(line)
    if not isinstance(labels, NoInput) and len(labels) == len(lines):
        ax.legend(lines, labels)

    ax.grid(True)

    if isinstance(x[0][0], datetime):
        years = mdates.YearLocator()  # type: ignore[no-untyped-call]
        months = mdates.MonthLocator()  # type: ignore[no-untyped-call]
        yearsFmt = mdates.DateFormatter("%Y")  # type: ignore[no-untyped-call]
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        fig.autofmt_xdate()
    return fig, ax, lines


def plot3d(
    x: list[Any], y: list[Any], z: np.ndarray[tuple[int, int], np.dtype[np.float64]]
) -> tuple[plt.Figure, plt.Axes, None]:  # type: ignore[name-defined]
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # import matplotlib.dates as mdates  # type: ignore[import]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y = np.meshgrid(x, y)
    # Plot the surface.
    ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)  # type: ignore[attr-defined]
    return fig, ax, None


def _make_py_json(json: str, class_name: str) -> str:
    """Modifies the output JSON output for Rust structs wrapped by Python classes."""
    return '{"PyWrapped":' + json + "}"


__all__ = ["Defaults"]
