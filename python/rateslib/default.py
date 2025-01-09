from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, ParamSpec, TypeVar

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas
from packaging import version
from pandas import Series, read_csv

from rateslib._spec_loader import INSTRUMENT_SPECS
from rateslib.rs import Cal, NamedCal, UnionCal

PlotOutput = tuple[plt.Figure, plt.Axes, list[plt.Line2D]]  # type: ignore[name-defined]

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class NoInput(Enum):
    """Enumerable type to handle setting default values.

    See :ref:`default values <defaults-doc>`.
    """

    blank = 0
    inherit = 1
    negate = -1


class Fixings:
    """Class to lazy load fixing data from CSV files.

    .. warning::

       *Rateslib* does not come pre-packaged with accurate, nor upto date fixing data.
       This is for a number of reasons; one being a lack of data licensing to distribute such
       data, and the second being a statically uploaded package relative to daily, dynamic fixing
       information is not practical.

    To use this class effectively the CSV files must be populated by the user, ideally scheduled
    regularly to continuously update these files with incoming fixing data.
    See :ref:`working with fixings <cook-fixings-doc>`.

    """

    @staticmethod
    def _load_csv(directory: str, path: str) -> Series[float]:
        target = os.path.join(directory, path)
        if version.parse(pandas.__version__) < version.parse("2.0"):  # pragma: no cover
            # this is tested by the minimum version gitflow actions.
            # TODO (low:dependencies) remove when pandas min version is bumped to 2.0
            df = read_csv(target)
            df["reference_date"] = df["reference_date"].map(
                lambda x: datetime.strptime(x, "%d-%m-%Y"),
            )
            df = df.set_index("reference_date")
        else:
            df = read_csv(target, index_col=0, parse_dates=[0], date_format="%d-%m-%Y")
        return df["rate"].sort_index(ascending=True)

    def __getitem__(self, item: str) -> Series[float]:
        if item in self.loaded:
            return self.loaded[item]

        try:
            s = self._load_csv(self.directory, f"{item}.csv")
        except FileNotFoundError:
            raise ValueError(
                f"Fixing data for the index '{item}' has been attempted, but there is no file:\n"
                f"'{item}.csv' located in the search directory: '{self.directory}'\n"
                "Create a CSV file in the directory with the above name and the exact "
                "template structure:\n###################\n"
                "reference_date,rate\n26-08-2023,5.6152\n27-08-2023,5.6335\n##################\n"
                "For further info see 'Working with Fixings' in the documentation cookbook.",
            )

        self.loaded[item] = s
        return s

    def __init__(self) -> None:
        self.directory = os.path.dirname(os.path.abspath(__file__)) + "/data"
        self.loaded: dict[str, Series[float]] = {}


class Defaults:
    """The *defaults* object used by initialising objects. Values are printed below:

    .. ipython:: python

       from rateslib import defaults
       print(defaults.print())

    """

    def __init__(self) -> None:
        # Scheduling
        self.stub = "SHORTFRONT"
        self.stub_length = "SHORT"
        self.eval_mode = "swaps_align"
        self.modifier = "MF"
        self.calendars: dict[str, NamedCal | UnionCal | Cal] = {
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
            "wlg": NamedCal("wlg"),
        }
        self.frequency_months = {
            "M": 1,
            "B": 2,
            "Q": 3,
            "T": 4,
            "S": 6,
            "A": 12,
            "Z": 120000,  # corresponds to 10,000 years
        }
        self.eom = False
        self.eom_fx = True

        # Instrument parameterisation

        self.convention = "ACT360"
        self.notional = 1.0e6
        self.index_lag = 3
        self.index_method = "daily"
        self.payment_lag = 2
        self.payment_lag_exchange = 0
        self.payment_lag_specific = {
            "IRS": 2,
            "STIRFuture": 0,
            "IIRS": 2,
            "ZCS": 2,
            "ZCIS": 0,
            "FXSwap": 0,
            "SBS": 2,
            "Swap": 2,
            "XCS": 2,
            "FixedRateBond": 0,
            "IndexFixedRateBond": 0,
            "FloatRateNote": 0,
            "Bill": 0,
            "FRA": 0,
            "CDS": 0,
        }
        self.fixing_method = "rfr_payment_delay"
        self.fixing_method_param = {
            "rfr_payment_delay": 0,  # no observation shift - use payment_delay param
            "rfr_observation_shift": 2,
            "rfr_lockout": 2,
            "rfr_lookback": 2,
            "rfr_payment_delay_avg": 0,  # no observation shift - use payment_delay param
            "rfr_observation_shift_avg": 2,
            "rfr_lockout_avg": 2,
            "rfr_lookback_avg": 2,
            "ibor": 2,
        }
        self.spread_compound_method = "none_simple"
        self.base_currency = "usd"

        self.fx_delivery_lag = 2
        self.fx_delta_type = "spot"
        self.fx_option_metric = "pips"

        self.cds_premium_accrued = True
        self.cds_recovery_rate = 0.40
        self.cds_protection_discretization = 23

        # Curves

        self.interpolation = {
            "Curve": "log_linear",
            "LineCurve": "linear",
            "IndexCurve": "linear_index",
        }
        self.endpoints = "natural"
        # fmt: off
        self.multi_csa_steps = [
           2, 5, 10, 20, 30, 50, 77, 81, 86, 91, 96, 103, 110, 119, 128, 140, 153,
           169, 188, 212, 242, 281, 332, 401, 498, 636, 835, 1104, 1407, 1646, 1766,
           1808, 1821, 1824, 1825,
        ]
        # fmt: on
        self.curve_caching = True

        # Solver

        self.tag = "v"
        self.algorithm = "levenberg_marquardt"
        self.curve_not_in_solver = "ignore"
        self.ini_lambda = (1000.0, 0.25, 2.0)

        # bonds
        self.calc_mode = {
            "FixedRateBond": "uk_gb",
            "FloatRateNote": "uk_gb",
            "Bill": "us_gbb",
            "IndexFixedRateBond": "uk_gb",
        }
        self.settle = 1
        self.ex_div = 1
        self.calc_mode_futures = "ytm"

        # misc

        self.pool = 1
        self.no_fx_fixings_for_xcs = "warn"
        self.headers = {
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
            "currency": "Ccy",
            "rate": "Rate",
            "spread": "Spread",
            "npv": "NPV",
            "cashflow": "Cashflow",
            "fx": "FX Rate",
            "npv_fx": "NPV Ccy",
            "real_cashflow": "Real Cashflow",
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
        }
        self._global_ad_order = 1

        # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
        # Commercial use of this code, and/or copying and redistribution is prohibited.
        # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

        # fixings data
        self.fixings = Fixings()

        self.spec = INSTRUMENT_SPECS

    def reset_defaults(self) -> None:
        """Revert defaults back to their initialisation status.

        Examples
        --------
        .. ipython:: python

           from rateslib import defaults
           defaults.reset_defaults()

        """
        base = Defaults()
        for attr in [_ for _ in dir(self) if _[:2] != "__"]:
            setattr(self, attr, getattr(base, attr))

    def print(self) -> str:
        """Return a string representation of the current values in the defaults object.
        """

        def _t_n(v: str) -> str:  # teb-newline
            return f"\t{v}\n"

        _: str = f"""\
Scheduling:\n
{''.join([_t_n(f'{attribute}: {getattr(self, attribute)}') for attribute in [
    'stub', 
    'stub_length', 
    'modifier', 
    'eom', 
    'eom_fx',
    'eval_mode', 
    'frequency_months',
]])}
Instruments:\n
{''.join([_t_n(f'{attribute}: {getattr(self, attribute)}') for attribute in [
    'convention', 
    'payment_lag', 
    'payment_lag_exchange', 
    'payment_lag_specific', 
    'notional', 
    'fixing_method',
    'fixing_method_param',
    'spread_compound_method',
    'base_currency',
    'fx_delivery_lag',
    'fx_delta_type',
    'fx_option_metric',
    'cds_premium_accrued',
    'cds_recovery_rate',
    'cds_protection_discretization',   
]])}
Curves:\n
{''.join([_t_n(f'{attribute}: {getattr(self, attribute)}') for attribute in [
    'interpolation',
    'endpoints',
    'multi_csa_steps',    
    'curve_caching',        
]])}
Solver:\n
{''.join([_t_n(f'{attribute}: {getattr(self, attribute)}') for attribute in [
    'algorithm',
    'tag',
    'curve_not_in_solver',         
]])}
Miscellaneous:\n
{''.join([_t_n(f'{attribute}: {getattr(self, attribute)}') for attribute in [
    'headers',
    'no_fx_fixings_for_xcs',
    'pool',         
]])}
"""  # noqa: W291
        return _


def plot(x: list[Any], y: list[list[Any]], labels: list[str] | NoInput = NoInput(0)) -> PlotOutput:
    labels = _drb([], labels)
    fig, ax = plt.subplots(1, 1)
    lines = []
    for _y in y:
        (line,) = ax.plot(x, _y)
        lines.append(line)
    if not isinstance(labels, NoInput) and len(labels) == len(lines):
        ax.legend(lines, labels)

    ax.grid(True)

    if isinstance(x[0], datetime):
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


def _drb(default: Any, possible_ni: Any | NoInput) -> Any:
    """(D)efault (r)eplaces (b)lank"""
    return default if isinstance(possible_ni, NoInput) else possible_ni


def _make_py_json(json: str, class_name: str) -> str:
    """Modifies the output JSON output for Rust structs wrapped by Python classes."""
    return '{"Py":' + json + "}"


P = ParamSpec("P")
R = TypeVar("R")


def _validate_states(func: Callable[P, R]) -> Callable[P, R]:
    """Add a decorator to a class instance method to first validate the cache before performing
    additional operations. If a change is detected the implemented `validate_cache` function
    is responsible for resetting the cache and updating any `cache_id`s.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        self._validate_state()  # type: ignore[attr-defined]
        return func(*args, **kwargs)

    return wrapper


class _WithState:
    """Record and manage the `state_id` of mutable classes.

    Attributes
    ----------
    _state: int: This is the most recent recorded state reference of this object.
    _mutable_by_association: bool: This is a rateslib definition of whether this object is
        directly mutable and therefore generates its own state id, or whether its state is
        derived from the most recently evaluated state of its associated objects.

    """

    _state: int = 0
    _mutable_by_association = False

    def _set_new_state(self) -> None:
        """Set the state_id of a superclass. Some objects which are 'mutable by association'
        will overload the `get_compoisted_state` method to derive a state from their
        associated items.
        """
        if self._mutable_by_association:
            self._state = self._get_composited_state()
        else:
            self._state = hash(os.urandom(8))  # 64-bit entropy

    def _validate_state(self) -> None:
        """Used by 'mutable by association' objects to evaluate if their own record of
        associated objects states matches the current state of those objects.
        """
        raise NotImplementedError("Must be implemented for 'mutable by association' types")

    def _get_composited_state(self) -> int:
        """Used by 'mutable by association' objects to record the state of their associated
        objects and set this as the object's own state.
        """
        raise NotImplementedError("Must be implemented for 'mutable by association' types")
