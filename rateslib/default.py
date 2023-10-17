from pandas import read_csv
import pandas
import os
from enum import Enum
from packaging import version
from datetime import datetime
from rateslib._spec_loader import INSTRUMENT_SPECS

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class NoInput(Enum):
    blank = 0
    inherit = 1
    negate = -1


class Fixings:
    @staticmethod
    def _load_csv(path):
        abspath = os.path.dirname(os.path.abspath(__file__))
        target = os.path.join(abspath, path)
        if version.parse(pandas.__version__) < version.parse("2.0"):  # pragma: no cover
            # this is tested by the minimum version gitflow actions.
            # TODO (low:dependencies) remove when pandas min version is bumped to 2.0
            df = read_csv(target)
            df["reference_date"] = df["reference_date"].map(
                lambda x: datetime.strptime(x, "%d-%m-%Y")
            )
            df = df.set_index("reference_date")
        else:
            df = read_csv(target, index_col=0, parse_dates=[0], date_format="%d-%m-%Y")
        return df["rate"].sort_index(ascending=True)

    def __init__(self):
        self._sonia = None
        self._estr = None
        self._sofr = None
        self._swestr = None
        self._nowa = None
        self._corra = None

    @property
    def sonia(self):
        if self._sonia is None:
            self._sonia = self._load_csv("data/sonia.csv")
        return self._sonia

    @property
    def estr(self):
        if self._estr is None:
            self._estr = self._load_csv("data/estr.csv")
        return self._estr

    @property
    def sofr(self):
        if self._sofr is None:
            self._sofr = self._load_csv("data/sofr.csv")
        return self._sofr

    @property
    def swestr(self):
        if self._swestr is None:
            self._swestr = self._load_csv("data/swestr.csv")
        return self._swestr

    @property
    def nowa(self):
        if self._nowa is None:
            self._nowa = self._load_csv("data/nowa.csv")
        return self._nowa

    @property
    def corra(self):
        if self._corra is None:
            self._corra = self._load_csv("data/corra.csv")
        return self._corra

    @property
    def saron(self):
        raise NotImplementedError("Swiss SIX exchange licence not available.")


class Defaults:
    """
    The *defaults* object used by initialising objects. Values are printed below:

    .. ipython:: python

       from rateslib import defaults
       print(defaults.print())

    """

    # Scheduling
    stub = "SHORTFRONT"
    stub_length = "SHORT"
    eval_mode = "swaps_align"
    modifier = "MF"
    # calendar = BusinessDay()
    frequency_months = {
        "M": 1,
        "B": 2,
        "Q": 3,
        "T": 4,
        "S": 6,
        "A": 12,
        "Z": 1e8,
    }
    eom = False

    # Instrument parameterisation

    convention = "ACT360"
    notional = 1.0e6
    index_lag = 3
    index_method = "daily"
    payment_lag = 2
    payment_lag_exchange = 0
    payment_lag_specific = {
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
    }
    fixing_method = "rfr_payment_delay"
    fixing_method_param = {
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
    spread_compound_method = "none_simple"
    base_currency = "usd"

    # Curves

    interpolation = {
        "Curve": "log_linear",
        "LineCurve": "linear",
        "IndexCurve": "linear_index",
    }
    endpoints = "natural"
    # fmt: off
    multi_csa_steps = [
       2, 5, 10, 20, 30, 50, 77, 81, 86, 91, 96, 103, 110, 119, 128, 140, 153,
       169, 188, 212, 242, 281, 332, 401, 498, 636, 835, 1104, 1407, 1646, 1766,
       1808, 1821, 1824, 1825,
    ]
    # fmt: on

    # Solver

    tag = "v"
    algorithm = "levenberg_marquardt"
    curve_not_in_solver = "ignore"
    ini_lambda = (1000.0, 0.25, 2.0)

    # bonds
    calc_mode = "ukg"
    settle = 1
    ex_div = 1

    # misc

    pool = 1
    no_fx_fixings_for_xcs = "warn"
    headers = {
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
    }

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    # fixings data
    fixings = Fixings()

    def reset_defaults(self):
        """
        Revert defaults back to their initialisation status.

        Examples
        --------
        .. ipython:: python

           from rateslib import defaults
           defaults.reset_defaults()
        """
        base = Defaults()
        for attr in [_ for _ in dir(self) if "__" != _[:2]]:
            setattr(self, attr, getattr(base, attr))

    spec = INSTRUMENT_SPECS

    def print(self):
        """
        Return a string representation of the current values in the defaults object.
        """

        def _t_n(v):
            return f"\t{v}\n"

        _ = f"""\
Scheduling:\n
{''.join([_t_n(f'{attribute}: {getattr(self, attribute)}') for attribute in [
    'stub', 
    'stub_length', 
    'modifier', 
    'eom', 
    'eval_mode', 
    'frequency_months'
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
]])}
Curves:\n
{''.join([_t_n(f'{attribute}: {getattr(self, attribute)}') for attribute in [
    'interpolation',
    'endpoints',
    'multi_csa_steps',        
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
"""
        return _


def plot(x, y: list, labels=[]):
    import matplotlib.pyplot as plt  # type: ignore[import]
    import matplotlib.dates as mdates  # type: ignore[import]

    fig, ax = plt.subplots(1, 1)
    lines = []
    for _y in y:
        (line,) = ax.plot(x, _y)
        lines.append(line)
    if labels and len(labels) == len(lines):
        ax.legend(lines, labels)
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter("%Y")
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.grid(True)
    fig.autofmt_xdate()
    return fig, ax, lines
