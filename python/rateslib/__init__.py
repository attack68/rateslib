__docformat__ = "restructuredtext"

# Let users know if they're missing any of our hard dependencies
_hard_dependencies = ("pandas", "matplotlib", "numpy")
_missing_dependencies = []

for _dependency in _hard_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:  # pragma: no cover
        raise ImportError(f"`rateslib` requires installation of {_dependency}: {_e}")

from datetime import datetime as dt

from rateslib.default import Defaults, NoInput

defaults = Defaults()

from contextlib import ContextDecorator


class default_context(ContextDecorator):
    """
    Context manager to temporarily set options in the `with` statement context.

    You need to invoke as ``option_context(pat, val, [(pat, val), ...])``.

    Examples
    --------
    >>> with option_context('convention', "act360", 'frequency', "S"):
    ...     pass
    """

    def __init__(self, *args) -> None:
        if len(args) % 2 != 0 or len(args) < 2:
            raise ValueError("Need to invoke as option_context(pat, val, [(pat, val), ...]).")

        self.ops = list(zip(args[::2], args[1::2]))

    def __enter__(self) -> None:
        self.undo = [(pat, getattr(defaults, pat, None)) for pat, _ in self.ops]

        for pat, val in self.ops:
            setattr(defaults, pat, val)

    def __exit__(self, *args) -> None:
        if self.undo:
            for pat, val in self.undo:
                setattr(defaults, pat, val)


from rateslib.dual import Dual, Dual2, dual_log, dual_exp, dual_solve

from rateslib.calendars import create_calendar, get_calendar, add_tenor, dcf

from rateslib.splines import (
    bsplev_single,
    bspldnev_single,
    PPSpline,
    PPSplineF64,
    PPSplineDual,
    PPSplineDual2,
)

from rateslib.scheduling import Schedule

from rateslib.curves import (
    Curve,
    LineCurve,
    interpolate,
    index_left,
    IndexCurve,
    CompositeCurve,
    ProxyCurve,
)

from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface

from rateslib.fx import (
    FXRates,
    FXForwards,
)

from rateslib.solver import Solver

from rateslib.periods import (
    FixedPeriod,
    FloatPeriod,
    Cashflow,
    IndexFixedPeriod,
    IndexCashflow,
    FXPutPeriod,
    FXCallPeriod,
)

from rateslib.legs import (
    FixedLeg,
    FixedLegMtm,
    FloatLeg,
    FloatLegMtm,
    ZeroFloatLeg,
    ZeroFixedLeg,
    ZeroIndexLeg,
    IndexFixedLeg,
    CustomLeg,
)

from rateslib.instruments import (
    Value,
    VolValue,
    Bill,
    FixedRateBond,
    IndexFixedRateBond,
    FloatRateNote,
    BondFuture,
    IRS,
    STIRFuture,
    IIRS,
    ZCS,
    ZCIS,
    FRA,
    SBS,
    FXSwap,
    FXExchange,
    XCS,
    Spread,
    Fly,
    Portfolio,
    FXCall,
    FXPut,
    FXRiskReversal,
    FXStraddle,
    FXStrangle,
    FXBrokerFly,
)

# module level doc-string
__doc__ = """
RatesLib - An efficient and interconnected fixed income library for Python
==========================================================================

**rateslib** is a Python package providing fast, flexible, and accurate
fixed income instrument configuration and calculation.
It aims to be the fundamental high-level building block for practical analysis of
fixed income securities, derivatives, FX representation and curve construction
in Python.
"""

# Use __all__ to let type checkers know what is part of the public API.
# Rateslib is not (yet) a py.typed library: the public API is determined
# based on the documentation.
__all__ = [
    "dt",
    "defaults",
    "NoInput",
    # dual.py
    "Dual",
    "Dual2",
    "dual_log",
    "dual_exp",
    "dual_solve",
    # splines.py
    "bsplev_single",
    "bspldnev_single",
    "PPSpline",
    "PPSplineF64",
    "PPSplineDual",
    "PPSplineDual2",
    # calendars.py
    "create_calendar",
    "get_calendar",
    "add_tenor",
    "dcf",
    # scheduling.py
    "Schedule",
    # curves.py
    "Curve",
    "LineCurve",
    "IndexCurve",
    "CompositeCurve",
    "ProxyCurve",
    "interpolate",
    "index_left",
    # fx_volatility.py
    "FXDeltaVolSmile",
    "FXDeltaVolSurface",
    # solver.py
    "Solver",
    # fx.py
    "FXRates",
    "FXForwards",
    # periods.py,
    "FixedPeriod",
    "FloatPeriod",
    "Cashflow",
    "IndexCashflow",
    "IndexFixedPeriod",
    "FXCallPeriod",
    "FXPutPeriod",
    # legs.py
    "FixedLeg",
    "FloatLeg",
    "ZeroFloatLeg",
    "ZeroFixedLeg",
    "FixedLegMtm",
    "FloatLegMtm",
    "IndexFixedLeg",
    "ZeroIndexLeg",
    "CustomLeg",
    # instruments.py
    "FixedRateBond",
    "IndexFixedRateBond",
    "FloatRateNote",
    "BondFuture",
    "FRA",
    "Value",
    "VolValue",
    "Bill",
    "IRS",
    "STIRFuture",
    "IIRS",
    "ZCS",
    "ZCIS",
    "SBS",
    "FXSwap",
    "FXExchange",
    "XCS",
    "Spread",
    "Fly",
    "Portfolio",
    "FXCall",
    "FXPut",
    "FXRiskReversal",
    "FXStraddle",
    "FXStrangle",
    "FXBrokerFly",
]

__version__ = "1.2.3"
