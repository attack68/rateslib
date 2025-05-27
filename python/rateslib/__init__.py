__docformat__ = "restructuredtext"

# Let users know if they're missing any of our hard dependencies
_hard_dependencies = ("pandas", "matplotlib", "numpy")
_missing_dependencies: list[str] = []

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

    def __init__(self, *args) -> None:  # type: ignore[no-untyped-def]
        if len(args) % 2 != 0 or len(args) < 2:
            raise ValueError("Need to invoke as option_context(pat, val, [(pat, val), ...]).")

        self.ops = list(zip(args[::2], args[1::2], strict=False))

    def __enter__(self) -> None:
        self.undo = [(pat, getattr(defaults, pat, None)) for pat, _ in self.ops]

        for pat, val in self.ops:
            setattr(defaults, pat, val)

    def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
        if self.undo:
            for pat, val in self.undo:
                setattr(defaults, pat, val)


from rateslib.calendars import (
    Cal,
    NamedCal,
    UnionCal,
    add_tenor,
    create_calendar,
    dcf,
    get_calendar,
    get_imm,
)
from rateslib.curves import (
    CompositeCurve,
    Curve,
    LineCurve,
    MultiCsaCurve,
    ProxyCurve,
    index_left,
    index_value,
)
from rateslib.dual import Dual, Dual2, Variable, dual_exp, dual_log, dual_solve, gradient
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile, FXSabrSurface
from rateslib.instruments import (
    CDS,
    FRA,
    IIRS,
    IRS,
    NDF,
    SBS,
    XCS,
    ZCIS,
    ZCS,
    Bill,
    BillCalcMode,
    BondCalcMode,
    BondFuture,
    FixedRateBond,
    FloatRateNote,
    Fly,
    FXBrokerFly,
    FXCall,
    FXExchange,
    FXPut,
    FXRiskReversal,
    FXStraddle,
    FXStrangle,
    FXSwap,
    IndexFixedRateBond,
    Portfolio,
    Spread,
    STIRFuture,
    Value,
    VolValue,
)
from rateslib.legs import (
    CreditPremiumLeg,
    CreditProtectionLeg,
    CustomLeg,
    FixedLeg,
    FixedLegMtm,
    FloatLeg,
    FloatLegMtm,
    IndexFixedLeg,
    ZeroFixedLeg,
    ZeroFloatLeg,
    ZeroIndexLeg,
)
from rateslib.periods import (
    Cashflow,
    CreditPremiumPeriod,
    CreditProtectionPeriod,
    FixedPeriod,
    FloatPeriod,
    FXCallPeriod,
    FXPutPeriod,
    IndexCashflow,
    IndexFixedPeriod,
    NonDeliverableCashflow,
)
from rateslib.scheduling import Schedule
from rateslib.serialization import from_json
from rateslib.solver import Solver
from rateslib.splines import (
    PPSpline,
    PPSplineDual,
    PPSplineDual2,
    PPSplineF64,
    bspldnev_single,
    bsplev_single,
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
"""  # noqa: A001

# Use __all__ to let type checkers know what is part of the public API.
# Rateslib is not (yet) a py.typed library: the public API is determined
# based on the documentation.
__all__ = [
    "dt",
    "defaults",
    "NoInput",
    "from_json",
    # dual.py
    "Dual",
    "Dual2",
    "Variable",
    "dual_log",
    "dual_exp",
    "dual_solve",
    "gradient",
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
    "get_imm",
    "add_tenor",
    "dcf",
    "Cal",
    "UnionCal",
    "NamedCal",
    # scheduling.py
    "Schedule",
    # curves.py
    "Curve",
    "LineCurve",
    "MultiCsaCurve",
    "CompositeCurve",
    "ProxyCurve",
    "index_left",
    "index_value",
    # fx_volatility.py
    "FXDeltaVolSmile",
    "FXDeltaVolSurface",
    "FXSabrSmile",
    "FXSabrSurface",
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
    "NonDeliverableCashflow",
    "CreditPremiumPeriod",
    "CreditProtectionPeriod",
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
    "CreditPremiumLeg",
    "CreditProtectionLeg",
    # instruments.py
    "FixedRateBond",
    "IndexFixedRateBond",
    "FloatRateNote",
    "BondFuture",
    "BondCalcMode",
    "CDS",
    "FRA",
    "Value",
    "VolValue",
    "Bill",
    "BillCalcMode",
    "IRS",
    "NDF",
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

__version__ = "2.0.0"
