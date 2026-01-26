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

from rateslib.data.loader import Fixings
from rateslib.default import Defaults

defaults = Defaults()
fixings = Fixings()

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


from rateslib.curves import (
    CompositeCurve,
    Curve,
    LineCurve,
    MultiCsaCurve,
    ProxyCurve,
    index_left,
    index_value,
)
from rateslib.curves.academic import (
    NelsonSiegelCurve,
    NelsonSiegelSvenssonCurve,
    SmithWilsonCurve,
)
from rateslib.data.fixings import (
    FloatRateIndex,
    FloatRateSeries,
    FXFixing,
    FXIndex,
    IBORFixing,
    IBORStubFixing,
    IndexFixing,
    RFRFixing,
)
from rateslib.dual import ADOrder, Dual, Dual2, Variable, dual_exp, dual_log, dual_solve, gradient
from rateslib.enums.generics import NoInput
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile, FXSabrSurface
from rateslib.instruments import (
    CDS,
    FRA,
    IIRS,
    IRS,
    NDF,
    NDXCS,
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
    FXForward,
    FXPut,
    FXRiskReversal,
    FXStraddle,
    FXStrangle,
    FXSwap,
    FXVolValue,
    IndexFixedRateBond,
    Portfolio,
    Spread,
    STIRFuture,
    Value,
)
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
from rateslib.periods import (
    Cashflow,
    CreditPremiumPeriod,
    CreditProtectionPeriod,
    FixedPeriod,
    FloatPeriod,
    FXCallPeriod,
    FXPutPeriod,
    ZeroFixedPeriod,
    ZeroFloatPeriod,
)
from rateslib.scheduling import (
    Adjuster,
    Cal,
    Convention,
    Frequency,
    Imm,
    NamedCal,
    RollDay,
    Schedule,
    StubInference,
    UnionCal,
    add_tenor,
    dcf,
    get_calendar,
    get_imm,
    next_imm,
)
from rateslib.serialization import from_json
from rateslib.solver import Solver
from rateslib.splines import (
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

__all__ = [
    "dt",
    "defaults",
    "fixings",
    "NoInput",
    "from_json",
    # dual.py
    "ADOrder",
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
    "PPSplineF64",
    "PPSplineDual",
    "PPSplineDual2",
    # scheduling.py
    "get_calendar",
    "get_imm",
    "next_imm",
    "add_tenor",
    "dcf",
    "Cal",
    "UnionCal",
    "NamedCal",
    "Schedule",
    "Frequency",
    "RollDay",
    "Adjuster",
    "StubInference",
    "Convention",
    "Imm",
    # curves.py
    "Curve",
    "LineCurve",
    "MultiCsaCurve",
    "CompositeCurve",
    "ProxyCurve",
    "index_left",
    "index_value",
    # academic curves
    "NelsonSiegelCurve",
    "NelsonSiegelSvenssonCurve",
    "SmithWilsonCurve",
    # fixings.py
    "FXFixing",
    "IBORFixing",
    "IBORStubFixing",
    "IndexFixing",
    "RFRFixing",
    "FXIndex",
    "FloatRateIndex",
    "FloatRateSeries",
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
    "ZeroFixedPeriod",
    "ZeroFloatPeriod",
    "Cashflow",
    "FXCallPeriod",
    "FXPutPeriod",
    "CreditPremiumPeriod",
    "CreditProtectionPeriod",
    # legs.py
    "Amortization",
    "FixedLeg",
    "FloatLeg",
    "ZeroFloatLeg",
    "ZeroFixedLeg",
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
    "FXVolValue",
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
    "FXForward",
    "XCS",
    "NDXCS",
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

__version__ = "2.5.1"
