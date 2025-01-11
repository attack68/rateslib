from collections.abc import Callable as Callable
from collections.abc import Sequence as Sequence
from datetime import datetime as datetime
from typing import Any as Any
from typing import NoReturn as NoReturn
from typing import TypeAlias

import numpy as np
from pandas import DataFrame as DataFrame

from rateslib.curves import Curve
from rateslib.default import NoInput
from rateslib.dual.variable import Variable
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface
from rateslib.instruments import CDS as CDS
from rateslib.instruments import FRA as FRA
from rateslib.instruments import IIRS as IIRS
from rateslib.instruments import IRS as IRS
from rateslib.instruments import SBS as SBS
from rateslib.instruments import XCS as XCS
from rateslib.instruments import ZCIS as ZCIS
from rateslib.instruments import ZCS as ZCS
from rateslib.instruments import Bill as Bill
from rateslib.instruments import FixedRateBond as FixedRateBond
from rateslib.instruments import FloatRateNote as FloatRateNote
from rateslib.instruments import FXBrokerFly as FXBrokerFly
from rateslib.instruments import FXCall as FXCall
from rateslib.instruments import FXExchange as FXExchange
from rateslib.instruments import FXOptionStrat as FXOptionStrat
from rateslib.instruments import FXPut as FXPut
from rateslib.instruments import FXRiskReversal as FXRiskReversal
from rateslib.instruments import FXStraddle as FXStraddle
from rateslib.instruments import FXStrangle as FXStrangle
from rateslib.instruments import FXSwap as FXSwap
from rateslib.instruments import IndexFixedRateBond as IndexFixedRateBond
from rateslib.instruments import STIRFuture as STIRFuture
from rateslib.legs import (
    CreditPremiumLeg,
    CreditProtectionLeg,
    FixedLeg,
    FloatLeg,
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
    IndexCashflow,
    IndexFixedPeriod,
)
from rateslib.rs import (
    Cal,
    Dual,
    Dual2,
    FlatBackwardInterpolator,
    FlatForwardInterpolator,
    LinearInterpolator,
    LinearZeroRateInterpolator,
    LogLinearInterpolator,
    NamedCal,
    NullInterpolator,
    UnionCal,
)
from rateslib.solver import Solver as Solver

CalTypes: TypeAlias = "Cal | UnionCal | NamedCal"
CalInput: TypeAlias = "CalTypes | str | NoInput"

DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"
Number: TypeAlias = "float | Dual | Dual2"

# https://stackoverflow.com/questions/68916893/
Arr1dF64: TypeAlias = "np.ndarray[tuple[int], np.dtype[np.float64]]"
Arr2dF64: TypeAlias = "np.ndarray[tuple[int, int], np.dtype[np.float64]]"
Arr1dObj: TypeAlias = "np.ndarray[tuple[int], np.dtype[np.object_]]"
Arr2dObj: TypeAlias = "np.ndarray[tuple[int, int], np.dtype[np.object_]]"

Curves_: TypeAlias = (
    "list[str | Curve | dict[str, Curve | str] | NoInput] | Curve | str | dict[str, Curve | str]"  # noqa: E501
)
Curves: TypeAlias = "Curves_ | NoInput"
CurveInput_: TypeAlias = "str | Curve | dict[str, str | Curve]"
CurveInput: TypeAlias = "CurveInput_ | NoInput"
CurveOption_: TypeAlias = "Curve | dict[str, Curve]"
CurveOption: TypeAlias = "CurveOption_ | NoInput"
CurvesTuple: TypeAlias = "tuple[CurveOption, CurveOption, CurveOption, CurveOption]"

Vol_: TypeAlias = "DualTypes | FXDeltaVolSmile | FXDeltaVolSurface | str"
Vol: TypeAlias = "Vol_ | NoInput"

VolInput_: TypeAlias = "str | FXDeltaVolSmile | FXDeltaVolSurface"
VolInput: TypeAlias = "VolInput_ | NoInput"

VolOption_: TypeAlias = "FXDeltaVolSmile | DualTypes | FXDeltaVolSurface"
VolOption: TypeAlias = "VolOption_ | NoInput"

FX_: TypeAlias = "DualTypes | FXRates | FXForwards"
FX: TypeAlias = "FX_ | NoInput"

NPV: TypeAlias = "DualTypes | dict[str, DualTypes]"

CurveInterpolator: TypeAlias = "FlatBackwardInterpolator | FlatForwardInterpolator | LinearInterpolator | LogLinearInterpolator | LinearZeroRateInterpolator | NullInterpolator"  # noqa: E501
Leg: TypeAlias = "FixedLeg | FloatLeg | IndexFixedLeg | ZeroFloatLeg | ZeroFixedLeg | ZeroIndexLeg | CreditPremiumLeg | CreditProtectionLeg"  # noqa: E501
Period: TypeAlias = "FixedPeriod | FloatPeriod | Cashflow | IndexFixedPeriod | IndexCashflow | CreditPremiumPeriod | CreditProtectionPeriod"  # noqa: E501

Security: TypeAlias = "FixedRateBond | FloatRateNote | Bill | IndexFixedRateBond"
FXOptionTypes: TypeAlias = (
    "FXCall | FXPut | FXRiskReversal | FXStraddle | FXStrangle | FXBrokerFly | FXOptionStrat"
)
RatesDerivative: TypeAlias = "IRS | SBS | FRA | ZCS | STIRFuture"
IndexDerivative: TypeAlias = "IIRS | ZCIS"
CurrencyDerivative: TypeAlias = "XCS | FXSwap | FXExchange"

Instrument: TypeAlias = "Security | FXOptionTypes | RatesDerivative | CDS | CurrencyDerivative"
