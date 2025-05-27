# This module is reserved only for typing purposes.
# It avoids all circular import by performing a TYPE_CHECKING check on any component.

from collections.abc import Callable as Callable
from collections.abc import Sequence as Sequence
from datetime import datetime as datetime
from typing import Any as Any
from typing import NoReturn as NoReturn
from typing import TypeAlias

import numpy as np
from pandas import DataFrame as DataFrame
from pandas import Series as Series

from rateslib.default import NoInput as NoInput
from rateslib.dual.variable import Variable as Variable
from rateslib.fx import FXForwards as FXForwards
from rateslib.fx import FXRates as FXRates
from rateslib.fx_volatility import FXDeltaVolSmile as FXDeltaVolSmile
from rateslib.fx_volatility import FXDeltaVolSurface as FXDeltaVolSurface
from rateslib.fx_volatility import FXSabrSmile as FXSabrSmile
from rateslib.fx_volatility import FXSabrSurface as FXSabrSurface
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
from rateslib.instruments import Fly as Fly
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
from rateslib.instruments import Portfolio as Portfolio
from rateslib.instruments import Spread as Spread
from rateslib.instruments import STIRFuture as STIRFuture
from rateslib.instruments import Value as Value
from rateslib.instruments import VolValue as VolValue
from rateslib.instruments.bonds import BondMixin as BondMixin
from rateslib.legs import CreditPremiumLeg as CreditPremiumLeg
from rateslib.legs import CreditProtectionLeg as CreditProtectionLeg
from rateslib.legs import FixedLeg as FixedLeg
from rateslib.legs import FloatLeg as FloatLeg
from rateslib.legs import IndexFixedLeg as IndexFixedLeg
from rateslib.legs import ZeroFixedLeg as ZeroFixedLeg
from rateslib.legs import ZeroFloatLeg as ZeroFloatLeg
from rateslib.legs import ZeroIndexLeg as ZeroIndexLeg
from rateslib.periods import Cashflow as Cashflow
from rateslib.periods import CreditPremiumPeriod as CreditPremiumPeriod
from rateslib.periods import CreditProtectionPeriod as CreditProtectionPeriod
from rateslib.periods import FixedPeriod as FixedPeriod
from rateslib.periods import FloatPeriod as FloatPeriod
from rateslib.periods import FXCallPeriod as FXCallPeriod
from rateslib.periods import FXOptionPeriod as FXOptionPeriod
from rateslib.periods import FXPutPeriod as FXPutPeriod
from rateslib.periods import IndexCashflow as IndexCashflow
from rateslib.periods import IndexFixedPeriod as IndexFixedPeriod
from rateslib.rs import (
    Cal,
    FlatBackwardInterpolator,
    FlatForwardInterpolator,
    LinearInterpolator,
    LinearZeroRateInterpolator,
    LogLinearInterpolator,
    NamedCal,
    NullInterpolator,
    UnionCal,
)
from rateslib.rs import Dual as Dual
from rateslib.rs import Dual2 as Dual2
from rateslib.rs import PPSplineDual as PPSplineDual
from rateslib.rs import PPSplineDual2 as PPSplineDual2
from rateslib.rs import PPSplineF64 as PPSplineF64
from rateslib.scheduling import Schedule as Schedule
from rateslib.solver import Solver as Solver

Solver_: TypeAlias = "Solver | NoInput"

CalTypes: TypeAlias = "Cal | UnionCal | NamedCal"
CalInput: TypeAlias = "CalTypes | str | NoInput"

DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"
DualTypes_: TypeAlias = "DualTypes | NoInput"

Number: TypeAlias = "float | Dual | Dual2"

# https://stackoverflow.com/questions/68916893/
Arr1dF64: TypeAlias = "np.ndarray[tuple[int], np.dtype[np.float64]]"
Arr2dF64: TypeAlias = "np.ndarray[tuple[int, int], np.dtype[np.float64]]"
Arr1dObj: TypeAlias = "np.ndarray[tuple[int], np.dtype[np.object_]]"
Arr2dObj: TypeAlias = "np.ndarray[tuple[int, int], np.dtype[np.object_]]"

FixingsRates: TypeAlias = "Series[DualTypes] | list[DualTypes | list[DualTypes] | Series[DualTypes] | NoInput] | tuple[DualTypes, Series[DualTypes]] | DualTypes"
FixingsRates_: TypeAlias = "FixingsRates | NoInput"

FixingsFx: TypeAlias = (
    "DualTypes | list[DualTypes] | Series[DualTypes] | tuple[DualTypes, Series[DualTypes]]"
)
FixingsFx_: TypeAlias = "FixingsFx | NoInput"

str_: TypeAlias = "str | NoInput"
bool_: TypeAlias = "bool | NoInput"
int_: TypeAlias = "int | NoInput"
datetime_: TypeAlias = "datetime | NoInput"
float_: TypeAlias = "float | NoInput"

from rateslib.curves import Curve as Curve  # noqa: E402

Curve_: TypeAlias = "Curve | NoInput"
CurveDict: TypeAlias = "dict[str, Curve | str] | dict[str, Curve] | dict[str, str]"

CurveOrId: TypeAlias = "Curve | str"
CurveOrId_: TypeAlias = "CurveOrId | NoInput"

CurveInput: TypeAlias = "CurveOrId | CurveDict"
CurveInput_: TypeAlias = "CurveInput | NoInput"

CurveOption: TypeAlias = "Curve | dict[str, Curve]"
CurveOption_: TypeAlias = "CurveOption | NoInput"

Curves: TypeAlias = "CurveOrId | CurveDict | Sequence[CurveOrId | CurveDict]"
Curves_: TypeAlias = "CurveOrId_ | CurveDict | Sequence[CurveOrId_ | CurveDict]"

Curves_Tuple: TypeAlias = "tuple[CurveOption_, CurveOption_, CurveOption_, CurveOption_]"
Curves_DiscTuple: TypeAlias = "tuple[CurveOption_, Curve_, CurveOption_, Curve_]"

FXVolObj: TypeAlias = "FXDeltaVolSurface | FXDeltaVolSmile | FXSabrSmile | FXSabrSurface"

FXVolOption: TypeAlias = "FXVolObj | DualTypes"
FXVolOption_: TypeAlias = "FXVolOption | NoInput"

FXVol: TypeAlias = "FXVolOption | str"
FXVol_: TypeAlias = "FXVol | NoInput"

VolInput_: TypeAlias = "str | FXDeltaVolSmile | FXDeltaVolSurface | FXSabrSmile | FXSabrSurface"
VolInput: TypeAlias = "VolInput_ | NoInput"

FXVolStrat_: TypeAlias = "Sequence[FXVolStrat_] | FXVol_"
ListFXVol_: TypeAlias = "list[ListFXVol_ | FXVol_]"

FX: TypeAlias = "DualTypes | FXRates | FXForwards"
FX_: TypeAlias = "FX | NoInput"

NPV: TypeAlias = "DualTypes | dict[str, DualTypes]"

CurveInterpolator: TypeAlias = "FlatBackwardInterpolator | FlatForwardInterpolator | LinearInterpolator | LogLinearInterpolator | LinearZeroRateInterpolator | NullInterpolator"
Leg: TypeAlias = "FixedLeg | FloatLeg | IndexFixedLeg | ZeroFloatLeg | ZeroFixedLeg | ZeroIndexLeg | CreditPremiumLeg | CreditProtectionLeg"
Period: TypeAlias = "FixedPeriod | FloatPeriod | Cashflow | IndexFixedPeriod | IndexCashflow | CreditPremiumPeriod | CreditProtectionPeriod"

Security: TypeAlias = "FixedRateBond | FloatRateNote | Bill | IndexFixedRateBond"
FXOptionTypes: TypeAlias = (
    "FXCall | FXPut | FXRiskReversal | FXStraddle | FXStrangle | FXBrokerFly | FXOptionStrat"
)
RatesDerivative: TypeAlias = "IRS | SBS | FRA | ZCS | STIRFuture"
IndexDerivative: TypeAlias = "IIRS | ZCIS"
CurrencyDerivative: TypeAlias = "XCS | FXSwap | FXExchange"
Combinations: TypeAlias = "Portfolio | Fly | Spread | Value | VolValue"

Instrument: TypeAlias = (
    "Combinations | Security | FXOptionTypes | RatesDerivative | CDS | CurrencyDerivative"
)


class SupportsRate:
    def rate(self, *args: Any, **kwargs: Any) -> DualTypes: ...  # type: ignore[empty-body]

    _rate_scalar: float


class SupportsMetrics:
    def rate(self, *args: Any, **kwargs: Any) -> DualTypes: ...  # type: ignore[empty-body]
    def npv(self, *args: Any, **kwargs: Any) -> NPV: ...  # type: ignore[empty-body]
    def delta(self, *args: Any, **kwargs: Any) -> DataFrame: ...  # type: ignore[empty-body]
    def gamma(self, *args: Any, **kwargs: Any) -> DataFrame: ...  # type: ignore[empty-body]
    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame: ...  # type: ignore[empty-body]
    def cashflows_table(self, *args: Any, **kwargs: Any) -> DataFrame: ...  # type: ignore[empty-body]
