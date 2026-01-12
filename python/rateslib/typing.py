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

# This module is reserved only for typing purposes.
# It avoids all circular import by performing a TYPE_CHECKING check on any component.

from collections.abc import Callable as Callable
from collections.abc import Sequence as Sequence
from datetime import datetime as datetime
from typing import Any as Any
from typing import NoReturn as NoReturn
from typing import Protocol, TypeAlias

import numpy as np
from pandas import DataFrame as DataFrame
from pandas import Series as Series

from rateslib.curves import RolledCurve as RolledCurve
from rateslib.curves import ShiftedCurve as ShiftedCurve
from rateslib.curves import TranslatedCurve as TranslatedCurve
from rateslib.curves import _BaseCurve as _BaseCurve
from rateslib.curves import _CurveMeta as _CurveMeta
from rateslib.data.fixings import FloatRateIndex as FloatRateIndex
from rateslib.data.fixings import FloatRateSeries as FloatRateSeries
from rateslib.data.fixings import FXFixing as FXFixing
from rateslib.data.fixings import FXIndex as FXIndex
from rateslib.data.fixings import IBORFixing as IBORFixing
from rateslib.data.fixings import IBORStubFixing as IBORStubFixing
from rateslib.data.fixings import IndexFixing as IndexFixing
from rateslib.data.fixings import RFRFixing as RFRFixing
from rateslib.data.loader import Fixings as Fixings
from rateslib.data.loader import _BaseFixingsLoader as _BaseFixingsLoader
from rateslib.dual.variable import Variable as Variable
from rateslib.enums.generics import NoInput as NoInput
from rateslib.enums.generics import Result as Result
from rateslib.enums.parameters import FloatFixingMethod as FloatFixingMethod
from rateslib.enums.parameters import FXDeltaMethod as FXDeltaMethod
from rateslib.enums.parameters import IndexMethod as IndexMethod
from rateslib.enums.parameters import OptionType as OptionType
from rateslib.enums.parameters import SpreadCompoundMethod as SpreadCompoundMethod
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
from rateslib.instruments.protocols.kwargs import _KWArgs as _KWArgs
from rateslib.instruments.protocols.pricing import _Curves as _Curves
from rateslib.instruments.protocols.pricing import _Vol as _Vol
from rateslib.legs import CreditPremiumLeg as CreditPremiumLeg
from rateslib.legs import CreditProtectionLeg as CreditProtectionLeg
from rateslib.legs import FixedLeg as FixedLeg
from rateslib.legs import FloatLeg as FloatLeg
from rateslib.legs import ZeroFixedLeg as ZeroFixedLeg
from rateslib.legs import ZeroFloatLeg as ZeroFloatLeg
from rateslib.legs import ZeroIndexLeg as ZeroIndexLeg
from rateslib.legs.protocols import _BaseLeg as _BaseLeg
from rateslib.periods import Cashflow as Cashflow
from rateslib.periods import CreditPremiumPeriod as CreditPremiumPeriod
from rateslib.periods import CreditProtectionPeriod as CreditProtectionPeriod
from rateslib.periods import FixedPeriod as FixedPeriod
from rateslib.periods import FloatPeriod as FloatPeriod
from rateslib.periods import FXCallPeriod as FXCallPeriod
from rateslib.periods import FXPutPeriod as FXPutPeriod
from rateslib.periods import _BaseFXOptionPeriod as _BaseFXOptionPeriod
from rateslib.periods.parameters import _FloatRateParams as _FloatRateParams
from rateslib.periods.parameters import _IndexParams as _IndexParams
from rateslib.periods.parameters import _NonDeliverableParams as _NonDeliverableParams
from rateslib.periods.parameters import _PeriodParams as _PeriodParams
from rateslib.periods.parameters import _SettlementParams as _SettlementParams
from rateslib.periods.protocols import _BasePeriod as _BasePeriod
from rateslib.rs import Adjuster as Adjuster
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

CurveInterpolator: TypeAlias = "FlatBackwardInterpolator | FlatForwardInterpolator | LinearInterpolator | LogLinearInterpolator | LinearZeroRateInterpolator | NullInterpolator"

from rateslib.rs import Convention as Convention
from rateslib.rs import Dual as Dual
from rateslib.rs import Dual2 as Dual2
from rateslib.rs import Frequency as Frequency
from rateslib.rs import PPSplineDual as PPSplineDual
from rateslib.rs import PPSplineDual2 as PPSplineDual2
from rateslib.rs import PPSplineF64 as PPSplineF64
from rateslib.rs import RollDay as RollDay
from rateslib.scheduling import Schedule as Schedule
from rateslib.solver import Solver as Solver

Solver_: TypeAlias = "Solver | NoInput"

CalTypes: TypeAlias = "Cal | UnionCal | NamedCal"
CalInput: TypeAlias = "CalTypes | str | NoInput"
Adjuster_: TypeAlias = "Adjuster | NoInput"
FXIndex_: TypeAlias = "FXIndex | NoInput"

DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"
DualTypes_: TypeAlias = "DualTypes | NoInput"

Number: TypeAlias = "float | Dual | Dual2"

# https://stackoverflow.com/questions/68916893/
Arr1dF64: TypeAlias = "np.ndarray[tuple[int], np.dtype[np.float64]]"
Arr2dF64: TypeAlias = "np.ndarray[tuple[int, int], np.dtype[np.float64]]"
Arr1dObj: TypeAlias = "np.ndarray[tuple[int], np.dtype[np.object_]]"
Arr2dObj: TypeAlias = "np.ndarray[tuple[int, int], np.dtype[np.object_]]"

PeriodFixings: TypeAlias = "DualTypes | Series[DualTypes] | str | NoInput"
LegFixings: TypeAlias = "PeriodFixings | list[PeriodFixings] | tuple[PeriodFixings, PeriodFixings]"

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

# _BaseCurve is an ABC
_BaseCurve_: TypeAlias = "_BaseCurve | NoInput"
_BaseCurveOrId: TypeAlias = "_BaseCurve | str"  # used as best practice for Solver mappings
_BaseCurveOrId_: TypeAlias = "_BaseCurveOrId | NoInput"
_BaseCurveOrIdDict: TypeAlias = (
    "dict[str, _BaseCurve | str] | dict[str, _BaseCurve] | dict[str, str]"
)
_BaseCurveDict: TypeAlias = "dict[str, _BaseCurve]"
_BaseCurveOrDict: TypeAlias = "_BaseCurve | _BaseCurveDict"
_BaseCurveOrIdOrIdDict: TypeAlias = "_BaseCurveOrId | _BaseCurveOrIdDict"
_BaseCurveOrDict_: TypeAlias = "_BaseCurve | _BaseCurveDict | NoInput"
_BaseCurveOrIdOrIdDict_: TypeAlias = "_BaseCurveOrId | _BaseCurveOrIdDict | NoInput"
CurvesT: TypeAlias = "_BaseCurveOrIdOrIdDict | Sequence[CurveOrId | CurveDict] | _Curves"
CurvesT_: TypeAlias = "CurvesT | NoInput"

_FXVolObj: TypeAlias = "FXDeltaVolSurface | FXDeltaVolSmile | FXSabrSmile | FXSabrSurface"
_FXVolOption: TypeAlias = "_FXVolObj | DualTypes"
_FXVolOption_: TypeAlias = "_FXVolOption | NoInput"

FXVol: TypeAlias = "_FXVolOption | str"
FXVol_: TypeAlias = "FXVol | NoInput"

VolT: TypeAlias = "FXVol | _Vol"
VolT_: TypeAlias = "VolT | NoInput"
FXVolStrat_: TypeAlias = "Sequence[FXVolStrat_] | VolT | NoInput"
SeqVolT_: TypeAlias = "Sequence[VolT_]"

CurveDict: TypeAlias = "dict[str, _BaseCurve | str] | dict[str, _BaseCurve] | dict[str, str]"
CurveOrId: TypeAlias = "_BaseCurve | str"
CurveOrId_: TypeAlias = "CurveOrId | NoInput"

CurveInput: TypeAlias = "CurveOrId | CurveDict"
CurveInput_: TypeAlias = "CurveInput | NoInput"

CurveOption: TypeAlias = "_BaseCurve | dict[str, _BaseCurve]"
CurveOption_: TypeAlias = "CurveOption | NoInput"

Curves: TypeAlias = "CurveOrId | CurveDict | Sequence[CurveOrId | CurveDict]"
Curves_: TypeAlias = "CurveOrId_ | CurveDict | Sequence[CurveOrId_ | CurveDict]"

Curves_Tuple: TypeAlias = "tuple[CurveOption_, CurveOption_, CurveOption_, CurveOption_]"
Curves_DiscTuple: TypeAlias = "tuple[CurveOption_, _BaseCurve_, CurveOption_, _BaseCurve_]"

# this is a type for a wrapped `rate_curve`, `disc_curve` and `index_curve`
PeriodCurves: TypeAlias = "tuple[CurveOption_, _BaseCurve_, _BaseCurve_]"

FX: TypeAlias = "DualTypes | FXRates | FXForwards"
FX_: TypeAlias = "FX | NoInput"
FXRevised_: TypeAlias = "FXRates | FXForwards | NoInput"
FXForwards_: TypeAlias = "FXForwards | NoInput"

# NPV: TypeAlias = "DualTypes | dict[str, DualTypes]"
#

# Leg: TypeAlias = "FixedLeg | FloatLeg | ZeroFloatLeg | ZeroFixedLeg | ZeroIndexLeg | CreditPremiumLeg | CreditProtectionLeg"
# Period: TypeAlias = "FixedPeriod | FloatPeriod | Cashflow | CreditPremiumPeriod | CreditProtectionPeriod"
#
# Security: TypeAlias = "FixedRateBond | FloatRateNote | Bill | IndexFixedRateBond"
# FXOptionTypes: TypeAlias = (
#     "FXCall | FXPut | FXRiskReversal | FXStraddle | FXStrangle | FXBrokerFly | FXOptionStrat"
# )
# RatesDerivative: TypeAlias = "IRS | SBS | FRA | ZCS | STIRFuture"
# IndexDerivative: TypeAlias = "IIRS | ZCIS"
# CurrencyDerivative: TypeAlias = "XCS | FXSwap | FXForward"
# Combinations: TypeAlias = "Portfolio | Fly | Spread | Value | VolValue"
#
# Instrument: TypeAlias = (
#     "Combinations | Security | FXOptionTypes | RatesDerivative | CDS | CurrencyDerivative"
# )


class SupportsRate:
    def rate(self, *args: Any, **kwargs: Any) -> DualTypes: ...  # type: ignore[empty-body]

    _rate_scalar: float


class SupportsMetrics:
    def rate(self, *args: Any, **kwargs: Any) -> DualTypes: ...  # type: ignore[empty-body]
    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]: ...  # type: ignore[empty-body]
    def delta(self, *args: Any, **kwargs: Any) -> DataFrame: ...  # type: ignore[empty-body]
    def gamma(self, *args: Any, **kwargs: Any) -> DataFrame: ...  # type: ignore[empty-body]
    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame: ...  # type: ignore[empty-body]
    def cashflows_table(self, *args: Any, **kwargs: Any) -> DataFrame: ...  # type: ignore[empty-body]


class _SupportsFixedFloatLeg1(Protocol):
    @property
    def leg1(self) -> FixedLeg | FloatLeg: ...
