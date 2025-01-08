from typing import TypeAlias

import numpy as np

from rateslib.default import NoInput
from rateslib.dual.variable import Variable
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface
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

CalTypes: TypeAlias = "Cal | UnionCal | NamedCal"
CalInput: TypeAlias = "CalTypes | str | NoInput"

DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"
Number: TypeAlias = "float | Dual | Dual2"

# https://stackoverflow.com/questions/68916893/
Arr1dF64: TypeAlias = "np.ndarray[tuple[int], np.dtype[np.float64]]"
Arr2dF64: TypeAlias = "np.ndarray[tuple[int, int], np.dtype[np.float64]]"
Arr1dObj: TypeAlias = "np.ndarray[tuple[int], np.dtype[np.object_]]"
Arr2dObj: TypeAlias = "np.ndarray[tuple[int, int], np.dtype[np.object_]]"

Vol: TypeAlias = "DualTypes | FXDeltaVolSmile | FXDeltaVolSurface | str | NoInput"
VolInput: TypeAlias = "str | FXDeltaVolSmile | FXDeltaVolSurface | NoInput"
VolOption: TypeAlias = "FXDeltaVolSmile | DualTypes | FXDeltaVolSurface | NoInput"

FX: TypeAlias = "DualTypes | FXRates | FXForwards | NoInput"
NPV: TypeAlias = "DualTypes | dict[str, DualTypes]"

CurveInterpolator: TypeAlias = (
    FlatBackwardInterpolator
    | FlatForwardInterpolator
    | LinearInterpolator
    | LogLinearInterpolator
    | LinearZeroRateInterpolator
    | NullInterpolator
)
