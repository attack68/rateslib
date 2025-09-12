from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pandas import DataFrame

import rateslib.errors as err
from rateslib.enums.generics import Err

if TYPE_CHECKING:
    from rateslib.typing import (
        Result,
    )

class _WithRateFixingsExposureStatic(Protocol):
    def try_unindexed_reference_fixings_exposure(self) -> Result[DataFrame]:
        return Err(TypeError(err.TE_NO_FIXING_EXPOSURE_ON_OBJ.format(type(self).__name__)))