from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import Series

from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        LegFixings,
        PeriodFixings,
    )


def _leg_fixings_to_list(rate_fixings: LegFixings, n_periods: int) -> list[PeriodFixings]:  # type: ignore[type-var]
    """Perform a conversion of 'LegRateFixings' into a list of PeriodFixings."""
    if isinstance(rate_fixings, NoInput):
        # NoInput is converted to a list of NoInputs
        return [NoInput(0)] * n_periods
    elif isinstance(rate_fixings, tuple):
        # A tuple must be a 2-tuple which is converted to a first item and then multiplied.
        return [rate_fixings[0]] + [rate_fixings[1]] * (n_periods - 1)
    elif isinstance(rate_fixings, list):
        # A list is padded with NoInputs
        return rate_fixings + [NoInput(0)] * (n_periods - len(rate_fixings))
    elif isinstance(rate_fixings, str | Series):
        # A string or seried is multiplied
        return [rate_fixings] * n_periods
    else:
        # A scalar value is padded with NoInputs.
        return [rate_fixings] + [NoInput(0)] * (n_periods - 1)  # type: ignore[return-value]
