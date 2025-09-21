from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from pandas import Series

from rateslib.enums import Err, Ok
from rateslib.periods.components.parameters.settlement import _init_fx_fixing

if TYPE_CHECKING:
    from rateslib.typing import DualTypes, FXFixing, FXForwards_, Result, datetime, str_


class _MtmParams:
    _fx_fixing_start: FXFixing
    _fx_fixing_end: FXFixing
    _currency: str

    def __init__(
        self,
        _fx_fixing_start: FXFixing,
        _fx_fixing_end: FXFixing,
        _currency: str,
    ) -> None:
        self._fx_fixing_start = _fx_fixing_start
        self._fx_fixing_end = _fx_fixing_end
        self._currency = _currency

    @property
    def fx_fixing_start(self) -> FXFixing:
        """The FX fixing measured at the start of the period."""
        return self._fx_fixing_start

    @property
    def fx_fixing_end(self) -> FXFixing:
        """The FX fixing measured at the start of the period."""
        return self._fx_fixing_end

    @property
    def currency(self) -> str:
        """The settlement currency of the period."""
        return self._currency

    @property
    def pair(self) -> str:
        """The pair that defined the FX fixings."""
        return self.fx_fixing_start.pair

    @cached_property
    def fx_reversed(self) -> bool:
        return self.currency == self.pair[:3]

    def try_fixing_change(self, fx: FXForwards_) -> Result[DualTypes]:
        """Calculate the change between the FX fixing at the start and end of the period."""
        fx0 = self.fx_fixing_start.try_value_or_forecast(fx=fx)
        fx1 = self.fx_fixing_end.try_value_or_forecast(fx=fx)
        if isinstance(fx0, Err):
            return fx0
        if isinstance(fx1, Err):
            return fx1
        else:
            return Ok(fx1.unwrap() - fx0.unwrap())


def _init_MtmParams(
    _pair: str,
    _start: datetime,
    _end: datetime,
    _fx_fixings_start: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
    _fx_fixings_end: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
    _currency: str,
) -> _MtmParams:
    return _MtmParams(
        _fx_fixing_start=_init_fx_fixing(date=_start, pair=_pair, fixings=_fx_fixings_start),
        _fx_fixing_end=_init_fx_fixing(date=_end, pair=_pair, fixings=_fx_fixings_end),
        _currency=_currency,
    )
