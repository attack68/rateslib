from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import Series

from rateslib.data.fixings import FXFixing
from rateslib.enums.parameters import FXOptionMetric, _get_fx_option_metric

if TYPE_CHECKING:
    from rateslib.typing import (
        DualTypes,
        DualTypes_,
        FXDeltaMethod,
        OptionType,
        datetime,
        str_,
    )


class _FXOptionParams:
    """
    Parameters for *FX Option Period* cashflows.
    """
    _expiry: datetime
    _delivery: datetime
    _pair: str
    _delta_type: FXDeltaMethod
    _metric: FXOptionMetric
    _option_fixing: FXFixing
    _strike: DualTypes_
    _currency: str
    _direction: OptionType

    def __init__(
        self,
        _direction: OptionType,
        _expiry: datetime,
        _delivery: datetime,
        _pair: str,
        _delta_type: FXDeltaMethod,
        _metric: str | FXOptionMetric,
        _option_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        _strike: DualTypes_,
    ):
        self._direction = _direction
        self._expiry = _expiry
        self._delivery = _delivery
        self._pair = _pair.lower()
        self._delta_type = _delta_type
        self._metric = _get_fx_option_metric(_metric)
        self._strike = _strike
        if isinstance(_option_fixings, Series):
            value = FXFixing._lookup(timeseries=_option_fixings, date=self.delivery)
            self._option_fixing = FXFixing(
                date=self.delivery,
                value=value,
                pair=self.pair,
            )
        elif isinstance(_option_fixings, str):
            self._option_fixing = FXFixing(
                date=self.delivery,
                identifier=_option_fixings,
                pair=self.pair,
            )
        else:
            self._option_fixing = FXFixing(
                date=self.delivery,
                value=_option_fixings,
                pair=self.pair,
            )

    @property
    def expiry(self) -> datetime:
        """The expiry date of the option."""
        return self._expiry

    @property
    def delivery(self) -> datetime:
        """The date of the FX rate exchange for the FX rate used for settlement of the option."""
        return self._delivery

    @property
    def pair(self) -> str:
        """The currency pair for settlement of the option."""
        return self._pair

    @property
    def direction(self) -> OptionType:
        """The direction of the option."""
        return self._direction

    @property
    def strike(self) -> DualTypes_:
        """The strike price of the option."""
        return self._strike

    @strike.setter
    def strike(self, val: DualTypes_) -> None:
        self._strike = val

    @property
    def option_fixing(self) -> FXFixing:
        """The FX fixing related to settlement of the option."""
        return self._option_fixing

    @property
    def metric(self) -> FXOptionMetric:
        """The default pricing quoting of the option."""
        return self._metric

    @property
    def delta_type(self) -> FXDeltaMethod:
        """The delta type used by the option to define its delta."""
        return self._delta_type

    def time_to_expiry(self, now: datetime) -> float:
        """The time to expiry of the option in years measured by calendar days from ``now``."""
        # TODO make this a dual, associated with theta
        return (self.expiry - now).days / 365.0
