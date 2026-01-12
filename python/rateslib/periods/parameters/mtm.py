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

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from pandas import Series

from rateslib.enums import Err, Ok
from rateslib.periods.parameters.settlement import _init_fx_fixing

if TYPE_CHECKING:
    from rateslib.typing import (
        DualTypes,
        FXFixing,
        FXForwards_,
        FXIndex,
        Result,
        datetime,
        str_,
    )


class _MtmParams:
    """
    Parameters for *Period* cashflows associated with multiple
    :class:`~rateslib.data.fixings.FXFixing`.

    Parameters
    ----------
    _fx_fixing_start: FXFixing
        The :class:`~rateslib.data.fixings.FXFixing` that is determined at the start of the
        *Period*.
    _fx_fixing_end: FXFixing
        The :class:`~rateslib.data.fixings.FXFixing` that is determined at the end of the *Period*.
    _currency: str
        The local *settlement currency* of the *Period*.
    """

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
        """The  :class:`~rateslib.data.fixings.FXFixing` measured at the start of the period."""
        return self._fx_fixing_start

    @property
    def fx_fixing_end(self) -> FXFixing:
        """The :class:`~rateslib.data.fixings.FXFixing` measured at the end of the period."""
        return self._fx_fixing_end

    @property
    def currency(self) -> str:
        """The settlement currency of the period."""
        return self._currency

    @property
    def pair(self) -> str:
        """The pair that defines each  :class:`~rateslib.data.fixings.FXFixing`."""
        return self.fx_fixing_start.pair

    @property
    def reference_currency(self) -> str:
        """The *reference currency* of the period."""
        ccy1, ccy2 = self.pair[:3], self.pair[3:]
        return ccy1 if ccy2 == self.currency else ccy2

    @cached_property
    def fx_reversed(self) -> bool:
        """Whether the ``reference_currency`` and ``currency`` are reversed in the ``pair``."""
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
    _fx_index: FXIndex,
    _start: datetime,
    _end: datetime,
    _fx_fixings_start: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
    _fx_fixings_end: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
    _currency: str,
) -> _MtmParams:
    # FX fixing publication dates are derived under the ISDA conventions associated with FXIndex.
    return _MtmParams(
        _fx_fixing_start=_init_fx_fixing(
            delivery=_start,
            fx_index=_fx_index,
            fixings=_fx_fixings_start,
        ),
        _fx_fixing_end=_init_fx_fixing(
            delivery=_end,
            fx_index=_fx_index,
            fixings=_fx_fixings_end,
        ),
        _currency=_currency,
    )
