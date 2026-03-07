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

from typing import TYPE_CHECKING

from pandas import Series

from rateslib.data.fixings import IRSFixing
from rateslib.enums.generics import NoInput
from rateslib.enums.parameters import _get_ir_option_metric, _get_swaption_settlement_method

if TYPE_CHECKING:
    from rateslib.local_types import (
        DualTypes,
        DualTypes_,
        IROptionMetric,
        IRSSeries,
        OptionType,
        SwaptionSettlementMethod,
        datetime,
        str_,
    )


class _IROptionParams:
    """
    Parameters for *IR Option Period* cashflows.
    """

    _expiry: datetime
    _metric: IROptionMetric
    _option_fixing: IRSFixing
    _strike: DualTypes_
    _currency: str
    _direction: OptionType

    def __init__(
        self,
        _direction: OptionType,
        _expiry: datetime,
        _tenor: str | datetime,
        _irs_series: IRSSeries,
        _metric: str | IROptionMetric,
        _option_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        _strike: DualTypes_,
        _settlement_method: SwaptionSettlementMethod | str,
    ):
        self._direction = _direction
        self._expiry = _expiry
        self._metric = _get_ir_option_metric(_metric)
        self._strike = _strike
        self._settlement_method = _get_swaption_settlement_method(_settlement_method)

        if isinstance(_option_fixings, Series):
            value = IRSFixing._lookup(timeseries=_option_fixings, date=self.expiry)
            self._option_fixing = IRSFixing(
                tenor=_tenor,
                value=value,
                irs_series=_irs_series,
                publication=_expiry,
                identifier=NoInput(0),
            )
        elif isinstance(_option_fixings, str):
            self._option_fixing = IRSFixing(
                tenor=_tenor,
                value=NoInput(0),
                irs_series=_irs_series,
                publication=_expiry,
                identifier=_option_fixings,
            )
        else:
            self._option_fixing = IRSFixing(
                tenor=_tenor,
                value=_option_fixings,
                publication=_expiry,
                irs_series=_irs_series,
                identifier=NoInput(0),
            )

        self._option_fixing.irs.fixed_rate = self.strike

    @property
    def settlement_method(self) -> SwaptionSettlementMethod:
        """The settlement method of the option."""
        return self._settlement_method

    @property
    def expiry(self) -> datetime:
        """The expiry date of the option."""
        return self._expiry

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
        self.option_fixing.irs.fixed_rate = val
        self._strike = val

    @property
    def option_fixing(self) -> IRSFixing:
        """The FX fixing related to settlement of the option."""
        return self._option_fixing

    @property
    def metric(self) -> IROptionMetric:
        """The default pricing quoting of the option."""
        return self._metric

    def time_to_expiry(self, now: datetime) -> float:
        """The time to expiry of the option in years measured by calendar days from ``now``."""
        # TODO make this a dual, associated with theta
        return (self.expiry - now).days / 365.0
