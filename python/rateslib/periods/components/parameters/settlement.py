from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from pandas import Series

import rateslib.errors as err
from rateslib import defaults
from rateslib.data.fixings import FXFixing
from rateslib.enums.generics import (
    Err,
    NoInput,
    Ok,
    _drb,
)
from rateslib.periods.utils import _try_validate_fx_as_forwards

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        DualTypes,
        DualTypes_,
        FXForwards_,
        Result,
        datetime,
        datetime_,
        str_,
    )


class _SettlementParams:
    """
    Parameters for settlement of *Period* cashflows.

    Parameters
    ----------
    _currency: str
        The physical *settlement currency* of the *Period*.
    _notional: float, Dual, Dual2, Variable
        The notional amount of the *Period* expressed in *reference currency*.
    _payment: datetime
        The payment date of the cashflow.
    _pair: str, optional
        For non-deliverable *Periods* only.
        The currency pair of the *FX* rate fixing that determines settlement. The
        *reference currency* is implied from ``pair`` when it is not equal to ``currency``.
    _fx_fixings: float, Dual, Dual2, Variable, str, Series, optional
        For non-deliverable *Periods* only.
        An element which can determine the ``fx_fixing``.
    _delivery: datetime, optional
        For non-deliverable *Periods* only.
        The settlement delivery date of the *FX* rate fixing.
    _ex_dividend: datetime, optional
        The ex-dividend date of the *Period*. Settlements occurring **after** this date
        are assumed to be non-receivable.

    """

    _currency: str
    _notional: DualTypes
    _notional_currency: str
    _payment: datetime
    _ex_dividend: datetime

    def __init__(
        self,
        _currency: str,
        _notional: DualTypes,
        _notional_currency: str,
        _payment: datetime,
        _ex_dividend: datetime_ = NoInput(0),
    ) -> None:
        self._currency = _currency.lower()
        self._notional = _notional
        self._notional_currency = _notional_currency
        self._payment = _payment
        self._ex_dividend = _drb(self.payment, _ex_dividend)

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def notional(self) -> DualTypes:
        return self._notional

    @property
    def notional_currency(self) -> str:
        return self._notional_currency

    @property
    def payment(self) -> datetime:
        return self._payment

    @property
    def ex_dividend(self) -> datetime:
        return self._ex_dividend


class _NonDeliverableParams:
    _currency: str
    _pair: str
    _fx_fixing: FXFixing
    _delivery: datetime

    def __init__(
        self,
        _currency: str,
        _pair: str,
        _delivery: datetime,
        _fx_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
    ) -> None:
        self._currency = _currency.lower()
        self._pair = _pair
        self._delivery = _delivery

        if isinstance(_fx_fixings, Series):
            value = FXFixing._lookup(timeseries=_fx_fixings, date=self.delivery)
            self._fx_fixing = FXFixing(
                date=self.delivery,
                value=value,
            )
        elif isinstance(_fx_fixings, str):
            self._fx_fixing = FXFixing(
                date=self.delivery,
                identifier=_fx_fixings,
            )
        else:
            self._fx_fixing = FXFixing(
                date=self.delivery,
                value=_fx_fixings,
            )

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def reference_currency(self) -> str:
        ccy1, ccy2 = self.pair[0:3], self.pair[3:6]
        return ccy1 if ccy1 != self.currency else ccy2

    @property
    def pair(self) -> str:
        return self._pair

    @property
    def fx_fixing(self) -> FXFixing:
        return self._fx_fixing

    @fx_fixing.setter
    def fx_fixing(self, val: Any) -> None:
        raise ValueError(err.VE_ATTRIBUTE_IS_IMMUTABLE.format("fx_fixing"))

    @property
    def delivery(self) -> datetime:
        return self._delivery

    @cached_property
    def fx_reversed(self) -> bool:
        return self.pair[3:6] == self.reference_currency

    def try_fx_fixing(self, fx: FXForwards_) -> Result[DualTypes]:
        fx_fixing: DualTypes_ = self.fx_fixing.value
        if isinstance(fx_fixing, NoInput):
            fx_result = _try_validate_fx_as_forwards(fx)
            if isinstance(fx_result, Err):
                return fx_result
            fx_fixing_: DualTypes = fx_result.unwrap().rate(self.pair, self.delivery)
        else:
            fx_fixing_ = fx_fixing
        return Ok(fx_fixing_)


def _init_or_none_NonDeliverableParams(
    _currency: str,
    _pair: str_,
    _delivery: datetime,
    _fx_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
) -> _NonDeliverableParams | None:
    if isinstance(_pair, NoInput):
        return None
    else:
        return _NonDeliverableParams(
            _currency=_currency,
            _pair=_pair,
            _delivery=_delivery,
            _fx_fixings=_fx_fixings,
        )


def _init_SettlementParams_with_nd_pair(
    _currency: str_,
    _payment: datetime,
    _notional: DualTypes_,
    _ex_dividend: datetime,
    _non_deliverable_pair: str_,
) -> _SettlementParams:
    notional = _drb(defaults.notional, _notional)
    ccy = _drb(defaults.base_currency, _currency).lower()
    if isinstance(_non_deliverable_pair, NoInput):
        return _SettlementParams(
            _currency=ccy,
            _notional_currency=ccy,
            _payment=_payment,
            _notional=notional,
            _ex_dividend=_ex_dividend,
        )
    else:
        c1, c2 = _non_deliverable_pair.lower()[:3], _non_deliverable_pair.lower()[3:]
        # other parameters will also be determined.
        return _SettlementParams(
            _currency=ccy,
            _notional_currency=c1 if c1 != ccy else c2,
            _payment=_payment,
            _notional=notional,
            _ex_dividend=_ex_dividend,
        )
