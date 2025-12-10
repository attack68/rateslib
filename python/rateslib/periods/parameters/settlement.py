from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from pandas import Series

import rateslib.errors as err
from rateslib import defaults
from rateslib.data.fixings import FXFixing
from rateslib.enums.generics import (
    NoInput,
    _drb,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        DualTypes,
        DualTypes_,
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
        The notional amount of the *Period* expressed in ``notional currency``.
    _notional_currency: str
        The currency for the expresion of ``notional`` amount.
    _payment: datetime
        The payment date of the *Period* cashflow.
    _ex_dividend: datetime, optional
        The ex-dividend date of the *Period*. Settlements occurring **after** this date
        are assumed to be non-receivable. If not given is assumed to be equal to ``payment``
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
        self._notional_currency = _notional_currency.lower()
        self._payment = _payment
        self._ex_dividend = _drb(self.payment, _ex_dividend)

    @property
    def currency(self) -> str:
        """The local settlement currency of the *Period* cashflow."""
        return self._currency

    @property
    def notional(self) -> DualTypes:
        """The notional amount of the *Period* expressed in units of ``notional_currency``."""
        return self._notional

    @property
    def notional_currency(self) -> str:
        """The currency for the expression of ``notional`` amount."""
        return self._notional_currency

    @property
    def payment(self) -> datetime:
        """The payment date of the *Period* cashflow."""
        return self._payment

    @property
    def ex_dividend(self) -> datetime:
        """The ex-dividend date for settlement of the *Period* cashflow."""
        return self._ex_dividend


class _NonDeliverableParams:
    """
    Parameters for determination of non-deliverable *Period* cashflows.

    Parameters
    ----------
    _currency: str
        The physical *settlement currency* of the *Period*.
    _pair: str, optional
        For non-deliverable *Periods* only.
        The currency pair of the *FX* rate fixing that determines settlement. The
        *reference currency* is implied from ``pair`` when it is not equal to ``currency``.
    _fx_fixings: float, Dual, Dual2, Variable, Series, str, optional
        The value of the :class:`~rateslib.data.fixings.FXFixing`. If a scalar is used directly.
        If a string identifier will link to the central ``fixings`` object and data loader.
    _delivery: datetime, optional
        For non-deliverable *Periods* only.
        The settlement delivery date of the *FX* rate fixing.

    """

    def __init__(
        self,
        _currency: str,
        _pair: str,
        _delivery: datetime,
        _fx_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
    ) -> None:
        self._currency = _currency.lower()
        self._pair = _pair.lower()
        self._fx_fixing = _init_fx_fixing(date=_delivery, pair=self.pair, fixings=_fx_fixings)

    @property
    def currency(self) -> str:
        """The physical *settlement currency* of the *Period*."""
        return self._currency

    @property
    def reference_currency(self) -> str:
        """The *reference currency* of underlying *Period* cashflows."""
        ccy1, ccy2 = self.pair[0:3], self.pair[3:6]
        return ccy1 if ccy1 != self.currency else ccy2

    @property
    def pair(self) -> str:
        """The currency pair associated with the :class:`~rateslib.data.fixings.FXFixing`."""
        return self._pair

    @property
    def fx_fixing(self) -> FXFixing:
        """The :class:`~rateslib.data.fixings.FXFixing` associated with the *Period* cashflow."""
        return self._fx_fixing

    @fx_fixing.setter
    def fx_fixing(self, val: Any) -> None:
        raise ValueError(err.VE_ATTRIBUTE_IS_IMMUTABLE.format("fx_fixing"))

    @property
    def delivery(self) -> datetime:
        """The settlement delivery date of the :class:`~rateslib.data.fixings.FXFixing`."""
        return self.fx_fixing.date

    @cached_property
    def fx_reversed(self) -> bool:
        """Whether the *FX* rate fixing is reversed relative to the ``referency_currency``
        and ``currency``."""
        return self.pair[3:6] == self.reference_currency


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


def _init_SettlementParams_with_fx_pair(
    _currency: str_,
    _payment: datetime,
    _notional: DualTypes_,
    _ex_dividend: datetime,
    _fx_pair: str_,
) -> _SettlementParams:
    notional = _drb(defaults.notional, _notional)
    ccy = _drb(defaults.base_currency, _currency).lower()
    if isinstance(_fx_pair, NoInput):
        return _SettlementParams(
            _currency=ccy,
            _notional_currency=ccy,
            _payment=_payment,
            _notional=notional,
            _ex_dividend=_ex_dividend,
        )
    else:
        c1, c2 = _fx_pair.lower()[:3], _fx_pair.lower()[3:]
        # other parameters will also be determined.
        if ccy != c1 and ccy != c2:
            raise ValueError(err.VE_MISMATCHED_ND_PAIR.format(ccy, _fx_pair))
        return _SettlementParams(
            _currency=ccy,
            _notional_currency=c1 if c1 != ccy else c2,
            _payment=_payment,
            _notional=notional,
            _ex_dividend=_ex_dividend,
        )


def _init_fx_fixing(
    date: datetime,
    pair: str,
    fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
) -> FXFixing:
    if isinstance(fixings, Series):
        value = FXFixing._lookup(timeseries=fixings, date=date)
        return FXFixing(date=date, value=value, pair=pair)
    elif isinstance(fixings, str):
        return FXFixing(date=date, identifier=fixings, pair=pair)
    else:
        return FXFixing(date=date, value=fixings, pair=pair)
