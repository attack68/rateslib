from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

from rateslib import defaults
from rateslib.enums.generics import NoInput, Ok, _drb
from rateslib.enums.parameters import IndexMethod
from rateslib.periods.components.parameters import (
    _init_MtmParams,
    _init_or_none_IndexParams,
    _init_or_none_NonDeliverableParams,
    _init_SettlementParams_with_fx_pair,
)
from rateslib.periods.components.parameters.mtm import _MtmParams
from rateslib.periods.components.protocols import (
    _BasePeriodStatic,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXVolOption_,
        Result,
        Series,
        _BaseCurve_,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class Cashflow(_BasePeriodStatic):
    r"""
    A *Period* defined by a specific amount.

    The expected unindexed reference cashflow under the risk neutral distribution is defined as,

    .. math::

       \mathbb{E^Q} [\bar{C}_t] = -N

    There is no *analytical delta* for this *Period* type and hence :math:`\xi` is not defined.

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib.periods.components import Cashflow
       from datetime import datetime as dt

    .. ipython:: python

       period = Cashflow(
           payment=dt(2025, 10, 22),
           ex_dividend=dt(2025, 10, 21),
           currency="eur",
           notional=125000,
       )
       period.cashflows()

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .
        .. note::

           The following define generalised **settlement** parameters.

    currency: str, :green:`optional (set by 'defaults')`
        The physical *settlement currency* of the *Period*.
    notional: float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The notional amount of the *Period* expressed in ``notional currency``.
    payment: datetime, :red:`required`
        The payment date of the *Period* cashflow.
    ex_dividend: datetime, :green:`optional (set as 'payment')`
        The ex-dividend date of the *Period*. Settlements occurring **after** this date
        are assumed to be non-receivable.

        .. note::

           The following parameters define **non-deliverability**. If the *Period* is directly
           deliverable do not supply these parameters.

    pair: str, :green:`optional`
        The currency pair of the :class:`~rateslib.data.fixings.FXFixing` that determines
        settlement. The *reference currency* is implied from ``pair``. Must include ``currency``.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing`. If a scalar is used directly.
        If a string identifier will link to the central ``fixings`` object and data loader.
    delivery: datetime, :green:`optional (set as 'payment')`
        The settlement delivery date of the :class:`~rateslib.data.fixings.FXFixing`.

        .. note::

           The following parameters define **indexation**. The *Period* will be considered
           indexed if any of ``index_method``, ``index_lag``, ``index_base``, ``index_fixings``
           are given.

    index_method : IndexMethod, str, :green:`optional (set by 'defaults')`
        The interpolation method, or otherwise, to determine index values from reference dates.
    index_lag: int, :green:`optional (set by 'defaults')`
        The indexation lag, in months, applied to the determination of index values.
    index_base: float, Dual, Dual2, Variable, :green:`optional`
        The specific value set of the base index value.
        If not given and ``index_fixings`` is a str fixings identifier that will be
        used to determine the base index value.
    index_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The index value for the reference date.
        If a scalar value this is used directly. If a string identifier will link to the
        central ``fixings`` object and data loader.
    index_base_date: datetime, :green:`optional`
        The reference date for determining the base index value. Not required if ``_index_base``
        value is given directly.
    index_reference_date: datetime, :green:`optional (set as 'end')`
        The reference date for determining the index value. Not required if ``_index_fixings``
        is given as a scalar value.
    index_only: bool, :green:`optional (set as False)`
        A flag which determines non-payment of notional on supported *Periods*.

    """

    def __init__(
        self,
        *,
        # currency args:
        payment: datetime,
        notional: DualTypes,
        currency: str_ = NoInput(0),
        ex_dividend: datetime_ = NoInput(0),
        # non-deliverable args:
        pair: str_ = NoInput(0),
        fx_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        delivery: datetime_ = NoInput(0),
        # index-args:
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        index_only: bool_ = NoInput(0),
        index_base_date: datetime_ = NoInput(0),
        index_reference_date: datetime_ = NoInput(0),
    ):
        self._settlement_params = _init_SettlementParams_with_fx_pair(
            _notional=notional,
            _payment=payment,
            _currency=_drb(defaults.base_currency, currency).lower(),
            _ex_dividend=_drb(payment, ex_dividend),
            _fx_pair=pair,
        )
        self._non_deliverable_params = _init_or_none_NonDeliverableParams(
            _currency=self.settlement_params.currency,
            _pair=pair,
            _fx_fixings=fx_fixings,
            _delivery=_drb(self.settlement_params.payment, delivery),
        )
        self._index_params = _init_or_none_IndexParams(
            _index_base=index_base,
            _index_lag=index_lag,
            _index_method=index_method,
            _index_fixings=index_fixings,
            _index_base_date=index_base_date,
            _index_reference_date=_drb(self.settlement_params.payment, index_reference_date),
            _index_only=index_only,
        )

    def unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        **kwargs: Any,
    ) -> DualTypes:
        return -self.settlement_params.notional

    def try_unindexed_reference_cashflow_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        return Ok(0.0)

    def try_unindexed_reference_cashflow_analytic_rate_fixings(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        return Ok(DataFrame())


# class NonDeliverableCashflow(Cashflow):
#     """
#     Deprecated
#
#     .. warning::
#
#        This object is deprecated. Use a :class:`~rateslib.periods.components.Cashflow` instead.
#
#     """
#
#     def __init__(self, **kwargs: Any) -> None:
#         super().__init__(**kwargs)
#         if not self.is_non_deliverable:
#             raise ValueError(err.VE_NEEDS_ND_CURRENCY_PARAMS.format(type(self).__name__))
#         if self.is_indexed:
#             raise ValueError(err.VE_HAS_INDEX_PARAMS.format(type(self).__name__))
#
#
# class IndexCashflow(Cashflow):
#     """
#     Deprecated
#
#     .. warning::
#
#        This object is deprecated. Use a :class:`~rateslib.periods.components.Cashflow` instead.
#
#     """
#
#     def __init__(self, **kwargs: Any) -> None:
#         super().__init__(**kwargs)
#         if not self.is_indexed:
#             raise ValueError(err.VE_NEEDS_INDEX_PARAMS.format(type(self).__name__))
#         if self.is_non_deliverable:
#             raise ValueError(err.VE_HAS_ND_CURRENCY_PARAMS.format(type(self).__name__))
#
#
# class NonDeliverableIndexCashflow(Cashflow):
#     """
#     Deprecated
#
#     .. warning::
#
#        This object is deprecated. Use a :class:`~rateslib.periods.components.Cashflow` instead.
#
#     """
#
#     def __init__(self, **kwargs: Any) -> None:
#         super().__init__(**kwargs)
#         if not self.is_indexed:
#             raise ValueError(err.VE_NEEDS_INDEX_PARAMS.format(type(self).__name__))
#         if not self.is_non_deliverable:
#             raise ValueError(err.VE_NEEDS_ND_CURRENCY_PARAMS.format(type(self).__name__))


class MtmCashflow(_BasePeriodStatic):
    r"""
    A *Period* defined by a specific amount calculated from the difference between two
    :class:`~rateslib.data.fixings.FXFixing`.

    This type does not permit non-deliverability, although its notional is expressed in a
    notional currency which is different to the settlement currency.

    The expected unindexed reference cashflow under the risk neutral distribution is defined as,

    .. math::

       \mathbb{E^Q} [\bar{C}_t] = -N ( f_{not:ref}(m_{a.e}) - f_{not:ref}(m_{a.s}) )

    There is no *analytical delta* for this *Period* type and hence :math:`\xi` is not defined.

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib.periods.components import MtmCashflow
       from datetime import datetime as dt

    .. ipython:: python

       period = MtmCashflow(
           payment=dt(2025, 10, 22),
           start=dt(2025, 7, 22),
           currency="usd",
           pair="eurusd",
           notional=125000,
           fx_fixings_start=1.10,
           fx_fixings_end=1.20,
       )
       period.cashflows()

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .
        .. note::

           The following define generalised **settlement** parameters.

    currency: str, :green:`optional (set by 'defaults')`
        The physical *settlement currency* of the *Period*.
    notional: float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The notional amount of the *Period* expressed in ``notional currency``.
    payment: datetime, :red:`required`
        The payment date of the *Period* cashflow.
    ex_dividend: datetime, :green:`optional (set as 'payment')`
        The ex-dividend date of the *Period*. Settlements occurring **after** this date
        are assumed to be non-receivable.

        .. note::

           The following parameters define the specific **mtm** aspects of the *cashflow*.

    pair: str, :red:`required`
        The currency pair of the two :class:`~rateslib.data.fixings.FXFixing` that determines
        settlement. The *reference currency* is implied from ``pair``. Must include ``currency``.
    start: datetime, :red:`required`
        The delivery date of the first :class:`~rateslib.data.fixings.FXFixing` at the start of
        the *Period*.
    end: datetime, :green:`optional (set as 'payment')`
        The delivery date of the second :class:`~rateslib.data.fixings.FXFixing` at the end of
        the *Period*.
    fx_fixings_start: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the first :class:`~rateslib.data.fixings.FXFixing`. If a scalar, is used
        directly. If a string identifier will link to the central ``fixings`` object and
        data loader.
    fx_fixings_end: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the second :class:`~rateslib.data.fixings.FXFixing`. If a scalar, is used
        directly. If a string identifier will link to the central ``fixings`` object and
        data loader.

        .. note::

           The following parameters define **indexation**. The *Period* will be considered
           indexed if any of ``index_method``, ``index_lag``, ``index_base``, ``index_fixings``
           are given.

    index_method : IndexMethod, str, :green:`optional (set by 'defaults')`
        The interpolation method, or otherwise, to determine index values from reference dates.
    index_lag: int, :green:`optional (set by 'defaults')`
        The indexation lag, in months, applied to the determination of index values.
    index_base: float, Dual, Dual2, Variable, :green:`optional`
        The specific value set of the base index value.
        If not given and ``index_fixings`` is a str fixings identifier that will be
        used to determine the base index value.
    index_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The index value for the reference date.
        If a scalar value this is used directly. If a string identifier will link to the
        central ``fixings`` object and data loader.
    index_base_date: datetime, :green:`optional`
        The reference date for determining the base index value. Not required if ``_index_base``
        value is given directly.
    index_reference_date: datetime, :green:`optional (set as 'end')`
        The reference date for determining the index value. Not required if ``_index_fixings``
        is given as a scalar value.
    index_only: bool, :green:`optional (set as False)`
        A flag which determines non-payment of notional on supported *Periods*.

    """

    @property
    def mtm_params(self) -> _MtmParams:
        """The :class:`~rateslib.periods.components.parameters._MtmParams` of the
        *Period*."""
        return self._mtm_params

    def __init__(
        self,
        *,
        payment: datetime,
        notional: DualTypes,
        pair: str,
        start: datetime,
        end: datetime_ = NoInput(0),
        currency: str_ = NoInput(0),
        ex_dividend: datetime_ = NoInput(0),
        fx_fixings_start: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        fx_fixings_end: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        # index-args:
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        index_only: bool_ = NoInput(0),
        index_base_date: datetime_ = NoInput(0),
        index_reference_date: datetime_ = NoInput(0),
    ):
        self._settlement_params = _init_SettlementParams_with_fx_pair(
            _notional=notional,
            _payment=payment,
            _currency=_drb(defaults.base_currency, currency).lower(),
            _ex_dividend=_drb(payment, ex_dividend),
            _fx_pair=pair,
        )
        self._mtm_params = _init_MtmParams(
            _pair=pair,
            _currency=_drb(defaults.base_currency, currency).lower(),
            _start=start,
            _end=_drb(payment, end),
            _fx_fixings_start=fx_fixings_start,
            _fx_fixings_end=fx_fixings_end,
        )
        self._non_deliverable_params = None
        self._index_params = _init_or_none_IndexParams(
            _index_base=index_base,
            _index_lag=index_lag,
            _index_method=index_method,
            _index_fixings=index_fixings,
            _index_base_date=index_base_date,
            _index_reference_date=_drb(self.settlement_params.payment, index_reference_date),
            _index_only=index_only,
        )

    def unindexed_reference_cashflow(  # type: ignore[override]
        self,
        *,
        fx: FXForwards_ = NoInput(0),
        **kwargs: Any,
    ) -> DualTypes:
        fx0 = self.mtm_params.fx_fixing_start.try_value_or_forecast(fx).unwrap()
        fx1 = self.mtm_params.fx_fixing_end.try_value_or_forecast(fx).unwrap()
        if self.mtm_params.fx_reversed:
            diff = 1.0 / fx1 - 1.0 / fx0
        else:
            diff = fx1 - fx0
        return -self.settlement_params.notional * diff

    def try_unindexed_reference_cashflow_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        return Ok(0.0)

    def try_unindexed_reference_cashflow_analytic_rate_fixings(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        return Ok(DataFrame())
