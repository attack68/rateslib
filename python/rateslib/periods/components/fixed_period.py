from __future__ import annotations

from typing import TYPE_CHECKING

import rateslib.errors as err
from rateslib.curves._parsers import _try_disc_required_maybe_from_curve
from rateslib.enums.generics import Err, NoInput, Ok
from rateslib.periods.components.base_period import BasePeriod
from rateslib.periods.components.parameters import (
    _FixedRateParams,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CurveOption_,
        DualTypes,
        DualTypes_,
        Result,
        _BaseCurve_,
    )


class FixedPeriod(BasePeriod):
    r"""
    A *Period* defined by a fixed interest rate.

    The expected unindexed reference cashflow under the risk neutral distribution is defined as,

    .. math::

       \mathbb{E^Q} [\bar{C}_t] = -N d R

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

           The following parameters are scheduling **period** parameters

    start: datetime, :red:`required`
        The identified start date of the *Period*.
    end: datetime, :red:`required`
        The identified end date of the *Period*.
    frequency: Frequency, str, :red:`required`
        The :class:`~rateslib.scheduling.Frequency` associated with the *Period*.
    convention: Convention, str, :green:`optional` (set by 'defaults')
        The day count :class:`~rateslib.scheduling.Convention` associated with the *Period*.
    termination: datetime, :green:`optional`
        The termination date of an external :class:`~rateslib.scheduling.Schedule`.
    calendar: Calendar, :green:`optional`
         The calendar associated with the *Period*.
    stub: bool, str, :green:`optional (set as False)`
        Whether the *Period* is defined as a stub according to some external
        :class:`~rateslib.scheduling.Schedule`.
    adjuster: Adjuster, :green:`optional`
        The date :class:`~rateslib.scheduling.Adjuster` applied to unadjusted dates in the
        external :class:`~rateslib.scheduling.Schedule` to arrive at adjusted accrual dates.

        .. note::

           The following define **fixed rate** parameters.

    fixed_rate: float, Dual, Dual2, Variable, :green:`optional`
        The fixed rate to determine the *Period* cashflow.

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


    ..  Examples
        --------

        A typical RFR type :class:`~rateslib.periods.components.FloatPeriod`.

        .. ipython:: python
           :supress:

           from rateslib.periods.components import FloatPeriod
           from rateslib.data.fixings import FloatRateIndex
           from datetime import datetime as dt

        .. ipython:: python

           period = FloatPeriod(
               start=dt(2025, 9, 22),
               end=dt(2025, 10, 20),
               payment=dt(2025, 10, 22),
               frequency="1M",
           )

        A typical IBOR tenor type :class:`~rateslib.periods.components.FloatPeriod`.

        .. ipython:: python

           period = FloatPeriod(
               start=dt(2025, 9, 22),
               end=dt(2025, 10, 22),
               payment=dt(2025, 10, 22),
               frequency="1M",
               currency="eur",
               fixing_method="IBOR",
               fixing_series="eur_IBOR",
           )

    """

    @property
    def rate_params(self) -> _FixedRateParams:
        """The :class:`~rateslib.periods.components.parameters._FixedRateParams` of the *Period*."""
        return self._rate_params

    def __init__(self, *, fixed_rate: DualTypes_ = NoInput(0), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rate_params = _FixedRateParams(fixed_rate)

    def try_unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        **kwargs: Any,
    ) -> Result[DualTypes]:
        if isinstance(self.rate_params.fixed_rate, NoInput):
            return Err(ValueError(err.VE_NEEDS_FIXEDRATE))
        else:
            return Ok(
                -self.settlement_params.notional
                * self.rate_params.fixed_rate
                * 0.01
                * self.period_params.dcf
            )

    def try_unindexed_reference_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        disc_curve_ = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        if isinstance(disc_curve_, Err):
            return disc_curve_
        return Ok(
            self.settlement_params.notional
            * 0.0001
            * self.period_params.dcf
            * disc_curve_.unwrap()[self.settlement_params.payment]
        )


class NonDeliverableFixedPeriod(FixedPeriod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_non_deliverable:
            raise ValueError(err.VE_NEEDS_ND_CURRENCY_PARAMS.format(type(self).__name__))
        if self.is_indexed:
            raise ValueError(err.VE_HAS_INDEX_PARAMS.format(type(self).__name__))


class IndexFixedPeriod(FixedPeriod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_indexed:
            raise ValueError(err.VE_NEEDS_INDEX_PARAMS.format(type(self).__name__))
        if self.is_non_deliverable:
            raise ValueError(err.VE_HAS_ND_CURRENCY_PARAMS.format(type(self).__name__))


class NonDeliverableIndexFixedPeriod(FixedPeriod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_indexed:
            raise ValueError(err.VE_NEEDS_INDEX_PARAMS.format(type(self).__name__))
        if not self.is_non_deliverable:
            raise ValueError(err.VE_NEEDS_ND_CURRENCY_PARAMS.format(type(self).__name__))
