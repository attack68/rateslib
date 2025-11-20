from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import rateslib.errors as err
from rateslib import defaults
from rateslib.curves._parsers import (
    _disc_required_maybe_from_curve,
)
from rateslib.data.fixings import _leg_fixings_to_list
from rateslib.enums.generics import NoInput, _drb
from rateslib.legs.components.amortization import Amortization, _AmortizationType, _get_amortization
from rateslib.legs.components.protocols import (
    _BaseLeg,
    _WithExDiv,
)
from rateslib.periods.components import (
    Cashflow,
    FixedPeriod,
    MtmCashflow,
    ZeroFixedPeriod,
    _BasePeriod,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CurveOption_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXVolOption_,
        IndexMethod,
        LegFixings,
        Schedule,
        Series,
        _BaseCurve_,
        _SettlementParams,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FixedLeg(_BaseLeg, _WithExDiv):
    """
    Define a *Leg* containing :class:`~rateslib.periods.components.FixedPeriod`.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    schedule: Schedule, :red:`required`
        The :class:`~rateslib.scheduling.Schedule` object which structures contiguous *Periods*.
        The schedule object also contains data for payment dates, payment dates for notional
        exchanges and ex-dividend dates for each period.

        .. note::

           The following define generalised **settlement** parameters.

    currency : str, :green:`optional (set by 'defaults')`
        The local settlement currency of the leg (3-digit code).
    notional : float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The initial leg notional, defined in units of *reference currency*.
    amortization: float, Dual, Dual2, Variable, str, Amortization, :green:`optional (set as zero)`
        Set a non-constant notional per *Period*. If a scalar value, adjusts the ``notional`` of
        each successive period by that same value. Should have
        sign equal to that of notional if the notional is to reduce towards zero.
    initial_exchange : bool, :green:`optional (set as False)`
        Whether to also include an initial notional exchange. If *True* then ``final_exchange``
        **will** also be set to *True*.
    final_exchange : bool, :green:`optional (set as initial_exchange)`
        Whether to also include a final notional exchange and interim amortization
        notional exchanges.

        .. note::

           The following define **non-deliverable** parameters. If the *Leg* is directly
           deliverable then do not set a non-deliverable ``pair`` or any ``fx_fixings``.

    pair: str, :green:`optional`
        The currency pair for :class:`~rateslib.data.fixings.FXFixing` that determines *Period*
        settlement. The *reference currency* is implied from ``pair``. Must include ``currency``.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for each *Period* according
        to non-deliverability. Review the **notes** section non-deliverability.
    mtm: bool, :green:`optional (set to False)`
        Define whether the non-deliverability depends on a single
        :class:`~rateslib.data.fixings.FXFixing` defined at the start of the *Leg*, or
        multiple throughout its settlement. Review the **notes** section non-deliverability.

        .. note::

           The following are **period parameters** combined with the ``schedule``.

    convention: str, :green:`optional (set by 'defaults')`
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.scheduling.dcf`.

        .. note::

           The following define **rate parameters**.

    fixed_rate: float, Dual, Dual2, Variable, :green:`optional`
        The fixed rate of each composited :class:`~rateslib.periods.components.FixedPeriod`.

        .. note::

           The following parameters define **indexation**. The *Period* will be considered
           indexed if any of ``index_method``, ``index_lag``, ``index_base``, ``index_fixings``
           are given.

    index_method : IndexMethod, str, :green:`optional (set by 'defaults')`
        The interpolation method, or otherwise, to determine index values from reference dates.
    index_lag: int, :green:`optional (set by 'defaults')`
        The indexation lag, in months, applied to the determination of index values.
    index_base: float, Dual, Dual2, Variable, :green:`optional`
        The specific value applied as the base index value for all *Periods*.
        If not given and ``index_fixings`` is a string fixings identifier that will be
        used to determine the base index value.
    index_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The index value for the reference date.
        Best practice is to supply this value as string identifier relating to the global
        ``fixings`` object.
    index_only: bool, :green:`optional (set as False)`
        A flag which indicates that the nominal amount is deducted from the cashflow leaving only
        the indexed up quantity.

    Notes
    -----
    **Typical Fixed Legs**

    A typical *FixedLeg* has no amortization, no indexation, is directly deliverable and offers
    no notional exchanges. This represents one component of, for example, an
    :class:`~rateslib.instruments.IRS`.

    .. ipython:: python
       :suppress:

       from rateslib import fixings
       from pandas import Series
       from rateslib.legs.components import FixedLeg
       from rateslib import Schedule
       from datetime import datetime as dt

    .. ipython:: python

       leg = FixedLeg(
           schedule=Schedule(dt(2000, 1, 1), dt(2000, 7, 1), "Q"),
           fixed_rate=2.0,
           convention="Act360",
           notional=5000000,
       )
       print(leg.cashflows())

    **Notional Exchanges**

    Notional exchanges are common elements on securities, e.g. a
    :class:`~rateslib.instruments.FixedRateBond`. These can be specifically included using the
    ``final_exchange`` and ``initial_exchange`` parameters.

    .. ipython:: python

       leg = FixedLeg(
           schedule=Schedule(dt(2000, 1, 1), dt(2000, 7, 1), "Q"),
           fixed_rate=2.0,
           convention="Act360",
           notional=5000000,
           final_exchange=True,
       )
       print(leg.cashflows())

    Initial and final notional exchanges have opposite directions.

    **Amortization**

    Amortization can be applied either with customised schedules, or with simpler consistent
    amounts per period.

    If ``final_exchange`` is *True* then amortization will also create interim notional exchange
    cashflows. Note that a same sign ``amortization`` value is translated into
    a notional reduction. If ``final_exchange`` is *False*, or amortization is zero, there are no
    interim notional exchange cashflows generated.

    .. ipython:: python

       leg = FixedLeg(
           schedule=Schedule(dt(2000, 1, 1), dt(2000, 7, 1), "Q"),
           fixed_rate=2.0,
           convention="Act360",
           notional=5000000,
           amortization=1000000,
           final_exchange=True,
       )
       print(leg.cashflows())

    **Indexation**

    An :class:`~rateslib.instruments.IndexFixedRateBond` is the most common instrument that
    uses an index-linked *FixedLeg*. Setting *index* parameters creates the necessary
    indexation of cashflows. Note that all previous features such as notional exchanges and
    amortization are all adjusted appropriately.

    .. ipython:: python

       fixings.add("MY_RPI", Series(
           index=[dt(2000, 1, 1), dt(2000, 4, 1), dt(2000, 7, 1)],
           data=[101.0, 102.0, 103.0]
       ))
       leg = FixedLeg(
           schedule=Schedule(dt(2000, 1, 1), dt(2000, 7, 1), "Q"),
           fixed_rate=2.0,
           convention="Act360",
           notional=5000000,
           amortization=1000000,
           final_exchange=True,
           index_fixings="MY_RPI",
           index_lag=0,
           index_method="monthly",
       )
       print(leg.cashflows())

    Any interim notional exchange cashflows generated by ``amortization`` are also indexed.

    **Non-Deliverability**

    There are three types of *non-deliverability* that can be applied to *Legs*:

    - **One initial** :class:`~rateslib.data.fixings.FXFixing`, which is the type used by
      the foreign leg of a non-MTM :class:`~rateslib.instruments.XCS`. This effectively fixes the
      foreign notional, relative to a reference notional and FX rate, for all *Periods* at the
      start of the *Leg*.

      In this case, the ``fx_fixings`` input is usually a known, single scalar value agreed at
      trade execution time. It can also be input as a string identifier from which the
      :class:`~rateslib.data.fixings.FXFixing` can be lookup up.
      The relevant delivery date for the fixing date is taken as the first payment exchange
      date from the schedule, i.e. ``schedule.pschedule2[0]``.

      In the example below the *Leg* is settled is USD but is expressed with a notional in EUR.
      All *Periods* assume the USD notional is converted under the single 1.25 EURUSD FX rate.

      .. ipython:: python

         leg = FixedLeg(
             schedule=Schedule(
                 effective=dt(2000, 1, 1),
                 termination=dt(2000, 10, 1),
                 frequency="Q",
                 payment_lag=1,
                 payment_lag_exchange=0,
             ),
             fixed_rate=1.0,
             currency="usd",
             pair="eurusd",
             initial_exchange=True,
             notional=5e6,
             fx_fixings=1.25,
         )
         print(leg.cashflows())

    - **Multiple** :class:`~rateslib.data.fixings.FXFixing`, at future deliveries,
      without notional exchanges. This is the type used by ND-IRS. In this case each future
      payment is converted to a deliverable currency at the time of payment. Therefore the fixing
      delivery dates are the payment dates for accrual periods, i.e. ``schedule.pschedule[i+1]``.

      In the example below the *Leg* is settled is USD but is expressed with a notional in EUR.
      Each *Period's* cashflow is determined in EUR and then converted to USD with an FX rate
      that is determined specifically for that particular payment date, i.e. in the future.

      .. ipython:: python

         fixings.add("EURUSD_1600", Series(
             index=[dt(2000, 4, 2), dt(2000, 7, 2), dt(2000, 10, 2)],
             data=[1.27, 1.29, 1.32])
         )
         leg = FixedLeg(
             schedule=Schedule(
                 effective=dt(2000, 1, 1),
                 termination=dt(2000, 10, 1),
                 frequency="Q",
                 payment_lag=1,
                 payment_lag_exchange=0
             ),
             fixed_rate=1.0,
             currency="usd",
             pair="eurusd",
             mtm=True,
             notional=5e6,
             fx_fixings="EURUSD_1600",
         )
         print(leg.cashflows())


    - **Multiple** :class:`~rateslib.data.fixings.FXFixing`, at future deliveries,
      with notional exchanges. This is the type used by MTM :class:`~rateslib.instruments.XCS`.
      In this case the foreign notional is determined at the start of each period by a known
      fixing and there an additional MTM cashflow exchange at the start of a period to adjust
      for that fixing, i.e. ``schedule.pschedule2[i]``.

      .. ipython:: python

         leg = FixedLeg(
             schedule=Schedule(
                 effective=dt(2000, 1, 1),
                 termination=dt(2000, 10, 1),
                 frequency="Q",
                 payment_lag=2,
                 payment_lag_exchange=1
             ),
             fixed_rate=1.0,
             currency="usd",
             pair="eurusd",
             mtm=True,
             initial_exchange=True,
             notional=5e6,
             fx_fixings=(1.25, "EURUSD_1600"),
         )
         print(leg.cashflows())

    These modes are controlled by the ``mtm`` parameter (*False* for single fixing and *True* for
    multiple fixings), as well as being determined by whether ``final_exchange`` is *True* to
    separate the cases with exchanges.

    When entering ``fx_fixings`` this should be appropriate as to the relevant mode of
    non-deliverability. If the first case, then only 1 single scalar fixing should be provided.
    A 2-tuple is also functional but only the first element will be used.

    For the second case, the ND-IRS type, then a single string identifier is best practice. This
    can also be entered as a list of FX fixing values for each regular period in the schedule.

    For the last case, a tuple represents best practice and allows an arbitrary first FX fixing
    with a string identifier for the remaining future fixings. This reflects a typical
    :class:`~rateslib.instruments.XCS` traded agreement.

    **Amortization and Non-Deliverability**

    When amortization is combined with non-deliverability, the interim notional exchange cashflows
    are adjusted appropriately in both the non-mtm and mtm cases.

    .. ipython:: python

       leg = FixedLeg(
           schedule=Schedule(
               effective=dt(2000, 1, 1),
               termination=dt(2000, 10, 1),
               frequency="Q",
               payment_lag=2,
               payment_lag_exchange=1
           ),
           fixed_rate=1.0,
           currency="usd",
           pair="eurusd",
           initial_exchange=True,
           notional=5e6,
           amortization=1000000,
           fx_fixings=(1.25, "EURUSD_1600"),
       )
       print(leg.cashflows())

    .. ipython:: python

       leg = FixedLeg(
           schedule=Schedule(
               effective=dt(2000, 1, 1),
               termination=dt(2000, 10, 1),
               frequency="Q",
               payment_lag=2,
               payment_lag_exchange=1
           ),
           fixed_rate=1.0,
           currency="usd",
           pair="eurusd",
           initial_exchange=True,
           notional=5e6,
           amortization=1000000,
           mtm=True,
           fx_fixings=(1.25, "EURUSD_1600"),
       )
       print(leg.cashflows())

    **Indexation, Non-Deliverability and Amortization**

    In the most complicated case, which rarely even relates to real tradable instruments all
    of the parameters may be combined. The :meth:`~rateslib.legs.components.FixedLeg.cashflows`
    method outlines the relevant fixing values and dates used in calculations.

    .. ipython:: python

       leg = FixedLeg(
           schedule=Schedule(
               effective=dt(2000, 1, 1),
               termination=dt(2000, 7, 1),
               frequency="Q",
               payment_lag=2,
               payment_lag_exchange=1
           ),
           fixed_rate=1.0,
           currency="usd",
           pair="eurusd",
           initial_exchange=True,
           notional=5e6,
           amortization=1000000,
           mtm=True,
           fx_fixings=(1.25, "EURUSD_1600"),
           index_lag=0,
           index_fixings="MY_RPI",
           index_method="monthly",
       )
       print(leg.cashflows())

    .. ipython:: python
       :suppress:

       fixings.pop("EURUSD_1600")
       fixings.pop("MY_RPI")

    Examples
    --------
    See :ref:`Leg Examples<legs-doc>`

    """

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.components.parameters._SettlementParams` associated with
        the first :class:`~rateslib.periods.components.FloatPeriod`."""
        return self._regular_periods[0].settlement_params

    @cached_property
    def periods(self) -> list[_BasePeriod]:
        """Combine all period collection types into an ordered list."""
        periods_: list[_BasePeriod] = []

        if self._exchange_periods[0] is not None:
            periods_.append(self._exchange_periods[0])

        args: tuple[tuple[_BasePeriod], ...] = (self._regular_periods[:-1],)  # type: ignore[assignment]
        if self._mtm_exchange_periods is not None:
            args = args + (self._mtm_exchange_periods,)  # type: ignore[operator]
        if self._amortization_exchange_periods is not None:
            args = args + (self._amortization_exchange_periods,)  # type: ignore[operator]
        interleaved_periods_: list[_BasePeriod] = [
            item for combination in zip(*args, strict=True) for item in combination
        ]
        interleaved_periods_.append(self._regular_periods[-1])  # add last regular period
        periods_.extend(interleaved_periods_)

        if self._exchange_periods[1] is not None:
            periods_.append(self._exchange_periods[1])

        return periods_

    @property
    def fixed_rate(self) -> DualTypes_:
        """The fixed rate parameter of each composited
        :class:`~rateslib.periods.components.FixedPeriod`."""
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes_) -> None:
        self._fixed_rate = value
        for period in self._regular_periods:
            period.rate_params.fixed_rate = value

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    @property
    def amortization(self) -> Amortization:
        return self._amortization

    def __init__(
        self,
        schedule: Schedule,
        *,
        # settlement and currency
        notional: DualTypes_ = NoInput(0),
        amortization: DualTypes_ | list[DualTypes] | Amortization | str = NoInput(0),
        currency: str_ = NoInput(0),
        # non-deliverable
        pair: str_ = NoInput(0),
        fx_fixings: LegFixings = NoInput(0),  # type: ignore[type-var]
        mtm: bool = False,
        # period
        convention: str_ = NoInput(0),
        initial_exchange: bool = False,
        final_exchange: bool = False,
        # rate params
        fixed_rate: NoInput = NoInput(0),
        # index params
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        index_only: bool = False,
    ) -> None:
        self._fixed_rate = fixed_rate
        self._schedule = schedule
        self._notional: DualTypes = _drb(defaults.notional, notional)
        self._amortization: Amortization = _get_amortization(
            amortization, self._notional, self.schedule.n_periods
        )
        self._currency: str = _drb(defaults.base_currency, currency).lower()
        self._convention: str = _drb(defaults.convention, convention)

        index_fixings_ = _leg_fixings_to_list(index_fixings, self.schedule.n_periods)
        fx_fixings_ = _leg_fixings_to_list(fx_fixings, self.schedule.n_periods)
        # Exchange periods
        if not initial_exchange:
            _ini_cf: Cashflow | None = None
        else:
            _ini_cf = Cashflow(  # type: ignore[abstract]
                payment=self.schedule.pschedule2[0],
                notional=-self._amortization.outstanding[0],
                currency=self._currency,
                ex_dividend=self.schedule.pschedule3[0],
                # non-deliverable
                pair=pair,
                fx_fixings=fx_fixings_[0],
                delivery=self.schedule.pschedule2[0],
                # index params
                index_base=index_base,
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings_[0],
                index_base_date=self.schedule.aschedule[0],
                index_reference_date=self.schedule.aschedule[0],
                index_only=index_only,
            )
        final_exchange_ = final_exchange or initial_exchange
        if not final_exchange_:
            _final_cf: Cashflow | None = None
        else:
            _final_cf = Cashflow(  # type: ignore[abstract]
                payment=self.schedule.pschedule2[-1],
                notional=self._amortization.outstanding[-1],
                currency=self._currency,
                ex_dividend=self.schedule.pschedule3[-1],
                # non-deliverable
                pair=pair,
                fx_fixings=fx_fixings_[0] if not mtm else fx_fixings_[-1],
                delivery=self.schedule.pschedule2[0] if not mtm else self.schedule.pschedule2[-2],
                # index parameters
                index_base=index_base,
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings_[-1],
                index_base_date=self.schedule.aschedule[0],
                index_reference_date=self.schedule.aschedule[-1],
                index_only=index_only,
            )
        self._exchange_periods = (_ini_cf, _final_cf)

        def fx_delivery(i: int) -> datetime:
            if not mtm:
                # then ND type is a one-fixing only
                return self.schedule.pschedule2[0]
            else:
                if final_exchange_:
                    # then ND type is a XCS
                    return self.schedule.pschedule2[i]
                else:
                    # then ND type is IRS
                    return self.schedule.pschedule[i + 1]

        self._regular_periods: tuple[FixedPeriod, ...] = tuple(
            [
                FixedPeriod(  # type: ignore[abstract]
                    fixed_rate=fixed_rate,
                    # currency args
                    payment=self.schedule.pschedule[i + 1],
                    currency=self._currency,
                    notional=self.amortization.outstanding[i],
                    ex_dividend=self.schedule.pschedule3[i + 1],
                    # period params
                    start=self.schedule.aschedule[i],
                    end=self.schedule.aschedule[i + 1],
                    frequency=self.schedule.frequency_obj,
                    convention=self._convention,
                    termination=self.schedule.aschedule[-1],
                    stub=self.schedule._stubs[i],
                    roll=NoInput(0),  #  defined by Frequency
                    calendar=self.schedule.calendar,
                    adjuster=self.schedule.accrual_adjuster,
                    # non-deliverable : Not allowed with notional exchange
                    pair=pair,
                    fx_fixings=fx_fixings_[0] if not mtm else fx_fixings_[i],
                    delivery=fx_delivery(i),
                    # index params
                    index_base=index_base,
                    index_lag=index_lag,
                    index_method=index_method,
                    index_fixings=index_fixings_[i],
                    index_base_date=self.schedule.aschedule[0],
                    index_reference_date=self.schedule.aschedule[i + 1],
                    index_only=index_only,
                )
                for i in range(self.schedule.n_periods)
            ]
        )

        # amortization exchanges
        if not final_exchange_ or self.amortization._type == _AmortizationType.NoAmortization:
            self._amortization_exchange_periods: tuple[_BasePeriod, ...] | None = None
        else:
            # only with notional exchange and some Amortization amount
            self._amortization_exchange_periods = tuple(
                [
                    Cashflow(  # type: ignore[abstract]
                        notional=self.amortization.amortization[i],
                        payment=self.schedule.pschedule2[i + 1],
                        currency=self._currency,
                        ex_dividend=self.schedule.pschedule3[i + 1],
                        # non-deliverable params
                        pair=pair,
                        fx_fixings=fx_fixings_[0] if not mtm else fx_fixings_[i + 1],
                        delivery=self.schedule.pschedule2[0]
                        if not mtm
                        else self.schedule.pschedule2[i + 1],  # schedule for exchanges
                        # index params
                        index_base=index_base,
                        index_lag=index_lag,
                        index_method=index_method,
                        index_fixings=index_fixings_[i],
                        index_base_date=self.schedule.aschedule[0],
                        index_reference_date=self.schedule.aschedule[i + 1],
                        index_only=index_only,
                    )
                    for i in range(self.schedule.n_periods - 1)
                ]
            )

        # mtm exchanges
        if mtm and final_exchange_:
            if isinstance(pair, NoInput):
                raise ValueError(err.VE_PAIR_AND_LEG_MTM)
            self._mtm_exchange_periods: tuple[_BasePeriod, ...] | None = tuple(
                [
                    MtmCashflow(  # type: ignore[abstract]
                        payment=self.schedule.pschedule2[i + 1],
                        notional=-self.amortization.outstanding[i],
                        pair=pair,
                        start=self.schedule.pschedule2[i],
                        end=self.schedule.pschedule2[i + 1],
                        currency=self._currency,
                        ex_dividend=self.schedule.pschedule3[i + 1],
                        fx_fixings_start=fx_fixings_[i],
                        fx_fixings_end=fx_fixings_[i + 1],
                        # index params
                        index_base=index_base,
                        index_lag=index_lag,
                        index_method=index_method,
                        index_fixings=index_fixings_[i],
                        index_base_date=self.schedule.aschedule[0],
                        index_reference_date=self.schedule.aschedule[i + 1],
                        index_only=index_only,
                    )
                    for i in range(self.schedule.n_periods - 1)
                ]
            )
        else:
            self._mtm_exchange_periods = None

    def spread(
        self,
        *,
        target_npv: DualTypes,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        # local_npv is calculated to identify the isolated NPV component of cashflow exchanges.
        _ = self.fixed_rate
        self.fixed_rate = 0.0
        local_npv = self.local_npv(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            forward=forward,
            settlement=settlement,
        )
        self.fixed_rate = _

        a_delta = self.local_analytic_delta(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            forward=forward,
            settlement=settlement,
        )
        return -(target_npv - local_npv) / a_delta


class ZeroFixedLeg(_BaseLeg):
    """
    Create a zero coupon fixed leg composed of a single
    :class:`~rateslib.periods.FixedPeriod` .

    """

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.components.parameters._SettlementParams` associated with
        the first :class:`~rateslib.periods.components.FloatPeriod`."""
        return self._regular_periods[0].settlement_params

    @cached_property
    def periods(self) -> list[_BasePeriod]:
        """Combine all period collection types into an ordered list."""
        periods_: list[_BasePeriod] = []

        if self._exchange_periods[0] is not None:
            periods_.append(self._exchange_periods[0])
        periods_.extend(self._regular_periods)
        if self._exchange_periods[1] is not None:
            periods_.append(self._exchange_periods[1])

        return periods_

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    @property
    def amortization(self) -> Amortization:
        return self._amortization

    def __init__(
        self,
        schedule: Schedule,
        *,
        fixed_rate: NoInput = NoInput(0),
        # settlement and currency
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        # non-deliverable
        pair: str_ = NoInput(0),
        fx_fixings: LegFixings = NoInput(0),
        mtm: bool = False,
        # period
        convention: str_ = NoInput(0),
        initial_exchange: bool = False,
        final_exchange: bool = False,
        # index params
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        index_only: bool = False,
    ) -> None:
        self._schedule = schedule
        if self.schedule.frequency == "Z":
            raise ValueError(
                "`frequency` for a ZeroFixedLeg should not be 'Z'. The Leg is zero frequency by "
                "construction. Set the `frequency` equal to the compounding frequency of the "
                "expressed fixed rate, e.g. 'S' for semi-annual compounding.",
            )
        self._notional: DualTypes = _drb(defaults.notional, notional)
        self._currency: str = _drb(defaults.base_currency, currency).lower()
        self._convention: str = _drb(defaults.convention, convention)
        self._amortization = Amortization(n=self.schedule.n_periods, initial=self._notional)

        index_fixings_ = _leg_fixings_to_list(index_fixings, self.schedule.n_periods)
        fx_fixings_ = _leg_fixings_to_list(fx_fixings, self.schedule.n_periods)

        # Exchange periods
        if not initial_exchange:
            _ini_cf: Cashflow | None = None
        else:
            _ini_cf = Cashflow(  # type: ignore[abstract]
                payment=self.schedule.pschedule2[0],
                notional=-self._amortization.outstanding[0],
                currency=self._currency,
                ex_dividend=self.schedule.pschedule3[0],
                # non-deliverable
                pair=pair,
                fx_fixings=fx_fixings_[0],
                delivery=self.schedule.pschedule2[0],
                # index params
                index_base=index_base,
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings_[0],
                index_base_date=self.schedule.aschedule[0],
                index_reference_date=self.schedule.aschedule[0],
                index_only=index_only,
            )
        final_exchange_ = final_exchange or initial_exchange
        if not final_exchange_:
            _final_cf: Cashflow | None = None
        else:
            _final_cf = Cashflow(  # type: ignore[abstract]
                payment=self.schedule.pschedule2[-1],
                notional=self._amortization.outstanding[-1],
                currency=self._currency,
                ex_dividend=self.schedule.pschedule3[-1],
                # non-deliverable
                pair=pair,
                fx_fixings=fx_fixings_[0] if not mtm else fx_fixings_[-1],
                delivery=self.schedule.pschedule2[0] if not mtm else self.schedule.pschedule2[-2],
                # index parameters
                index_base=index_base,
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings_[0],
                index_base_date=self.schedule.aschedule[0],
                index_reference_date=self.schedule.aschedule[-1],
                index_only=index_only,
            )
        self._exchange_periods = (_ini_cf, _final_cf)

        self._regular_periods = (
            ZeroFixedPeriod(  # type: ignore[abstract]
                fixed_rate=NoInput(0),
                schedule=self.schedule,
                # currency args
                currency=self._currency,
                notional=self._notional,
                # period params
                convention=self._convention,
                # non-deliverable : Not allowed with notional exchange
                pair=pair,
                fx_fixings=fx_fixings_[0],
                delivery=self.schedule.pschedule2[0]
                if (not mtm or final_exchange)
                else self.schedule.pschedule2[-1],
                # index params
                index_base=index_base,
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings_[0],
                index_only=index_only,
            ),
        )

        self.fixed_rate = fixed_rate

    @property
    def fixed_rate(self) -> DualTypes_:
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes_) -> None:
        self._fixed_rate = value
        for period in self._regular_periods:
            period.rate_params.fixed_rate = value

    def spread(
        self,
        *,
        target_npv: DualTypes,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        disc_curve_ = _disc_required_maybe_from_curve(rate_curve, disc_curve)

        if not isinstance(settlement, NoInput):
            if settlement > self.settlement_params.ex_dividend:
                raise ZeroDivisionError(
                    "A `spread` cannot be determined when the *Leg* always has zero value.\n"
                    "The given `settlement` is after the `ex_dividend` date."
                )
            else:
                w_fwd = disc_curve_[_drb(settlement, forward)]
        else:
            if isinstance(forward, NoInput):
                w_fwd = 1.0
            else:
                w_fwd = disc_curve_[forward]

        immediate_target_npv = target_npv * w_fwd
        unindexed_target_npv = immediate_target_npv / self._regular_periods[0].index_up(
            1.0, index_curve=index_curve
        )
        unindexed_reference_target_npv = unindexed_target_npv / self._regular_periods[
            0
        ].convert_deliverable(1.0, fx=fx)

        f = self.schedule.periods_per_annum
        d = self._regular_periods[0].dcf
        N = self.settlement_params.notional
        w = disc_curve_[self.settlement_params.payment]
        R = ((-unindexed_reference_target_npv / (N * w) + 1) ** (1 / (d * f)) - 1) * f * 10000.0
        return R


class ZeroIndexLeg(_BaseLeg):
    """
    Create a zero coupon fixed leg composed of a single
    :class:`~rateslib.periods.FixedPeriod` .

    """

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.components.parameters._SettlementParams` associated with
        the first :class:`~rateslib.periods.components.FloatPeriod`."""
        return self._regular_periods[0].settlement_params

    @cached_property
    def periods(self) -> list[_BasePeriod]:
        """Combine all period collection types into an ordered list."""
        periods_: list[_BasePeriod] = []

        if self._exchange_periods[0] is not None:
            periods_.append(self._exchange_periods[0])
        periods_.extend(self._regular_periods)

        return periods_

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    @property
    def amortization(self) -> Amortization:
        return self._amortization

    def __init__(
        self,
        schedule: Schedule,
        *,
        # settlement and currency
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        # non-deliverable
        pair: str_ = NoInput(0),
        fx_fixings: LegFixings = NoInput(0),
        mtm: bool = False,
        # period
        convention: str_ = NoInput(0),
        initial_exchange: bool = False,
        final_exchange: bool = False,
        # index params
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
    ) -> None:
        self._schedule = schedule
        self._notional: DualTypes = _drb(defaults.notional, notional)
        self._currency: str = _drb(defaults.base_currency, currency).lower()
        self._convention: str = _drb(defaults.convention, convention)
        self._amortization = Amortization(n=self.schedule.n_periods, initial=self._notional)

        index_fixings_ = _leg_fixings_to_list(index_fixings, self.schedule.n_periods)
        fx_fixings_ = _leg_fixings_to_list(fx_fixings, self.schedule.n_periods)

        # Exchange periods
        if not initial_exchange:
            _ini_cf: Cashflow | None = None
        else:
            _ini_cf = Cashflow(  # type: ignore[abstract]
                payment=self.schedule.pschedule2[0],
                notional=-self._amortization.outstanding[0],
                currency=self._currency,
                ex_dividend=self.schedule.pschedule3[0],
                # non-deliverable
                pair=pair,
                fx_fixings=fx_fixings_[0],
                delivery=self.schedule.pschedule2[0],
                # index params
                index_base=index_base,
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings_[0],
                index_base_date=self.schedule.aschedule[0],
                index_reference_date=self.schedule.aschedule[0],
                index_only=False,  #  is only True if there is not final exchange
            )
        final_exchange_ = final_exchange or initial_exchange
        _final_cf = Cashflow(  # type: ignore[abstract]
            payment=self.schedule.pschedule2[-1],
            notional=self._amortization.outstanding[-1],
            currency=self._currency,
            ex_dividend=self.schedule.pschedule3[-1],
            # non-deliverable
            pair=pair,
            fx_fixings=fx_fixings_[0] if not mtm else fx_fixings_[-1],
            delivery=self.schedule.pschedule2[0] if not mtm else self.schedule.pschedule2[-2],
            # index parameters
            index_base=index_base,
            index_lag=index_lag,
            index_method=index_method,
            index_fixings=index_fixings_[0],
            index_base_date=self.schedule.aschedule[0],
            index_reference_date=self.schedule.aschedule[-1],
            index_only=not final_exchange_,
        )
        self._exchange_periods = (_ini_cf,)
        self._regular_periods = (_final_cf,)
