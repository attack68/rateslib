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

import rateslib.errors as err
from rateslib import defaults
from rateslib.curves._parsers import (
    _disc_required_maybe_from_curve,
)
from rateslib.data.fixings import _leg_fixings_to_list
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import LegMtm, _get_let_mtm
from rateslib.legs.amortization import Amortization, _AmortizationType, _get_amortization
from rateslib.legs.protocols import (
    _BaseLeg,
    _WithExDiv,
)
from rateslib.periods import (
    Cashflow,
    FixedPeriod,
    MtmCashflow,
    ZeroFixedPeriod,
)
from rateslib.periods.protocols import (
    _BasePeriod,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXIndex,
        IndexMethod,
        LegFixings,
        Schedule,
        Series,
        _BaseCurve_,
        _FXVolOption_,
        _SettlementParams,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FixedLeg(_BaseLeg, _WithExDiv):
    """
    A *Leg* containing :class:`~rateslib.periods.FixedPeriod`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib import fixings, Schedule
       from pandas import Series
       from rateslib.legs import FixedLeg
       from datetime import datetime as dt

    .. ipython:: python

       fl = FixedLeg(
           schedule=Schedule(
                effective=dt(2000, 2, 1),
                termination=dt(2002, 2, 1),
                frequency="S",
           ),
           convention="ActActICMA",
           fixed_rate=2.5,
           notional=10e6,
       )
       fl.cashflows()

    .. role:: red

    .. role:: green

    Parameters
    ----------
    schedule: Schedule, :red:`required`
        The :class:`~rateslib.scheduling.Schedule` object which structures contiguous *Periods*.
        The schedule object also contains data for payment dates, payment dates for notional
        exchanges and ex-dividend dates for each period.

        .. note::

           The following are **period parameters** combined with the ``schedule``.

    convention: str, :green:`optional (set by 'defaults')`
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.scheduling.dcf`.

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

           The following define **rate parameters**.

    fixed_rate: float, Dual, Dual2, Variable, :green:`optional`
        The fixed rate of each composited :class:`~rateslib.periods.FixedPeriod`.

        .. note::

           The following define **non-deliverable** parameters. If the *Leg* is directly
           deliverable then do not set a non-deliverable ``pair`` or any ``fx_fixings``.

    pair: FXIndex, str, :green:`optional`
        The :class:`~rateslib.data.fixings.FXIndex` for :class:`~rateslib.data.fixings.FXFixing`
        defining the currency pair that determines *Period*
        settlement. The *reference currency* is implied from ``pair``. Must include ``currency``.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for each *Period* according
        to non-deliverability. Review the **notes** section non-deliverability. This should only
        ever be entered as either:

        - scalar value: 1.15,
        - fixings series: "Reuters_ZBS",
        - tuple of transaction rate and fixing series: (1.25, "Reuters_ZBC")
    mtm: LegMtm or str, :green:`optional (set to 'initial')`
        Define how the fixing dates are determined for each :class:`~rateslib.data.fixings.FXFixing`
        See **Notes** regarding non-deliverability.

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

    The leg uses a ``mtm`` argument to define the types of non-deliverability that it can
    construct. Currently there are three kinds which cater to the various type of requirements
    for, *ND-IRS*, *MTM-XCS*, *non-MTM XCS*, *ND-XCS*

    .. tabs::

       .. tab:: Initial

          This uses the *Initial* variant of a :class:`~rateslib.enums.LegMtm` and it
          defines all :class:`~rateslib.data.fixings.FXFixing` on the *Leg* to be a single date
          at the start of the *Leg* (derived from ``schedule.pschedule2[0]``). Usually this fixing is
          directly specified being agreed at execution of the transaction and not dependent
          upon a published financial fixing.

          This type of *non-deliverability* is suitable to define a *Leg* of one currency, but
          expressed by a notional in another currency, and is used for a *non-MTM XCS*.

          Since only one fixing is required, ``fx_fixings`` can be entered either as
          a known scalar value or string series identifier.

          .. ipython:: python

             leg = FixedLeg(
                 schedule=Schedule(
                     effective=dt(2000, 1, 1),
                     termination=dt(2000, 7, 1),
                     frequency="Q",
                     payment_lag=1,
                     payment_lag_exchange=0,
                 ),
                 fixed_rate=1.0,
                 initial_exchange=True,
                 mtm="initial",
                 currency="usd",
                 pair="eurusd",
                 notional=10e6,      # <- Leg is a USD leg but expressed with a EUR notional
                 fx_fixings=1.25,    # <- All periods are treated as 12.5mm USD
             )
             print(leg.cashflows())

       .. tab:: Payment

          Under the *Payment* variant of a :class:`~rateslib.enums.LegMtm`
          all reference currency cashflows are converted to settlement
          currency using an :class:`~rateslib.data.fixings.FXFixing` with a date of the payment.
          This is probably the most traditional type of non-deliverability and is suitable
          for *NDIRS* and *NDXCS* *Instruments*.

          The best practice entry for ``fx_fixings`` depends if the *Leg* has
          notional exchanges or not. If there is an initial notional exchange then
          a 2-tuple, with the first element being the transacted exchange rate and
          the second element referring to the fixing
          series for future *FX Fixings*. If only future fixings are required then a string
          series is used.

          .. ipython:: python

             fixings.add("WMR_10AM_TY0_T+2_EURUSD", Series(
                 index=[dt(2000, 1, 1), dt(2000, 4, 2), dt(2000, 7, 1), dt(2000, 7, 2)],
                 data=[1.26, 1.27, 1.29, 1.295])
             )
             leg = FixedLeg(
                 schedule=Schedule(
                     effective=dt(2000, 1, 1),
                     termination=dt(2000, 7, 1),
                     frequency="Q",
                     payment_lag=1,
                     payment_lag_exchange=0,
                 ),
                 fixed_rate=1.0,
                 initial_exchange=True,
                 mtm="payment",
                 currency="usd",
                 pair="eurusd",
                 notional=10e6,                          # <- Leg settles in USD leg but reference cashflows in EUR
                 fx_fixings=(1.25, "WMR_10AM_TY0_T+2"),  # <- Initial exchange rate and future fixings
             )
             print(leg.cashflows())

       .. tab:: XCS

          The *XCS* variant of a :class:`~rateslib.enums.LegMtm` is specially configured
          for *MTM-XCS*. These *Legs* have their
          cashflows determined with :class:`~rateslib.data.fixings.FXFixing` at the start of
          each *Period*, in a manner slightly similar  to the *Initial* variant, and specifically
          generated :class:`~rateslib.periods.MtmCashflow` *Periods* adjusting the value of the
          notional by an *FXFixing* at the end of each *Period*.

          The best practice entry for ``fx_fixings`` is as a 2-tuple, with the first
          element the transacted exchange rate and the second element referring to the fixing
          series for future *FX Fixings*.

          .. ipython:: python

             fixings.add("WMR_4PM_GMT_T+2_EURUSD", Series(
                 index=[dt(2000, 4, 1), dt(2000, 4, 2), dt(2000, 7, 2)],
                 data=[1.265, 1.27, 1.29])
             )
             leg = FixedLeg(
                 schedule=Schedule(
                     effective=dt(2000, 1, 1),
                     termination=dt(2000, 7, 1),
                     frequency="Q",
                     payment_lag=1,
                     payment_lag_exchange=0,
                 ),
                 fixed_rate=1.0,
                 initial_exchange=True,
                 currency="usd",
                 pair="eurusd",
                 mtm="xcs",
                 notional=10e6,
                 fx_fixings=(1.25, "WMR_4PM_GMT_T+2"),
             )
             print(leg.cashflows())

    **Amortization and Non-Deliverability**

    When amortization is combined with non-deliverability, the interim notional exchange cashflows
    are adjusted appropriately in both the non-mtm and mtm cases.

    .. tabs::

       .. tab:: Initial

          Amortization under this method adopts the same singular fixing as all other *Periods*.

          .. ipython:: python

             leg = FixedLeg(
                 schedule=Schedule(
                     effective=dt(2000, 1, 1),
                     termination=dt(2000, 7, 1),
                     frequency="Q",
                     payment_lag=1,
                     payment_lag_exchange=0,
                 ),
                 fixed_rate=1.0,
                 initial_exchange=True,
                 mtm="initial",
                 currency="usd",
                 pair="eurusd",
                 notional=10e6,      # <- Leg is a USD leg but expressed with a EUR notional
                 amortization=4e6,
                 fx_fixings=1.25,    # <- All periods are treated as 12.5mm USD
             )
             print(leg.cashflows())

       .. tab:: Payment

          Amortization under this method settles according to the payment date.

          .. ipython:: python

             leg = FixedLeg(
                 schedule=Schedule(
                     effective=dt(2000, 1, 1),
                     termination=dt(2000, 7, 1),
                     frequency="Q",
                     payment_lag=1,
                     payment_lag_exchange=0,
                 ),
                 fixed_rate=1.0,
                 initial_exchange=True,
                 mtm="payment",
                 currency="usd",
                 pair="eurusd",
                 notional=10e6,                     # <- Leg settles in USD leg but reference cashflows in EUR
                 amortization=4e6,
                 fx_fixings=(1.25, "WMR_10AM_TY0_T+2"),  # <- Initial exchange rate and future fixings
             )
             print(leg.cashflows())

       .. tab:: XCS

          Amortization for a *XCS* takes places after the :class:`~rateslib.periods.MtmCashflow`.

          .. ipython:: python

             leg = FixedLeg(
                 schedule=Schedule(
                     effective=dt(2000, 1, 1),
                     termination=dt(2000, 7, 1),
                     frequency="Q",
                     payment_lag=1,
                     payment_lag_exchange=0,
                 ),
                 fixed_rate=1.0,
                 initial_exchange=True,
                 currency="usd",
                 pair="eurusd",
                 mtm="xcs",
                 notional=10e6,
                 amortization=4e6,
                 fx_fixings=(1.25, "WMR_4PM_GMT_T+2"),
             )
             print(leg.cashflows())

    **Indexation, Non-Deliverability and Amortization**

    In the most complicated case, which rarely even relates to real tradable instruments all
    of the parameters may be combined. The :meth:`~rateslib.legs.FixedLeg.cashflows`
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
           mtm="xcs",
           fx_fixings=(1.25, "WMR_10AM_TY0_T+2"),
           index_lag=0,
           index_fixings="MY_RPI",
           index_method="monthly",
       )
       print(leg.cashflows())

    .. ipython:: python
       :suppress:

       fixings.pop("WMR_10AM_TY0_T+2_EURUSD")
       fixings.pop("WMR_4PM_GMT_T+2_EURUSD")
       fixings.pop("MY_RPI")

    """  # noqa: E501

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.parameters._SettlementParams` associated with
        the first :class:`~rateslib.periods.FixedPeriod`."""
        return self._regular_periods[0].settlement_params

    @cached_property
    def periods(self) -> list[_BasePeriod]:
        """A list of all contained *Periods*."""
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
        :class:`~rateslib.periods.FixedPeriod`."""
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes_) -> None:
        self._fixed_rate = value
        for period in self._regular_periods:
            period.rate_params.fixed_rate = value

    @property
    def schedule(self) -> Schedule:
        """The :class:`~rateslib.scheduling.Schedule` object of *Leg*."""
        return self._schedule

    @property
    def amortization(self) -> Amortization:
        """
        The :class:`~rateslib.legs.Amortization` object associated with the schedule.
        """
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
        pair: FXIndex | str_ = NoInput(0),
        fx_fixings: LegFixings = NoInput(0),
        mtm: LegMtm | str = LegMtm.Initial,
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
        index_fixings: LegFixings = NoInput(0),
        index_only: bool = False,
    ) -> None:
        self._fixed_rate = fixed_rate
        del fixed_rate
        self._schedule = schedule
        del schedule
        self._notional: DualTypes = _drb(defaults.notional, notional)
        del notional
        self._amortization: Amortization = _get_amortization(
            amortization, self._notional, self.schedule.n_periods
        )
        del amortization
        self._currency: str = _drb(defaults.base_currency, currency).lower()
        del currency
        self._convention: str = _drb(defaults.convention, convention)
        del convention
        self._mtm = _get_let_mtm(mtm)
        del mtm

        index_fixings_ = _leg_fixings_to_list(index_fixings, self.schedule.n_periods)
        del index_fixings

        # if initial and final exchange with MtM.Payment then there is an extra fixing date
        _mtm_param = 1 if (self._mtm == LegMtm.Payment and initial_exchange) else 0
        fx_fixings_ = _leg_fixings_to_list(fx_fixings, self.schedule.n_periods + _mtm_param)
        del fx_fixings

        # Exchange periods
        if not initial_exchange:
            _ini_cf: Cashflow | None = None
        else:
            _ini_cf = Cashflow(
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
            delivery_ = {
                LegMtm.Initial: self.schedule.pschedule2[0],
                LegMtm.XCS: self.schedule.pschedule2[-2],
                LegMtm.Payment: self.schedule.pschedule2[-1],
            }
            _final_cf = Cashflow(
                payment=self.schedule.pschedule2[-1],
                notional=self._amortization.outstanding[-1],
                currency=self._currency,
                ex_dividend=self.schedule.pschedule3[-1],
                # non-deliverable
                pair=pair,
                fx_fixings=fx_fixings_[0] if self._mtm == LegMtm.Initial else fx_fixings_[-1],
                delivery=delivery_[self._mtm],
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

        self._regular_periods: tuple[FixedPeriod, ...] = tuple(
            [
                FixedPeriod(
                    fixed_rate=self.fixed_rate,
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
                    fx_fixings=fx_fixings_[0]
                    if self._mtm == LegMtm.Initial
                    else fx_fixings_[i + _mtm_param],
                    delivery=_fx_delivery(i, self._mtm, self.schedule, False, False),
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
                    Cashflow(
                        notional=self.amortization.amortization[i],
                        payment=self.schedule.pschedule2[i + 1],
                        currency=self._currency,
                        ex_dividend=self.schedule.pschedule3[i + 1],
                        # non-deliverable params
                        pair=pair,
                        fx_fixings=fx_fixings_[0]
                        if self._mtm == LegMtm.Initial
                        else fx_fixings_[i + 1],
                        delivery=_fx_delivery(
                            i, self._mtm, self.schedule, True, True
                        ),  # schedule for exchanges
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
        if self._mtm == LegMtm.XCS and final_exchange_:
            if isinstance(pair, NoInput):
                raise ValueError(err.VE_PAIR_AND_LEG_MTM)
            self._mtm_exchange_periods: tuple[_BasePeriod, ...] | None = tuple(
                [
                    MtmCashflow(
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
        fx_vol: _FXVolOption_ = NoInput(0),
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
    A zero coupon *Leg* composed of a single
    :class:`~rateslib.periods.ZeroFixedPeriod` .

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.legs import ZeroFixedLeg
       from rateslib.scheduling import Schedule
       from datetime import datetime as dt
       from pandas import Series

    .. ipython:: python

       zfl = ZeroFixedLeg(
           schedule=Schedule(
                effective=dt(2000, 2, 1),
                termination=dt(2002, 2, 1),
                frequency="S",
           ),
           fixed_rate=2.5,
           notional=10e6,
       )
       zfl.cashflows()

    .. role:: red

    .. role:: green

    Parameters
    ----------
    schedule: Schedule, :red:`required`
        The :class:`~rateslib.scheduling.Schedule` object which structures contiguous *Periods*.
        The schedule object also contains data for payment dates, payment dates for notional
        exchanges and ex-dividend dates for each period.

        .. note::

           The following are **period parameters** combined with the ``schedule``.

    convention: str, :green:`optional (set by 'defaults')`
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.scheduling.dcf`.

        .. note::

           The following define generalised **settlement** parameters.

    currency : str, :green:`optional (set by 'defaults')`
        The local settlement currency of the leg (3-digit code).
    notional : float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The initial leg notional, defined in units of *reference currency*.
    initial_exchange : bool, :green:`optional (set as False)`
        Whether to also include an initial notional exchange. If *True* then ``final_exchange``
        **will** also be set to *True*.
    final_exchange : bool, :green:`optional (set as initial_exchange)`
        Whether to also include a final notional exchange and interim amortization
        notional exchanges.

        .. note::

           The following define **rate parameters**.

    fixed_rate: float, Dual, Dual2, Variable, :green:`optional`
        The IRR of the composited :class:`~rateslib.periods.ZeroFixedPeriod`.

        .. note::

           The following define **non-deliverable** parameters. If the *Leg* is directly
           deliverable then do not set a non-deliverable ``pair`` or any ``fx_fixings``.

    pair: FXIndex, str, :green:`optional`
        The :class:`~rateslib.data.fixings.FXIndex` for :class:`~rateslib.data.fixings.FXFixing`
        defining the currency pair that determines *Period*
        settlement. The *reference currency* is implied from ``pair``. Must include ``currency``.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for each *Period* according
        to non-deliverability. Review the **notes** section non-deliverability.
    mtm: bool, :green:`optional (set to False)`
        Define whether the non-deliverability depends on a single
        :class:`~rateslib.data.fixings.FXFixing` defined at the start of the *Leg*, or the end.
        Review the **notes** section non-deliverability.

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
    """

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.parameters._SettlementParams` associated with
        the :class:`~rateslib.periods.ZeroFixedPeriod`."""
        return self._regular_periods[0].settlement_params

    @cached_property
    def periods(self) -> list[_BasePeriod]:
        """A list of all contained *Periods*."""
        periods_: list[_BasePeriod] = []

        if self._exchange_periods[0] is not None:
            periods_.append(self._exchange_periods[0])
        periods_.extend(self._regular_periods)
        if self._exchange_periods[1] is not None:
            periods_.append(self._exchange_periods[1])

        return periods_

    @property
    def schedule(self) -> Schedule:
        """The :class:`~rateslib.scheduling.Schedule` object of *Leg*."""
        return self._schedule

    @property
    def amortization(self) -> Amortization:
        """
        The :class:`~rateslib.legs.Amortization` object associated with the schedule.
        """
        return self._amortization

    def __init__(
        self,
        schedule: Schedule,
        *,
        # period
        convention: str_ = NoInput(0),
        # rate params
        fixed_rate: NoInput = NoInput(0),
        # settlement and currency
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        initial_exchange: bool = False,
        final_exchange: bool = False,
        # non-deliverable
        pair: FXIndex | str_ = NoInput(0),
        fx_fixings: LegFixings = NoInput(0),
        mtm: bool = False,
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
            _ini_cf = Cashflow(
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
            _final_cf = Cashflow(
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
            ZeroFixedPeriod(
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
        """The fixed rate parameter of the composited
        :class:`~rateslib.periods.ZeroFixedPeriod`."""
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
        fx_vol: _FXVolOption_ = NoInput(0),
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
    A *Leg* composed of *indexed* :class:`~rateslib.periods.Cashflow` at termination,
    and possibly effective.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.legs import ZeroIndexLeg
       from rateslib.scheduling import Schedule
       from datetime import datetime as dt
       from pandas import Series

    .. ipython:: python

       fixings.add("CPI_UK", Series(index=[dt(2000, 1, 1), dt(2002, 1, 1)], data=[100.0, 115.0]))
       zil = ZeroIndexLeg(
           schedule=Schedule(
                effective=dt(2000, 2, 1),
                termination=dt(2002, 2, 1),
                frequency="Z",
           ),
           index_lag=1,
           index_fixings="CPI_UK",
           notional=10e6,
       )
       zil.cashflows()

    .. ipython:: python
       :suppress:

       fixings.pop("CPI_UK")

    .. role:: red

    .. role:: green

    Parameters
    ----------
    schedule: Schedule, :red:`required`
        The :class:`~rateslib.scheduling.Schedule` object which structures contiguous *Periods*.
        The schedule object also contains data for payment dates, payment dates for notional
        exchanges and ex-dividend dates for each period. Only the start and end of the schedule are
        relevant for this *Zero* type *Leg*.

        .. note::

           The following define generalised **settlement** parameters.

    currency : str, :green:`optional (set by 'defaults')`
        The local settlement currency of the leg (3-digit code).
    notional : float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The initial leg notional, defined in units of *reference currency*.
    initial_exchange : bool, :green:`optional (set as False)`
        Whether to also include an initial notional exchange. If *True* then ``final_exchange``
        **will** also be set to *True*.
    final_exchange : bool, :green:`optional (set as initial_exchange)`
        Whether to also include a final notional exchange and interim amortization
        notional exchanges.

    .. note::

           The following are **period parameters** combined with the ``schedule``.

    convention: str, :green:`optional (set by 'defaults')`
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.scheduling.dcf`.

        .. note::

           The following define **non-deliverable** parameters. If the *Leg* is directly
           deliverable then do not set a non-deliverable ``pair`` or any ``fx_fixings``.

    pair: FXIndex, str, :green:`optional`
        The :class:`~rateslib.data.fixings.FXIndex` for :class:`~rateslib.data.fixings.FXFixing`
        defining the currency pair that determines *Period*
        settlement. The *reference currency* is implied from ``pair``. Must include ``currency``.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for each *Period* according
        to non-deliverability. Review the **notes** section non-deliverability.
    mtm: bool, :green:`optional (set to False)`
        Define whether the non-deliverability depends on a single
        :class:`~rateslib.data.fixings.FXFixing` defined at the start of the *Leg*, or the end.
        Review the **notes** section non-deliverability.

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

    Notes
    -----
    A :class:`~rateslib.legs.ZeroIndexLeg` contains, at most, two
    :class:`~rateslib.periods.Cashflow`. Three structures can be configured:

    - One cashflow consisting of only the **indexed amount** relating to some notional value (
      ``initial_exchange`` and ``final_exchange`` are both *False*)
    - One cashflow consisting of a notional amount **plus its indexed amount** (``final_exchange``
      is *True*)
    - Two cashflows (of opposite directions) exchanging notionals (``initial_exchange`` and
      ``final_exchange`` are both *True*)

    **Non-deliverability**

    Non-deliverability behaves in the same way as a :class:`~rateslib.legs.FixedLeg`.
    If ``mtm`` is *False* then a single :class:`~rateslib.data.fixings.FXFixing` defined by
    the ``effective`` date or an agreed transactional value is used for all cashflows.

    With notional exchanges this same principle applies, since there are only upto two cashflows.

    Without notional exchanges and setting ``mtm`` to *True* allows the
    :class:`~rateslib.data.fixings.FXFixing` to have a delivery date equal to the future payment
    date of the cashflow.

    """

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.parameters._SettlementParams` associated with
        the :class:`~rateslib.periods.Cashflow` at maturity."""
        return self._regular_periods[0].settlement_params

    @cached_property
    def periods(self) -> list[_BasePeriod]:
        """A list of all contained *Periods*."""
        periods_: list[_BasePeriod] = []

        if self._exchange_periods[0] is not None:
            periods_.append(self._exchange_periods[0])
        periods_.extend(self._regular_periods)

        return periods_

    @property
    def schedule(self) -> Schedule:
        """The :class:`~rateslib.scheduling.Schedule` object of *Leg*."""
        return self._schedule

    @property
    def amortization(self) -> Amortization:
        """
        The :class:`~rateslib.legs.Amortization` object associated with the schedule.
        """
        return self._amortization

    def __init__(
        self,
        schedule: Schedule,
        *,
        # period
        convention: str_ = NoInput(0),
        # settlement and currency
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        initial_exchange: bool = False,
        final_exchange: bool = False,
        # non-deliverable
        pair: FXIndex | str_ = NoInput(0),
        fx_fixings: LegFixings = NoInput(0),
        mtm: bool = False,
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
            _ini_cf = Cashflow(
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
        _final_cf = Cashflow(
            payment=self.schedule.pschedule2[-1],
            notional=self._amortization.outstanding[-1],
            currency=self._currency,
            ex_dividend=self.schedule.pschedule3[-1],
            # non-deliverable
            pair=pair,
            fx_fixings=fx_fixings_[0] if not mtm else fx_fixings_[-1],
            delivery=self.schedule.pschedule2[-1]
            if (mtm and not final_exchange_)
            else self.schedule.pschedule2[0],
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

    def spread(self, *args: Any, **kwargs: Any) -> DualTypes:
        return super().spread(*args, **kwargs)  # type: ignore[safe-super]


def _fx_delivery(
    i: int,
    mtm: LegMtm,
    schedule: Schedule,
    is_exchange: bool,
    is_amortisation: bool,
) -> datetime:
    """Based on the `mtm` parameter determine the FX fixing dates for regular period 'i'."""
    if mtm == LegMtm.Initial:
        # then ND type is a one-fixing only, so is determined by only a single rate of exchange
        # this date is set to the initial payment exchange date of the schedule
        return schedule.pschedule2[0]
    elif mtm == LegMtm.Payment:
        # then the ND type is a NDXCS or a NDIRS which determines FX at payment
        if is_exchange:
            return schedule.pschedule2[i + 1]
        else:
            return schedule.pschedule[i + 1]
    else:  # LegMtm.XCS
        # then the ND type is a MTM-XCS which has special MTMCashflow periods
        # the relevant FX fixing is set in advance of the period using notional exchange dates
        if is_amortisation:
            return schedule.pschedule2[i + 1]
        else:
            return schedule.pschedule2[i]
