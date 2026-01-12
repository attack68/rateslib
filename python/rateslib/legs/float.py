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

import rateslib.errors as err
from rateslib import defaults
from rateslib.data.fixings import _leg_fixings_to_list
from rateslib.dual import ift_1dim
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import FloatFixingMethod, LegMtm, SpreadCompoundMethod, _get_let_mtm
from rateslib.legs.amortization import Amortization, _AmortizationType, _get_amortization
from rateslib.legs.custom import CustomLeg
from rateslib.legs.fixed import _fx_delivery
from rateslib.legs.protocols import _BaseLeg, _WithExDiv
from rateslib.periods import Cashflow, FloatPeriod, MtmCashflow, ZeroFloatPeriod
from rateslib.periods.parameters import _FloatRateParams, _SettlementParams

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CurveOption_,
        DualTypes,
        DualTypes_,
        FloatRateSeries,
        Frequency,
        FXForwards_,
        FXIndex,
        IndexMethod,
        LegFixings,
        Schedule,
        Sequence,
        _BaseCurve_,
        _BasePeriod,
        _FXVolOption_,
        datetime_,
        int_,
        str_,
    )


class FloatLeg(_BaseLeg, _WithExDiv):
    """
    A *Leg* containing :class:`~rateslib.periods.FloatPeriod`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib import fixings, Schedule
       from pandas import Series
       from rateslib.legs import FloatLeg
       from datetime import datetime as dt

    .. ipython:: python

       fl = FloatLeg(
           schedule=Schedule(
                effective=dt(2000, 2, 1),
                termination=dt(2002, 2, 1),
                frequency="S",
           ),
           convention="Act360",
           float_spread=25.0,
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

           The following are **period parameters** combined with the ``schedule``.

    convention: str, :green:`optional (set by 'defaults')`
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.scheduling.dcf`.

        .. note::

           The following define **rate parameters**.

    fixing_method: FloatFixingMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.FloatFixingMethod` describing the determination
        of the floating rate for each period.
    method_param: int, :green:`optional (set by 'defaults')`
        A specific parameter that is used by the specific ``fixing_method``.
    fixing_frequency: Frequency, str, :green:`optional (set by 'frequency' or '1B')`
        The :class:`~rateslib.scheduling.Frequency` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given is assumed to match the
        frequency of the schedule for an IBOR type ``fixing_method`` or '1B' if RFR type.
    fixing_series: FloatRateSeries, str, :green:`optional (implied by other parameters)`
        The :class:`~rateslib.data.fixings.FloatRateSeries` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given inherits attributes given
        such as the ``calendar``, ``convention``, ``method_param`` etc.
    float_spread: float, Dual, Dual2, Variable, :green:`optional (set as 0.0)`
        The amount (in bps) added to the rate in each period rate determination.
    spread_compound_method: SpreadCompoundMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.SpreadCompoundMethod` used in the calculation
        of the period rate when combining a ``float_spread``. Used **only** with RFR type
        ``fixing_method``.
    rate_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        See :ref:`Fixings <fixings-doc>`.
        The value of the rate fixing. If a scalar, is used directly. If a string identifier, links
        to the central ``fixings`` object and data loader.

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
    The various combinations of **amortisation**, **non-deliverability**, **indexation**,
    and **notional exchanges** are identical to, and demonstrated in the documentation for, a
    :class:`~rateslib.legs.FixedLeg` object.

    """

    @property
    def rate_params(self) -> _FloatRateParams:
        """The :class:`~rateslib.periods.parameters._FloatRateParams` associated with
        the first :class:`~rateslib.periods.FloatPeriod`."""
        return self._regular_periods[0].rate_params

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.parameters._SettlementParams` associated with
        the first :class:`~rateslib.periods.FloatPeriod`."""
        return self._regular_periods[0].settlement_params

    @property
    def periods(self) -> list[_BasePeriod]:
        """A list of all contained *Periods*."""
        periods_: list[_BasePeriod] = []

        if self._exchange_periods[0] is not None:
            periods_.append(self._exchange_periods[0])

        args: tuple[tuple[FloatPeriod | MtmCashflow | Cashflow, ...], ...] = (
            self._regular_periods[:-1],
        )
        if self._mtm_exchange_periods is not None:
            args += (self._mtm_exchange_periods,)
        if self._amortization_exchange_periods is not None:
            args += (self._amortization_exchange_periods,)
        interleaved_periods_: list[_BasePeriod] = [
            item for combination in zip(*args, strict=True) for item in combination
        ]
        interleaved_periods_.append(self._regular_periods[-1])  # add last regular period
        periods_.extend(interleaved_periods_)

        if self._exchange_periods[1] is not None:
            periods_.append(self._exchange_periods[1])

        return periods_

    @property
    def float_spread(self) -> DualTypes:
        """The float spread parameter of each composited
        :class:`~rateslib.periods.FloatPeriod`."""
        return self._regular_periods[0].rate_params.float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        for period in self._regular_periods:
            period.rate_params.float_spread = value

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
        float_spread: DualTypes_ = NoInput(0),
        rate_fixings: LegFixings = NoInput(0),
        fixing_method: FloatFixingMethod | str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        spread_compound_method: SpreadCompoundMethod | str_ = NoInput(0),
        fixing_frequency: Frequency | str_ = NoInput(0),
        fixing_series: FloatRateSeries | str_ = NoInput(0),
        # index params
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: LegFixings = NoInput(0),
    ) -> None:
        self._schedule = schedule
        del schedule
        self._notional: DualTypes = _drb(defaults.notional, notional)
        del notional
        self._amortization: Amortization = _get_amortization(
            amortization, self._notional, self._schedule.n_periods
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
            )
        self._exchange_periods = (_ini_cf, _final_cf)

        rate_fixings_list = _leg_fixings_to_list(rate_fixings, self._schedule.n_periods)
        self._regular_periods = tuple(
            [
                FloatPeriod(
                    float_spread=float_spread,
                    rate_fixings=rate_fixings_list[i],
                    fixing_method=fixing_method,
                    method_param=method_param,
                    spread_compound_method=spread_compound_method,
                    fixing_frequency=fixing_frequency,
                    fixing_series=fixing_series,
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
                    index_base_date=self._schedule.aschedule[0],
                    index_reference_date=self._schedule.aschedule[i + 1],
                )
                for i in range(self._schedule.n_periods)
            ]
        )

        # amortization exchanges
        if not final_exchange_ or self.amortization._type == _AmortizationType.NoAmortization:
            self._amortization_exchange_periods: tuple[Cashflow, ...] | None = None
        else:
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
                        index_base_date=self._schedule.aschedule[0],
                        index_reference_date=self._schedule.aschedule[i + 1],
                    )
                    for i in range(self._schedule.n_periods - 1)
                ]
            )

        # mtm exchanges
        if self._mtm == LegMtm.XCS and final_exchange_:
            if isinstance(pair, NoInput):
                raise ValueError(err.VE_PAIR_AND_LEG_MTM)
            self._mtm_exchange_periods: tuple[MtmCashflow, ...] | None = tuple(
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
                    )
                    for i in range(self.schedule.n_periods - 1)
                ]
            )
        else:
            self._mtm_exchange_periods = None

    @property
    def _is_linear(self) -> bool:
        """
        Tests if analytic delta spread is a linear function affecting NPV.

        This is non-linear if the spread is itself compounded, which only occurs
        on RFR trades with *"isda_compounding"* or *"isda_flat_compounding"*, which
        should typically be avoided anyway.

        Returns
        -------
        bool
        """
        # ruff: noqa: SIM103
        if (
            self.rate_params.fixing_method != FloatFixingMethod.IBOR
            and self.rate_params.spread_compound_method != SpreadCompoundMethod.NoneSimple
        ):
            return False
        return True

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
        if self._is_linear:
            local_npv = self.local_npv(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                fx=fx,
                forward=forward,
                settlement=settlement,
            )
            a_delta = self.local_analytic_delta(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                fx=fx,
                forward=forward,
                settlement=settlement,
            )
            return -(target_npv - local_npv) / a_delta + self.float_spread
        else:
            original_z = self.float_spread

            def s(g: DualTypes) -> DualTypes:
                """
                This determines the NPV change subject to a given float spread change denoted, g.
                """
                self.float_spread = g
                return self.local_npv(
                    rate_curve=rate_curve,
                    disc_curve=disc_curve,
                    index_curve=index_curve,
                    fx=fx,
                    forward=forward,
                    settlement=settlement,
                )

            result = ift_1dim(
                s=s,
                s_tgt=target_npv,
                h="ytm_quadratic",
                ini_h_args=(-300, 300, 1200),
                # h="modified_brent",
                # ini_h_args=(-10000, 10000),
                func_tol=1e-6,
                conv_tol=1e-6,
            )

            self.float_spread = original_z
            _: DualTypes = result["g"]
            return _


class ZeroFloatLeg(_BaseLeg):
    """
    A zero coupon *Leg* composed of a single :class:`~rateslib.periods.ZeroFloatPeriod`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.legs import ZeroFloatLeg
       from rateslib.scheduling import Schedule
       from datetime import datetime as dt
       from pandas import Series

    .. ipython:: python

       zfl = ZeroFloatLeg(
           schedule=Schedule(
                effective=dt(2000, 2, 1),
                termination=dt(2002, 2, 1),
                frequency="S",
           ),
           notional=10e6,
       )
       zfl.cashflows()
       zfl.float_periods.cashflows()

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

    fixing_method: FloatFixingMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.FloatFixingMethod` describing the determination
        of the floating rate for each period.
    method_param: int, :green:`optional (set by 'defaults')`
        A specific parameter that is used by the specific ``fixing_method``.
    fixing_frequency: Frequency, str, :green:`optional (set by 'frequency' or '1B')`
        The :class:`~rateslib.scheduling.Frequency` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given is assumed to match the
        frequency of the schedule for an IBOR type ``fixing_method`` or '1B' if RFR type.
    fixing_series: FloatRateSeries, str, :green:`optional (implied by other parameters)`
        The :class:`~rateslib.data.fixings.FloatRateSeries` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given inherits attributes given
        such as the ``calendar``, ``convention``, ``method_param`` etc.
    float_spread: float, Dual, Dual2, Variable, :green:`optional (set as 0.0)`
        The amount (in bps) added to the rate in each period rate determination.
    spread_compound_method: SpreadCompoundMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.SpreadCompoundMethod` used in the calculation
        of the period rate when combining a ``float_spread``. Used **only** with RFR type
        ``fixing_method``.
    rate_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        See :ref:`Fixings <fixings-doc>`.
        The value of the rate fixing. If a scalar, is used directly. If a string identifier, links
        to the central ``fixings`` object and data loader.

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
        the first :class:`~rateslib.periods.FloatPeriod`."""
        return self._regular_periods[0].settlement_params

    @cached_property
    def periods(self) -> Sequence[_BasePeriod]:
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

    @property
    def rate_params(self) -> _FloatRateParams:
        """The :class:`~rateslib.periods.parameters._FloatRateParams` associated with
        the first :class:`~rateslib.periods.FloatPeriod`."""
        return self._regular_periods[0].rate_params

    @property
    def float_spread(self) -> DualTypes:
        """The float spread parameter of each composited
        :class:`~rateslib.periods.FloatPeriod`."""
        return self._regular_periods[0].rate_params.float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        for period in self._regular_periods:
            period.rate_params.float_spread = value

    @property
    def float_periods(self) -> CustomLeg:
        """A :class:`~rateslib.legs.CustomLeg` containing the individual
        :class:`~rateslib.periods.FloatPeriod`."""
        return CustomLeg(self._regular_periods[0].float_periods)

    def __init__(
        self,
        schedule: Schedule,
        *,
        float_spread: DualTypes_ = NoInput(0),
        rate_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: FloatFixingMethod | str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        spread_compound_method: SpreadCompoundMethod | str_ = NoInput(0),
        fixing_frequency: Frequency | str_ = NoInput(0),
        fixing_series: FloatRateSeries | str_ = NoInput(0),
        # settlement and currency
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        # non-deliverable
        pair: FXIndex | str_ = NoInput(0),
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
        if self.schedule.frequency == "Z":
            raise ValueError(
                "`frequency` for a ZeroFloatLeg should not be 'Z'. The Leg is zero frequency by "
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
                index_fixings=index_fixings_[-1],
                index_base_date=self.schedule.aschedule[0],
                index_reference_date=self.schedule.aschedule[-1],
            )
        self._exchange_periods = (_ini_cf, _final_cf)

        self._regular_periods = (
            ZeroFloatPeriod(
                float_spread=float_spread,
                rate_fixings=rate_fixings,
                fixing_method=fixing_method,
                method_param=method_param,
                spread_compound_method=spread_compound_method,
                fixing_frequency=fixing_frequency,
                fixing_series=fixing_series,
                schedule=self.schedule,
                # currency args
                currency=self._currency,
                notional=self._notional,
                # period params
                convention=self._convention,
                # non-deliverable: Not allowed with notional exchange
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
            ),
        )

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
        original_z = self.float_spread

        def s(g: DualTypes) -> DualTypes:
            """
            This determines the NPV of the *Leg* subject to a given float spread change denoted, g.
            """
            self.float_spread = g
            iteration_local_npv = self.local_npv(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                fx=fx,
                forward=forward,
                settlement=settlement,
            )
            return iteration_local_npv

        result = ift_1dim(
            s=s,
            s_tgt=target_npv,
            h="ytm_quadratic",
            ini_h_args=(-300, 300, 1200),
            # h="modified_brent",
            # ini_h_args=(-10000, 10000),
            func_tol=1e-6,
            conv_tol=1e-6,
        )

        self.float_spread = original_z
        _: DualTypes = result["g"]
        return _
