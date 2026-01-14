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

from rateslib import defaults
from rateslib.curves import index_left
from rateslib.enums.generics import NoInput, _drb
from rateslib.legs.amortization import Amortization, _get_amortization
from rateslib.legs.protocols import _BaseLeg, _WithExDiv
from rateslib.periods import CreditPremiumPeriod, CreditProtectionPeriod

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        FX_,
        Any,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        Schedule,
        _BaseCurve_,
        _FXVolOption_,
        _SettlementParams,
        bool_,
        datetime,
        datetime_,
        str_,
    )


class CreditPremiumLeg(_BaseLeg, _WithExDiv):
    """
    Define a *Leg* containing :class:`~rateslib.periods.CreditPremiumPeriod`.

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

    convention: str, optional
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

        .. note::

           The following define **rate parameters**.

    fixed_rate: float, Dual, Dual2, Variable, :green:`optional`
        The fixed rate of each composited :class:`~rateslib.periods.CreditPremiumPeriod`.

        .. note::

           The following parameters define **credit specific** elements.

    premium_accrued: bool, :green:`optional (set by 'defaults')`
        Whether an accrued premium is paid on the event of mid-period credit default.


    Notes
    -----
    TODO

    Examples
    --------
    See :ref:`Leg Examples<legs-doc>`

    """

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.parameters._SettlementParams` associated with
        the first :class:`~rateslib.periods.FloatPeriod`."""
        return self._regular_periods[0].settlement_params

    @property
    def periods(self) -> list[CreditPremiumPeriod]:
        """Combine all period collection types into an ordered list."""
        return list(self._regular_periods)

    @property
    def fixed_rate(self) -> DualTypes_:
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

    def accrued(self, settlement: datetime) -> DualTypes:
        """
        Calculate the amount of premium accrued until a specific date within the relevant *Period*.

        Parameters
        ----------
        settlement: datetime
            The date against which accrued is measured.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        _ = index_left(
            self.schedule.uschedule,
            len(self.schedule.uschedule),
            settlement,
        )
        # This index is valid because this Leg only contains CreditPremiumPeriods and no exchanges.
        return self.periods[_].accrued(settlement)

    def __init__(
        self,
        schedule: Schedule,
        *,
        fixed_rate: NoInput = NoInput(0),
        premium_accrued: bool_ = NoInput(0),
        # settlement and currency
        notional: DualTypes_ = NoInput(0),
        amortization: DualTypes_ | list[DualTypes] | Amortization | str = NoInput(0),
        currency: str_ = NoInput(0),
        # period
        convention: str_ = NoInput(0),
    ) -> None:
        self._fixed_rate = fixed_rate
        self._schedule = schedule
        self._notional: DualTypes = _drb(defaults.notional, notional)
        self._amortization: Amortization = _get_amortization(
            amortization, self._notional, self.schedule.n_periods
        )
        self._currency: str = _drb(defaults.base_currency, currency).lower()
        self._convention: str = _drb(defaults.convention, convention)

        self._regular_periods = tuple(
            [
                CreditPremiumPeriod(
                    fixed_rate=fixed_rate,
                    premium_accrued=premium_accrued,
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
                )
                for i in range(self.schedule.n_periods)
            ]
        )

        # # No amortization exchanges
        # self._interim_exchange_periods = None
        # self._exchange_periods = (None, None)
        # self._mtm_exchange_periods = None

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
        a_delta = self.local_analytic_delta(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            forward=forward,
            settlement=settlement,
        )
        return -target_npv / a_delta


class CreditProtectionLeg(_BaseLeg):
    """
    Create a credit protection leg composed of :class:`~rateslib.periods.CreditProtectionPeriod` s.

    Parameters
    ----------
    args : tuple
        Required positional args to :class:`BaseLeg`.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    The NPV of a credit protection leg is the sum of the period NPVs.

    .. math::

       P = \\sum_{i=1}^n P_i

    The analytic delta is the sum of the period analytic deltas.

    .. math::

       A = -\\frac{\\partial P}{\\partial S} = \\sum_{i=1}^n -\\frac{\\partial P_i}{\\partial S}

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib.curves import Curve
       from rateslib.scheduling import Schedule
       from rateslib.legs import CreditProtectionLeg
       from datetime import datetime as dt

    .. ipython:: python

       disc_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
       hazard_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995})
       protection_leg = CreditProtectionLeg(
           schedule=Schedule(dt(2022, 1, 1), "9M", "Z"),
           notional=1000000,
       )
       protection_leg.cashflows(rate_curve=hazard_curve, disc_curve=disc_curve)
       protection_leg.npv(rate_curve=hazard_curve, disc_curve=disc_curve)
    """  # noqa: E501

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.parameters._SettlementParams` associated with
        the first :class:`~rateslib.periods.FloatPeriod`."""
        return self._regular_periods[0].settlement_params

    @property
    def periods(self) -> list[CreditProtectionPeriod]:
        """Combine all period collection types into an ordered list."""
        return list(self._regular_periods)

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
        # period
        convention: str_ = NoInput(0),
    ) -> None:
        self._schedule = schedule
        self._notional: DualTypes = _drb(defaults.notional, notional)
        self._amortization: Amortization = _get_amortization(
            amortization, self._notional, self.schedule.n_periods
        )
        self._currency: str = _drb(defaults.base_currency, currency).lower()
        self._convention: str = _drb(defaults.convention, convention)

        self._regular_periods = tuple(
            [
                CreditProtectionPeriod(
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
                )
                for i in range(self.schedule.n_periods)
            ]
        )

        # # No amortization exchanges
        # self._interim_exchange_periods = None
        # self._exchange_periods = (None, None)
        # self._mtm_exchange_periods = None

    def analytic_rec_risk(
        self,
        rate_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> float:
        """
        Return the analytic recovery risk of the *CreditProtectionLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        _ = (
            period.analytic_rec_risk(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                fx=fx,
                base=base,
            )
            for period in self.periods
        )
        ret: float = sum(_)
        return ret

    def spread(self, *args: Any, **kwargs: Any) -> DualTypes:
        raise NotImplementedError(f"{type(self).__name__} does not implement `spread`.")
