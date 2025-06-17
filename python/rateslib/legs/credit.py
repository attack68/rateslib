from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.curves import index_left
from rateslib.default import NoInput, _drb
from rateslib.legs.base import BaseLeg, _FixedLegMixin
from rateslib.periods.credit import CreditPremiumPeriod, CreditProtectionPeriod

if TYPE_CHECKING:
    from pandas import DataFrame

    from rateslib.typing import Any, DualTypes, DualTypes_, Schedule, datetime


class CreditPremiumLeg(_FixedLegMixin, BaseLeg):
    """
    Create a credit premium leg composed of :class:`~rateslib.periods.CreditPremiumPeriod` s.

    Parameters
    ----------
    args : tuple
        Required positional args to :class:`BaseLeg`.
    fixed_rate : float, optional
        The credit spread applied to determine cashflows in percentage points (i.e 50bps = 0.50).
        Can be left unset and
        designated later, perhaps after a mid-market rate for all periods has been calculated.
    premium_accrued : bool, optional
        Whether the premium is accrued within the period to default.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    The NPV of a credit premium leg is the sum of the period NPVs.

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
       from rateslib.legs import CreditPremiumLeg
       from datetime import datetime as dt

    .. ipython:: python

       disc_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
       hazard_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995})
       premium_leg = CreditPremiumLeg(
           dt(2022, 1, 1), "9M", "Q",
           fixed_rate=2.60,
           notional=1000000,
       )
       premium_leg.cashflows(hazard_curve, disc_curve)
       premium_leg.npv(hazard_curve, disc_curve)
    """  # noqa: E501

    periods: list[CreditPremiumPeriod]  # type: ignore[assignment]
    schedule: Schedule

    _regular_periods: tuple[CreditPremiumPeriod, ...]

    def __init__(
        self,
        *args: Any,
        fixed_rate: DualTypes_ = NoInput(0),
        premium_accrued: bool | NoInput = NoInput(0),
        **kwargs: Any,
    ):
        self._fixed_rate = fixed_rate
        self.premium_accrued = _drb(defaults.cds_premium_accrued, premium_accrued)
        super().__init__(*args, **kwargs)
        if self.initial_exchange or self.final_exchange:
            raise ValueError(
                "`initial_exchange` and `final_exchange` cannot be True on CreditPremiumLeg."
            )
        self._set_periods()

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *CreditPremiumLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *CreditPremiumLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *CreditPremiumLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)

    def accrued(self, settlement: datetime) -> DualTypes | None:
        """
        Calculate the amount of premium accrued until a specific date within the relevant *Period*.

        Parameters
        ----------
        settlement: datetime
            The date against which accrued is measured.

        Returns
        -------
        float
        """
        _ = index_left(
            self.schedule.uschedule,
            len(self.schedule.uschedule),
            settlement,
        )
        # This index is valid because this Leg only contains CreditPremiumPeriods and no exchanges.
        return self.periods[_].accrued(settlement)

    def _set_periods(self) -> None:
        return super()._set_periods()

    def _regular_period(  # type: ignore[override]
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        notional: DualTypes,
        stub: bool,
        iterator: int,
    ) -> CreditPremiumPeriod:
        return CreditPremiumPeriod(
            fixed_rate=self.fixed_rate,
            premium_accrued=self.premium_accrued,
            start=start,
            end=end,
            payment=payment,
            frequency=self.schedule.frequency,
            notional=notional,
            currency=self.currency,
            convention=self.convention,
            termination=self.schedule.termination,
            stub=stub,
            roll=self.schedule.roll,
            calendar=self.schedule.calendar,
        )


class CreditProtectionLeg(BaseLeg):
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
       from rateslib.legs import CreditProtectionLeg
       from datetime import datetime as dt

    .. ipython:: python

       disc_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
       hazard_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995})
       protection_leg = CreditProtectionLeg(
           dt(2022, 1, 1), "9M", "Z",
           notional=1000000,
       )
       protection_leg.cashflows(hazard_curve, disc_curve)
       protection_leg.npv(hazard_curve, disc_curve)
    """  # noqa: E501

    periods: list[CreditProtectionPeriod]  # type: ignore[assignment]

    def __init__(
        self,
        *args: Any,
        recovery_rate: DualTypes | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._recovery_rate: DualTypes = _drb(defaults.cds_recovery_rate, recovery_rate)
        super().__init__(*args, **kwargs)
        if self.initial_exchange or self.final_exchange:
            raise ValueError(
                "`initial_exchange` and `final_exchange` cannot be True on CreditProtectionLeg."
            )
        self._set_periods()

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *CreditProtectionLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def analytic_rec_risk(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic recovery risk of the *CreditProtectionLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        _ = (period.analytic_rec_risk(*args, **kwargs) for period in self.periods)
        ret: DualTypes = sum(_)
        return ret

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *CreditProtectionLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *CreditProtectionLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)

    def _set_periods(self) -> None:
        return super()._set_periods()

    def _regular_period(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        stub: bool,
        notional: DualTypes,
        iterator: int,
    ) -> CreditProtectionPeriod:
        return CreditProtectionPeriod(
            start=start,
            end=end,
            payment=payment,
            frequency=self.schedule.frequency,
            notional=notional,
            currency=self.currency,
            convention=self.convention,
            termination=self.schedule.termination,
            stub=stub,
            roll=self.schedule.roll,
            calendar=self.schedule.calendar,
        )
