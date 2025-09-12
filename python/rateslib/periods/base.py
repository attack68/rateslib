from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.curves._parsers import _disc_maybe_from_curve, _disc_required_maybe_from_curve
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.periods.utils import _get_fx_and_base
from rateslib.scheduling import Adjuster, Frequency, dcf, get_calendar
from rateslib.scheduling.adjuster import _get_adjuster
from rateslib.scheduling.convention import _get_convention
from rateslib.scheduling.frequency import _get_frequency

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        Any,
        CalInput,
        Convention,
        CurveOption_,
        DualTypes,
        RollDay,
        _BaseCurve,
        _BaseCurve_,
        datetime,
        datetime_,
        float_,
        str_,
    )


class BasePeriod(metaclass=ABCMeta):
    """
    Abstract base class with common parameters for all ``Period`` subclasses.

    See also: :ref:`User guide for Periods <periods-doc>`.

    Parameters
    ----------
    start : Datetime
        The adjusted start date of the calculation period.
    end: Datetime
        The adjusted end date of the calculation period.
    payment : Datetime
        The adjusted payment date of the period.
    frequency : Frequency, str
        The frequency of the corresponding leg. If it does not contain a ``roll`` or
        ``calendar`` will be inferred from the additional arguments as necessary.
    notional : float, optional, set by Default
        The notional amount of the period (positive implies paying a cashflow).
    currency : str, optional
        The currency of the cashflow (3-digit code), set by default.
    convention : str, optional, set by Default
        The day count convention of the calculation period accrual.
        See :meth:`~rateslib.scheduling.dcf`.
    termination : Datetime, optional
        The termination date of the corresponding leg. Required only with
        specific values for ``convention``.
    stub : bool, optional
        Records whether the period is a stub or regular. Used by certain day count
        convention calculations.
    roll : int, str, optional
        Used to yield a :class:`~rateslib.scheduling.Frequency` if ``frequency`` does
        not already specify a valid and necessary value.
    calendar : Calendar, str, optional
        Used only by ``stub`` periods and for specific values of ``convention``.
    adjuster : Adjuster, str, optional
        The adjuster used to derive adjusted period accrual dates.

    """

    @abstractmethod
    def __init__(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        frequency: Frequency | str,
        notional: float_ = NoInput(0),
        currency: str_ = NoInput(0),
        convention: str_ = NoInput(0),
        termination: datetime_ = NoInput(0),
        stub: bool = False,
        roll: RollDay | int | str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        adjuster: Adjuster | str_ = NoInput(0),
    ):
        if end < start:
            raise ValueError("`end` cannot be before `start`.")
        self.start: datetime = start
        self.end: datetime = end
        self.payment: datetime = payment
        self.frequency: Frequency = _get_frequency(frequency, roll, calendar)
        self.calendar: CalInput = get_calendar(calendar)
        self.adjuster: Adjuster | NoInput = (
            adjuster if isinstance(adjuster, NoInput) else _get_adjuster(adjuster)
        )
        self.stub: bool = stub
        self.notional: float = _drb(defaults.notional, notional)
        self.currency: str = _drb(defaults.base_currency, currency).lower()
        self.convention: Convention = _get_convention(_drb(defaults.convention, convention))
        self.termination = termination
        self.freq_months: int = int(12.0 / self.frequency.periods_per_annum())

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def __str__(self) -> str:
        return (
            f"<{type(self).__name__}: {self.start.strftime('%Y-%m-%d')}->"
            f"{self.end.strftime('%Y-%m-%d')},{self.notional},{self.convention}>"
        )

    @cached_property
    def dcf(self) -> float:
        """
        float : Calculated with appropriate ``convention`` over the period.
        """
        return dcf(
            start=self.start,
            end=self.end,
            convention=self.convention,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            roll=_get_roll_from_frequency(self.frequency),
            calendar=self.calendar,
            adjuster=self.adjuster,
        )

    @abstractmethod
    def analytic_delta(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the period object.

        Parameters
        ----------
        curve : Curve
            The forecasting curve object. Not used unless it is set equal to
            ``disc_curve``, or if a rate in a :class:`FloatPeriod` is required.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards`
            object, converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.

        Returns
        -------
        float, Dual, Dual2

        Examples
        --------
        .. ipython:: python

           curve = Curve({dt(2021,1,1): 1.00, dt(2025,1,1): 0.83}, interpolation="log_linear", id="SONIA")
           fxr = FXRates({"gbpusd": 1.25}, base="usd")

        .. ipython:: python

           period = FixedPeriod(
               start=dt(2022, 1, 1),
               end=dt(2022, 7, 1),
               payment=dt(2022, 7, 1),
               frequency="S",
               currency="gbp",
               fixed_rate=4.00,
           )
           period.analytic_delta(curve, curve)
           period.analytic_delta(curve, curve, fxr)
           period.analytic_delta(curve, curve, fxr, "gbp")
        """  # noqa: E501
        disc_curve_: _BaseCurve = _disc_required_maybe_from_curve(curve, disc_curve)
        fx_, _ = _get_fx_and_base(self.currency, fx, base)
        ret: DualTypes = fx_ * self.notional * self.dcf * disc_curve_[self.payment] / 10000
        return ret

    @abstractmethod
    def cashflows(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the properties of the period used in calculating cashflows.

        Parameters
        ----------
        curve : Curve, optional
            The forecasting curve object. Not used unless it is set equal to
            ``disc_curve``, or if a rate in a :class:`FloatPeriod` is required.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards`
            object, converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code).
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object. If not given defaults to
            ``fx.base``.

        Returns
        -------
        dict

        Examples
        --------
        .. ipython:: python

           period.cashflows(curve, curve, fxr)
        """
        disc_curve_: _BaseCurve_ = _disc_maybe_from_curve(curve, disc_curve)
        if isinstance(disc_curve_, NoInput):
            df: float | None = None
            collateral: str | None = None
        else:
            df = _dual_float(disc_curve_[self.payment])
            collateral = disc_curve_.meta.collateral

        return {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["stub_type"]: "Stub" if self.stub else "Regular",
            defaults.headers["currency"]: self.currency.upper(),
            defaults.headers["a_acc_start"]: self.start,
            defaults.headers["a_acc_end"]: self.end,
            defaults.headers["payment"]: self.payment,
            defaults.headers["convention"]: str(self.convention),
            defaults.headers["dcf"]: self.dcf,
            defaults.headers["notional"]: _dual_float(self.notional),
            defaults.headers["df"]: df,
            defaults.headers["collateral"]: collateral,
        }

    @abstractmethod
    def cashflow(
        self,
        curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
    ) -> DualTypes | None:
        """
        Return the forecast cashflow of the period.

        Parameters
        ----------
        curve : Curve, optional
            The forecasting curve object. Not used unless it is set equal to
            ``disc_curve``, or if a rate in a :class:`FloatPeriod` is required.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards`
            object, converts from local currency into ``base``.

        Returns
        -------
        float, Dual, Dual2, or None
        """
        pass

    @abstractmethod
    def npv(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the period object.

        Calculates the cashflow for the period and multiplies it by the DF associated
        with the payment date.

        Parameters
        ----------
        curve : Curve, optional
            The forecasting curve object. Not used unless it is set equal to
            ``disc_curve``, or if a rate in a :class:`FloatPeriod` is required.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards`
            object, converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.
        local : bool, optional
            If `True` will ignore the ``base`` request and return a dict identifying
            local currency NPV.

        Returns
        -------
        float, Dual, Dual2, or dict of such

        Examples
        --------
        .. ipython:: python

           period.npv(curve, curve)
           period.npv(curve, curve, fxr)
           period.npv(curve, curve, fxr, "gbp")
           period.npv(curve, curve, fxr, local=True)
        """
        pass  # pragma: no cover


def _get_roll_from_frequency(frequency: Frequency) -> RollDay | NoInput:
    if isinstance(frequency, Frequency.Months):
        return NoInput(0) if frequency.roll is None else frequency.roll
    else:
        return NoInput(0)
