from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.calendars import dcf
from rateslib.curves._parsers import _disc_maybe_from_curve, _disc_required_maybe_from_curve
from rateslib.default import NoInput, _drb
from rateslib.dual.utils import _dual_float
from rateslib.periods.utils import _get_fx_and_base

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        Any,
        CalInput,
        Curve,
        Curve_,
        CurveOption_,
        DualTypes,
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
    frequency : str
        The frequency of the corresponding leg. Also used
        with specific values for ``convention``, or floating rate calculation.
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
        Used only by ``stub`` periods and for specific values of ``convention``.
    calendar : CustomBusinessDay, str, optional
        Used only by ``stub`` periods and for specific values of ``convention``.

    """

    @abstractmethod
    def __init__(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        frequency: str,
        notional: float_ = NoInput(0),
        currency: str_ = NoInput(0),
        convention: str_ = NoInput(0),
        termination: datetime_ = NoInput(0),
        stub: bool = False,
        roll: int | str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
    ):
        if end < start:
            raise ValueError("`end` cannot be before `start`.")
        self.start: datetime = start
        self.end: datetime = end
        self.payment: datetime = payment
        self.frequency: str = frequency.upper()
        self.notional: float = _drb(defaults.notional, notional)
        self.currency: str = _drb(defaults.base_currency, currency).lower()
        self.convention: str = _drb(defaults.convention, convention)
        self.termination = termination
        self.freq_months: int = defaults.frequency_months[self.frequency]
        self.stub: bool = stub
        self.roll: int | str | NoInput = roll
        self.calendar: CalInput = calendar

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def __str__(self) -> str:
        return (
            f"<{type(self).__name__}: {self.start.strftime('%Y-%m-%d')}->"
            f"{self.end.strftime('%Y-%m-%d')},{self.notional},{self.convention}>"
        )

    @property
    def dcf(self) -> float:
        """
        float : Calculated with appropriate ``convention`` over the period.
        """
        return dcf(
            self.start,
            self.end,
            self.convention,
            self.termination,
            self.freq_months,
            self.stub,
            self.roll,
            self.calendar,
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
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        fx_, _ = _get_fx_and_base(self.currency, fx, base)
        ret: DualTypes = fx_ * self.notional * self.dcf * disc_curve_[self.payment] / 10000
        return ret

    @abstractmethod
    def cashflows(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
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
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
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
            defaults.headers["convention"]: self.convention,
            defaults.headers["dcf"]: self.dcf,
            defaults.headers["notional"]: _dual_float(self.notional),
            defaults.headers["df"]: df,
            defaults.headers["collateral"]: collateral,
        }

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
