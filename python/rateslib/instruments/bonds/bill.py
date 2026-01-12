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
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.dual import Variable, gradient
from rateslib.dual.utils import _dual_float, _to_number
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.bonds.conventions import (
    BillCalcMode,
    _get_bill_calc_mode,
)
from rateslib.instruments.bonds.fixed_rate_bond import FixedRateBond
from rateslib.instruments.bonds.protocols import _BaseBondInstrument
from rateslib.instruments.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _maybe_get_curve_maybe_from_solver,
    _Vol,
)
from rateslib.legs import FixedLeg
from rateslib.scheduling import Schedule
from rateslib.scheduling.frequency import _get_frequency

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurvesT_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        Number,
        RollDay,
        Sequence,
        Solver_,
        VolT_,
        _BaseLeg,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class Bill(_BaseBondInstrument):
    """
    A *bill*, or discount security, composed of a :class:`~rateslib.legs.FixedLeg`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import Bill
       from datetime import datetime as dt

    .. ipython:: python

       bill = Bill(
           effective=dt(2000, 1, 1),
           termination="3y",
           spec="us_gbb",
       )
       bill.cashflows()

    .. rubric:: Pricing

    A *Bill* requires one *disc curve*. The following input formats are
    allowed:

    .. code-block:: python

       curves = curve | [curve]           #  a single curve is repeated for all required curves
       curves = {"disc_curve": disc_curve}  # dict form is explicit

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .

        .. note::

           The following define generalised **scheduling** parameters.

    effective : datetime, :red:`required`
        The unadjusted effective date. If given as adjusted, unadjusted alternatives may be
        inferred.
    termination : datetime, str, :red:`required`
        The unadjusted termination date. If given as adjusted, unadjusted alternatives may be
        inferred. If given as string tenor will be calculated from ``effective``.
    frequency : Frequency, str, :red:`required`
        The frequency of the schedule.
        If given as string will derive a :class:`~rateslib.scheduling.Frequency` aligning with:
        monthly ("M"), quarterly ("Q"), semi-annually ("S"), annually("A") or zero-coupon ("Z"), or
        a set number of calendar or business days ("_D", "_B"), weeks ("_W"), months ("_M") or
        years ("_Y").
        Where required, the :class:`~rateslib.scheduling.RollDay` is derived as per ``roll``
        and business day calendar as per ``calendar``.
    roll : RollDay, int in [1, 31], str in {"eom", "imm", "som"}, :green:`optional`
        The roll day of the schedule. If not given or not available in ``frequency`` will be
        inferred for monthly frequency variants.
    eom : bool, :green:`optional`
        Use an end of month preference rather than regular rolls for ``roll`` inference. Set by
        default. Not required if ``roll`` is defined.
    modifier : Adjuster, str in {"NONE", "F", "MF", "P", "MP"}, :green:`optional`
        The :class:`~rateslib.scheduling.Adjuster` used for adjusting unadjusted schedule dates
        into adjusted dates. If given as string must define simple date rolling rules.
    calendar : calendar, str, :green:`optional`
        The business day calendar object to use. If string will call
        :meth:`~rateslib.scheduling.get_calendar`.
    payment_lag: Adjuster, int, :green:`optional`
        The :class:`~rateslib.scheduling.Adjuster` to use to map adjusted schedule dates into
        a payment date. If given as integer will define the number of business days to
        lag payments by.
    ex_div: Adjuster, int, :green:`optional`
        The :class:`~rateslib.scheduling.Adjuster` to use to map adjusted schedule dates into
        additional dates, which may be used, for example by fixings schedules. If given as integer
        will define the number of business days to lag dates by.
    convention: str, :green:`optional (set by 'defaults')`
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.scheduling.dcf`.

        .. note::

           The following define generalised **settlement** parameters.

    currency : str, :green:`optional (set by 'defaults')`
        The local settlement currency of the *Instrument* (3-digit code).
    notional : float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The initial leg notional, defined in units of *reference currency*.

        .. note::

           The following are **meta parameters**.

    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    calc_mode : str or BillCalcMode
        A calculation mode for dealing with bonds under different conventions. See notes.
    settle: int
        The number of days by which to lag 'today' to arrive at standard settlement.
    metric : str, :green:`optional` (set as 'price')
        The pricing metric returned by :meth:`~rateslib.instruments.FixedRateBond.rate`.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.

    """

    _rate_scalar = 1.0

    @property
    def leg1(self) -> FixedLeg:
        """The :class:`~rateslib.legs.FixedLeg` of the *Instrument*."""
        return self._leg1

    @property
    def legs(self) -> Sequence[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    def __init__(
        self,
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: str_ = NoInput(0),
        roll: int | RollDay | str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        convention: str_ = NoInput(0),
        ex_div: int_ = NoInput(0),
        settle: int_ = NoInput(0),
        calc_mode: BillCalcMode | str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        spec: str_ = NoInput(0),
        metric: str = "price",
    ):
        user_args = dict(
            effective=effective,
            termination=termination,
            frequency=frequency,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            ex_div=ex_div,
            roll=roll,
            eom=eom,
            notional=notional,
            currency=currency,
            convention=convention,
            settle=settle,
            calc_mode=calc_mode,
            curves=self._parse_curves(curves),
            metric=metric,
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            initial_exchange=False,
            final_exchange=True,
            fixed_rate=0.0,
            vol=_Vol(),
        )

        default_args = dict(
            notional=defaults.notional,
            calc_mode=defaults.calc_mode[type(self).__name__],
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            payment_lag_exchange=defaults.payment_lag_specific[type(self).__name__],
            ex_div=defaults.ex_div,
            settle=defaults.settle,
        )

        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "calc_mode", "settle", "metric", "frequency", "vol"],
        )
        self.kwargs.meta["calc_mode"] = _get_bill_calc_mode(self.kwargs.meta["calc_mode"])
        if isinstance(self.kwargs.leg1["termination"], str):
            s_ = Schedule(
                effective=self.kwargs.leg1["effective"],
                termination=self.kwargs.leg1["termination"],
                frequency=self.kwargs.leg1["termination"],
                modifier=self.kwargs.leg1["modifier"],
                calendar=self.kwargs.leg1["calendar"],
                roll=self.kwargs.leg1["roll"],
                eom=self.kwargs.leg1["eom"],
            )
            self._kwargs.leg1["termination"] = s_.termination
        self._kwargs.leg1["frequency"] = "Z"
        self._kwargs.meta["frequency"] = _drb(
            self.kwargs.meta["calc_mode"]._ytm_clone_kwargs["frequency"],
            self.kwargs.meta["frequency"],
        )

        self._leg1 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self._legs = [self.leg1]

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        A Bill has one curve requirements: a disc_curve.

        When given as only 1 element this curve is applied to all of the those components

        When given as 2 elements the first is treated as the rate curve and the 2nd as disc curve.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
            return _Curves(
                disc_curve=curves.get("disc_curve", NoInput(0)),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 1:
                return _Curves(
                    disc_curve=curves[0],
                )
            elif len(curves) == 2:
                return _Curves(
                    disc_curve=curves[1],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires only 1 curve types. Got {len(curves)}."
                )
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                disc_curve=curves,  # type: ignore[arg-type]
            )

    def rate(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        metric : str in {"price", "discount_rate", "ytm", "simple_rate"}
            Metric returned by the method. Uses the *Instrument* default if not given.

        Returns
        -------
        float, Dual, Dual2
        """
        disc_curve_ = _validate_obj_not_no_input(
            _maybe_get_curve_maybe_from_solver(
                solver=solver,
                name="disc_curve",
                curves=self._parse_curves(curves),
                curves_meta=self.kwargs.meta["curves"],
            ),
            "disc_curve",
        )
        settlement_ = self._maybe_get_settlement(settlement=settlement, disc_curve=disc_curve_)

        # scale price to par 100 and make a fwd adjustment according to curve
        price = (
            self.npv(curves=curves, solver=solver, local=False)  # type: ignore[operator]
            * 100
            / (-self.leg1.settlement_params.notional * disc_curve_[settlement_])
        )
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()
        if metric_ in ["price", "clean_price", "dirty_price"]:
            return price
        elif metric_ == "discount_rate":
            return self.discount_rate(price, settlement_)
        elif metric_ == "simple_rate":
            return self.simple_rate(price, settlement_)
        elif metric_ == "ytm":
            return self.ytm(price, settlement_, NoInput(0))
        raise ValueError("`metric` must be in {'price', 'discount_rate', 'ytm', 'simple_rate'}")

    def simple_rate(self, price: DualTypes, settlement: datetime) -> DualTypes:
        """
        Return the simple rate of the security from its ``price``.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The price of the security.
        settlement : datetime
            The settlement date of the security.

        Returns
        -------
        float, Dual, or Dual2
        """
        acc_frac = self.kwargs.meta["calc_mode"]._settle_accrual(self, settlement, 0)
        dcf = (1 - acc_frac) * self.leg1._regular_periods[0].period_params.dcf
        return ((100 / price - 1) / dcf) * 100  # type: ignore[no-any-return]

    def discount_rate(self, price: DualTypes, settlement: datetime) -> DualTypes:
        """
        Return the discount rate of the security from its ``price``.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The price of the security.
        settlement : datetime
            The settlement date of the security.

        Returns
        -------
        float, Dual, or Dual2
        """
        acc_frac = self.kwargs.meta["calc_mode"]._settle_accrual(self, settlement, 0)
        dcf = (1 - acc_frac) * self.leg1._regular_periods[0].period_params.dcf
        rate = ((1 - price / 100) / dcf) * 100
        return rate  # type: ignore[no-any-return]

    def price(
        self,
        rate: DualTypes,
        settlement: datetime,
        dirty: bool = False,
        calc_mode: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the price of the bill given the ``discount_rate``.

        Parameters
        ----------
        rate : float
            The rate used by the pricing formula.
        settlement : datetime
            The settlement date.
        dirty : bool, not required
            Discount securities have no coupon, the concept of clean or dirty is not
            relevant. Argument is included for signature consistency with
            :meth:`FixedRateBond.price<rateslib.instruments.FixedRateBond.price>`.
        calc_mode : str, optional
            A calculation mode to force, which is used instead of that attributed the
            *Bill* instance.

        Returns
        -------
        float, Dual, Dual2
        """
        calc_mode_ = _get_bill_calc_mode(_drb(self.kwargs.meta["calc_mode"], calc_mode))
        price_func = getattr(self, f"_price_{calc_mode_._price_type}")
        return price_func(rate, settlement)  # type: ignore[no-any-return]

    def _price_discount(self, rate: DualTypes, settlement: datetime) -> DualTypes:
        acc_frac = self.kwargs.meta["calc_mode"]._settle_accrual(self, settlement, 0)
        dcf = (1 - acc_frac) * self.leg1._regular_periods[0].period_params.dcf
        return 100 - rate * dcf  # type: ignore[no-any-return]

    def _price_simple(self, rate: DualTypes, settlement: datetime) -> DualTypes:
        acc_frac = self.kwargs.meta["calc_mode"]._settle_accrual(self, settlement, 0)
        dcf = (1 - acc_frac) * self.leg1._regular_periods[0].period_params.dcf
        return 100 / (1 + rate * dcf / 100)  # type: ignore[no-any-return]

    def ytm(  # type: ignore[override]
        self,
        price: DualTypes,
        settlement: datetime,
        calc_mode: BillCalcMode | str_ = NoInput(0),
    ) -> Number:
        """
        Calculate the yield-to-maturity on an equivalent bond with a coupon of 0%.

        Parameters
        ----------
        price: float, Dual, Dual2
            The price of the *Bill*.
        settlement: datetime
            The settlement date of the *Bill*.
        calc_mode : str, optional
            A calculation mode to force, which is used instead of that attributed the
            *Bill* instance.

        Notes
        -----
        Maps the following *Bill* ``calc_mode`` to the following *Bond* specifications:

        - *NoInput* -> "ust"
        - *"ustb"* -> "ust"
        - *"uktb"* -> "ukt"
        - *"sgbb"* -> "sgb"

        This method calculates by constructing a :class:`~rateslib.instruments.FixedRateBond`
        with a regular 0% coupon measured from the termination date of the bill.
        """
        calc_mode_ = _get_bill_calc_mode(_drb(self.kwargs.meta["calc_mode"], calc_mode))
        freq = calc_mode_._ytm_clone_kwargs["frequency"]

        frequency = _get_frequency(
            freq, self.leg1.schedule.utermination.day, self.leg1.schedule.calendar
        )
        quasi_ustart = frequency.uprevious(self.leg1.schedule.uschedule[-1])
        while quasi_ustart > settlement:
            quasi_ustart = frequency.uprevious(quasi_ustart)

        equiv_bond = FixedRateBond(  # type: ignore[abstract]
            effective=quasi_ustart,
            termination=self.leg1.schedule.utermination,
            fixed_rate=0.0,
            **calc_mode_._ytm_clone_kwargs,  # type: ignore[arg-type]
        )
        return equiv_bond.ytm(price, settlement)

    def duration(self, ytm: DualTypes, settlement: datetime, metric: str = "risk") -> float:
        """
        Return the duration of the *Bill*. See
        :class:`~rateslib.instruments.FixedRateBond.duration` for arguments.

        Notes
        ------

        .. warning::

           This function returns a *duration* that is consistent with a
           *FixedRateBond* yield-to-maturity definition. It currently does not use the
           specified ``convention`` of the *Bill*, and can be sensitive to the
           ``frequency`` of the representative *FixedRateBond* equivalent.

        .. ipython:: python

           bill = Bill(effective=dt(2024, 2, 29), termination=dt(2024, 8, 29), spec="us_gbb")
           bill.duration(settlement=dt(2024, 5, 30), ytm=5.2525, metric="duration")

           bill = Bill(effective=dt(2024, 2, 29), termination=dt(2024, 8, 29), spec="us_gbb", frequency="A")
           bill.duration(settlement=dt(2024, 5, 30), ytm=5.2525, metric="duration")

        """  # noqa: E501
        # TODO: this is not AD safe: returns only float
        ytm_: float = _dual_float(ytm)
        if metric == "duration":
            price_ = _to_number(self.price(Variable(ytm_, ["y"]), settlement, dirty=True))
            freq = _get_frequency(
                self.kwargs.meta["frequency"],
                self.leg1.schedule.utermination.day,
                self.leg1.schedule.calendar,
            )
            f = freq.periods_per_annum()
            v = 1 + ytm_ / (100 * f)
            _: float = -gradient(price_, ["y"])[0] / _dual_float(price_) * v * 100
            return _
        else:
            return super().duration(ytm, settlement, metric)
