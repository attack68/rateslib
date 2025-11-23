from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.dual import Variable, gradient
from rateslib.dual.utils import _dual_float, _to_number
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.components.bonds.conventions import (
    BillCalcMode,
    _get_bill_calc_mode,
)
from rateslib.instruments.components.bonds.fixed_rate_bond import FixedRateBond
from rateslib.instruments.components.bonds.protocols import _BaseBondInstrument
from rateslib.instruments.components.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.components.protocols.pricing import (
    _Curves,
    _get_maybe_curve_maybe_from_solver,
)
from rateslib.legs.components import FixedLeg
from rateslib.scheduling.frequency import _get_frequency

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurveOption_,
        Curves_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXVolOption_,
        Number,
        Solver_,
        _BaseLeg,
        datetime,
        datetime_,
        int_,
        str_,
    )


class Bill(_BaseBondInstrument):
    """
    Create a discount security.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime or str
        The adjusted or unadjusted termination date. If a string, then a tenor must be
        given expressed in days (`"D"`), months (`"M"`) or years (`"Y"`), e.g. `"7M"`.
    frequency : str in {"M", "B", "Q", "T", "S", "A"}, optional
        The frequency used only by the :meth:`~rateslib.instruments.Bill.ytm` method.
        All *Bills* have an implicit frequency of "Z" for schedule construction.
    modifier : str, optional
        The modification rule, in {"F", "MF", "P", "MP"}
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, looks up named calendar from
        static data.
    payment_lag : int, optional
        The number of business days to lag payments by.
    notional : float, optional
        The leg notional, which is applied to each period.
    currency : str, optional
        The currency of the leg (3-digit code).
    convention: str, optional
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.scheduling.dcf`.
    settle : int
        The number of business days for regular settlement time, i.e, 1 is T+1.
    calc_mode : str, optional (defaults.calc_mode["Bill"])
        A calculation mode for dealing with bonds that are in short stub or accrual
        periods. All modes give the same value for YTM at issue date for regular
        bonds but differ slightly for bonds with stubs or with accrued.
    curves : CurveType, str or list of such, optional
        A single *Curve* or string id or a list of such.

        A list defines the following curves in the order:

        - Forecasting *Curve* for ``leg1``.
        - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.
    metric: str, optional
        The pricing metric returned by the ``rate`` method of the *Instrument*.

    Examples
    --------
    This example is taken from the US Treasury Federal website. A copy of
    which is available :download:`here<_static/ofcalc6decTbill.pdf>`.

    We demonstrate the use of **analogue methods** which do not need *Curves* or
    *Solvers*,
    :meth:`~rateslib.instruments.Bill.price`,
    :meth:`~rateslib.instruments.Bill.simple_rate`,
    :meth:`~rateslib.instruments.Bill.discount_rate`,
    :meth:`~rateslib.instruments.FixedRateBond.ytm`,
    :meth:`~rateslib.instruments.FixedRateBond.ex_div`,
    :meth:`~rateslib.instruments.FixedRateBond.accrued`,
    :meth:`~rateslib.instruments.FixedRateBond.repo_from_fwd`
    :meth:`~rateslib.instruments.FixedRateBond.fwd_from_repo`
    :meth:`~rateslib.instruments.FixedRateBond.duration`,
    :meth:`~rateslib.instruments.FixedRateBond.convexity`.

    .. ipython:: python
       :suppress:

       from rateslib import Bill, Solver

    .. ipython:: python

       bill = Bill(
           effective=dt(2004, 1, 22),
           termination=dt(2004, 2, 19),
           calendar="nyc",
           modifier="NONE",
           currency="usd",
           convention="Act360",
           settle=1,
           notional=-1e6,  # negative notional receives fixed, i.e. buys a bill
           curves="bill_curve",
           calc_mode="us_gbb",
       )
       bill.ex_div(dt(2004, 1, 22))
       bill.price(rate=0.80, settlement=dt(2004, 1, 22))
       bill.simple_rate(price=99.937778, settlement=dt(2004, 1, 22))
       bill.discount_rate(price=99.937778, settlement=dt(2004, 1, 22))
       bill.ytm(price=99.937778, settlement=dt(2004, 1, 22))
       bill.accrued(dt(2004, 1, 22))
       bill.fwd_from_repo(
           price=99.937778,
           settlement=dt(2004, 1, 22),
           forward_settlement=dt(2004, 2, 19),
           repo_rate=0.8005,
           convention="Act360",
       )
       bill.repo_from_fwd(
           price=99.937778,
           settlement=dt(2004, 1, 22),
           forward_settlement=dt(2004, 2, 19),
           forward_price=100.00,
           convention="Act360",
       )
       bill.duration(settlement=dt(2004, 1, 22), ytm=0.8005, metric="risk")
       bill.duration(settlement=dt(2004, 1, 22), ytm=0.8005, metric="modified")
       bill.convexity(settlement=dt(2004, 1, 22), ytm=0.8005)


    The following **digital methods** consistent with the library's ecosystem are
    also available,
    :meth:`~rateslib.instruments.Bill.rate`,
    :meth:`~rateslib.instruments.FixedRateBond.npv`,
    :meth:`~rateslib.instruments.FixedRateBond.analytic_delta`,
    :meth:`~rateslib.instruments.FixedRateBond.cashflows`,
    :meth:`~rateslib.instruments.FixedRateBond.delta`,
    :meth:`~rateslib.instruments.FixedRateBond.gamma`,

    .. ipython:: python

       bill_curve = Curve({dt(2004, 1, 21): 1.0, dt(2004, 3, 21): 1.0}, id="bill_curve")
       instruments = [
           (bill, (), {"metric": "ytm"}),
       ]
       solver = Solver(
           curves=[bill_curve],
           instruments=instruments,
           s=[0.8005],
           instrument_labels=["Feb04 Tbill"],
           id="bill_solver",
       )
       bill.npv(solver=solver)
       bill.analytic_delta(disc_curve=bill_curve)
       bill.rate(solver=solver, metric="price")

    The sensitivities are also available. In this case the *Solver* is calibrated
    with *instruments* priced in yield terms so sensitivities are measured in basis
    points (bps).

    .. ipython:: python

       bill.delta(solver=solver)
       bill.gamma(solver=solver)

    The DataFrame of cashflows.

    .. ipython:: python

       bill.cashflows(solver=solver)

    """

    _rate_scalar = 1.0

    @property
    def leg1(self) -> FixedLeg:
        """The :class:`~rateslib.legs.components.FixedLeg` of the *Instrument*."""
        return self._leg1

    @property
    def legs(self) -> list[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    def __init__(
        self,
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: str_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        convention: str_ = NoInput(0),
        settle: int_ = NoInput(0),
        calc_mode: BillCalcMode | str_ = NoInput(0),
        curves: Curves_ = NoInput(0),
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
            meta_args=["curves", "calc_mode", "settle", "metric", "frequency"],
        )
        self.kwargs.meta["calc_mode"] = _get_bill_calc_mode(self.kwargs.meta["calc_mode"])
        self._kwargs.leg1["frequency"] = "Z"
        self._kwargs.meta["frequency"] = _drb(
            self.kwargs.meta["calc_mode"]._ytm_clone_kwargs["frequency"],
            self.kwargs.meta["frequency"],
        )

        self._leg1 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self._legs = [self.leg1]

    def _parse_curves(self, curves: CurveOption_) -> _Curves:
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
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                disc_curve=curves,
            )

    def rate(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes_:
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
        disc_curve_ = _get_maybe_curve_maybe_from_solver(
            solver=solver,
            name="disc_curve",
            curves=self._parse_curves(curves),
            curves_meta=self.kwargs.meta["curves"],
        )
        settlement_ = self._maybe_get_settlement(settlement=settlement, disc_curve=disc_curve_)

        # scale price to par 100 and make a fwd adjustment according to curve
        price = (
            self.npv(curves=curves, solver=solver, fx=fx, base=base, local=False)
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
        return ((100 / price - 1) / dcf) * 100

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
        return rate

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
        return 100 - rate * dcf

    def _price_simple(self, rate: DualTypes, settlement: datetime) -> DualTypes:
        acc_frac = self.kwargs.meta["calc_mode"]._settle_accrual(self, settlement, 0)
        dcf = (1 - acc_frac) * self.leg1._regular_periods[0].period_params.dcf
        return 100 / (1 + rate * dcf / 100)

    def ytm(
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

        equiv_bond = FixedRateBond(
            effective=quasi_ustart,
            termination=self.leg1.schedule.utermination,
            fixed_rate=0.0,
            **calc_mode_._ytm_clone_kwargs,
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
            price_: Dual | Dual2 = _to_number(  # type: ignore[assignment]
                self.price(Variable(ytm_, ["y"]), settlement, dirty=True)
            )
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
