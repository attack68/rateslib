from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import FloatFixingMethod
from rateslib.instruments.components.bonds.conventions import (
    BondCalcMode,
    _get_bond_calc_mode,
)
from rateslib.instruments.components.bonds.protocols import _BaseBondInstrument
from rateslib.instruments.components.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.components.protocols.pricing import (
    _Curves,
    _maybe_get_curve_or_dict_maybe_from_solver,
    _Vol,
)
from rateslib.legs.components import FloatLeg
from rateslib.periods.components import FloatPeriod
from rateslib.scheduling import Frequency

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurveOption_,
        Curves_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        Solver_,
        VolT_,
        _BaseLeg,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FloatRateNote(_BaseBondInstrument):
    """
    Create a floating rate note (FRN) security.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime or str
        The adjusted or unadjusted termination date. If a string, then a tenor must be
        given expressed in days (`"D"`), months (`"M"`) or years (`"Y"`), e.g. `"48M"`.
    frequency : str in {"M", "B", "Q", "T", "S", "A"}, optional
        The frequency of the schedule. "Z" is not permitted.
    stub : str combining {"SHORT", "LONG"} with {"FRONT", "BACK"}, optional
        The stub type to enact on the swap. Can provide two types, for
        example "SHORTFRONTLONGBACK".
    front_stub : datetime, optional
        An adjusted or unadjusted date for the first stub period.
    back_stub : datetime, optional
        An adjusted or unadjusted date for the back stub period.
        See notes for combining ``stub``, ``front_stub`` and ``back_stub``
        and any automatic stub inference.
    roll : int in [1, 31] or str in {"eom", "imm", "som"}, optional
        The roll day of the schedule. Inferred if not given.
    eom : bool, optional
        Use an end of month preference rather than regular rolls for inference. Set by
        default. Not required if ``roll`` is specified.
    modifier : str, optional
        The modification rule, in {"F", "MF", "P", "MP"}
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, looks up named calendar from
        static data.
    payment_lag : int, optional
        The number of business days to lag regular coupon payments by.
    payment_lag_exchange : int, optional
        The number of business days to lag notional exchange payments by.
    notional : float, optional
        The leg notional, which is applied to each period.
    currency : str, optional
        The currency of the leg (3-digit code).
    amortization: float, optional
        The amount by which to adjust the notional each successive period. Should have
        sign equal to that of notional if the notional is to reduce towards zero.
    convention: str, optional
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.scheduling.dcf`.
    float_spread : float, optional
        The spread applied to determine cashflows. Can be set to `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given as the first *m* RFR fixings
        within individual curve and composed into the overall rate.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    ex_div : int
        The number of business days prior to a cashflow which determines the last settlement date
        for which a coupon payment is still receivable. See :meth:`BondMixin.ex_div`.
    settle : int
        The number of business days for regular settlement time, i.e, 1 is T+1.
    calc_mode : str
        A calculation mode for dealing with bonds under different conventions. See notes.
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

    Notes
    -----
    .. warning::

       FRNs based on RFR rates which have ex-div days must ensure that fixings are
       available to define the entire period. This means that `ex_div` days must be less
       than the `fixing_method` `method_param` lag minus the time to settlement time.

        That is, a bond with a `method_param` of 5 and a settlement time of 2 days
        can have an `ex_div` period of at maximum 3.

        A bond with a `method_param` of 2 and a settlement time of 1 day cnan have an
        `ex_div` period of at maximum 1.

    Attributes
    ----------
    ex_div_days : int
    leg1 : FloatLeg
    """

    _rate_scalar = 1.0

    @property
    def float_spread(self) -> DualTypes:
        """The float spread parameter of the composited
        :class:`~rateslib.legs.components.FloatLeg`."""
        return self.leg1.float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        self.kwargs.leg1["float_spread"] = value
        self.leg1.float_spread = value

    @property
    def leg1(self) -> FloatLeg:
        """The :class:`~rateslib.legs.components.FloatLeg` of the *Instrument*."""
        return self._leg1

    @property
    def legs(self) -> list[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    def __init__(
        self,
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: int_ = NoInput(0),
        *,
        stub: str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: str | int_ = NoInput(0),
        eom: bool | NoInput = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        payment_lag_exchange: int_ = NoInput(0),
        ex_div: int_ = NoInput(0),
        convention: str_ = NoInput(0),
        # settlement params
        currency: str_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        amortization: DualTypes_ = NoInput(0),
        # rate params
        float_spread: DualTypes_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        rate_fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        fixing_frequency: Frequency | str_ = NoInput(0),
        fixing_series: FloatRateSeries | str_ = NoInput(0),
        # meta parameters
        curves: Curves_ = NoInput(0),
        calc_mode: BondCalcMode | str_ = NoInput(0),
        settle: int_ = NoInput(0),
        spec: str_ = NoInput(0),
        metric: str = "clean_price",
    ) -> None:
        user_args = dict(
            # scheduling
            effective=effective,
            termination=termination,
            frequency=frequency,
            stub=stub,
            front_stub=front_stub,
            back_stub=back_stub,
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            payment_lag_exchange=payment_lag_exchange,
            ex_div=ex_div,
            convention=convention,
            # settlement
            currency=currency,
            notional=notional,
            amortization=amortization,
            # rate
            float_spread=float_spread,
            spread_compound_method=spread_compound_method,
            rate_fixings=rate_fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            fixing_frequency=fixing_frequency,
            fixing_series=fixing_series,
            # meta
            curves=self._parse_curves(curves),
            calc_mode=calc_mode,
            settle=settle,
            metric=metric,
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            initial_exchange=False,
            final_exchange=True,
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
            meta_args=["curves", "calc_mode", "settle", "metric", "vol"],
        )
        self.kwargs.meta["calc_mode"] = _get_bond_calc_mode(self.kwargs.meta["calc_mode"])

        self._leg1 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        if self._leg1.schedule.frequency_obj == Frequency.Zero():
            raise ValueError("A `FloatRateNote` cannot have a 'zero' frequency.")
        self._legs = [self.leg1]

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    def _parse_curves(self, curves: CurveOption_) -> _Curves:
        """
        An FRN has two curve requirements: a leg2_rate_curve and a disc_curve used by both legs.

        When given as only 1 element this curve is applied to all of the those components

        When given as 2 elements the first is treated as the rate curve and the 2nd as disc curve.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
            return _Curves(
                rate_curve=curves.get("rate_curve", NoInput(0)),
                disc_curve=curves.get("disc_curve", NoInput(0)),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 2:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                )
            elif len(curves) == 1:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[0],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires only 2 curve types. Got {len(curves)}."
                )
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                rate_curve=curves,
                disc_curve=curves,
            )

    def rate(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        metric = _drb(self.kwargs.meta["metric"], metric).lower()
        _curves = self._parse_curves(curves)
        if metric in ["clean_price", "dirty_price", "spread", "ytm"]:
            disc_curve = _maybe_get_curve_or_dict_maybe_from_solver(
                solver=solver,
                curves=_curves,
                curves_meta=self.kwargs.meta["curves"],
                name="disc_curve",
            )
            rate_curve = _maybe_get_curve_or_dict_maybe_from_solver(
                solver=solver,
                curves=_curves,
                curves_meta=self.kwargs.meta["curves"],
                name="rate_curve",
            )
            settlement_ = self._maybe_get_settlement(settlement, disc_curve)

            if metric == "spread":
                _: DualTypes = self.leg1.spread(
                    # target_npv=-(npv + self.leg1.settlement_params.notional),
                    target_npv=-(self.leg1.settlement_params.notional),
                    rate_curve=rate_curve,
                    disc_curve=disc_curve,
                    settlement=settlement_,
                    forward=settlement_,
                )
                return _
            else:
                npv = self.leg1.local_npv(
                    rate_curve=rate_curve,
                    disc_curve=disc_curve,
                    settlement=settlement_,
                    forward=settlement_,
                )
                # scale price to par 100 (npv is already projected forward to settlement)
                dirty_price = npv * 100 / -self.leg1.settlement_params.notional

                if metric == "dirty_price":
                    return dirty_price
                elif metric == "clean_price":
                    return dirty_price - self.accrued(settlement_, rate_curve=rate_curve)
                elif metric == "ytm":
                    return self.ytm(
                        price=dirty_price, settlement=settlement_, dirty=True, rate_curve=rate_curve
                    )

        raise ValueError("`metric` must be in {'dirty_price', 'clean_price', 'spread', 'ytm'}.")

    def accrued(
        self,
        settlement: datetime,
        rate_curve: CurveOption_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculate the accrued amount per nominal par value of 100.

        Parameters
        ----------
        settlement : datetime
            The settlement date which to measure accrued interest against.
        curve : Curve, optional
            If ``forecast`` is *True* and fixings are future based then must provide
            a forecast curve.

        Notes
        -----
        The settlement of an FRN will always be a definite amount. The
        ``fixing_method``, ``method_param`` and ``ex_div`` will contain a
        valid combination of parameters such that when payments need to be
        cleared these definitive amounts can be calculated
        via previously published fixings.

        If the coupon is IBOR based then the accrued
        fractionally apportions the coupon payment based on calendar days, including
        negative accrued during ex div periods. This rarely poses a problem since
        IBOR is fixed well in advance of settlement.

        .. math::

           \\text{Accrued} = \\text{Coupon} \\times \\frac{\\text{Settle - Last Coupon}}{\\text{Next Coupon - Last Coupon}}

        With RFR rates, however, and since ``settlement`` typically occurs
        in the future, e.g. T+2, it may be
        possible, particularly if the bond is *ex-div* that some fixings are not known
        today, but they will be known by ``settlement``. This is also true if we
        wish to calculate the forward dirty price of a bond and need to forecast
        the accrued amount (and also for a forecast IBOR period).

        Thus, there are two options:

        - In the analogue mode where very few fixings might be missing, and we require
          these values to calculate negative accrued in an ex-div period we do not
          require a ``curve`` but repeat the last historic fixing.
        - In the digital mode where the ``settlement`` is likely in the future we
          use a ``curve`` to forecast rates,

        Examples
        --------
        An RFR based FRN where the fixings are known up to the end of period.

        .. ipython:: python
           :suppress:

           from rateslib import FloatRateNote
           from pandas import date_range

        .. ipython:: python

           fixings = Series(2.0, index=date_range(dt(1999, 12, 1), dt(2000, 6, 2)))
           frn = FloatRateNote(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               currency="gbp",
               convention="Act365F",
               ex_div=3,
               fixings=fixings,
               fixing_method="rfr_observation_shift",
               method_param=5,
           )
           frn.accrued(dt(2000, 3, 27))
           frn.accrued(dt(2000, 6, 4))


        An IBOR based FRN where the coupon is known in advance.

        .. ipython:: python

           fixings = Series(2.0, index=[dt(1999, 12, 5)])
           frn = FloatRateNote(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               currency="gbp",
               convention="Act365F",
               ex_div=7,
               fixings=fixings,
               fixing_method="ibor",
               method_param=2,
           )
           frn.accrued(dt(2000, 3, 27))
           frn.accrued(dt(2000, 6, 4))
        """  # noqa: E501
        acc_idx = self.leg1._period_index(settlement)
        if self.leg1.rate_params.fixing_method == FloatFixingMethod.IBOR:
            frac = self.kwargs.meta["calc_mode"]._settle_accrual(self, settlement, acc_idx)
            if self.ex_div(settlement):
                frac = frac - 1  # accrued is negative in ex-div period
            rate = self.leg1._regular_periods[acc_idx].rate(rate_curve)
            cashflow = (
                -self.leg1._regular_periods[acc_idx].settlement_params.notional
                * self.leg1._regular_periods[acc_idx].period_params.dcf
                * rate
                / 100
            )
            return frac * cashflow / -self.leg1.settlement_params.notional * 100

        else:  # is "rfr"
            p = FloatPeriod(
                start=self.leg1.schedule.aschedule[acc_idx],
                end=settlement,
                payment=settlement,
                termination=self.leg1.schedule.aschedule[acc_idx + 1],
                stub=True,
                frequency=self.leg1.schedule.frequency_obj,
                notional=-100,
                currency=self.leg1.settlement_params.currency,
                convention=self.leg1._regular_periods[acc_idx].period_params.convention,
                float_spread=self.float_spread,
                fixing_method=self.leg1.rate_params.fixing_method,
                rate_fixings=self.leg1.rate_params.fixing_identifier,
                method_param=self.leg1.rate_params.method_param,
                spread_compound_method=self.leg1.rate_params.spread_compound_method,
                fixing_series=self.leg1.rate_params.fixing_series,
                fixing_frequency=self.leg1.rate_params.fixing_frequency,
                # roll=self.leg1.schedule.roll,
                calendar=self.leg1.schedule.calendar,
                adjuster=self.leg1.schedule.accrual_adjuster,
            )
            if p.period_params.start == p.period_params.end and acc_idx == 0:
                # bond settlement on issue date so there is no accrued
                return 0.0

            is_ex_div = self.ex_div(settlement)
            if is_ex_div and settlement == self.leg1._regular_periods[acc_idx].period_params.end:
                # then settlement is on a coupon date so no accrued
                return 0.0

            rate_to_settle = p.rate(rate_curve)
            accrued_to_settle = 100 * p.period_params.dcf * rate_to_settle / 100

            if is_ex_div:
                rate_to_end = self.leg1._regular_periods[acc_idx].rate(rate_curve=rate_curve)
                accrued_to_end = 100 * self.leg1._regular_periods[acc_idx].dcf * rate_to_end / 100
                return accrued_to_settle - accrued_to_end
            else:
                return accrued_to_settle
