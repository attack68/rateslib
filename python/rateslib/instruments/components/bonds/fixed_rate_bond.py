from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.curves._parsers import (
    _maybe_set_ad_order,
    _validate_curve_not_no_input,
)
from rateslib.dual import ift_1dim
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.components.bonds.conventions import (
    BondCalcMode,
    _get_bond_calc_mode,
)
from rateslib.instruments.components.bonds.protocols import (
    _WithDuration,
    _WithExDiv,
    _WithOASpread,
    _WithRepo,
    _WithYTM,
)
from rateslib.instruments.components.protocols import _BaseInstrument
from rateslib.instruments.components.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.components.protocols.pricing import (
    _Curves,
    _get_maybe_curve_maybe_from_solver,
)
from rateslib.legs.components import FixedLeg

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurveOption_,
        Curves_,
        DataFrame,
        DualTypes,
        DualTypes_,
        Frequency,
        FXForwards_,
        FXVolOption_,
        RollDay,
        Solver_,
        _BaseCurve,
        _BaseLeg,
        bool_,
        datetime,
        datetime_,
        float_,
        int_,
        str_,
    )


class FixedRateBond(_BaseInstrument, _WithExDiv, _WithYTM, _WithDuration, _WithRepo, _WithOASpread):
    """
    A *interest rate swap (IRS)* composing a :class:`~rateslib.legs.components.FixedLeg`
    and a :class:`~rateslib.legs.components.FloatLeg`.

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
    stub : StubInference, str in {"ShortFront", "LongFront", "ShortBack", "LongBack"}, :green:`optional`
        The stub type used if stub inference is required. If given as string will derive a
        :class:`~rateslib.scheduling.StubInference`.
    front_stub : datetime, :green:`optional`
        The unadjusted date for the start stub period. If given as adjusted, unadjusted
        alternatives may be inferred.
    back_stub : datetime, :green:`optional`
        The unadjusted date for the back stub period. If given as adjusted, unadjusted
        alternatives may be inferred.
        See notes for combining ``stub``, ``front_stub`` and ``back_stub``
        and any automatic stub inference.
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
    payment_lag_exchange: Adjuster, int, :green:`optional`
        The :class:`~rateslib.scheduling.Adjuster` to use to map adjusted schedule dates into
        additional payment date. If given as integer will define the number of business days to
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
    amortization: float, Dual, Dual2, Variable, str, Amortization, :green:`optional (set as zero)`
        Set a non-constant notional per *Period*. If a scalar value, adjusts the ``notional`` of
        each successive period by that same value. Should have
        sign equal to that of notional if the notional is to reduce towards zero.

        .. note::

           The following are **rate parameters**.

    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.

        .. note::

           The following are **meta parameters**.

    curves : XXX
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument.
    calc_mode : str or BondCalcMode
        A calculation mode for dealing with bonds under different conventions. See notes.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.

    """  # noqa: E501

    _rate_scalar = 1.0

    @property
    def fixed_rate(self) -> DualTypes_:
        """The fixed rate parameter of the composited
        :class:`~rateslib.legs.components.FixedLeg`."""
        return self.leg1.fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes_) -> None:
        self.kwargs.leg1["fixed_rate"] = value
        self.leg1.fixed_rate = value

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
        # scheduling
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: Frequency | str_ = NoInput(0),
        *,
        stub: str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: int | RollDay | str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        payment_lag_exchange: int_ = NoInput(0),
        ex_div: int_ = NoInput(0),
        convention: str_ = NoInput(0),
        # settlement parameters
        currency: str_ = NoInput(0),
        notional: float_ = NoInput(0),
        amortization: float_ = NoInput(0),
        # rate parameters
        fixed_rate: DualTypes_ = NoInput(0),
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
            fixed_rate=fixed_rate,
            # meta
            curves=self._parse_curves(curves),
            calc_mode=calc_mode,
            settle=settle,
            metric=metric,
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            initial_exchange=False,
            final_exchange=True,
        )

        default_args = dict(
            notional=defaults.notional,
            calc_mode=defaults.calc_mode[type(self).__name__],
            initial_exchange=False,
            final_exchange=True,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            payment_lag_exchange=defaults.payment_lag_specific[type(self).__name__],
            ex_div=defaults.ex_div,
            settle=defaults.settle,
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "calc_mode", "settle", "metric"],
        )
        self.kwargs.meta["calc_mode"] = _get_bond_calc_mode(self.kwargs.meta["calc_mode"])

        if isinstance(self.kwargs.leg1["fixed_rate"], NoInput):
            raise ValueError(f"`fixed_rate` must be provided for {type(self).__name__}.")

        self._leg1 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self._legs = [self.leg1]

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
        _curves = self._parse_curves(curves)
        disc_curve = _get_maybe_curve_maybe_from_solver(
            curves_meta=self.kwargs.meta["curves"], curves=_curves, name="disc_curve", solver=solver
        )
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()
        if metric_ in ["clean_price", "dirty_price", "ytm"]:
            if isinstance(settlement, NoInput):
                settlement_ = self.leg1.schedule.calendar.lag_bus_days(
                    disc_curve.nodes.initial,
                    self.kwargs.meta["settle"],
                    True,
                )
            else:
                settlement_ = settlement
            npv = self.leg1.local_npv(
                disc_curve=disc_curve, settlement=settlement_, forward=settlement_
            )
            # scale price to par 100 (npv is already projected forward to settlement)
            dirty_price = npv * 100 / -self.leg1.settlement_params.notional

            if metric_ == "dirty_price":
                return dirty_price
            elif metric_ == "clean_price":
                return dirty_price - self.accrued(settlement_)
            elif metric_ == "ytm":
                return self.ytm(dirty_price, settlement_, True)

        raise ValueError("`metric` must be in {'dirty_price', 'clean_price', 'ytm'}.")

    def spread(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        NotImplementedError()

    def npv(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        _curves = self._parse_curves(curves)
        disc_curve = _get_maybe_curve_maybe_from_solver(
            curves_meta=self.kwargs.meta["curves"], curves=_curves, name="disc_curve", solver=solver
        )
        if isinstance(settlement, NoInput):
            settlement_ = self.leg1.schedule.calendar.lag_bus_days(
                disc_curve.nodes.initial,
                self.kwargs.meta["settle"],
                True,
            )
            forward_ = _drb(disc_curve.nodes.initial, forward)
        else:
            settlement_ = settlement
            forward_ = forward  # if NoInput adopts the usual default settings from 'settlement'

        return super().npv(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            local=local,
            settlement=settlement_,
            forward=forward_,
        )

    def _parse_curves(self, curves: CurveOption_) -> _Curves:
        """
        An FRB has one curve requirements: a disc_curve.

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

    def cashflows(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return super()._cashflows_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            settlement=settlement,
            forward=forward,
        )

    def local_analytic_rate_fixings(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return self._local_analytic_rate_fixings_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            settlement=settlement,
            forward=forward,
        )

    def price(self, ytm: DualTypes, settlement: datetime, dirty: bool = False) -> DualTypes:
        """
        Calculate the price of the security per nominal value of 100, given
        yield-to-maturity.

        Parameters
        ----------
        ytm : float
            The yield-to-maturity against which to determine the price.
        settlement : datetime
            The settlement date on which to determine the price.
        dirty : bool, optional
            If `True` will include the
            :meth:`rateslib.instruments.FixedRateBond.accrued` in the price.

        Returns
        -------
        float, Dual, Dual2

        Examples
        --------
        This example is taken from the UK debt management office website.
        The result should be `141.070132` and the bond is ex-div.

        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.ex_div(dt(1999, 5, 27))
           gilt.price(
               ytm=4.445,
               settlement=dt(1999, 5, 27),
               dirty=True
           )

        This example is taken from the Swedish national debt office website.
        The result of accrued should, apparently, be `0.210417` and the clean
        price should be `99.334778`.

        .. ipython:: python

           bond = FixedRateBond(
               effective=dt(2017, 5, 12),
               termination=dt(2028, 5, 12),
               frequency="A",
               calendar="stk",
               currency="sek",
               convention="ActActICMA",
               ex_div=5,
               fixed_rate=0.75
           )
           bond.ex_div(dt(2017, 8, 23))
           bond.accrued(dt(2017, 8, 23))
           bond.price(
               ytm=0.815,
               settlement=dt(2017, 8, 23),
               dirty=False
           )

        """
        return self._price_from_ytm(
            ytm=ytm,
            settlement=settlement,
            calc_mode=NoInput(0),  # will be set to kwargs.meta
            dirty=dirty,
            curve=NoInput(0),
        )

    def oaspread(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        base: str_ = NoInput(0),
        price: DualTypes_ = NoInput(0),
        metric: str_ = NoInput(0),
        func_tol: float_ = NoInput(0),
        conv_tol: float_ = NoInput(0),
    ) -> DualTypes:
        """
        The option adjusted spread added to the discounting *Curve* to value the security
        at ``price``.

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
        price : float, Dual, Dual2
            The price of the bond to match.
        metric : str, optional
            The metric to use when evaluating the price/rate of the instrument. If not
            given uses the instrument's :meth:`~rateslib.instruments.FixedRateBond.rate` method
            default.
        func_tol: float, optional
            The tolerance for the objective function value when iteratively solving. If not given
            uses `defaults.oaspread_func_tol`.
        conv_tol: float, optional
            The tolerance used for stopping criteria of successive iteration values. If not
            given uses `defaults.oaspread_conv_tol`.

        Returns
        -------
        float, Dual, Dual2

        Notes
        ------
        The discount curve must be of type :class:`~rateslib.curves._BaseCurve` with a
        provided :meth:`~rateslib.curves._BaseCurve.shift` method available.

        .. warning::
           The sensitivity of variables is preserved for the input argument ``price``, but this
           function does **not** preserve AD towards variables associated with the ``curves`` or
           ``solver``.

        Examples
        --------

        .. ipython:: python
           :suppress:

           from rateslib import Variable

        .. ipython:: python

           bond = FixedRateBond(dt(2000, 1, 1), "3Y", fixed_rate=2.5, spec="us_gb")
           curve = Curve({dt(2000, 7, 1): 1.0, dt(2005, 7, 1): 0.80})
           # Add AD variables to the curve without a Solver
           curve._set_ad_order(1)

           bond.oaspread(curves=curve, price=Variable(95.0, ["price"], []))

        This result excludes curve sensitivities but includes sensitivity to the
        constructed *'price'* variable. Accuracy can be observed through numerical simulation.

        .. ipython:: python

           bond.oaspread(curves=curve, price=96.0)
           bond.oaspread(curves=curve, price=94.0)

        """
        if isinstance(price, NoInput):
            raise ValueError("`price` must be supplied in order to derive the `oaspread`.")

        _curves = self._parse_curves(curves)
        disc_curve = _get_maybe_curve_maybe_from_solver(
            curves_meta=self.kwargs.meta["curves"], curves=_curves, name="disc_curve", solver=solver
        )
        rate_curve = _get_maybe_curve_maybe_from_solver(
            curves_meta=self.kwargs.meta["curves"], curves=_curves, name="rate_curve", solver=solver
        )

        def s_with_args(
            g: DualTypes, curve: CurveOption_, disc_curve: _BaseCurve, metric: str_
        ) -> DualTypes:
            """
            Return the price of a bond given an OASpread.

            Parameters
            ----------
            g: DualTypes
                The OASpread value in basis points.
            curve:
                The forecasting curve.
            disc_curve:
                The discount curve.

            Returns
            -------
            DualTypes
            """
            _shifted_discount_curve = disc_curve.shift(g)
            return self.rate(curves=[curve, _shifted_discount_curve], metric=metric)

        disc_curve_: _BaseCurve = _validate_curve_not_no_input(disc_curve)
        _ad_disc = _maybe_set_ad_order(disc_curve_, 0)
        _ad_fore = _maybe_set_ad_order(rate_curve, 0)

        s = partial(
            s_with_args,
            curve=rate_curve,
            disc_curve=disc_curve_,
            metric=metric,
        )

        result = ift_1dim(
            s,
            price,
            "ytm_quadratic",
            (-300, 200, 1200),
            func_tol=_drb(defaults.oaspread_func_tol, func_tol),
            conv_tol=_drb(defaults.oaspread_conv_tol, conv_tol),
        )

        _maybe_set_ad_order(disc_curve_, _ad_disc)
        _maybe_set_ad_order(rate_curve, _ad_fore)
        ret: DualTypes = result["g"]
        return ret

    def analytic_delta(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        leg: int = 1,
    ) -> DualTypes | dict[str, DualTypes]:
        if isinstance(settlement, NoInput):
            _curves = self._parse_curves(curves)
            disc_curve = _get_maybe_curve_maybe_from_solver(
                curves_meta=self.kwargs.meta["curves"],
                curves=_curves,
                name="disc_curve",
                solver=solver,
            )
            settlement_ = self.leg1.schedule.calendar.lag_bus_days(
                disc_curve.nodes.initial,
                self.kwargs.meta["settle"],
                True,
            )
        else:
            settlement_ = settlement

        return super().analytic_delta(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            local=local,
            settlement=settlement_,
            forward=forward,
            leg=leg,
        )
