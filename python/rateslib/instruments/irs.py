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
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import LegMtm
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _get_fx_forwards_maybe_from_solver,
    _maybe_get_curve_maybe_from_solver,
    _maybe_get_curve_or_dict_maybe_from_solver,
    _Vol,
)
from rateslib.legs import FixedLeg, FloatLeg

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurvesT_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FloatRateSeries,
        Frequency,
        FXForwards_,
        LegFixings,
        RollDay,
        Solver_,
        VolT_,
        _BaseLeg,
        bool_,
        datetime,
        datetime_,
        float_,
        int_,
        str_,
    )


class IRS(_BaseInstrument):
    """
    An *interest rate swap (IRS)* composing a :class:`~rateslib.legs.FixedLeg`
    and a :class:`~rateslib.legs.FloatLeg`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import IRS
       from rateslib.data.fixings import FXIndex
       from datetime import datetime as dt
       from rateslib import fixings

    .. ipython:: python

       irs = IRS(
           effective=dt(2000, 1, 1),
           termination="2y",
           spec="usd_irs",
           fixed_rate=2.0,
       )
       irs.cashflows()

    .. rubric:: Pricing

    An *IRS* requires a *disc curve* on both legs (which should be the same *Curve*) and a
    *leg2 rate curve* to forecast rates on the *FloatLeg*. The following input formats are
    allowed:

    .. code-block:: python

       curves = curve | [curve]           #  a single curve is repeated for all required curves
       curves = [rate_curve, disc_curve]  #  two curves are applied in the given order
       curves = [None, disc_curve, rate_curve, disc_curve]     # four curves applied to each leg
       curves = {"leg2_rate_curve": rate_curve, "disc_curve": disc_curve}  # dict form is explicit

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
    leg2_effective : datetime, :green:`optional (inherited from leg1)`
    leg2_termination : datetime, str, :green:`optional (inherited from leg1)`
    leg2_frequency : Frequency, str, :green:`optional (inherited from leg1)`
    leg2_stub : StubInference, str, :green:`optional (inherited from leg1)`
    leg2_front_stub : datetime, :green:`optional (inherited from leg1)`
    leg2_back_stub : datetime, :green:`optional (inherited from leg1)`
    leg2_roll : RollDay, int, str, :green:`optional (inherited from leg1)`
    leg2_eom : bool, :green:`optional (inherited from leg1)`
    leg2_modifier : Adjuster, str, :green:`optional (inherited from leg1)`
    leg2_calendar : calendar, str, :green:`optional (inherited from leg1)`
    leg2_payment_lag: Adjuster, int, :green:`optional (inherited from leg1)`
    leg2_payment_lag_exchange: Adjuster, int, :green:`optional (inherited from leg1)`
    leg2_ex_div: Adjuster, int, :green:`optional (inherited from leg1)`
    leg2_convention: str, :green:`optional (inherited from leg1)`

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
    leg2_notional : float, Dual, Dual2, Variable, :green:`optional (negatively inherited from leg1)`
    leg2_amortization : float, Dual, Dual2, Variable, str, Amortization, :green:`optional (negatively inherited from leg1)`

        .. note::

           The following are **rate parameters**.

    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.
    leg2_fixing_method: FloatFixingMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.FloatFixingMethod` describing the determination
        of the floating rate for each period.
    leg2_method_param: int, :green:`optional (set by 'defaults')`
        A specific parameter that is used by the specific ``fixing_method``.
    leg2_fixing_frequency: Frequency, str, :green:`optional (set by 'frequency' or '1B')`
        The :class:`~rateslib.scheduling.Frequency` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given is assumed to match the
        frequency of the schedule for an IBOR type ``fixing_method`` or '1B' if RFR type.
    leg2_fixing_series: FloatRateSeries, str, :green:`optional (implied by other parameters)`
        The :class:`~rateslib.data.fixings.FloatRateSeries` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given inherits attributes given
        such as the ``calendar``, ``convention``, ``method_param`` etc.
    leg2_float_spread: float, Dual, Dual2, Variable, :green:`optional (set as 0.0)`
        The amount (in bps) added to the rate in each period rate determination.
    leg2_spread_compound_method: SpreadCompoundMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.SpreadCompoundMethod` used in the calculation
        of the period rate when combining a ``float_spread``. Used **only** with RFR type
        ``fixing_method``.
    leg2_rate_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        See :ref:`Fixings <fixings-doc>`.
        The value of the rate fixing. If a scalar, is used directly. If a string identifier, links
        to the central ``fixings`` object and data loader.

        .. note::

           The following define **non-deliverability** parameters. If the swap is
           directly deliverable do not use these parameters. Review the **notes** section
           non-deliverability.

    pair: FXIndex, str, :green:`optional`
        The currency pair for :class:`~rateslib.data.fixings.FXFixing` that determines *Period*
        settlement. The *reference currency* is implied from ``pair``. Must include ``currency``.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for each *Period* according
        to non-deliverability.
    leg2_fx_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for each *Period* on *Leg2*
        according to non-deliverability.

        .. note::

           The following are **meta parameters**.

    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.

    Notes
    -----

    **Non-Deliverable IRS (NDIRS)**

    An *NDIRS* can be constructed by using the ``pair`` argument. The ``currency`` defines the
    *settlement currency*, whilst the *reference currency* is derived from ``pair`` and the
    ``notional`` is expressed *reference currency* units.

    The ``fx_fixings`` argument is typically used to provide an FX fixing series from which to
    extract non-deliverable :class:`~rateslib.data.fixings.FXFixing` data. The ``leg2_fx_fixings``
    inherits from the former and is likely to always be omitted, unless the fixings are provided
    as a list (against best practice) and the schedules do not align.

    For **pricing**, whilst a traditional *IRS* can be priced with just one *Curve*, e.g. "sofr"
    for a conventional USD IRS, an ND-IRS will always require 2 different curves: a *leg2 rate
    curve* for forecasting rates in the non-deliverable reference currency, and a *disc curve* for
    discounting cashflows in the settlement currency.

    The following is an example of a THB ND-IRS settled in USD with notional of 10mm THB.

    .. ipython:: python

       fixings.add("WMR_10AM_TYO_USDTHB", Series(index=[dt(2000, 6, 30), dt(2001, 1, 2)], data=[35.25, 37.0]))
       irs = IRS(
           effective=dt(2000, 1, 1),
           termination="2y",
           frequency="S",
           currency="usd",                              #  <- USD set as the settlement currency
           pair=FXIndex("usdthb", "fed", 1, "fed", -1), #  <- THB inferred as the reference currency
           fx_fixings="WMR_10AM_TYO",
           fixed_rate=2.0,
           # all other arguments set as normal IRS
       )
       irs.cashflows()

    .. ipython:: python
       :suppress:

       fixings.pop("WMR_10AM_TYO_USDTHB")

    Further information is available in the documentation for a :class:`~rateslib.legs.FixedLeg`.

    """  # noqa: E501

    _rate_scalar = 1.0

    @property
    def fixed_rate(self) -> DualTypes_:
        """The fixed rate parameter of the composited
        :class:`~rateslib.legs.FixedLeg`."""
        return self.leg1.fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes_) -> None:
        self.kwargs.leg1["fixed_rate"] = value
        self.leg1.fixed_rate = value

    @property
    def leg2_float_spread(self) -> DualTypes_:
        """The float spread parameter of the composited
        :class:`~rateslib.legs.FloatLeg`."""
        return self.leg2.float_spread

    @leg2_float_spread.setter
    def leg2_float_spread(self, value: DualTypes) -> None:
        self.kwargs.leg2["float_spread"] = value
        self.leg2.float_spread = value

    @property
    def leg1(self) -> FixedLeg:
        """The :class:`~rateslib.legs.FixedLeg` of the *Instrument*."""
        return self._leg1

    @property
    def leg2(self) -> FloatLeg:
        """The :class:`~rateslib.legs.FloatLeg` of the *Instrument*."""
        return self._leg2

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
        leg2_effective: datetime_ = NoInput(1),
        leg2_termination: datetime | str_ = NoInput(1),
        leg2_frequency: Frequency | str_ = NoInput(1),
        leg2_stub: str_ = NoInput(1),
        leg2_front_stub: datetime_ = NoInput(1),
        leg2_back_stub: datetime_ = NoInput(1),
        leg2_roll: int | RollDay | str_ = NoInput(1),
        leg2_eom: bool_ = NoInput(1),
        leg2_modifier: str_ = NoInput(1),
        leg2_calendar: CalInput = NoInput(1),
        leg2_payment_lag: int_ = NoInput(1),
        leg2_payment_lag_exchange: int_ = NoInput(1),
        leg2_ex_div: int_ = NoInput(1),
        leg2_convention: str_ = NoInput(1),
        # settlement parameters
        currency: str_ = NoInput(0),
        notional: float_ = NoInput(0),
        amortization: float_ = NoInput(0),
        leg2_notional: float_ = NoInput(-1),
        leg2_amortization: float_ = NoInput(-1),
        # non-deliverability
        pair: str_ = NoInput(0),
        fx_fixings: LegFixings = NoInput(0),
        leg2_fx_fixings: LegFixings = NoInput(1),
        # rate parameters
        fixed_rate: DualTypes_ = NoInput(0),
        leg2_float_spread: DualTypes_ = NoInput(0),
        leg2_spread_compound_method: str_ = NoInput(0),
        leg2_rate_fixings: LegFixings = NoInput(0),
        leg2_fixing_method: str_ = NoInput(0),
        leg2_method_param: int_ = NoInput(0),
        leg2_fixing_frequency: Frequency | str_ = NoInput(0),
        leg2_fixing_series: FloatRateSeries | str_ = NoInput(0),
        # meta parameters
        curves: CurvesT_ = NoInput(0),
        spec: str_ = NoInput(0),
    ) -> None:
        user_args = dict(
            # scheduling
            effective=effective,
            leg2_effective=leg2_effective,
            termination=termination,
            leg2_termination=leg2_termination,
            frequency=frequency,
            leg2_frequency=leg2_frequency,
            stub=stub,
            leg2_stub=leg2_stub,
            front_stub=front_stub,
            leg2_front_stub=leg2_front_stub,
            back_stub=back_stub,
            leg2_back_stub=leg2_back_stub,
            roll=roll,
            leg2_roll=leg2_roll,
            eom=eom,
            leg2_eom=leg2_eom,
            modifier=modifier,
            leg2_modifier=leg2_modifier,
            calendar=calendar,
            leg2_calendar=leg2_calendar,
            payment_lag=payment_lag,
            leg2_payment_lag=leg2_payment_lag,
            payment_lag_exchange=payment_lag_exchange,
            leg2_payment_lag_exchange=leg2_payment_lag_exchange,
            ex_div=ex_div,
            leg2_ex_div=leg2_ex_div,
            convention=convention,
            leg2_convention=leg2_convention,
            # settlement
            currency=currency,
            notional=notional,
            leg2_notional=leg2_notional,
            amortization=amortization,
            leg2_amortization=leg2_amortization,
            # non-deliverability
            pair=pair,
            fx_fixings=fx_fixings,
            leg2_fx_fixings=leg2_fx_fixings,
            # rate
            fixed_rate=fixed_rate,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_rate_fixings=leg2_rate_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
            leg2_fixing_series=leg2_fixing_series,
            leg2_fixing_frequency=leg2_fixing_frequency,
            # meta
            curves=self._parse_curves(curves),
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            leg2_currency=NoInput(1),
            leg2_pair=NoInput(1),
            initial_exchange=False,
            final_exchange=False,
            leg2_initial_exchange=False,
            leg2_final_exchange=False,
            mtm=LegMtm.Payment,
            leg2_mtm=LegMtm.Payment,
            vol=_Vol(),
        )

        default_args = dict(
            notional=defaults.notional,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            payment_lag_exchange=defaults.payment_lag_exchange,
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "vol"],
        )

        self._leg1 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self._leg2 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg2, 1))
        self._legs = [self.leg1, self.leg2]

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
        _curves = self._parse_curves(curves)
        fx_ = _get_fx_forwards_maybe_from_solver(solver, fx)

        leg2_npv: DualTypes = self.leg2.local_npv(
            rate_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "leg2_rate_curve", solver
            ),
            disc_curve=_maybe_get_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "leg2_disc_curve", solver
            ),
            index_curve=NoInput(0),
            fx=fx_,
            settlement=settlement,
            forward=forward,
        )
        return (
            self.leg1.spread(
                target_npv=-leg2_npv,
                rate_curve=NoInput(0),
                disc_curve=_maybe_get_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "disc_curve", solver
                ),
                fx=fx_,
                index_curve=NoInput(0),
                settlement=settlement,
                forward=forward,
            )
            / 100
        )

    def spread(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        _curves = self._parse_curves(curves)
        fx_ = _get_fx_forwards_maybe_from_solver(solver, fx)
        leg2_rate_curve = _maybe_get_curve_or_dict_maybe_from_solver(
            self.kwargs.meta["curves"], _curves, "leg2_rate_curve", solver
        )
        disc_curve = _maybe_get_curve_maybe_from_solver(
            self.kwargs.meta["curves"], _curves, "disc_curve", solver
        )
        leg1_npv: DualTypes = self.leg1.local_npv(
            rate_curve=NoInput(0),
            disc_curve=disc_curve,
            index_curve=NoInput(0),
            fx=fx_,
            settlement=settlement,
            forward=forward,
        )
        return self.leg2.spread(
            target_npv=-leg1_npv,
            rate_curve=leg2_rate_curve,
            fx=fx_,
            disc_curve=disc_curve,
            index_curve=NoInput(0),
            settlement=settlement,
            forward=forward,
        )

    def npv(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        self._set_pricing_mid(
            curves=curves,
            solver=solver,
            fx=fx,
            settlement=settlement,
            forward=forward,
        )
        return super().npv(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            local=local,
            settlement=settlement,
            forward=forward,
        )

    def _set_pricing_mid(
        self,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> None:
        # the test for an unpriced IRS is that its fixed rate is not set.
        if isinstance(self.kwargs.leg1["fixed_rate"], NoInput):
            # set a fixed rate for the purpose of generic methods NPV will be zero.
            mid_market_rate = self.rate(
                curves=curves,
                solver=solver,
                settlement=settlement,
                forward=forward,
                fx=fx,
            )
            self.leg1.fixed_rate = _dual_float(mid_market_rate)

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        An IRS has two curve requirements: a leg2_rate_curve and a disc_curve used by both legs.

        When given as only 1 element this curve is applied to all of the those components

        When given as 2 elements the first is treated as the rate curve and the 2nd as disc curve.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        elif isinstance(curves, list | tuple):
            if len(curves) == 2:
                return _Curves(
                    leg2_rate_curve=curves[0],
                    disc_curve=curves[1],
                    leg2_disc_curve=curves[1],
                )
            elif len(curves) == 1:
                return _Curves(
                    leg2_rate_curve=curves[0],
                    disc_curve=curves[0],
                    leg2_disc_curve=curves[0],
                )
            elif len(curves) == 4:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                    leg2_rate_curve=curves[2],
                    leg2_disc_curve=curves[3],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires only 2 curve types. Got {len(curves)}."
                )
        elif isinstance(curves, dict):
            return _Curves(
                rate_curve=curves.get("rate_curve", NoInput(0)),
                disc_curve=curves.get("disc_curve", NoInput(0)),
                leg2_rate_curve=_drb(
                    curves.get("rate_curve", NoInput(0)),
                    curves.get("leg2_rate_curve", NoInput(0)),
                ),
                leg2_disc_curve=_drb(
                    curves.get("disc_curve", NoInput(0)),
                    curves.get("leg2_disc_curve", NoInput(0)),
                ),
            )
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                leg2_rate_curve=curves,  # type: ignore[arg-type]
                disc_curve=curves,  # type: ignore[arg-type]
                leg2_disc_curve=curves,  # type: ignore[arg-type]
            )

    def cashflows(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return super()._cashflows_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            settlement=settlement,
            forward=forward,
        )

    def local_analytic_rate_fixings(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return self._local_analytic_rate_fixings_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            settlement=settlement,
            forward=forward,
        )
