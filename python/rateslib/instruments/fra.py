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
from rateslib.enums.generics import NoInput, Ok, Result, _drb
from rateslib.enums.parameters import FloatFixingMethod, SpreadCompoundMethod
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _maybe_get_curve_maybe_from_solver,
    _maybe_get_curve_or_dict_maybe_from_solver,
    _maybe_get_curve_or_dict_object_maybe_from_solver,
    _Vol,
)
from rateslib.legs import FixedLeg, FloatLeg
from rateslib.scheduling import Adjuster

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurvesT_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FixingsRates_,
        FloatRateSeries,
        Frequency,
        FXForwards_,
        RollDay,
        Solver_,
        VolT_,
        _BaseCurveOrDict_,
        _BaseLeg,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FRA(_BaseInstrument):
    """
    A *forward rate agreement (FRA)* compositing a
    :class:`~rateslib.legs.FixedLeg` and :class:`~rateslib.legs.FloatLeg`.

    These *Legs* have *Instrument* level overloads in order to satisfy the cashflow determination
    conventions of a *FRA* instruments.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import FRA
       from datetime import datetime as dt

    .. ipython:: python

       fra = FRA(
           effective=dt(2000, 1, 1),
           termination="6m",
           spec="eur_fra6",
           fixed_rate=2.0,
       )
       fra.cashflows()

    .. rubric:: Pricing

    An *FRA* requires a *disc curve* on both legs (which should be the same *Curve*) and a
    *leg2 rate curve* to forecast the IBOR type rate on the *FloatLeg*. The following input
    formats are allowed:

    .. code-block:: python

       curves = curve | [curve]           #  a single curve is repeated for all required curves
       curves = [rate_curve, disc_curve]  #  two curves are applied in the given order
       curves = [None, disc_curve, rate_curve, disc_curve]     # four curves applied to each leg
       curves = {"leg2_rate_curve": rate_curve, "disc_curve": disc_curve}  # dict form is explicit

    The only ``metric`` is *'rate'*.

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
    payment_lag: int, :green:`optional (set as 0)`
        A number of business days by which to lag a traditional *FRA* payment date.

        .. warning::

           *FRAs* are defined by a payment structure that has a cashflow at the accrual start
           date and an amount adjusted by the rate fixing. An input to this parameter, say 5,
           will apply an adjuster: `Adjuster.BusDaysLagSettleInAdvance(5)`.

    ex_div: int, :green:`optional (set as 0)`
        Applied in the same manner as the ``payment_lag``, except negated. An input of 1 will
        apply an adjuster: `Adjuster.BusDaysLagSettleInAdvance(-1)`.
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

           The following are **rate parameters**.

    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.
    leg2_method_param: int, :green:`optional (set by 'defaults')`
        A specific parameter that is used by the specific ``fixing_method``.
    leg2_fixing_frequency: Frequency, str, :green:`optional (set by 'frequency' or '1B')`
        The :class:`~rateslib.scheduling.Frequency` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given is assumed to match the
        frequency of the schedule.
    leg2_fixing_series: FloatRateSeries, str, :green:`optional (implied by other parameters)`
        The :class:`~rateslib.data.fixings.FloatRateSeries` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given inherits attributes given
        such as the ``calendar``, ``convention``, ``method_param`` etc.
    leg2_rate_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        See :ref:`Fixings <fixings-doc>`.
        The value of the rate fixing. If a scalar, is used directly. If a string identifier, links
        to the central ``fixings`` object and data loader.

        .. note::

           The following are **meta parameters**.

    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.
    metric : str, :green:`optional` (set as 'rate')`
        The pricing metric returned by :meth:`~rateslib.instruments.FRA.rate`.

    Notes
    -----

    A *FRA* is modelled as a single period *IRS* whose payment date is overloaded to be
    based on the 'accrual' effective date, and whose cashflow values are adjusted by a scaling
    factor related to the floating rate, i.e. :math:`\\frac{1}{1 + d r}`, thus replicating the
    payoff calculation for a traditional *FRA*.

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

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        An STIRFuture has two curve requirements: a leg2_rate_curve and a disc_curve used by
        both legs.

        When given as only 1 element this curve is applied to all of the those components

        When given as 2 elements the first is treated as the rate curve and the 2nd as disc curve.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
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
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                leg2_rate_curve=curves,  # type: ignore[arg-type]
                disc_curve=curves,  # type: ignore[arg-type]
                leg2_disc_curve=curves,  # type: ignore[arg-type]
            )

    def __init__(
        self,
        # scheduling
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: Frequency | str_ = NoInput(0),
        *,
        roll: int | RollDay | str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int = 0,
        ex_div: int = 0,
        convention: str_ = NoInput(0),
        # settlement parameters
        currency: str_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        # rate parameters
        fixed_rate: DualTypes_ = NoInput(0),
        leg2_rate_fixings: FixingsRates_ = NoInput(0),
        leg2_method_param: int_ = NoInput(0),
        leg2_fixing_frequency: Frequency | str_ = NoInput(0),
        leg2_fixing_series: FloatRateSeries | str_ = NoInput(0),
        # meta parameters
        curves: CurvesT_ = NoInput(0),
        spec: str_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> None:
        user_args = dict(
            # scheduling
            effective=effective,
            termination=termination,
            frequency=frequency,
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
            convention=convention,
            # settlement
            currency=currency,
            notional=notional,
            # rate
            fixed_rate=fixed_rate,
            leg2_rate_fixings=leg2_rate_fixings,
            leg2_method_param=leg2_method_param,
            leg2_fixing_series=leg2_fixing_series,
            leg2_fixing_frequency=leg2_fixing_frequency,
            # meta
            curves=self._parse_curves(curves),
            metric=metric,
        )
        instrument_args = dict(
            leg2_effective=NoInput.inherit,
            leg2_termination=NoInput.inherit,
            leg2_frequency=NoInput.inherit,
            leg2_roll=NoInput.inherit,
            leg2_eom=NoInput.inherit,
            leg2_modifier=NoInput.inherit,
            leg2_calendar=NoInput.inherit,
            leg2_payment_lag=NoInput.inherit,
            leg2_ex_div=NoInput.inherit,
            leg2_convention=NoInput.inherit,
            leg2_float_spread=0.0,
            leg2_fixing_method=FloatFixingMethod.IBOR,
            leg2_spread_compound_method=SpreadCompoundMethod.NoneSimple,
            leg2_notional=NoInput.negate,
            leg2_currency=NoInput.inherit,
            payment_lag=Adjuster.BusDaysLagSettleInAdvance(payment_lag),
            ex_div=Adjuster.BusDaysLagSettleInAdvance(-ex_div),
            initial_exchange=False,
            final_exchange=False,
            leg2_initial_exchange=False,
            leg2_final_exchange=False,
            vol=_Vol(),
        )
        default_args = dict(
            notional=defaults.notional,
            metric="rate",
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "metric", "vol"],
        )

        self._leg1 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self._leg2 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg2, 1))
        self._legs = [self.leg1, self.leg2]

        if self._leg1.schedule.n_periods != 1:
            raise ValueError(
                "The scheduling parameters of the STIRFuture must define exactly "
                f"one regular period. Got '{self.leg1.schedule.n_periods}'."
            )

    def _fra_rate_scalar(self, leg2_rate_curve: _BaseCurveOrDict_) -> DualTypes:
        r = self.leg2._regular_periods[0].rate(rate_curve=leg2_rate_curve)
        return 1 / (1 + self.leg2._regular_periods[0].period_params.dcf * r / 100.0)

    def _try_fra_rate_scalar(self, leg2_rate_curve: _BaseCurveOrDict_) -> Result[DualTypes]:
        r = self.leg2._regular_periods[0].try_rate(rate_curve=leg2_rate_curve)
        if r.is_err:
            return r
        else:
            return Ok(
                1 / (1 + self.leg2._regular_periods[0].period_params.dcf * r.unwrap() / 100.0)
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
            settlement=settlement,
            forward=forward,
        )
        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        leg2_rate_curve = _maybe_get_curve_or_dict_maybe_from_solver(
            solver=solver, curves_meta=_curves_meta, curves=_curves, name="leg2_rate_curve"
        )
        fra_scalar = self._fra_rate_scalar(leg2_rate_curve=leg2_rate_curve)

        npv = super().npv(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            local=local,
            settlement=settlement,
            forward=forward,
        )
        if isinstance(npv, dict):
            return {k: v * fra_scalar for k, v in npv.items()}
        else:
            return npv * fra_scalar

    def _set_pricing_mid(
        self,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
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
            )
            self.leg1.fixed_rate = _dual_float(mid_market_rate)

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
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()

        leg2_npv: DualTypes = self.leg2.local_npv(
            rate_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "leg2_rate_curve", solver
            ),
            disc_curve=_maybe_get_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "leg2_disc_curve", solver
            ),
            settlement=settlement,
            forward=forward,
        )
        rate = (
            self.leg1.spread(
                target_npv=-leg2_npv,
                rate_curve=NoInput(0),
                disc_curve=_maybe_get_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "disc_curve", solver
                ),
                index_curve=NoInput(0),
                settlement=settlement,
                forward=forward,
            )
            / 100
        )
        if metric_ == "rate":
            return rate
        else:
            raise ValueError("`metric` must be in {'rate'}.")

    def analytic_delta(
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
        leg: int = 1,
    ) -> DualTypes | dict[str, DualTypes]:
        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        leg2_rate_curve = _maybe_get_curve_or_dict_maybe_from_solver(
            solver=solver, curves_meta=_curves_meta, curves=_curves, name="leg2_rate_curve"
        )
        fra_scalar = self._fra_rate_scalar(leg2_rate_curve=leg2_rate_curve)
        a_delta = super().analytic_delta(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            local=local,
            settlement=settlement,
            forward=forward,
            leg=leg,
        )
        if isinstance(a_delta, dict):
            return {k: v * fra_scalar for k, v in a_delta.items()}
        else:
            return a_delta * fra_scalar

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
        df = self._local_analytic_rate_fixings_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            settlement=settlement,
            forward=forward,
        )
        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        return df * self._fra_rate_scalar(
            leg2_rate_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                solver=solver, curves_meta=_curves_meta, curves=_curves, name="leg2_rate_curve"
            )
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
        df = super()._cashflows_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            settlement=settlement,
            forward=forward,
        )

        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        leg2_rate_curve = _maybe_get_curve_or_dict_object_maybe_from_solver(
            solver=solver, curves_meta=_curves_meta, curves=_curves, name="leg2_rate_curve"
        )
        scalar = self._try_fra_rate_scalar(leg2_rate_curve=leg2_rate_curve)

        headers = [
            defaults.headers["cashflow"],
            defaults.headers["npv"],
            defaults.headers["npv_fx"],
        ]
        for header in headers:
            if scalar.is_err:
                df[header] = None
            else:
                df[header] = df[header] * _dual_float(scalar.unwrap())
        return df
