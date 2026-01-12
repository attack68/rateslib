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
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _get_fx_maybe_from_solver,
    _maybe_get_curve_maybe_from_solver,
    _maybe_get_curve_object_maybe_from_solver,
    _maybe_get_curve_or_dict_maybe_from_solver,
    _Vol,
)
from rateslib.legs import FixedLeg, FloatLeg
from rateslib.periods.utils import (
    _maybe_fx_converted,
    _maybe_local,
    _validate_base_curve,
)

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
        _BaseLeg,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class STIRFuture(_BaseInstrument):
    """
    A *short term interest rate (STIR) future* compositing a
    :class:`~rateslib.legs.FixedLeg` and :class:`~rateslib.legs.FloatLeg`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import STIRFuture
       from datetime import datetime as dt

    .. ipython:: python

       stir = STIRFuture(
           effective=dt(2022, 3, 16),
           termination=dt(2022, 6, 15),
           spec="usd_stir",
           price=99.50,
           contracts=10,
       )
       stir.cashflows()

    .. rubric:: Pricing

    A *STIRFuture* requires a *disc curve* on both legs (which should be the same *Curve*) and a
    *leg2 rate curve* to forecast rates on the *FloatLeg*. The following input formats are
    allowed:

    .. code-block:: python

       curves = curve | [curve]           #  a single curve is repeated for all required curves
       curves = [rate_curve, disc_curve]  #  two curves are applied in the given order
       curves = [None, disc_curve, rate_curve, disc_curve]     # four curves applied to each leg
       curves = {"leg2_rate_curve": rate_curve, "disc_curve": disc_curve}  # dict form is explicit

    The available pricing ``metric`` are in *{'rate', 'price'}* which will return the future's
    market price in the respective terms.

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

    contracts : int
        The number of traded contracts.
    nominal : float
        The nominal value of the contract. See **Notes**.
    currency : str, :green:`optional (set by 'defaults')`
        The local settlement currency of the *Instrument* (3-digit code).

        .. note::

           The following are **rate parameters**.

    price : float
        The traded price of the future. Defined as 100 minus the fixed rate.
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

           The following are **meta parameters**.

    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    metric : str, :green:`optional` (set by 'defaults')
        The pricing metric returned by :meth:`~rateslib.instruments.STIRFuture.rate`.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.

    Notes
    -----
    A *STIRFuture* is modelled as a single period *IRS* whose payment date is overloaded to always
    result in immediate settlement, thus replicating the behaviour of traditional exchanges.
    The immediate date is derived from the discount curve used during pricing.

    The ``nominal`` for one contract should be set according to the ``convention`` so that the
    correct amount of risk is allocated is to 1bp. For example, for a CME SOFR 3M future, setting
    a convention of *ActActICMA* yields a DCF of 0.25 and therefore a ``nominal`` of 1mm USD
    yields a 1bp sensitivity of 25 USD for any contract, as per the CME contract specification. The
    ``leg2_fixing_series`` argument allows full specification of the floating rate index
    conventions.

    """

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
        payment_lag: int_ = NoInput(0),
        ex_div: int_ = NoInput(0),
        convention: str_ = NoInput(0),
        # settlement parameters
        currency: str_ = NoInput(0),
        contracts: int = 1,
        nominal: float | NoInput = NoInput(0),
        # rate parameters
        price: DualTypes_ = NoInput(0),
        leg2_float_spread: DualTypes_ = NoInput(0),
        leg2_spread_compound_method: str_ = NoInput(0),
        leg2_rate_fixings: FixingsRates_ = NoInput(0),
        leg2_fixing_method: str_ = NoInput(0),
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
            payment_lag=payment_lag,
            ex_div=ex_div,
            convention=convention,
            # settlement
            currency=currency,
            nominal=nominal,
            contracts=contracts,
            # rate
            price=price,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_rate_fixings=leg2_rate_fixings,
            leg2_fixing_method=leg2_fixing_method,
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
            fixed_rate=NoInput(0) if isinstance(price, NoInput) else 100 - price,
            vol=_Vol(),
        )
        default_args = dict(
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            nominal=defaults.notional,
            metric="rate",
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "contracts", "nominal", "price", "metric", "vol"],
        )
        self._kwargs.leg1["notional"] = -self.kwargs.meta["nominal"] * self.kwargs.meta["contracts"]
        self._kwargs.leg2["notional"] = self.kwargs.meta["nominal"] * self.kwargs.meta["contracts"]

        self._leg1 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self._leg2 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg2, 1))
        self._legs = [self.leg1, self.leg2]

        if self._leg1.schedule.n_periods != 1:
            raise ValueError(
                "The scheduling parameters of the STIRFuture must define exactly "
                f"one regular period. Got '{self.leg1.schedule.n_periods}'."
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
        self._set_pricing_mid(curves=curves, solver=solver, settlement=settlement, forward=forward)
        local_npv = super().npv(  # type: ignore[index]
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            local=True,
            settlement=settlement,
            forward=forward,
        )[self.leg1.settlement_params.currency]

        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        npv_immediate = (
            local_npv
            / _validate_base_curve(
                _maybe_get_curve_maybe_from_solver(
                    solver=solver, curves_meta=_curves_meta, curves=_curves, name="disc_curve"
                )
            )[self.leg1.settlement_params.payment]
        )

        if not local:
            return _maybe_fx_converted(
                value=npv_immediate,
                currency=self.leg1.settlement_params.currency,
                fx=_get_fx_maybe_from_solver(solver=solver, fx=fx),
                base=_drb(self.leg1.settlement_params.currency, base),
                forward=forward,
            )
        else:
            return {self.leg1.settlement_params.currency: npv_immediate}

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
        if metric_ == "price":
            return 100 - rate
        elif metric_ == "rate":
            return rate
        else:
            raise ValueError("`metric` must be in {'rate', 'price'}.")

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
        unadjusted_local_analytic_delta = super().analytic_delta(  # type: ignore[index]
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            local=True,
            settlement=settlement,
            forward=forward,
            leg=leg,
        )[self.leg1.settlement_params.currency]
        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        prefix = "" if leg == 1 else "leg2_"
        adjusted_local_analytic_delta = (
            unadjusted_local_analytic_delta
            / _validate_base_curve(
                _maybe_get_curve_maybe_from_solver(
                    solver=solver,
                    curves_meta=_curves_meta,
                    curves=_curves,
                    name=f"{prefix}disc_curve",
                )
            )[self.leg1.settlement_params.payment]
        )
        return _maybe_local(
            value=adjusted_local_analytic_delta,
            local=local,
            currency=self.leg1.settlement_params.currency,
            fx=_get_fx_maybe_from_solver(solver=solver, fx=fx),
            base=base,
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
        return (
            df  # type: ignore[operator]
            / _validate_base_curve(
                _maybe_get_curve_or_dict_maybe_from_solver(
                    solver=solver, curves_meta=_curves_meta, curves=_curves, name="leg2_disc_curve"
                )
            )[self.leg1.settlement_params.payment]
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
        df[defaults.headers["payment"]] = None

        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        disc_curve = _maybe_get_curve_object_maybe_from_solver(
            solver=solver, curves_meta=_curves_meta, curves=_curves, name="disc_curve"
        )
        if isinstance(disc_curve, NoInput):
            pass
        else:
            df[defaults.headers["payment"]] = disc_curve.nodes.initial
            df[defaults.headers["npv"]] = df[defaults.headers["npv"]] / df[defaults.headers["df"]]
            df[defaults.headers["npv_fx"]] = (
                df[defaults.headers["npv_fx"]] / df[defaults.headers["df"]]
            )
            df[defaults.headers["df"]] = 1.0

        return df
