from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.components.protocols import _BaseInstrument
from rateslib.instruments.components.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.components.protocols.pricing import (
    _Curves,
    _get_fx_maybe_from_solver,
    _get_maybe_curve_maybe_from_solver,
)
from rateslib.legs.components import FixedLeg, FloatLeg

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurveOption_,
        Curves_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FixingsRates_,
        FloatRateSeries,
        Frequency,
        FXForwards_,
        FXVolOption_,
        LegFixings,
        RollDay,
        Solver_,
        _BaseLeg,
        bool_,
        datetime,
        datetime_,
        float_,
        int_,
        str_,
    )


class XCS(_BaseInstrument):
    """
    A *cross-currency swap (XCS)* composing either
    :class:`~rateslib.legs.components.FixedLeg`
    and/or :class:`~rateslib.legs.components.FloatLeg` in different currencies.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments.components import XCS
       from datetime import datetime as dt
       from rateslib import fixings
       from pandas import Series

    .. ipython:: python

       fixings.add("EURUSD_1600_GMT", Series(index=[dt(2025, 4, 8)], data=[1.175]))
       xcs = XCS(
           effective=dt(2025, 1, 8),
           termination="6m",
           spec="eurusd_xcs",
           notional=5e6,
           leg2_fx_fixings=(1.15, "EURUSD_1600_GMT"),
           leg2_mtm=True,
       )
       xcs.cashflows()

    .. ipython:: python
       :suppress:

       fixings.pop("EURUSD_1600_GMT")

    .. rubric:: Pricing

    The methods of a *XCS* require an :class:`~rateslib.fx.FXForwards` object for ``fx``, and
    **four** :class:`~rateslib.curves._BaseCurve` for ``curves``;

    - a **rate_curve**,
    - a **disc_curve**,
    - a **leg2_rate_curve**,
    - a **leg2_disc_curve** (for pricing consistency the collateral of each discount curve should
      be the same).

    If given as a list these should be specified in this order.

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
        The local settlement currency of leg1 (3-digit code).
    notional : float, Dual, Dual2, Variable, :green:`optional (set from 'leg2_notional' or 'defaults' )`
        The initial leg1 notional, defined in units of the currency of the leg. Only one
        of ``notional`` and ``leg2_notional`` can be given. The alternate leg notional is derived
        via non-deliverability :class:`~rateslib.data.fixings.FXFixing`.
    amortization: float, Dual, Dual2, Variable, str, Amortization, :green:`optional (set as zero)`
        Set a non-constant notional per *Period*. If a scalar value, adjusts the ``notional`` of
        each successive period by that same value. Should have
        sign equal to that of notional if the notional is to reduce towards zero.
    leg2_currency : str, :red:`required`
        The currency of the leg2.
    leg2_notional : float, Dual, Dual2, Variable, :green:`optional (negatively inherited from leg1)`
    leg2_amortization : float, Dual, Dual2, Variable, str, Amortization, :green:`optional (negatively inherited from leg1)`

        .. note::

           The following are **rate parameters**.

    fixed : bool, :green:`optional (set as False)`
        Whether leg1 is a :class:`~rateslib.legs.FixedLeg` or a :class:`~rateslib.legs.FloatLeg`.
    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.
    fixing_method: FloatFixingMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.FloatFixingMethod` describing the determination
        of the floating rate for each period.
    method_param: int, :green:`optional (set by 'defaults')`
        A specific parameter that is used by the specific ``fixing_method``.
    fixing_frequency: Frequency, str, :green:`optional (set by 'frequency' or '1B')`
        The :class:`~rateslib.scheduling.Frequency` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given is assumed to match the
        frequency of the schedule for an IBOR type ``fixing_method`` or '1B' if RFR type.
    fixing_series: FloatRateSeries, str, :green:`optional (implied by other parameters)`
        The :class:`~rateslib.data.fixings.FloatRateSeries` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given inherits attributes given
        such as the ``calendar``, ``convention``, ``method_param`` etc.
    float_spread: float, Dual, Dual2, Variable, :green:`optional (set as 0.0)`
        The amount (in bps) added to the rate in each period rate determination.
    spread_compound_method: SpreadCompoundMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.SpreadCompoundMethod` used in the calculation
        of the period rate when combining a ``float_spread``. Used **only** with RFR type
        ``fixing_method``.
    rate_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        See XXX (working with fixings).
        The value of the rate fixing. If a scalar, is used directly. If a string identifier, links
        to the central ``fixings`` object and data loader.
    leg2_fixed : bool, :green:`optional (set as False)`
    leg2_fixed_rate : float or None
    leg2_fixing_method: FloatFixingMethod, str, :green:`optional (set by 'defaults')`
    leg2_method_param: int, :green:`optional (set by 'defaults')`
    leg2_fixing_frequency: Frequency, str, :green:`optional (set by 'frequency' or '1B')`
    leg2_fixing_series: FloatRateSeries, str, :green:`optional (implied by other parameters)`
    leg2_float_spread: float, Dual, Dual2, Variable, :green:`optional (set as 0.0)`
    leg2_spread_compound_method: SpreadCompoundMethod, str, :green:`optional (set by 'defaults')`
    leg2_rate_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`

       .. note::

          The following are the cross-currency **non-deliverable** parameters.

    fx_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for each *Period* according
        to non-deliverability. This can only be provided if ``leg2_notional`` is given. The
        currency pair is expressed in direction 'currency:leg2_currency'.
    mtm: bool, :green:`optional (set to False)`
        Define the *XCS* is mark-to-market on leg1. Only one leg can be mark-to-market.
    leg2_fx_fixings:
        This can only be provided if ``notional`` is given. The
        currency pair is expressed in direction 'currency:leg2_currency'.
    leg2_mtm: bool, :green:`optional (set to False)`

        .. note::

           The following are **meta parameters**.

    curves : XXX
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.
    metric: str, :green:`optional (set as 'leg1')`
        Determines which calculation metric to return by default when using the
        :meth:`~rateslib.instruments.components.XCS.rate` method.

    Notes
    -----
    A *XCS* is a flexible *Instrument* which can handle either fixed or floating legs, controlled
    by the ``fixed`` and ``leg2_fixed`` arguments. For a mark-to-market *XCS* one of ``mtm`` or
    ``leg2_mtm`` can be set to *True*. By default, non-mark-to-market *XCS* are constructed. Only
    one of ``notional`` or ``leg2_notional`` (and correspondingly ``amortization`` or
    ``leg2_amortization``) must be given depending upon whether the notional is expressed in
    units of ``currency`` or ``leg2_currency`` respectively. The derived notional is handled via
    non-deliverability and either ``leg2_fx_fixings`` or ``fx_fixings`` respectively. These fixings
    are always expressed using an FX rate of direction *'currency:leg2_currency'*. One requirement
    is that if any leg is ``mtm`` then that leg cannot set the defining notional; the notional must
    be set on the non-mtm leg.

    **For example**, we initialise a MTM GBP/USD XCS in £100m. The MTM leg is USD so the notional
    must be expressed on the GBP leg. The pricing spread is applied to the GBP leg.

    .. ipython:: python

       xcs = XCS(
           effective=dt(2025, 1, 8),
           termination="6m",
           frequency="Q",
           currency="gbp",
           notional=100e6,
           leg2_mtm=True,
           leg2_currency="usd",
           leg2_fx_fixings=1.35,
           metric="leg1",
       )

    Or, we initialise a MTM USD/JPY XCS in ¥1bn with ¥100m amortization. The MTM leg is USD so the
    notional must be expressed on the JPY leg. The pricing spread is applied to the JPY leg.

    .. ipython:: python

       xcs = XCS(
           effective=dt(2025, 1, 8),
           termination="6m",
           frequency="Q",
           currency="usd",
           mtm=True,
           fx_fixings=155.0,
           leg2_currency="jpy",
           leg2_notional=1e9,
           leg2_amortization=100e6,
           metric="leg2",
       )
       xcs.cashflows()

    """  # noqa: E501

    _rate_scalar = 1.0

    @property
    def fixed_rate(self) -> DualTypes_:
        """The fixed rate parameter of the composited
        :class:`~rateslib.legs.components.FixedLeg`."""
        if self.kwargs.meta["fixed"]:
            return self.leg1.fixed_rate
        else:
            raise AttributeError(f"Leg1 is of type: {type(self.leg1).__name__}")

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes_) -> None:
        if self.kwargs.meta["fixed"]:
            self.kwargs.leg1["fixed_rate"] = value
            self.leg1.fixed_rate = value
        else:
            raise AttributeError(f"Leg1 is of type: {type(self.leg1).__name__}")

    @property
    def float_spread(self) -> DualTypes:
        """The float spread parameter of the composited
        :class:`~rateslib.legs.components.FloatLeg`."""
        if not self.kwargs.meta["fixed"]:
            return self.leg1.float_spread
        else:
            raise AttributeError(f"Leg1 is of type: {type(self.leg1).__name__}")

    @float_spread.setter
    def float_spread(self, value: DualTypes_) -> None:
        if not self.kwargs.meta["fixed"]:
            self.kwargs.leg1["float_spread"] = value
            self.leg1.float_spread = value
        else:
            raise AttributeError(f"Leg1 is of type: {type(self.leg1).__name__}")

    @property
    def leg2_fixed_rate(self) -> DualTypes_:
        """The float spread parameter of the composited
        :class:`~rateslib.legs.components.FloatLeg`."""
        if self.kwargs.meta["leg2_fixed"]:
            return self.leg2.fixed_rate
        else:
            raise AttributeError(f"Leg2 is of type: {type(self.leg2).__name__}")

    @leg2_fixed_rate.setter
    def leg2_fixed_rate(self, value: DualTypes_) -> None:
        if self.kwargs.meta["leg2_fixed"]:
            self.kwargs.leg2["fixed_rate"] = value
            self.leg2.fixed_rate = value
        else:
            raise AttributeError(f"Leg2 is of type: {type(self.leg2).__name__}")

    @property
    def leg2_float_spread(self) -> DualTypes_:
        """The float spread parameter of the composited
        :class:`~rateslib.legs.components.FloatLeg`."""
        if not self.kwargs.meta["leg2_fixed"]:
            return self.leg2.float_spread
        else:
            raise AttributeError(f"Leg2 is of type: {type(self.leg2).__name__}")

    @leg2_float_spread.setter
    def leg2_float_spread(self, value: DualTypes) -> None:
        if not self.kwargs.meta["leg2_fixed"]:
            self.kwargs.leg2["float_spread"] = value
            self.leg2.float_spread = value
        else:
            raise AttributeError(f"Leg2 is of type: {type(self.leg2).__name__}")

    @property
    def leg1(self) -> FixedLeg | FloatLeg:
        """The first :class:`~rateslib.legs.components.FixedLeg` or
        :class:`~rateslib.legs.components.FloatLeg` of the *Instrument*."""
        return self._leg1

    @property
    def leg2(self) -> FixedLeg | FloatLeg:
        """The second :class:`~rateslib.legs.components.FixedLeg` or
        :class:`~rateslib.legs.components.FloatLeg` of the *Instrument*."""
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
        leg2_currency: str_ = NoInput(0),
        leg2_notional: float_ = NoInput(0),
        leg2_amortization: float_ = NoInput(0),
        # rate parameters
        fixed: bool = False,
        mtm: bool = False,
        fixed_rate: DualTypes_ = NoInput(0),
        float_spread: DualTypes_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        rate_fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        fixing_frequency: Frequency | str_ = NoInput(0),
        fixing_series: FloatRateSeries | str_ = NoInput(0),
        fx_fixings: LegFixings = NoInput(0),
        leg2_fixed: bool = False,
        leg2_mtm: bool = False,
        leg2_fixed_rate: DualTypes_ = NoInput(0),
        leg2_float_spread: DualTypes_ = NoInput(0),
        leg2_spread_compound_method: str_ = NoInput(0),
        leg2_rate_fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        leg2_fixing_method: str_ = NoInput(0),
        leg2_method_param: int_ = NoInput(0),
        leg2_fixing_frequency: Frequency | str_ = NoInput(0),
        leg2_fixing_series: FloatRateSeries | str_ = NoInput(0),
        leg2_fx_fixings: LegFixings = NoInput(0),
        # meta parameters
        curves: Curves_ = NoInput(0),
        spec: str_ = NoInput(0),
        metric: str = "leg1",
    ) -> None:
        if isinstance(notional, NoInput) and isinstance(leg2_notional, NoInput):
            notional = defaults.notional

        # validation
        if mtm and leg2_mtm:
            raise ValueError("`mtm` and `leg2_mtm` must define at most one MTM leg.")
        if not isinstance(notional, NoInput) and not isinstance(leg2_notional, NoInput):
            raise ValueError(
                "The `notional` can only be provided on one leg, expressed in its `currency`.\n"
                "The other leg's cashflows are derived via `fx_fixings` and non-deliverability."
            )

        if not isinstance(notional, NoInput) and not isinstance(fx_fixings, NoInput):
            raise ValueError(
                "When `notional` is given only `leg2_fx_fixings` are required to derive "
                "cashflows on leg2 via non-deliverability."
            )
        if not isinstance(leg2_notional, NoInput) and not isinstance(leg2_fx_fixings, NoInput):
            raise ValueError(
                "When `leg2_notional` is given only `fx_fixings` are required to derive "
                "cashflows on leg1 via non-deliverability."
            )

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
            leg2_currency=leg2_currency,
            notional=notional,
            leg2_notional=leg2_notional,
            amortization=amortization,
            leg2_amortization=leg2_amortization,
            # non-deliverability
            fx_fixings=fx_fixings,
            leg2_fx_fixings=leg2_fx_fixings,
            mtm=mtm,
            leg2_mtm=leg2_mtm,
            # rate
            fixed_rate=fixed_rate,
            float_spread=float_spread,
            spread_compound_method=spread_compound_method,
            rate_fixings=rate_fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            fixing_frequency=fixing_frequency,
            fixing_series=fixing_series,
            leg2_fixed_rate=leg2_fixed_rate,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_rate_fixings=leg2_rate_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
            leg2_fixing_frequency=leg2_fixing_frequency,
            leg2_fixing_series=leg2_fixing_series,
            # meta
            fixed=fixed,
            leg2_fixed=leg2_fixed,
            curves=self._parse_curves(curves),
            metric=metric.lower(),
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            initial_exchange=True,
            final_exchange=True,
            leg2_initial_exchange=True,
            leg2_final_exchange=True,
        )

        default_args = dict(
            currency=defaults.base_currency,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            payment_lag_exchange=defaults.payment_lag_exchange,
        )

        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "metric", "fixed", "leg2_fixed"],
        )

        # narrowing of fixed or floating
        float_attrs = [
            "float_spread",
            "spread_compound_method",
            "rate_fixings",
            "fixing_method",
            "method_param",
            "fixing_frequency",
            "fixing_series",
        ]
        if fixed:
            for item in float_attrs:
                self.kwargs.leg1.pop(item)
        else:
            self.kwargs.leg1.pop("fixed_rate")
        if leg2_fixed:
            for item in float_attrs:
                self.kwargs.leg2.pop(item)
        else:
            self.kwargs.leg2.pop("fixed_rate")

        # populate non-deliverable leg, based on which leg notional is given
        if isinstance(self.kwargs.leg1["notional"], NoInput):
            self._kwargs.leg1["notional"] = -1.0 * self._kwargs.leg2["notional"]
            self._kwargs.leg1["amortization"] = (
                NoInput(0)
                if isinstance(self._kwargs.leg2["amortization"], NoInput)
                else -1.0 * self._kwargs.leg2["amortization"]
            )
            self._kwargs.leg1["pair"] = (
                f"{self._kwargs.leg1['currency']}{self._kwargs.leg2['currency']}"
            )
        if isinstance(self.kwargs.leg2["notional"], NoInput):
            self._kwargs.leg2["notional"] = -1.0 * self._kwargs.leg1["notional"]
            self._kwargs.leg2["amortization"] = (
                NoInput(0)
                if isinstance(self._kwargs.leg1["amortization"], NoInput)
                else -1.0 * self._kwargs.leg1["amortization"]
            )
            self._kwargs.leg2["pair"] = (
                f"{self._kwargs.leg1['currency']}{self._kwargs.leg2['currency']}"
            )

        if fixed:
            self._leg1 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        else:
            self._leg1 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        if leg2_fixed:
            self._leg2 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg2, 1))
        else:
            self._leg2 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg2, 1))
        self._legs = [self.leg1, self.leg2]

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
        metric_ = _drb(self.kwargs.meta["metric"], metric)

        fx_ = _get_fx_maybe_from_solver(fx=fx, solver=solver)
        leg2_rate_curve = _get_maybe_curve_maybe_from_solver(
            self.kwargs.meta["curves"], _curves, "leg2_rate_curve", solver
        )
        leg2_disc_curve = _get_maybe_curve_maybe_from_solver(
            self.kwargs.meta["curves"], _curves, "leg2_disc_curve", solver
        )
        rate_curve = _get_maybe_curve_maybe_from_solver(
            self.kwargs.meta["curves"], _curves, "rate_curve", solver
        )
        disc_curve = _get_maybe_curve_maybe_from_solver(
            self.kwargs.meta["curves"], _curves, "disc_curve", solver
        )

        if metric_ == "leg1":
            leg2_npv: DualTypes = self.leg2.npv(
                rate_curve=leg2_rate_curve,
                disc_curve=leg2_disc_curve,
                base=self.leg1.settlement_params.currency,
                fx=fx_,
                settlement=settlement,
                forward=forward,
            )
            spread = self.leg1.spread(
                target_npv=-leg2_npv,
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                settlement=settlement,
                fx=fx_,
                forward=forward,
            )
            if self.kwargs.meta["fixed"]:
                return spread / 100.0
            else:
                return spread
        elif metric_ == "leg2":
            leg1_npv: DualTypes = self.leg1.npv(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                base=self.leg2.settlement_params.currency,
                fx=fx_,
                settlement=settlement,
                forward=forward,
            )
            spread = self.leg2.spread(
                target_npv=-leg1_npv,
                rate_curve=leg2_rate_curve,
                disc_curve=leg2_disc_curve,
                settlement=settlement,
                forward=forward,
                fx=fx_,
            )
            if self.kwargs.meta["leg2_fixed"]:
                return spread / 100.0
            else:
                return spread
        else:
            raise ValueError("`metric` must be in {'leg1', 'leg2'}")

    # def _set_rate(self, value: DualTypes, leg: int) -> DualTypes:
    #     if leg == 1:
    #         if self.kwargs.meta["fixed"]:
    #             ret = self.leg1.fixed_rate
    #             self.leg1.fixed_rate = value
    #         else:
    #             ret = self.leg1.float_spread
    #             self.leg1.float_spread = value
    #     else:  # leg 2
    #         if self.kwargs.meta["leg2_fixed"]:
    #             ret = self.leg2.fixed_rate
    #             self.leg2.fixed_rate = value
    #         else:
    #             ret = self.leg2.float_spread
    #             self.leg2.float_spread = value
    #     return ret

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
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        return self.rate(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            settlement=settlement,
            forward=forward,
            metric=metric,
        )

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
        self._set_pricing_mid(
            curves=curves,
            solver=solver,
            settlement=settlement,
            forward=forward,
            fx=fx,
        )
        return super().npv(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            local=local,
            settlement=settlement,
            forward=forward,
        )

    def _set_pricing_mid(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> None:
        # all float_spread are assumed to be equal to zero if not given.
        # missing fixed rates will be priced and set if possible.

        if self.kwargs.meta["fixed"] and isinstance(self.kwargs.leg1["fixed_rate"], NoInput):
            if self.kwargs.meta["leg2_fixed"] and isinstance(
                self.kwargs.leg2["fixed_rate"], NoInput
            ):
                raise ValueError("At least one leg must have a defined `fixed_rate`.")

            mid_price = self.rate(
                curves=curves,
                solver=solver,
                fx=fx,
                settlement=settlement,
                forward=forward,
                metric="leg1",
            )
            self.leg1.fixed_rate = _dual_float(mid_price)

        elif self.kwargs.meta["leg2_fixed"] and isinstance(self.kwargs.leg2["fixed_rate"], NoInput):
            # leg1 cannot be fixed with NoInput
            mid_price = self.rate(
                curves=curves,
                solver=solver,
                fx=fx,
                settlement=settlement,
                forward=forward,
                metric="leg2",
            )
            self.leg2.fixed_rate = _dual_float(mid_price)

    def _parse_curves(self, curves: CurveOption_) -> _Curves:
        """
        An IRS has two curve requirements: a leg2_rate_curve and a disc_curve used by both legs.

        When given as only 1 element this curve is applied to all of the those components

        When given as 2 elements the first is treated as the rate curve and the 2nd as disc curve.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
            return _Curves(
                rate_curve=curves.get("rate_curve", NoInput(0)),
                disc_curve=curves.get("disc_curve", NoInput(0)),
                leg2_rate_curve=curves.get("leg2_rate_curve", NoInput(0)),
                leg2_disc_curve=curves.get("leg2_disc_curve", NoInput(0)),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 4:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                    leg2_rate_curve=curves[2],
                    leg2_disc_curve=curves[3],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires 4 curve types. Got {len(curves)}."
                )
        else:  # `curves` is just a single input which is copied across all curves
            raise ValueError(f"{type(self).__name__} requires 4 curve types. Got 1.")

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
