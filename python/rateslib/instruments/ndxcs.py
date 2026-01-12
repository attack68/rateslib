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
        Sequence,
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


class NDXCS(_BaseInstrument):
    """
    A *non-deliverable cross-currency swap (XCS)* composing either
    :class:`~rateslib.legs.FixedLeg`
    and/or :class:`~rateslib.legs.FloatLeg` in different currencies.

    .. rubric:: Examples

    An INR NDXCS vs SOFR (IRUSON5 Curncy)

    .. ipython:: python
       :suppress:

       from rateslib.instruments import NDXCS
       from datetime import datetime as dt
       from rateslib import fixings
       from pandas import Series

    .. ipython:: python

       fixings.add("WMR_10AM_TY0_USDINR", Series(index=[dt(2025, 1, 8), dt(2025, 7, 4)], data=[92.0, 92.5]))
       ndxcs = NDXCS(
           effective=dt(2025, 1, 8),
           termination="1y",
           frequency="S",
           currency="usd",
           pair="usdinr",
           notional=5e6,           # <- INR Leg
           fixed=True,
           fx_fixings="WMR_10AM_TY0",
           leg2_fx_fixings=91.55,  # <- USD Notional at execution
           payment_lag=0,
       )
       ndxcs.cashflows()

    .. rubric:: Pricing

    The methods of a *NDXCS* require an :class:`~rateslib.fx.FXForwards` object for ``fx`` .

    They also require a *disc curve* for discounting both legs in the *settlement currency*
    and (if not *FixedLegs*) a *rate curve* and a *leg2 rate curve* for forecasting the floating
    rates on either *Leg*. The following input formats are allowed:

    .. code-block:: python

       curves = [rate_curve, disc_curve, leg2_rate_curve, disc_curve]  # four curves
       curves = {  # dict form is explicit
           "rate_curve": rate_curve,
           "disc_curve": disc_curve,
           "leg2_rate_curve": leg2_rate_curve,
       }

    The available pricing ``metric`` are in *{'leg1', 'leg2'}* which will return a *float spread*
    or a *fixed rate* on the specified leg, for the appropriate *Leg* type.

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

           The following are the **non-deliverability parameters**

    pair: str, :red:`required (if 'leg2_pair' not given)`
        The currency pair for :class:`~rateslib.data.fixings.FXFixing` that determines *Period*
        settlement on *Leg1*. The *reference currency* is implied from ``pair``.
        Must include ``currency``. Not required if this leg is not *non-deliverable*.
    leg2_pair: str, :green:`optional`
        The currency pair for :class:`~rateslib.data.fixings.FXFixing` that determines *Period*
        settlement on *Leg2*. Not required if not a 3-currency NDXCS.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for each *Period* according
        to non-deliverability. Not required if this leg is not *non-deliverable*.
    leg2_fx_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for each *Period* on *Leg2*
        according to non-deliverability. Not required if this leg is not *non-deliverable*.

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
        See :ref:`Fixings <fixings-doc>`.
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

          The following are the cross-currency **non-deliverable** parameters. For
          further details and examples see **Notes**.

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

    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.
    metric: str, :green:`optional (set as 'leg1')`
        Determines which calculation metric to return by default when using the
        :meth:`~rateslib.instruments.XCS.rate` method.

    Notes
    -----
    A non-deliverable *XCS* replicates a non-mtm cross-currency swap
    whose cashflows are paid out only in one *settlement currency*. This type of swap
    allows two configurations;

    - A **two currency** *NDXCS* where one leg is based on a *reference currency* and
      difference *settlement currency*, whilst the other leg is based purely on cashflows
      generated in the *settlement currency*.
    - A **three currency** *NDXCS* where one leg is based on *reference currency 1* with a
      *settlement currency* and the other leg is based on *reference currency 2* but
      also settling in *settlement currency*.

    .. tabs::

       .. tab:: Two Currency NDXCS

          The **required** parameters of a two currency NDXCS are as follows;

          - A ``currency`` which defines the *settlement currency* on both legs.
          - A ``pair`` which defines the currency pair and implicitly determines the *reference currency*.
          - A ``notional`` or ``leg2_notional``. The placement of the notional defines which *Leg* is the
            one that is based on the *reference currency*. Any notional quantity must be given in units of
            *reference currency*.
          - ``fx_fixings`` and ``leg2_fx_fixings``. These are FX fixings that are used by both legs;
            one leg will have a fixed rate of exchange for all periods (a single entry usually determined
            when the transaction is agreed), the other leg with base its ND FX Fixings on some future
            data series.

          This example swaps a 500mm INR *FloatLeg* non-deliverable into USD into a USD *FloatLeg*
          with an initially agreed FX rate of USDINR 92.0

          .. ipython:: python

             ndxcs = NDXCS(
                 effective=dt(2026, 1, 1),
                 termination="18M",
                 frequency="S",
                 currency="usd",          #  <-  USD settlement currency
                 pair="usdinr",           #  <-  INR reference currency implied
                 notional=500e6,          #  <-  Leg1 is based on the reference currency
                 fx_fixings="WMR_10AM_TY0",
                 leg2_fx_fixings=92.0,    #  <-  The USD Leg notional is implied as 5.43mm
             )
             ndxcs.cashflows()

          The *Leg* based on the *reference currency* is a non-deliverable *Leg* with a ``mtm``
          parameter set to *True*, whilst the other *Leg* is non-deliverable with ``mtm`` set to
          *False* and is based on one single FX rate.

       .. tab:: Three Currency NDXCS

          The **required** parameters of a three currency NDXCS are as follows;

          - A ``currency`` which defines the *settlement currency* on both legs.
          - A ``pair`` which defines the currency pair and implicitly determines
            the *reference currency 1*.
          - A ``leg2_pair`` which defines the currency pair of *Leg2* and implicitly
            determines the *reference currency 2*.
          - A ``notional`` and ``leg2_notional``. These must be pre-determined at an appropriate
            rate of exchange, usually this is agreed at transaction execution. These must be
            expressed in *reference currency 1* units and *reference currency 2* units
            respectively.
          - ``fx_fixings`` and ``leg2_fx_fixings`` which determine the future rates of exchange
            on both non-deliverable legs.

          This example swaps a 500mm INR *FloatLeg* non-deliverable into USD into a CHF *FloatLeg*
          non-deliverable into USD with an initial FX rate of CHFINR 125.0.

          .. ipython:: python

             fixings.add("WMR_10AM_TY0_USDCHF", Series(index=[dt(2025, 1, 6)], data=[0.9]))
             ndxcs = NDXCS(
                 effective=dt(2026, 1, 1),
                 termination="18M",
                 frequency="S",
                 currency="usd",               #  <-  USD settlement currency
                 pair="usdinr",                #  <-  INR reference currency 1 implied
                 leg2_pair="usdchf",           #  <-  CHF reference currency 2 implied
                 notional=500e6,               #  <-  Leg1 is based on the reference currency 1
                 leg2_notional=500e6/125.0,    #  <-  Leg2 entered directly in ref currency 2 units
                 fx_fixings="WMR_10AM_TY0",       #  <-  Data series tag for FXFixings on Leg1
                 leg2_fx_fixings="WMR_10AM_TY0",  #  <-  Data series tag for FXFixings on Leg2
             )
             ndxcs.cashflows()

          Both *Legs* are non-deliverable with their ``mtm`` parameters set to *True*.

    .. ipython:: python
       :suppress:

       fixings.pop("WMR_10AM_TY0_USDINR")
       fixings.pop("WMR_10AM_TY0_USDCHF")

    """  # noqa: E501

    def _rate_scalar_calc(self) -> float:
        if self.kwargs.meta["metric"] == "leg1":
            return 1.0 if isinstance(self.leg1, FixedLeg) else 100.0
        else:
            return 1.0 if isinstance(self.leg2, FixedLeg) else 100.0

    @property
    def fixed_rate(self) -> DualTypes_:
        """The fixed rate parameter of the composited
        :class:`~rateslib.legs.FixedLeg`."""
        if isinstance(self.leg1, FixedLeg):
            return self.leg1.fixed_rate
        else:
            raise AttributeError(f"Leg1 is of type: {type(self.leg1).__name__}")

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes_) -> None:
        if isinstance(self.leg1, FixedLeg):
            self.kwargs.leg1["fixed_rate"] = value
            self.leg1.fixed_rate = value
        else:
            raise AttributeError(f"Leg1 is of type: {type(self.leg1).__name__}")

    @property
    def float_spread(self) -> DualTypes:
        """The float spread parameter of the composited
        :class:`~rateslib.legs.FloatLeg`."""
        if isinstance(self.leg1, FloatLeg):
            return self.leg1.float_spread
        else:
            raise AttributeError(f"Leg1 is of type: {type(self.leg1).__name__}")

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        if isinstance(self.leg1, FloatLeg):
            self.kwargs.leg1["float_spread"] = value
            self.leg1.float_spread = value
        else:
            raise AttributeError(f"Leg1 is of type: {type(self.leg1).__name__}")

    @property
    def leg2_fixed_rate(self) -> DualTypes_:
        """The float spread parameter of the composited
        :class:`~rateslib.legs.FloatLeg`."""
        if isinstance(self.leg2, FixedLeg):
            return self.leg2.fixed_rate
        else:
            raise AttributeError(f"Leg2 is of type: {type(self.leg2).__name__}")

    @leg2_fixed_rate.setter
    def leg2_fixed_rate(self, value: DualTypes_) -> None:
        if isinstance(self.leg2, FixedLeg):
            self.kwargs.leg2["fixed_rate"] = value
            self.leg2.fixed_rate = value
        else:
            raise AttributeError(f"Leg2 is of type: {type(self.leg2).__name__}")

    @property
    def leg2_float_spread(self) -> DualTypes_:
        """The float spread parameter of the composited
        :class:`~rateslib.legs.FloatLeg`."""
        if isinstance(self.leg2, FloatLeg):
            return self.leg2.float_spread
        else:
            raise AttributeError(f"Leg2 is of type: {type(self.leg2).__name__}")

    @leg2_float_spread.setter
    def leg2_float_spread(self, value: DualTypes) -> None:
        if isinstance(self.leg2, FloatLeg):
            self.kwargs.leg2["float_spread"] = value
            self.leg2.float_spread = value
        else:
            raise AttributeError(f"Leg2 is of type: {type(self.leg2).__name__}")

    @property
    def leg1(self) -> FixedLeg | FloatLeg:
        """The first :class:`~rateslib.legs.FixedLeg` or
        :class:`~rateslib.legs.FloatLeg` of the *Instrument*."""
        return self._leg1

    @property
    def leg2(self) -> FixedLeg | FloatLeg:
        """The second :class:`~rateslib.legs.FixedLeg` or
        :class:`~rateslib.legs.FloatLeg` of the *Instrument*."""
        return self._leg2

    @property
    def legs(self) -> Sequence[_BaseLeg]:
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
        leg2_notional: float_ = NoInput(0),
        leg2_amortization: float_ = NoInput(0),
        # nondeliverable params
        pair: str_ = NoInput(0),
        leg2_pair: str_ = NoInput(0),
        fx_fixings: LegFixings = NoInput(0),
        leg2_fx_fixings: LegFixings = NoInput(0),
        # rate parameters
        fixed: bool_ = NoInput(0),
        fixed_rate: DualTypes_ = NoInput(0),
        float_spread: DualTypes_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        rate_fixings: LegFixings = NoInput(0),
        fixing_method: str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        fixing_frequency: Frequency | str_ = NoInput(0),
        fixing_series: FloatRateSeries | str_ = NoInput(0),
        leg2_fixed: bool_ = NoInput(0),
        leg2_mtm: bool_ = NoInput(0),
        leg2_fixed_rate: DualTypes_ = NoInput(0),
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
        metric: str_ = NoInput(0),
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
            leg2_pair=leg2_pair,
            fx_fixings=fx_fixings,
            leg2_fx_fixings=leg2_fx_fixings,
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
            metric=metric,
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            leg2_currency=NoInput(1),
            initial_exchange=True,
            final_exchange=True,
            leg2_initial_exchange=True,
            leg2_final_exchange=True,
            vol=_Vol(),
        )
        default_args = dict(
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            payment_lag_exchange=defaults.payment_lag_exchange,
            currency=defaults.base_currency,
            fixed=False,
            leg2_fixed=False,
            metric="leg1",
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "metric", "fixed", "leg2_fixed", "vol"],
        )

        # validation of currencies and pairs
        if isinstance(self.kwargs.leg1["notional"], NoInput) and isinstance(
            self.kwargs.leg2["notional"], NoInput
        ):
            self.kwargs.leg1["notional"] = defaults.notional

        (
            self.kwargs.leg1["mtm"],
            self.kwargs.leg2["mtm"],
            self.kwargs.leg1["pair"],
            self.kwargs.leg2["pair"],
        ) = self._init_args(
            currency=self.kwargs.leg1["currency"].lower(),
            pair=self.kwargs.leg1["pair"],
            leg2_pair=self.kwargs.leg2["pair"],
            notional=self.kwargs.leg1["notional"],
            leg2_notional=self.kwargs.leg2["notional"],
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
        if self.kwargs.meta["fixed"]:
            for item in float_attrs:
                self.kwargs.leg1.pop(item)
        else:
            self.kwargs.leg1.pop("fixed_rate")
        if self.kwargs.meta["leg2_fixed"]:
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
        if isinstance(self.kwargs.leg2["notional"], NoInput):
            self._kwargs.leg2["notional"] = -1.0 * self._kwargs.leg1["notional"]
            self._kwargs.leg2["amortization"] = (
                NoInput(0)
                if isinstance(self._kwargs.leg1["amortization"], NoInput)
                else -1.0 * self._kwargs.leg1["amortization"]
            )

        if self.kwargs.meta["fixed"]:
            self._leg1: FixedLeg | FloatLeg = FixedLeg(
                **_convert_to_schedule_kwargs(self.kwargs.leg1, 1)
            )
        else:
            self._leg1 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        if self.kwargs.meta["leg2_fixed"]:
            self._leg2: FixedLeg | FloatLeg = FixedLeg(
                **_convert_to_schedule_kwargs(self.kwargs.leg2, 1)
            )
        else:
            self._leg2 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg2, 1))
        self._legs = [self.leg1, self.leg2]
        self._rate_scalar = self._rate_scalar_calc()

    def _init_args(
        self,
        currency: str,
        pair: str_,
        leg2_pair: str_,
        notional: DualTypes_,
        leg2_notional: DualTypes_,
    ) -> tuple[LegMtm, LegMtm, str, str]:
        if isinstance(pair, NoInput):
            raise ValueError("`pair` must be given when creating a NDXCS.")
        else:
            pair_: str = pair.lower()
            if currency not in pair_:
                raise ValueError(f"`pair` must contain {currency}.")

        if isinstance(leg2_pair, str):
            leg2_pair_: str = leg2_pair.lower()
            if currency not in leg2_pair_:
                raise ValueError(f"`leg2_pair` must contain {currency}.")
            return NDXCS._init_three_currency(pair_, leg2_pair_, notional, leg2_notional)
        else:
            return NDXCS._init_two_currency(pair_, notional, leg2_notional)

    @staticmethod
    def _init_two_currency(
        pair: str,
        notional: DualTypes_,
        leg2_notional: DualTypes_,
    ) -> tuple[LegMtm, LegMtm, str, str]:
        if isinstance(notional, NoInput):
            # then reference Leg is leg2
            mtm, leg2_mtm = LegMtm.Initial, LegMtm.Payment
        else:
            if not isinstance(leg2_notional, NoInput):
                raise ValueError(
                    "Only one of `notional` or `leg2_notional` can be given for a two-currency "
                    "NDXCS.\nIf you are trying to set either notional based on a transacted "
                    "FX rate, then:\n1) Set the notional in reference currency units on the "
                    "reference currency leg.\n2) Set the ``fx_fixing`` or ``leg2_fx_fixing`` value "
                    "as this scalar for the leg that is solely based on the settlement currency."
                )
            mtm, leg2_mtm = LegMtm.Payment, LegMtm.Initial
        return mtm, leg2_mtm, pair, pair

    @staticmethod
    def _init_three_currency(
        pair: str,
        leg2_pair: str,
        notional: DualTypes_,
        leg2_notional: DualTypes_,
    ) -> tuple[LegMtm, LegMtm, str, str]:
        if isinstance(notional, NoInput) or isinstance(leg2_notional, NoInput):
            raise ValueError(
                "A three-currency NDXCS requires both `notional` and `leg2_notional` to be given.\n"
                "These should be given in their relevant reference currencies, according to the "
                "initially agreed FX Rate between them."
            )
        return LegMtm.Payment, LegMtm.Payment, pair, leg2_pair

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
        metric_ = _drb(self.kwargs.meta["metric"], metric)

        fx_ = _get_fx_forwards_maybe_from_solver(fx=fx, solver=solver)
        leg2_rate_curve = _maybe_get_curve_or_dict_maybe_from_solver(
            self.kwargs.meta["curves"], _curves, "leg2_rate_curve", solver
        )
        leg2_disc_curve = _maybe_get_curve_maybe_from_solver(
            self.kwargs.meta["curves"], _curves, "leg2_disc_curve", solver
        )
        rate_curve = _maybe_get_curve_or_dict_maybe_from_solver(
            self.kwargs.meta["curves"], _curves, "rate_curve", solver
        )
        disc_curve = _maybe_get_curve_maybe_from_solver(
            self.kwargs.meta["curves"], _curves, "disc_curve", solver
        )

        if metric_ == "leg1":
            leg2_npv: DualTypes = self.leg2.npv(  # type: ignore[assignment]
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
            leg1_npv: DualTypes = self.leg1.npv(  # type: ignore[assignment]
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
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        return self.rate(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            settlement=settlement,
            forward=forward,
            metric=metric,
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
            fx=fx,
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
        # all float_spread are assumed to be equal to zero if not given.
        # missing fixed rates will be priced and set if possible.

        if isinstance(self.leg1, FixedLeg) and isinstance(self.kwargs.leg1["fixed_rate"], NoInput):
            if isinstance(self.leg2, FixedLeg) and isinstance(
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

        elif isinstance(self.leg2, FixedLeg) and isinstance(
            self.kwargs.leg2["fixed_rate"], NoInput
        ):
            # leg1 cannot be fixed with NoInput - this branch is covered above
            mid_price = self.rate(
                curves=curves,
                solver=solver,
                fx=fx,
                settlement=settlement,
                forward=forward,
                metric="leg2",
            )
            self.leg2.fixed_rate = _dual_float(mid_price)

        elif (
            isinstance(self.leg1, FloatLeg)
            and isinstance(self.kwargs.leg1["float_spread"], NoInput)
            and isinstance(self.leg2, FloatLeg)
            and isinstance(self.kwargs.leg2["float_spread"], NoInput)
        ):
            # then no FloatLeg pricing parameters are provided
            mid_price = self.rate(
                curves=curves,
                solver=solver,
                fx=fx,
                settlement=settlement,
                forward=forward,
            )
            if self.kwargs.meta["metric"].lower() == "leg1":
                self.leg1.float_spread = _dual_float(mid_price)
            else:
                self.leg2.float_spread = _dual_float(mid_price)

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        A XCS requires 4 curves (mostly if float-float, otherwise it needs 2)
        """
        if isinstance(curves, NoInput):
            return _Curves()
        elif isinstance(curves, dict):
            return _Curves(
                rate_curve=curves.get("rate_curve", NoInput(0)),
                disc_curve=curves.get("disc_curve", NoInput(0)),
                leg2_rate_curve=curves.get("leg2_rate_curve", NoInput(0)),
                leg2_disc_curve=curves.get("leg2_disc_curve", NoInput(0)),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 4:
                return _Curves(
                    rate_curve=NoInput(0) if curves[0] is None else curves[0],
                    disc_curve=curves[1],
                    leg2_rate_curve=NoInput(0) if curves[2] is None else curves[2],
                    leg2_disc_curve=curves[3],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires 4 curve type input. Got {len(curves)}."
                )
        elif isinstance(curves, _Curves):
            return curves
        else:
            raise ValueError(f"{type(self).__name__} requires 4 curve type input. Got 1.")

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
