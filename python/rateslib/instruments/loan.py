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
from rateslib.dual import ift_1dim
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import LegMtm
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _get_curve,
    _parse_curves,
    _Vol,
)
from rateslib.legs import FixedLeg, FloatLeg
from rateslib.scheduling import Frequency

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Adjuster,
        CalInput,
        Convention,
        CurvesT_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FixingsRates_,
        FloatRateSeries,
        FXForwards_,
        IndexMethod,
        LegFixings,
        LegIndexBase,
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


class Loan(_BaseInstrument):
    """
    A *loan obligation* composing either a :class:`~rateslib.legs.FixedLeg` or a
    :class:`~rateslib.legs.FloatLeg`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import Loan
       from datetime import datetime as dt

    .. ipython:: python

       loan = Loan(dt(2022, 1, 4), "3m", "Q", notional=10e6, fixed_rate=10.0, calendar="nyc")
       loan.cashflows()

    .. rubric:: Pricing

    A *Loan* with a fixed rate requires one *disc curve* for discounting.
    A *Loan* with a floating rate may require an additional *rate curve* for forecasting if
    rate fixings have not been published.
    A *Loan* that is indexed may require an additional *index curve*.

    .. code-block:: python

       curves = curve | [curve]           #  a single curve is repeated for all required curves
       curves = [rate_curve, disc_curve]  #  two curves given in the specified order
       curves = [rate_curve, disc_curve, index_curve]  #  three curves given in the specified order
       curves = {  # dict form is explicit
           "rate_curve": rate_curve,
           "disc_curve": disc_curve,
           "index_curve": index_curve,
       }

    The *rate* method is not generally implemented for a *Loan*. However, for flexibility,
    one ``metric`` that is available:

    - *'npv'*: returns the result of the :meth:`~rateslib.instruments.Fee.npv` method.

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
        The local settlement currency of leg1 (3-digit code).
    notional : float, Dual, Dual2, Variable, :green:`optional (set from 'leg2_notional' or 'defaults' )`
        The initial leg1 notional, defined in units of the currency of the leg. Only one
        of ``notional`` and ``leg2_notional`` can be given. The alternate leg notional is derived
        via non-deliverability :class:`~rateslib.data.fixings.FXFixing`.
    amortization: float, Dual, Dual2, Variable, str, Amortization, :green:`optional (set as zero)`
        Set a non-constant notional per *Period*. If a scalar value, adjusts the ``notional`` of
        each successive period by that same value. Should have
        sign equal to that of notional if the notional is to reduce towards zero.

        .. note::

           The following are **rate parameters**.

    fixed : bool, :green:`optional (set as True)`
        Whether leg1 is a :class:`~rateslib.legs.FixedLeg` or a :class:`~rateslib.legs.FloatLeg`.
    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.
    fixing_method: FloatFixingMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.FloatFixingMethod` describing the determination
        of the floating rate for each period.
    fixing_frequency: Frequency, str, :green:`optional (set by 'frequency' or '1B')`
        The :class:`~rateslib.scheduling.Frequency` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given is assumed to match the
        frequency of the schedule for an IBOR type ``fixing_method`` or '1B' if RFR type.
    fixing_series: FloatRateSeries, str, :green:`optional (implied by other parameters)`
        The :class:`~rateslib.data.fixings.FloatRateSeries` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given inherits attributes given
        such as the ``calendar``, ``convention``, ``fixing_method`` etc.
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

    .. note::

           The following define **non-deliverability** parameters. If the fee is
           directly deliverable do not use these parameters.

    pair: FXIndex, str, :green:`optional`
        The currency pair for :class:`~rateslib.data.fixings.FXFixing` that determines *Period*
        settlement. The *reference currency* is implied from ``pair``. Must include ``currency``.
    mtm: bool, :green:`optional (set to False)`
        If *True* use non-deliverability defined by payment date, else use non-deliverability
        defined by a single fixing related to the effective date.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` according
        to non-deliverability.

        .. note::

           The following parameters define **indexation**. The *Period* will be considered
           indexed if any of ``index_method``, ``index_lag``, ``index_base``, ``index_fixings``
           are given.

    index_method : IndexMethod, str, :green:`optional (set by 'defaults')`
        The interpolation method, or otherwise, to determine index values from reference dates.
    index_lag: int, :green:`optional (set by 'defaults')`
        The indexation lag, in months, applied to the determination of index values.
    index_base: float, Dual, Dual2, Variable, :green:`optional`
        The specific value set of the base index value.
        If not given and ``index_fixings`` is a str fixings identifier that will be
        used to determine the base index value.
    index_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The index value for the reference date.
        If a scalar value this is used directly. If a string identifier will link to the
        central ``fixings`` object and data loader. See :ref:`fixings <fixings-doc>`.
    index_base_type: LegIndexBase, :green:`optional (set as 'initial')`
        A parameter to define how the ``index_base_date`` is set on each period. See notes.

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
        :meth:`~rateslib.instruments.Loan.rate` method.

    Notes
    -----
    How does a :class:`~rateslib.instruments.Loan` compare with
    a :class:`~rateslib.instruments.FixedRateBond` or :class:`~rateslib.instruments.FloatRateNote`?
    All of these *Instruments* consist of a single *Leg* with interest payments.
    However, a :class:`~rateslib.instruments.Loan` is modeled with its initial cashflow and final
    cashflow, whilst the :class:`~rateslib.instruments.FixedRateBond` and
    :class:`~rateslib.instruments.FloatRateNote` do not include their initial cashflow.

    This is a conceptual choice. *Bonds* typically trade in the primary and secondary market and
    therefore the initial cashflow, for the purchase of the security, is a transactional
    quantity based or price or YTM. Due to this variation the initial cashflow is excluded
    from a *Bonds* cashflow representation.

    *Loans* are *Instruments* that are considered to be accounting entries,
    so the initial cashflow is usually well defined between two counterparties, and is therefore
    included.

    **Indexation**

    The loan payments can be based on some indexed quantity. The ``index_base_date``
    for each payment will be set according to ``index_base_type``, and follows the
    logic applied to a :class:`~rateslib.legs.FixedLeg`.


    """  # noqa: E501

    _rate_scalar = 1.0

    @property
    def leg1(self) -> FixedLeg | FloatLeg:
        """The :class:`~rateslib.legs.FixedLeg` or :class:`~rateslib.legs.FloatLeg`
        of the *Instrument*."""
        return self._leg1

    @property
    def legs(self) -> list[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs  # type: ignore[return-value]

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
        modifier: Adjuster | str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: Adjuster | str | int_ = NoInput(0),
        payment_lag_exchange: Adjuster | str | int_ = NoInput(0),
        ex_div: Adjuster | str | int_ = NoInput(0),
        convention: Convention | str_ = NoInput(0),
        # settlement parameters
        currency: str_ = NoInput(0),
        notional: float_ = NoInput(0),
        amortization: float_ = NoInput(0),
        # rate parameters
        fixed: bool_ = NoInput(0),
        fixed_rate: DualTypes_ = NoInput(0),
        float_spread: DualTypes_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        rate_fixings: FixingsRates_ = NoInput(0),
        fixing_method: str_ = NoInput(0),
        fixing_frequency: Frequency | str_ = NoInput(0),
        fixing_series: FloatRateSeries | str_ = NoInput(0),
        # # non-deliverability
        pair: str_ = NoInput(0),
        fx_fixings: LegFixings = NoInput(0),
        mtm: bool_ = NoInput(0),
        # index params
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: LegFixings = NoInput(0),
        index_base_type: LegIndexBase | str_ = NoInput(0),
        # meta parameters
        metric: str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        spec: str_ = NoInput(0),
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
            # non-deliverability
            pair=pair,
            fx_fixings=fx_fixings,
            mtm=mtm,
            # indexation
            index_base=index_base,
            index_lag=index_lag,
            index_method=index_method,
            index_fixings=index_fixings,
            index_base_type=index_base_type,
            # rate
            fixed_rate=fixed_rate,
            float_spread=float_spread,
            spread_compound_method=spread_compound_method,
            rate_fixings=rate_fixings,
            fixing_method=fixing_method,
            fixing_frequency=fixing_frequency,
            fixing_series=fixing_series,
            # meta
            fixed=fixed,
            curves=self._parse_curves(curves),
            metric=metric,
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            initial_exchange=True,
            final_exchange=True,
            vol=_Vol(),
        )

        default_args = dict(
            currency=defaults.base_currency,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            payment_lag_exchange=defaults.payment_lag_exchange,
            fixed=True,
            mtm=False,
            metric="leg1",
        )

        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "metric", "fixed", "vol"],
        )

        # narrowing of fixed or floating
        float_attrs = [
            "float_spread",
            "spread_compound_method",
            "rate_fixings",
            "fixing_method",
            "fixing_frequency",
            "fixing_series",
        ]
        if self.kwargs.meta["fixed"]:
            for item in float_attrs:
                self.kwargs.leg1.pop(item)
        else:
            self.kwargs.leg1.pop("fixed_rate")

        # setting non-deliverability
        self.kwargs.leg1["mtm"] = LegMtm.Payment if self.kwargs.leg1["mtm"] else LegMtm.Initial

        if self.kwargs.meta["fixed"]:
            self._leg1: FixedLeg | FloatLeg = FixedLeg(
                **_convert_to_schedule_kwargs(self.kwargs.leg1, 1)
            )
        else:
            self._leg1 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))

        self._legs = [self._leg1]

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        A FixedRate Loan only requires one curve for discounting.
        A FloatRate Loan requires upto two, one for discounting and one for forecasting rates.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
            return _Curves(
                rate_curve=curves.get("rate_curve", NoInput(0)),
                disc_curve=curves.get("disc_curve", NoInput(0)),
                index_curve=curves.get("index_curve", NoInput(0)),
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
                    index_curve=curves[0],
                )
            elif len(curves) == 3:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                    index_curve=curves[2],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires upto 3 curve types. Got {len(curves)}."
                )
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                rate_curve=curves,  # type: ignore[arg-type]
                disc_curve=curves,  # type: ignore[arg-type]
                index_curve=curves,  # type: ignore[arg-type]
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
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()

        if metric_ == "npv":
            return self.npv(  # type: ignore[return-value]
                curves=curves,
                solver=solver,
                fx=fx,
                vol=vol,
                base=base,
                settlement=settlement,
                forward=forward,
                local=False,
            )

        c = _parse_curves(self, curves, solver)
        disc_curve = _get_curve("disc_curve", False, False, *c)
        settlement_ = _drb(disc_curve.nodes.initial, settlement)
        period_index = self.leg1._period_index(settlement_)
        tgt_notional = -self.leg1._regular_periods[period_index].settlement_params.notional

        if metric_ == "fixed_rate":
            raise NotImplementedError("metric 'float_rate' not implemented for Loan.")
            if not isinstance(self.leg1, FixedLeg):
                raise TypeError("Can only use 'fixed_rate' for FixedLeg Loan.")

            fixed_rate_ = self.leg1.fixed_rate

            def s(g):
                self.leg1.fixed_rate = g
                pv = self._npv_local_excluding_first_exchange(
                    curves=curves,
                    solver=solver,
                    settlement=settlement_,
                    forward=forward,
                )
                return pv

            result = ift_1dim(
                s=s,
                s_tgt=tgt_notional,
                h="ytm_quadratic",
                ini_h_args=(-3.0, 2.0, 12.0),
                func_tol=1e-5,
                conv_tol=1e-6,
                max_iter=20,
            )

            self.leg1.fixed_rate = fixed_rate_
            return result["g"]

        elif metric == "float_spread":
            raise NotImplementedError("metric 'float_rate' not implemented for Loan.")
            if not isinstance(self.leg1, FloatLeg):
                raise TypeError("Can only use 'float_spread' for FloatLeg Loan.")

            float_spread_ = self.leg1.float_spread

            def s(g):
                self.leg1.float_spread = g
                pv = self._npv_local_excluding_first_exchange(
                    curves=curves,
                    solver=solver,
                    settlement=settlement_,
                    forward=forward,
                )
                return pv

            result = ift_1dim(
                s=s,
                s_tgt=tgt_notional,
                h="ytm_quadratic",
                ini_h_args=(-300.0, 200.0, 1200.0),
                func_tol=1e-5,
                conv_tol=1e-6,
                max_iter=20,
            )

            self.leg1.fixed_rate = float_spread_
            return result["g"]

        else:
            raise ValueError("`metric`must be in {'npv', 'cashflow'}.")

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

    # def _npv_local_excluding_first_exchange(
    #     self,
    #     *,
    #     curves: CurvesT_ = NoInput(0),
    #     solver: Solver_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     vol: VolT_ = NoInput(0),
    #     settlement: datetime_ = NoInput(0),
    #     forward: datetime_ = NoInput(0),
    # ) -> DualTypes | dict[str, DualTypes]:
    #     c = _parse_curves(self, curves, solver)
    #     disc_curve = _get_curve("disc_curve", False, False, *c)
    #     first_npv = self.leg1.periods[0].npv(
    #         disc_curve=disc_curve, settlement=settlement, forward=forward
    #     )
    #     return (
    #         super().npv(
    #             curves=curves,
    #             solver=solver,
    #             fx=fx,
    #             vol=vol,
    #             settlement=settlement,
    #             forward=forward,
    #         )
    #         - first_npv
    #     )

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
        return super().analytic_delta(
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
