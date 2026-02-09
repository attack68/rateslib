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
from rateslib.enums.parameters import LegIndexBase
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _maybe_get_curve_maybe_from_solver,
    _Vol,
)
from rateslib.legs import FixedLeg

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        CalInput,
        CurvesT_,
        DataFrame,
        DualTypes,
        DualTypes_,
        Frequency,
        FXForwards_,
        IndexMethod,
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


class YoYIS(_BaseInstrument):
    r"""
    A *year-on-year indexed swap (YoYIS)* composing two :class:`~rateslib.legs.FixedLeg` with the
    second having ``index_params``.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import YoYIS
       from rateslib import fixings
       from datetime import datetime as dt
       from pandas import Series

    .. ipython:: python

       fixings.add("CPI_UK", Series(index=[dt(1999, 10, 1), dt(2000, 10, 1), dt(2001, 10, 1), dt(2002, 10, 1)], data=[110.0, 120.0, 125.0, 127.0]))
       yoyis = YoYIS(
           effective=dt(2000, 1, 1),
           termination="3y",
           frequency="A",
           fixed_rate=2.0,
           convention="One",
           leg2_index_fixings="CPI_UK",
           leg2_index_lag=3,
           leg2_index_method="monthly",
       )
       yoyis.cashflows()

    .. ipython:: python
       :suppress:

       fixings.pop("CPI_UK")

    .. rubric:: Pricing

    An *YoYIS* requires a *disc curve* on both legs (which should be the same *Curve*), and a
    *leg2 index curve* for index forecasting on the second *FixedLeg*.
    The following input formats are allowed:

    .. code-block:: python

       curves = [index_curve, disc_curve]  #  two curves are applied in order
       curves = {  # dict form is explicit
           "disc_curve": disc_curve,
           "leg2_index_curve": index_curve,
       }

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

        .. note::

           The following parameters define **indexation**.

    leg2_index_method : IndexMethod, str, :green:`optional (set by 'defaults')`
        The interpolation method, or otherwise, to determine index values from reference dates.
    leg2_index_lag: int, :green:`optional (set by 'defaults')`
        The indexation lag, in months, applied to the determination of index values.
    leg2_index_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The index value for the reference date.
        Best practice is to supply this value as string identifier relating to the global
        ``fixings`` object.

        .. note::

           The following are **meta parameters**.

    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.

    Notes
    -------

    A *YoYIS* has a nominal :class:`~rateslib.legs.FixedLeg` with a specific ``fixed_rate``, and
    a second *Leg*  whose cash flows are defined by some index values. *Rateslib* constructs this
    object as a second :class:`~rateslib.legs.FixedLeg` which inherits specific properties,
    namely:

    - The ``leg2_index_base_type`` is *LegIndexBase.PeriodOnPeriod*, to ensure that indexing is not
      calculated from one single ``leg2_index_base`` value, but by consecutive dates.
    - The ``leg2_fixed_rate`` is 100% to provide a coupon amount that matches the notional.
    - The ``leg2_index_only`` parameter is *True* to ensure that the cashflow paid only accounts for
      indexation and does not pay that 100% of notional.

    Under this definition the unindexed reference cashflow of each period of *Leg2* is the notional
    adjusted by the DCF:

    .. math::

       \mathbb{E^Q} [\bar{C}_t] = -N d

    and the indexed reference cashflow, accounting for indexation only, is:

    .. math::

       -N d ( \frac{I_v(m_i)}{I_v(m_{i-1})} - 1 )

    which matches the definition of the indexed *Leg* of a *YoYIS*.

    """  # noqa: E501

    _rate_scalar = 1.0

    @property
    def fixed_rate(self) -> DualTypes_:
        """The fixed rate of *Leg1*."""
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
    def leg2(self) -> FixedLeg:
        """The second :class:`~rateslib.legs.FixedLeg` of the *Instrument* with indexation."""
        return self._leg2

    @property
    def legs(self) -> Sequence[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    def __init__(
        self,
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
        leg2_convention: str_ = NoInput(1),
        leg2_ex_div: int_ = NoInput(1),
        # settlement params
        currency: str_ = NoInput(0),
        notional: float_ = NoInput(0),
        amortization: float_ = NoInput(0),
        leg2_notional: float_ = NoInput(-1),
        leg2_amortization: float_ = NoInput(-1),
        # index params
        leg2_index_lag: int_ = NoInput(0),
        leg2_index_method: IndexMethod | str_ = NoInput(0),
        leg2_index_fixings: LegFixings = NoInput(0),
        # rate params
        fixed_rate: DualTypes_ = NoInput(0),
        # meta params
        curves: CurvesT_ = NoInput(0),
        spec: str_ = NoInput(0),
    ) -> None:
        user_args = dict(
            effective=effective,
            termination=termination,
            frequency=frequency,
            fixed_rate=fixed_rate,
            leg2_index_lag=leg2_index_lag,
            leg2_index_method=leg2_index_method,
            leg2_index_fixings=leg2_index_fixings,
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
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            leg2_effective=leg2_effective,
            leg2_termination=leg2_termination,
            leg2_frequency=leg2_frequency,
            leg2_stub=leg2_stub,
            leg2_front_stub=leg2_front_stub,
            leg2_back_stub=leg2_back_stub,
            leg2_roll=leg2_roll,
            leg2_eom=leg2_eom,
            leg2_modifier=leg2_modifier,
            leg2_calendar=leg2_calendar,
            leg2_payment_lag=leg2_payment_lag,
            leg2_payment_lag_exchange=leg2_payment_lag_exchange,
            leg2_ex_div=leg2_ex_div,
            leg2_notional=leg2_notional,
            leg2_amortization=leg2_amortization,
            leg2_convention=leg2_convention,
            curves=self._parse_curves(curves),
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            leg2_currency=NoInput(1),
            initial_exchange=False,
            leg2_initial_exchange=False,
            final_exchange=False,
            leg2_final_exchange=False,
            leg2_index_base_type=LegIndexBase.PeriodOnPeriod,
            leg2_fixed_rate=100.0,  # combined with index_only this acts similarly to a cashflow
            leg2_index_only=True,  # but it is impacted by the DCF of the period.
            vol=_Vol(),
        )

        default_args = dict(
            notional=defaults.notional,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            payment_lag_exchange=defaults.payment_lag_exchange,
            index_lag=defaults.index_lag,
            index_method=defaults.index_method,
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "vol"],
        )

        self._leg1 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self._leg2 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg2, 1))
        self._legs = [self._leg1, self._leg2]

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

        leg2_npv: DualTypes = self.leg2.local_npv(
            rate_curve=NoInput(0),
            disc_curve=_maybe_get_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "leg2_disc_curve", solver
            ),
            index_curve=_maybe_get_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "leg2_index_curve", solver
            ),
            settlement=settlement,
            forward=forward,
        )

        return (
            self.leg1.spread(
                target_npv=-leg2_npv,  # - leg1_npv,
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

        leg1_npv: DualTypes = self.leg1.local_npv(
            rate_curve=NoInput(0),
            disc_curve=_maybe_get_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "disc_curve", solver
            ),
            index_curve=NoInput(0),
            settlement=settlement,
            forward=forward,
        )
        return self.leg2.spread(
            target_npv=-leg1_npv,
            rate_curve=NoInput(0),
            disc_curve=_maybe_get_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "leg2_disc_curve", solver
            ),
            index_curve=_maybe_get_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "leg2_index_curve", solver
            ),
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
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> None:
        # the test for an unpriced IIRS is that its fixed rate is not set.
        if isinstance(self.kwargs.leg1["fixed_rate"], NoInput):
            # set a fixed rate for the purpose of generic methods NPV will be zero.
            mid_market_rate = self.rate(
                curves=curves,
                solver=solver,
                settlement=settlement,
                forward=forward,
            )
            self.leg1.fixed_rate = _dual_float(mid_market_rate)

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        An IIRS has three curve requirements: an index_curve, a leg2_rate_curve and a
        disc_curve used by both legs.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        elif isinstance(curves, dict):
            return _Curves(
                disc_curve=curves.get("disc_curve", NoInput(0)),
                index_curve=curves.get("index_curve", NoInput(0)),
                leg2_index_curve=_drb(
                    curves.get("index_curve", NoInput(0)),
                    curves.get("leg2_index_curve", NoInput(0)),
                ),
                leg2_disc_curve=_drb(
                    curves.get("disc_curve", NoInput(0)),
                    curves.get("leg2_disc_curve", NoInput(0)),
                ),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 2:
                return _Curves(
                    disc_curve=curves[1],
                    leg2_index_curve=curves[0],
                    leg2_disc_curve=curves[1],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires 2 curve types. Got {len(curves)}."
                )
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input which is copied across all curves
            raise ValueError(f"{type(self).__name__} requires 2 curve types. Got 1.")

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

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
