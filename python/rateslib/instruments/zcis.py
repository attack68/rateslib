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
    _maybe_get_curve_maybe_from_solver,
    _Vol,
)
from rateslib.legs import ZeroFixedLeg, ZeroIndexLeg

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
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


class ZCIS(_BaseInstrument):
    """
    An *indexed zero coupon swap (ZCIS)* composing a :class:`~rateslib.legs.ZeroFixedLeg`
    and a :class:`~rateslib.legs.ZeroIndexLeg`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import ZCIS
       from datetime import datetime as dt
       from rateslib import fixings
       from pandas import Series

    .. ipython:: python

       fixings.add("CPI_UK", Series(index=[dt(1999, 10, 1), dt(1999, 11, 1)], data=[110.0, 112.0]))
       zcis = ZCIS(
           effective=dt(2000, 1, 10),
           termination="2Y",
           frequency="A",
           fixed_rate=3.5,
           currency="gbp",
           leg2_index_fixings="CPI_UK",
           leg2_index_method="daily",
       )
       zcis.cashflows()

    .. ipython:: python
       :suppress:

       fixings.pop("CPI_UK")

    .. rubric:: Pricing

    The methods of a *ZCIS* require a *disc curve* applicable to both legs and a *leg2 index curve*.
    The following input formats are allowed:

    .. code-block:: python

       curves = [index_curve, disc_curve]                         # two curves
       curves = [None, disc_curve, leg2_index_curve, disc_curve]  # four curves
       curves = {  # dict form is explicit
           "disc_curve": disc_curve,
           "leg2_index_curve": leg2_index_curve,
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
    leg2_notional : float, Dual, Dual2, Variable, :green:`optional (negatively inherited from leg1)`

        .. note::

           The following are **rate parameters**.

    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.ZeroFixedLeg`. If `None`
        will be set to mid-market when curves are provided.

        .. note::

           The following parameters define **indexation**.

    leg2_index_method : IndexMethod, str, :green:`optional (set by 'defaults')`
        The interpolation method, or otherwise, to determine index values from reference dates.
    leg2_index_lag: int, :green:`optional (set by 'defaults')`
        The indexation lag, in months, applied to the determination of index values.
    leg2_index_base: float, Dual, Dual2, Variable, :green:`optional`
        The specific value applied as the base index value for all *Periods*.
        If not given and ``index_fixings`` is a string fixings identifier that will be
        used to determine the base index value.
    leg2_index_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The index value for the reference date.
        Best practice is to supply this value as string identifier relating to the global
        ``fixings`` object.
        .. note::

           The following are **meta parameters**.

    curves : XXX
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.

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
    def leg1(self) -> ZeroFixedLeg:
        """The :class:`~rateslib.legs.ZeroFixedLeg` of the *Instrument*."""
        return self._leg1

    @property
    def leg2(self) -> ZeroIndexLeg:
        """The :class:`~rateslib.legs.ZeroFloatLeg` of the *Instrument*."""
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
        leg2_notional: float_ = NoInput(-1),
        # rate parameters
        fixed_rate: DualTypes_ = NoInput(0),
        # indexing
        leg2_index_base: DualTypes_ = NoInput(0),
        leg2_index_lag: int_ = NoInput(0),
        leg2_index_method: IndexMethod | str_ = NoInput(0),
        leg2_index_fixings: LegFixings = NoInput(0),
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
            # rate
            fixed_rate=fixed_rate,
            # indexing
            leg2_index_base=leg2_index_base,
            leg2_index_lag=leg2_index_lag,
            leg2_index_method=leg2_index_method,
            leg2_index_fixings=leg2_index_fixings,
            # meta
            curves=self._parse_curves(curves),
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            leg2_currency=NoInput(1),
            initial_exchange=False,
            final_exchange=False,
            leg2_initial_exchange=False,
            leg2_final_exchange=False,
            vol=_Vol(),
        )

        default_args = dict(
            notional=defaults.notional,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            payment_lag_exchange=defaults.payment_lag_exchange,
            leg2_index_lag=defaults.index_lag,
            leg2_index_method=defaults.index_method,
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "vol"],
        )

        self._leg1 = ZeroFixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self._leg2 = ZeroIndexLeg(**_convert_to_schedule_kwargs(self.kwargs.leg2, 1))
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
        raise NotImplementedError("ZCIS has no concept of `spread` - use `rate` instead.")

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

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        An ZCIS has two curve requirements: a leg2_index_curve and a disc_curve used by both legs.

        When given as 2 elements the first is treated as the rate curve and the 2nd as disc curve.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
            return _Curves(
                index_curve=curves.get("index_curve", NoInput(0)),
                disc_curve=curves.get("disc_curve", NoInput(0)),
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
                    leg2_index_curve=curves[0],
                    disc_curve=curves[1],
                    leg2_disc_curve=curves[1],
                )
            elif len(curves) == 4:
                return _Curves(
                    leg2_index_curve=curves[2],
                    disc_curve=curves[1],
                    leg2_disc_curve=curves[3],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires only 2 curve types. Got {len(curves)}."
                )
        elif isinstance(curves, _Curves):
            return curves
        else:
            raise ValueError(f"{type(self).__name__} requires only 2 curve types. Got 1.")

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
