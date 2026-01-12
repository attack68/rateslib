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
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.bonds.conventions import (
    BondCalcMode,
    _get_bond_calc_mode,
)
from rateslib.instruments.bonds.protocols import _BaseBondInstrument
from rateslib.instruments.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _maybe_get_curve_maybe_from_solver,
    _Vol,
)
from rateslib.legs import FixedLeg

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurvesT_,
        DualTypes,
        DualTypes_,
        Frequency,
        FXForwards_,
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


class FixedRateBond(_BaseBondInstrument):
    """
    A *fixed rate bond* composed of a :class:`~rateslib.legs.FixedLeg`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import FixedRateBond
       from datetime import datetime as dt

    .. ipython:: python

       frb = FixedRateBond(
           effective=dt(2000, 1, 1),
           termination="2y",
           spec="us_gb",
           fixed_rate=2.0,
       )
       frb.cashflows()

    .. rubric:: Pricing

    A *FixedRateBond* requires one *disc curve*. The following input formats are
    allowed:

    .. code-block:: python

       curves = curve | [curve]           #  a single curve is repeated for all required curves
       curves = {"disc_curve": disc_curve}  # dict form is explicit

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

        .. note::

           The following are **rate parameters**.

    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.

        .. note::

           The following are **meta parameters**.

    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    calc_mode : str or BondCalcMode
        A calculation mode for dealing with bonds under different conventions. See notes.
    settle: int
        The number of days by which to lag 'today' to arrive at standard settlement.
    metric : str, :green:`optional` (set as 'clean_price')
        The pricing metric returned by :meth:`~rateslib.instruments.FixedRateBond.rate`.
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
    def leg1(self) -> FixedLeg:
        """The :class:`~rateslib.legs.FixedLeg` of the *Instrument*."""
        return self._leg1

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
        # settlement parameters
        currency: str_ = NoInput(0),
        notional: float_ = NoInput(0),
        # amortization: float_ = NoInput(0),
        # rate parameters
        fixed_rate: DualTypes_ = NoInput(0),
        # meta parameters
        curves: CurvesT_ = NoInput(0),
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
            # amortization=amortization,
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

        if isinstance(self.kwargs.leg1["fixed_rate"], NoInput):
            raise ValueError(f"`fixed_rate` must be provided for {type(self).__name__}.")

        self._leg1 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self._legs = [self.leg1]

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
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
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                disc_curve=curves,  # type: ignore[arg-type]
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

        _curves = self._parse_curves(curves)
        disc_curve = _validate_obj_not_no_input(
            _maybe_get_curve_maybe_from_solver(
                curves_meta=self.kwargs.meta["curves"],
                curves=_curves,
                name="disc_curve",
                solver=solver,
            ),
            "disc_curve",
        )
        settlement_ = self._maybe_get_settlement(settlement=settlement, disc_curve=disc_curve)
        npv = self.leg1.local_npv(
            disc_curve=disc_curve,
            settlement=settlement_,
            forward=settlement_,
        )
        # scale price to par 100 (npv is already projected forward to settlement)
        dirty_price = npv * 100 / -self.leg1.settlement_params.notional

        if metric_ == "dirty_price":
            return dirty_price
        elif metric_ == "clean_price":
            return dirty_price - self.accrued(settlement_)
        elif metric_ == "ytm":
            return self.ytm(dirty_price, settlement_, True)
        else:
            raise ValueError("`metric` must be in {'dirty_price', 'clean_price', 'ytm'}.")
