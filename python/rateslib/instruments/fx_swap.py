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

from datetime import datetime
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.data.fixings import _fx_index_set_cross, _get_fx_index
from rateslib.enums.generics import NoInput
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _get_fx_forwards_maybe_from_solver,
    _maybe_get_curve_maybe_from_solver,
    _Vol,
)
from rateslib.legs import CustomLeg
from rateslib.periods import Cashflow
from rateslib.periods.utils import _validate_base_curve
from rateslib.scheduling import Schedule

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CalInput,
        CurvesT_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXIndex,
        FXIndex_,
        LegFixings,
        RollDay,
        Sequence,
        Solver_,
        VolT_,
        _BaseLeg,
        bool_,
        datetime_,
        str_,
    )


class FXSwap(_BaseInstrument):
    """
    An *FX swap* composing two
    :class:`~rateslib.legs.CustomLeg`
    of individual :class:`~rateslib.periods.Cashflow` of different currencies.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from datetime import datetime as dt
       from rateslib.instruments.fx_swap import FXSwap
       from rateslib import Curve, FXRates, FXForwards

    Paying a 3M EURUSD *FX Swap* expressed in USD notional at 56.5 swap points.

    .. ipython:: python

       fxs = FXSwap(
           effective=dt(2022, 1, 19),
           termination="3m",
           calendar="tgt|fed",
           pair="eurusd",
           leg2_notional=-10e6,
           split_notional=-10.25e6,
           fx_rate=1.15,
           points=56.5,
       )
       fxs.cashflows()

    .. rubric:: Pricing

    An *FX Swap* requires a *disc curve* and a *leg2 disc curve* to discount the cashflows
    of the respective currencies (typically with the same collateral definition).
    The following input formats are allowed:

    .. code-block:: python

       curves = [disc_curve, leg2_disc_curve]  #  two curves are applied in the given order
       curves = [None, disc_curve, None, leg2_disc_curve]  # four curves applied to each leg
       curves = {"disc_curve": disc_curve, "leg2_disc_curve": leg2_disc_curve}  # dict form is explicit

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .

        .. note::

           The following define generalised **scheduling** parameters.

    effective : datetime, :red:`required`
        The settlement date of the first currency pair.
    termination : datetime, str, :red:`required`
        The settlement of the second currency pair. If given as string requires additional
        scheduling arguments to derive from ``effective``.
    roll : RollDay, int in [1, 31], str in {"eom", "imm", "som"}, :green:`optional`
        If ``termination`` is str tenor, the roll day for its determination.
    eom : bool, :green:`optional`
        If ``termination`` is str tenor, the end-of-month preference if ``roll`` is not specified.
    modifier : Adjuster, str in {"NONE", "F", "MF", "P", "MP"}, :green:`optional (set by 'defaults')`
        If ``termination`` is str tenor, the adjustment to apply to its determination.
    calendar : calendar, str, :green:`optional (set as 'all')`
        If ``termination`` is str tenor, the calendar to apply to its determination.

        .. note::

           The following define generalised **settlement** parameters.

    pair : FXIndex, str, :red:`required`
        The FX pair of the *Instrument* (6-digit code).
    notional : float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        To define the notional of the trade in units of LHS pair use ``notional``.
    leg2_notional : float, Dual, Dual2, Variable, :green:`optional (negatively inherited from leg1)`
        To define the notional of the trade in units of RHS pair use ``leg2_notional``.
        Only one of ``notional`` or ``leg2_notional`` can be specified.
    split_notional: float, Variable, :green:`optional`
        If the second cashflow has a rate adjusted notional to mitigate spot FX risk this is
        entered as this argument. If not given the *FX Swap* is assumed not to have split notional.
        Expressed in the same units as that given for either ``notional`` or ``leg2_notional``.

        .. note::

           The following are **rate parameters**. Both must be given simultaneously or not
           at all.

    fx_rate : float, Dual, Dual2, Variable, :green:`optional`
        The ``fx_rate`` with direction according to ``pair`` to define the missing notional.
    points : float, Dual, Dual2, Variable, :green:`optional`
        The swap points valued (in 10,000ths) to add to ``fx_rate`` to arrive at the
        FX rate at maturity of the swap.

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
    An *FXSwap* is constructed from two *Legs* where one is non-deliverable. A fully
    specified *Instrument* is one whose non-deliverable *fx fixings* are set at initialisation
    via ``points`` and either ``fx_fixings`` or ``leg2_fx_fixings``. If these are not given then
    these values will be forecast :class:`~rateslib.data.fixings.FXFixing`, which will likely
    impact risk sensitivity calculations. This is best observed in the following example where
    two similar *FXSwaps* are created, but their risks (as demonstrated by the Dual gradients)
    are different.

    .. ipython:: python

       eurusd = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.95})
       usdusd = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.94})
       fxf = FXForwards(
           fx_rates=FXRates({"eurusd": 1.15}, settlement=dt(2000, 1, 3)),
           fx_curves={"usdusd": usdusd, "eureur": eurusd, "eurusd": eurusd},
       )
       fxs1 = FXSwap(
           dt(2000, 1, 10),
           dt(2000, 4, 10),
           pair="eurusd",
           notional=1e6,
           fx_rate=1.1502327721341274,  # <- mid-market value inserted as float
           points=30.303287307187343  # <- mid-market value inserted as float
       )
       fxs2 = FXSwap(
           dt(2000, 1, 10),
           dt(2000, 4, 10),
           pair="eurusd",
           notional=1e6,
       )
       fxs1.npv(curves=[eurusd, usdusd], fx=fxf)
       fxs2.npv(curves=[eurusd, usdusd], fx=fxf)

    """  # noqa: E501

    _rate_scalar = 1.0

    @property
    def leg1(self) -> CustomLeg:
        """The :class:`~rateslib.legs.CustomLeg` of the *Instrument*."""
        return self._leg1

    @property
    def leg2(self) -> CustomLeg:
        """The :class:`~rateslib.legs.CustomLeg` of the *Instrument*."""
        return self._leg2

    @property
    def legs(self) -> Sequence[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        An FXSwap requires a disc curve and a leg2 disc curve
        """
        if isinstance(curves, NoInput):
            return _Curves()
        elif isinstance(curves, dict):
            return _Curves(
                disc_curve=curves.get("disc_curve", NoInput(0)),
                leg2_disc_curve=curves.get("leg2_disc_curve", NoInput(0)),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 2:
                return _Curves(
                    disc_curve=curves[0],
                    leg2_disc_curve=curves[1],
                )
            elif len(curves) == 4:
                return _Curves(
                    disc_curve=curves[1],
                    leg2_disc_curve=curves[3],
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

    def __init__(
        self,
        # scheduling
        effective: datetime,
        termination: datetime | str,
        pair: FXIndex | str,
        *,
        roll: int | RollDay | str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        # settlement
        notional: DualTypes_ = NoInput(0),
        leg2_notional: DualTypes_ = NoInput(0),
        split_notional: DualTypes_ = NoInput(0),
        # rate
        fx_rate: DualTypes_ = NoInput(0),
        points: DualTypes_ = NoInput(0),
        # meta
        curves: CurvesT_ = NoInput(0),
        spec: str_ = NoInput(0),
    ):
        (
            fx_index_,
            notional_,
            leg2_notional_,
            fx_fixings_,
            leg2_fx_fixings_,
            pair_,
            leg2_pair_,
            fx_rate_,
            points_,
        ) = _validated_fxswap_input_combinations(
            pair=pair,
            notional=notional,
            leg2_notional=leg2_notional,
            split_notional=split_notional,
            fx_rate=fx_rate,
            points=points,
            spec=spec,
        )
        del pair, notional, leg2_notional, split_notional, fx_rate, points

        schedule = Schedule(
            effective=effective,
            termination=termination,
            frequency="Z",
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
        )

        user_args = dict(
            effective=schedule.aschedule[0],
            termination=schedule.aschedule[1],
            leg2_effective=schedule.aschedule[0],
            leg2_termination=schedule.aschedule[1],
            notional=notional_,
            leg2_notional=leg2_notional_,
            fx_fixings=fx_fixings_,
            leg2_fx_fixings=leg2_fx_fixings_,
            points=points_,
            curves=self._parse_curves(curves),
            fx_rate=fx_rate_,
            pair=pair_,
            leg2_pair=leg2_pair_,
        )

        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            currency=fx_index_.pair[:3],
            leg2_currency=fx_index_.pair[3:6],
            vol=_Vol(),
        )
        default_args: dict[str, Any] = dict()
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=[
                "curves",
                "points",
                "fx_rate",
                "vol",
            ],
        )

        self._leg1 = CustomLeg(
            periods=[
                Cashflow(
                    currency=self.kwargs.leg1["currency"],
                    notional=self.kwargs.leg1["notional"][0],
                    payment=self.kwargs.leg1["effective"],
                    pair=self.kwargs.leg1["pair"],
                    fx_fixings=self.kwargs.leg1["fx_fixings"][0],
                ),
                Cashflow(
                    currency=self.kwargs.leg1["currency"],
                    notional=self.kwargs.leg1["notional"][1],
                    payment=self.kwargs.leg1["termination"],
                    pair=self.kwargs.leg1["pair"],
                    fx_fixings=self.kwargs.leg1["fx_fixings"][1],
                ),
            ]
        )
        self._leg2 = CustomLeg(
            periods=[
                Cashflow(
                    currency=self.kwargs.leg2["currency"],
                    notional=self.kwargs.leg2["notional"][0],
                    payment=self.kwargs.leg2["effective"],
                    pair=self.kwargs.leg2["pair"],
                    fx_fixings=self.kwargs.leg2["fx_fixings"][0],
                ),
                Cashflow(
                    currency=self.kwargs.leg2["currency"],
                    notional=self.kwargs.leg2["notional"][1],
                    payment=self.kwargs.leg2["termination"],
                    pair=self.kwargs.leg2["pair"],
                    fx_fixings=self.kwargs.leg2["fx_fixings"][1],
                ),
            ]
        )
        self._legs = [self._leg1, self._leg2]

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
        if isinstance(self.kwargs.leg1["pair"], NoInput):
            # then non-deliverability and fx_fixing are on leg2
            return self._rate_on_leg(
                core_leg="leg1", nd_leg="leg2", curves=curves, fx=fx, solver=solver
            )
        else:
            # then non-deliverability and fx_fixing are on leg1
            return self._rate_on_leg(
                core_leg="leg2", nd_leg="leg1", curves=curves, fx=fx, solver=solver
            )

    def _rate_on_leg(
        self,
        core_leg: str,
        nd_leg: str,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
    ) -> DualTypes:
        _curves = self._parse_curves(curves)
        fx_ = _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)

        core_curve = "" if core_leg == "leg1" else "leg2_"
        nd_curve = "" if nd_leg == "leg1" else "leg2_"
        core_leg_: CustomLeg = getattr(self, core_leg)
        nd_leg_: CustomLeg = getattr(self, nd_leg)

        # then non-deliverability and fx_fixing are on leg2
        disc_curve = _validate_base_curve(
            _maybe_get_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, f"{core_curve}disc_curve", solver
            )
        )
        core_npv: DualTypes = core_leg_.npv(  # type: ignore[assignment]
            disc_curve=disc_curve,
            base=self.leg2.settlement_params.currency,
            fx=fx_,
            local=False,
        )
        nd_disc_curve = _validate_base_curve(
            _maybe_get_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, f"{nd_curve}disc_curve", solver
            )
        )
        nd_cf1_npv = self.leg2.periods[0].local_npv(disc_curve=nd_disc_curve, fx=fx_)
        net_zero_cf = (core_npv + nd_cf1_npv) / nd_disc_curve[
            nd_leg_.periods[1].settlement_params.payment
        ]
        required_fx = net_zero_cf / nd_leg_.periods[1].settlement_params.notional
        original_fx = nd_leg_.periods[0].non_deliverable_params.fx_fixing.value_or_forecast(fx=fx_)  # type: ignore[attr-defined]
        _: DualTypes = (required_fx - original_fx) * 10000.0
        return _

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


def _validated_fxswap_input_combinations(
    pair: FXIndex | str_,
    notional: DualTypes_,
    leg2_notional: DualTypes_,
    split_notional: DualTypes_,
    fx_rate: DualTypes_,
    points: DualTypes_,
    spec: str_,
) -> tuple[
    FXIndex,
    list[DualTypes],
    list[DualTypes],
    LegFixings,
    LegFixings,
    FXIndex_,
    FXIndex_,
    DualTypes_,
    DualTypes_,
]:
    """Method to handle arg parsing for 2 or 3 currency NDF instruments with default value
    setting and erroring raising.

    Returns
    -------
    (currency, pair, leg2_pair, notional, leg2_notional, fx_rate)
    """

    kw = _KWArgs(
        user_args=dict(
            pair=pair,
            notional=notional,
            leg2_notional=leg2_notional,
            split_notional=split_notional,
            fx_rate=fx_rate,
            points=points,
        ),
        default_args=dict(),
        spec=spec,
        meta_args=["pair", "fx_rate", "split_notional", "points"],
    )

    # FXSwaps are physically settled so do not allow WMR cross methodology to impact
    # forecast rates for FXFixings.
    fx_index_ = _fx_index_set_cross(_get_fx_index(kw.meta["pair"]), allow_cross=False)

    if isinstance(kw.leg1["notional"], NoInput) and isinstance(kw.leg2["notional"], NoInput):
        # set a default
        kw.leg1["notional"] = defaults.notional

    match (
        not isinstance(kw.leg1["notional"], NoInput),
        not isinstance(kw.leg2["notional"], NoInput),
        not isinstance(kw.meta["split_notional"], NoInput),
    ):
        case (True, True, _):
            raise ValueError(
                "The notional of an FXSwap can only be given on one Leg. Got two notionals.\n"
                "Use one notional and the `fx_rate` of `pair` to establish the implied "
                "transactional opposite notional."
            )
        case (False, True, False):
            # then leg2 notional is given
            kw.leg2["notional"] = [kw.leg2["notional"], -1.0 * kw.leg2["notional"]]
            kw.leg1["notional"] = [-1.0 * v for v in kw.leg2["notional"]]
            kw.leg1["pair"], kw.leg2["pair"] = fx_index_, NoInput(0)
        case (False, True, True):
            # then leg2 notional as a split
            if kw.meta["split_notional"] * kw.leg2["notional"] < 0:
                raise ValueError(
                    "A notional and the `split_notional` cannot be given with different signs."
                )
            kw.leg2["notional"] = [kw.leg2["notional"], -1.0 * kw.meta["split_notional"]]
            kw.leg1["notional"] = [-1.0 * v for v in kw.leg2["notional"]]
            kw.leg1["pair"], kw.leg2["pair"] = fx_index_, NoInput(0)
        case (True, False, False):
            # then leg1 notional is given
            kw.leg1["notional"] = [kw.leg1["notional"], -1.0 * kw.leg1["notional"]]
            kw.leg2["notional"] = [-1.0 * v for v in kw.leg1["notional"]]
            kw.leg1["pair"], kw.leg2["pair"] = NoInput(0), fx_index_
        case (True, False, True):
            kw.leg1["notional"] = [kw.leg1["notional"], -1.0 * kw.meta["split_notional"]]
            kw.leg2["notional"] = [-1.0 * v for v in kw.leg1["notional"]]
            kw.leg1["pair"], kw.leg2["pair"] = NoInput(0), fx_index_

    if (not isinstance(kw.meta["fx_rate"], NoInput) and isinstance(kw.meta["points"], NoInput)) or (
        isinstance(kw.meta["fx_rate"], NoInput) and not isinstance(kw.meta["points"], NoInput)
    ):
        raise ValueError(
            "For an FXSwap transaction both `fx_rate` and `points` must be given.\n"
            "Providing only one component is not allowed, please provide the missing element.\n"
            f"Got for `fx_rate`: {kw.meta['fx_rate']}\n"
            f"Got for `points`: {kw.meta['points']}\n"
        )
    elif not isinstance(kw.meta["fx_rate"], NoInput) and not isinstance(kw.meta["points"], NoInput):
        if not isinstance(kw.leg1["pair"], NoInput):
            kw.leg1["fx_fixings"] = [
                kw.meta["fx_rate"],
                kw.meta["fx_rate"] + kw.meta["points"] / 10000.0,
            ]
            kw.leg2["fx_fixings"] = [NoInput(0), NoInput(0)]
        else:
            kw.leg1["fx_fixings"] = [NoInput(0), NoInput(0)]
            kw.leg2["fx_fixings"] = [
                kw.meta["fx_rate"],
                kw.meta["fx_rate"] + kw.meta["points"] / 10000.0,
            ]
    else:
        kw.leg1["fx_fixings"] = [NoInput(0), NoInput(0)]
        kw.leg2["fx_fixings"] = [NoInput(0), NoInput(0)]

    return (
        fx_index_,
        kw.leg1["notional"],
        kw.leg2["notional"],
        kw.leg1["fx_fixings"],
        kw.leg2["fx_fixings"],
        kw.leg1["pair"],
        kw.leg2["pair"],
        kw.meta["fx_rate"],
        kw.meta["points"],
    )
