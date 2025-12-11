from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from rateslib import defaults
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
           fx_fixings=1.15,
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

    pair : str, :red:`required`
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

           The following are **rate parameters**.

    fx_fixings : float, Dual, Dual2, Variable, :green:`optional`
        If ``leg2_notional`` is given, this arguments can be provided to imply ``notional`` on leg1
        via non-deliverability. The direction of these rates should mirror ``pair``.
    leg2_fx_fixings : float, Dual, Dual2, Variable, :green:`optional`
        If ``notional`` is given, this argument can be provided to imply ``leg2_notional``, via
        non-deliverability. The direction of these rates should mirror ``pair``.
    points : float, Dual, Dual2, Variable, :green:`optional`
        If either ``fx_fixings`` or ``leg2_fx_fixings`` are given, this argument is required to
        imply the second FX fixing.

        .. note::

           The following are **meta parameters**.

    curves : XXX
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument.
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
           leg2_fx_fixings=1.1502327721341274,  # <- mid-market value inserted as float
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
        pair: str,
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
        fx_fixings: DualTypes_ = NoInput(0),
        leg2_fx_fixings: DualTypes_ = NoInput(0),
        points: DualTypes_ = NoInput(0),
        # meta
        curves: CurvesT_ = NoInput(0),
        spec: str_ = NoInput(0),
    ):
        pair_ = pair.lower()
        if isinstance(notional, NoInput) and isinstance(leg2_notional, NoInput):
            notional = defaults.notional
        elif not isinstance(notional, NoInput) and not isinstance(leg2_notional, NoInput):
            raise ValueError("Only one of `notional` and `leg2_notional` can be given.")

        schedule = Schedule(
            effective=effective,
            termination=termination,
            frequency="Z",
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
        )

        self._validate_init_combinations(
            notional=notional,
            leg2_notional=leg2_notional,
            fx_fixings=fx_fixings,
            leg2_fx_fixings=leg2_fx_fixings,
            points=points,
        )

        user_args = dict(
            effective=schedule.aschedule[0],
            termination=schedule.aschedule[1],
            leg2_effective=schedule.aschedule[0],
            leg2_termination=schedule.aschedule[1],
            notional=notional,
            leg2_notional=leg2_notional,
            fx_fixings=fx_fixings,
            leg2_fx_fixings=leg2_fx_fixings,
            points=points,
            split_notional=split_notional,
            curves=self._parse_curves(curves),
        )

        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            currency=pair[:3],
            leg2_currency=pair[3:6],
            pair=NoInput(0),
            leg2_pair=NoInput(0),
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
                "split_notional",
                "vol",
            ],
        )

        if isinstance(notional, NoInput):
            self.kwargs.leg1["notional"] = -1.0 * self.kwargs.leg2["notional"]
            self.kwargs.leg1["pair"] = pair_
            if isinstance(split_notional, NoInput):
                self.kwargs.leg1["split_notional"] = NoInput(0)
                self.kwargs.leg2["split_notional"] = NoInput(0)
            else:
                self.kwargs.leg1["split_notional"] = split_notional
                self.kwargs.leg2["split_notional"] = -split_notional
        else:  # notional set on leg1
            self.kwargs.leg2["notional"] = -1.0 * self.kwargs.leg1["notional"]
            self.kwargs.leg2["pair"] = pair_
            if isinstance(split_notional, NoInput):
                self.kwargs.leg2["split_notional"] = NoInput(0)
                self.kwargs.leg1["split_notional"] = NoInput(0)
            else:
                self.kwargs.leg2["split_notional"] = split_notional
                self.kwargs.leg1["split_notional"] = -split_notional

        # construct legs
        if isinstance(self.kwargs.leg1["fx_fixings"], NoInput):
            fx_fixings_2 = NoInput(0)
        else:
            fx_fixings_2 = self.kwargs.leg1["fx_fixings"] + self.kwargs.meta["points"] / 10000.0
        if isinstance(self.kwargs.leg2["fx_fixings"], NoInput):
            leg2_fx_fixings_2 = NoInput(0)
        else:
            leg2_fx_fixings_2 = (
                self.kwargs.leg2["fx_fixings"] + self.kwargs.meta["points"] / 10000.0
            )

        self._leg1 = CustomLeg(
            periods=[
                Cashflow(
                    currency=self.kwargs.leg1["currency"],
                    notional=self.kwargs.leg1["notional"],
                    payment=self.kwargs.leg1["effective"],
                    pair=self.kwargs.leg1["pair"],
                    fx_fixings=self.kwargs.leg1["fx_fixings"],
                ),
                Cashflow(
                    currency=self.kwargs.leg1["currency"],
                    notional=-1.0 * self.kwargs.leg1["notional"]
                    if isinstance(split_notional, NoInput)
                    else self.kwargs.leg1["split_notional"],
                    payment=self.kwargs.leg1["termination"],
                    pair=self.kwargs.leg1["pair"],
                    fx_fixings=fx_fixings_2,
                ),
            ]
        )
        self._leg2 = CustomLeg(
            periods=[
                Cashflow(
                    currency=self.kwargs.leg2["currency"],
                    notional=self.kwargs.leg2["notional"],
                    payment=self.kwargs.leg2["effective"],
                    pair=self.kwargs.leg2["pair"],
                    fx_fixings=self.kwargs.leg2["fx_fixings"],
                ),
                Cashflow(
                    currency=self.kwargs.leg2["currency"],
                    notional=-1.0 * self.kwargs.leg2["notional"]
                    if isinstance(split_notional, NoInput)
                    else self.kwargs.leg2["split_notional"],
                    payment=self.kwargs.leg2["termination"],
                    pair=self.kwargs.leg2["pair"],
                    fx_fixings=leg2_fx_fixings_2,
                ),
            ]
        )
        self._legs = [self._leg1, self._leg2]

    def _validate_init_combinations(
        self,
        notional: DualTypes_,
        leg2_notional: DualTypes_,
        fx_fixings: DualTypes_,
        leg2_fx_fixings: DualTypes_,
        points: DualTypes_,
    ) -> None:
        if not isinstance(fx_fixings, NoInput):
            if not isinstance(notional, NoInput):
                raise ValueError(
                    "When `notional` is given only `leg2_fx_fixings` are required to derive "
                    "cashflows on leg2 via non-deliverability."
                )
            if isinstance(points, NoInput):
                raise ValueError(
                    "An FXSwap must set ``fx_fixings`` and ``points`` simultaneously to determine"
                    "a properly initialized FXSwap object.\n Only ``fx_fixings`` was given."
                )
        if not isinstance(leg2_fx_fixings, NoInput):
            if not isinstance(leg2_notional, NoInput):
                raise ValueError(
                    "When `leg2_notional` is given only `fx_fixings` are required to derive "
                    "cashflows on leg1 via non-deliverability."
                )
            if isinstance(points, NoInput):
                raise ValueError(
                    "An FXSwap must set ``fx_fixings`` and ``points`` simultaneously to determine"
                    "a properly initialized FXSwap object.\n Only ``fx_fixings`` was given."
                )

        if not isinstance(points, NoInput) and (
            isinstance(leg2_fx_fixings, NoInput) and isinstance(fx_fixings, NoInput)
        ):
            raise ValueError(
                "`points` has been set on an FXSwap without a defined `fx_fixings` or "
                "`leg2_fx_fixings`.\nThe initial FXFixing is required to determine the cashflow "
                "exchanges at maturity."
            )

        if not isinstance(notional, NoInput) and not isinstance(fx_fixings, NoInput):
            raise ValueError(
                "When `notional` is given only `leg2_fx_fixings` is required to derive "
                "cashflows on leg2 via non-deliverability."
            )
        if not isinstance(leg2_notional, NoInput) and not isinstance(leg2_fx_fixings, NoInput):
            raise ValueError(
                "When `leg2_notional` is given only `fx_fixings` is required to derive "
                "cashflows on leg1 via non-deliverability."
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
