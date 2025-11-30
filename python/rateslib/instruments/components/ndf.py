from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.components.protocols import _BaseInstrument
from rateslib.instruments.components.protocols.kwargs import _KWArgs
from rateslib.instruments.components.protocols.pricing import (
    _Curves,
    _get_fx_forwards_maybe_from_solver,
)
from rateslib.legs.components import CustomLeg
from rateslib.periods.components import Cashflow
from rateslib.periods.components.utils import _validate_fx_as_forwards
from rateslib.scheduling.frequency import _get_fx_expiry_and_delivery_and_payment

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Adjuster,
        Any,
        CalInput,
        CurvesT_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXVolOption_,
        PeriodFixings,
        Sequence,
        Solver_,
        _BaseLeg,
        bool_,
        datetime_,
        int_,
        str_,
    )


class NDF(_BaseInstrument):
    """
    A *non-deliverable FX forward* (NDF), composing two
    :class:`~rateslib.legs.components.CustomLeg`
    of individual :class:`~rateslib.periods.components.Cashflow`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments.components import NDF
       from datetime import datetime as dt

    .. ipython:: python

       ndf = NDF(dt(2025, 1, 5), "usdbrl", fx_rate=5.5, currency="usd")
       ndf.cashflows()

    .. rubric:: Pricing

    The methods of an *NDF* require an :class:`~rateslib.fx.FXForwards` object for ``fx`` .

    They also require a *disc curve*, which is appropriate curve to discount the
    cashflows of the deliverable settlement currency. The following input
    formats are allowed:

    .. code-block:: python

       curves = disc_curve | [disc_curve]  # one curve
       curves = [None, disc_curve, None, disc_curve]  # four curves
       curves = {  # dict form is explicit
           "disc_curve": disc_curve,
           "leg2_disc_curve": disc_curve,
       }

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .
        .. note::

           The following are **settlement parameters**.

    settlement : datetime, str, :red:`required`
        The date of settlement for the currency ``pair`` and payment date.
    pair : str, :red:`required`
        The currency pair describing the notional of *leg1* and *leg2* respectively (6-digit code)
    currency : str, :green:`optional (set as RHS currency in pair)`
        The physical *settlement currency* of each leg. If not a currency in ``pair`` then each
        leg will be non-deliverable (3-digit code).
    notional : float
        The notional of *leg1* expressed in units of LHS currency of ``pair``. The notional for
        *leg2* is derived from ``fx_rate`` of ``pair``.

        .. note::

           The following are **scheduling parameters** required only if ``settlement`` given
           as string tenor.

    eval_date: datetime, :green:`optional`
        Today's date from which spot and other dates may be determined.
    calendar: str, Calendar, :green:`optional`
        The calendar associated with deriving dates for ``pair``.
    modifier: Adjuster, str, :green:`optional`
        The date adjuster for determining tenor dates under the convention for ``pair``.
    payment_lag: Adjuster, int, :green:`optional`
        The number of days by which to lag ``eval_date`` to derive the standard *spot* for ``pair``.
    eom: bool, :green:`optional`
        Whether tenors under ``pair`` adopt EOM convention or not.

        .. note::

           The following are **rate parameters** defining the settlement of the transaction.

    fx_rate : float, :green:`optional`
        The FX rate applied to the transaction. Used to derive the notional on *leg2*. If not
        given is determined as mid-market upon pricing.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for settlement of *leg1* if
        that leg is non-deliverable. If a scalar is used directly.
        If a string identifier will link to the central ``fixings`` object and data loader.
    reversed: bool, :green:`optional (set as False)`
        By default the ``fx_fixings`` are expressed in currency direction '*settlement:reference*'
        unless ``reversed`` is *True*, in which case '*reference:settlement*' is used.
    leg2_fx_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for settlement of *leg2* if
        that leg is non-deliverable. If a scalar is used directly.
        If a string identifier will link to the central ``fixings`` object and data loader.
    leg2_reversed: bool, :green:`optional (set as False)`
        By default the ``leg2_fx_fixings`` are expressed in currency direction
        '*settlement:reference*' unless ``reversed`` is *True*, in which
        case '*reference:settlement*' is used.

        .. note::

           The following are **meta parameters**.

    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.

    """

    _rate_scalar = 1.0

    @property
    def leg1(self) -> CustomLeg:
        """The :class:`~rateslib.legs.components.CustomLeg` of the *Instrument*."""
        return self._leg1

    @property
    def leg2(self) -> CustomLeg:
        """The :class:`~rateslib.legs.components.CustomLeg` of the *Instrument*."""
        return self._leg2

    @property
    def legs(self) -> Sequence[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        An NDF requires 1 disc curve for the cashflows in the delivery currency.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        elif isinstance(curves, dict):
            return _Curves(
                disc_curve=curves.get("disc_curve", NoInput(0)),
                leg2_disc_curve=_drb(
                    curves.get("disc_curve", NoInput(0)),
                    curves.get("leg2_disc_curve", NoInput(0)),
                ),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 1:
                return _Curves(
                    disc_curve=curves[0],
                    leg2_disc_curve=curves[0],
                )
            elif len(curves) == 4:
                return _Curves(
                    disc_curve=curves[1],
                    leg2_disc_curve=curves[3],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires 1 curve types. Got {len(curves)}."
                )
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                disc_curve=curves,  # type: ignore[arg-type]
                leg2_disc_curve=curves,  # type: ignore[arg-type]
            )

    def __init__(
        self,
        settlement: datetime,
        pair: str,
        *,
        # settlement
        currency: str_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        # scheduling
        eval_date: datetime_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        modifier: Adjuster | str_ = NoInput(0),
        payment_lag: Adjuster | int_ = NoInput(0),
        eom: bool_ = NoInput(0),
        # rate
        fx_rate: DualTypes_ = NoInput(0),
        fx_fixings: PeriodFixings = NoInput(0),
        leg2_fx_fixings: PeriodFixings = NoInput(0),
        reversed: bool = False,
        leg2_reversed: bool = False,
        # meta
        curves: CurvesT_ = NoInput(0),
        spec: str_ = NoInput(0),
    ):
        # determine 'currency' and each 'pair'
        # this coordinates the allowable combination inputs
        pair_ = pair.lower()
        if isinstance(currency, NoInput):
            user_args: dict[str, Any] = dict(
                currency=pair_[3:],
                pair=pair_,
                fx_fixings=fx_fixings,
                leg2_pair=NoInput(0),
                leg2_fx_fixings=NoInput(0),
            )
        else:
            currency_ = currency.lower()
            if currency_ == pair_[:3]:
                user_args = dict(
                    currency=currency_,
                    pair=NoInput(0),
                    fx_fixings=NoInput(0),
                    leg2_pair=pair_,
                    leg2_fx_fixings=fx_fixings,
                )
            elif currency_ == pair_[3:6]:
                user_args = dict(
                    currency=currency_,
                    pair=pair_,
                    fx_fixings=fx_fixings,
                    leg2_pair=NoInput(0),
                    leg2_fx_fixings=NoInput(0),
                )
            else:
                # settlement currency is a third currency (not in pair)
                user_args = dict(
                    currency=currency_,
                    pair=f"{currency_}{pair_[:3]}" if not reversed else f"{pair_[:3]}{currency_}",
                    fx_fixings=fx_fixings,
                    leg2_pair=f"{currency_}{pair_[3]}"
                    if not leg2_reversed
                    else f"{pair_[3:]}{currency_}",
                    leg2_fx_fixings=leg2_fx_fixings,
                )

        user_args.update(
            dict(
                settlement=settlement,
                notional=notional,
                fx_rate=fx_rate,
                curves=self._parse_curves(curves),
                settlement_pair=pair.lower(),
                eval_date=eval_date,
                calendar=calendar,
                modifier=modifier,
                payment_lag=payment_lag,
                eom=eom,
            )
        )

        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            leg2_currency=NoInput.inherit,
            leg2_settlement=NoInput.inherit,
            leg2_notional=NoInput(0),
        )
        default_args = dict(
            notional=defaults.notional,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            modifier=defaults.modifier,
            eom=defaults.eom_fx,
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=[
                "curves",
                "settlement_pair",
                "eval_date",
                "calendar",
                "modifier",
                "payment_lag",
                "eom",
            ],
        )

        # post input determination for 'settlement' and 'leg2_notional'
        if isinstance(self.kwargs.leg1["settlement"], datetime):
            settlement_: datetime = settlement
        else:
            _, settlement_, _ = _get_fx_expiry_and_delivery_and_payment(
                eval_date=self.kwargs.meta["eval_date"],
                expiry=self.kwargs.leg1["settlement"],
                delivery_lag=self.kwargs.meta["payment_lag"],
                calendar=self.kwargs.meta["calendar"],
                modifier=self.kwargs.meta["modifier"],
                eom=self.kwargs.meta["eom"],
                payment_lag=0,
            )
            self.kwargs.leg1["settlement"] = settlement_
            self.kwargs.leg2["settlement"] = settlement_

        if not isinstance(self.kwargs.leg1["fx_rate"], NoInput):
            self.kwargs.leg2["notional"] = (
                self.kwargs.leg1["notional"] * -self.kwargs.leg1["fx_rate"]
            )
        else:
            self.kwargs.leg2["notional"] = self.kwargs.leg1["notional"] * -1.0

        # construct legs
        self._leg1 = CustomLeg(
            periods=[
                Cashflow(
                    currency=self.kwargs.leg1["currency"],
                    notional=-1.0 * self.kwargs.leg1["notional"],
                    payment=self.kwargs.leg1["settlement"],
                    pair=self.kwargs.leg1["pair"],
                    fx_fixings=self.kwargs.leg1["fx_fixings"],
                ),
            ]
        )
        self._leg2 = CustomLeg(
            periods=[
                Cashflow(
                    currency=self.kwargs.leg2["currency"],
                    notional=-1.0 * self.kwargs.leg2["notional"],
                    payment=self.kwargs.leg2["settlement"],
                    pair=self.kwargs.leg2["pair"],
                    fx_fixings=self.kwargs.leg2["fx_fixings"],
                )
            ]
        )
        self._legs = [self._leg1, self._leg2]

    def cashflows(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
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

    def rate(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        fx_ = _validate_fx_as_forwards(_get_fx_forwards_maybe_from_solver(solver=solver, fx=fx))
        return fx_.rate(
            pair=self.kwargs.meta["settlement_pair"], settlement=self.kwargs.leg1["settlement"]
        )

    def _set_pricing_mid(
        self,
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
    ) -> None:
        if isinstance(self.kwargs.leg1["fx_rate"], NoInput):
            # determine the mid-market FX rate and set the notional of leg2
            mid_market_rate = self.rate(fx=fx, solver=solver)
            self.leg2.periods[0].settlement_params._notional = _dual_float(
                -mid_market_rate * self.leg1.periods[0].settlement_params.notional
            )

    def npv(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        self._set_pricing_mid(
            solver=solver,
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
