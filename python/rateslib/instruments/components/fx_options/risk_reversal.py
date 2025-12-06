from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

from rateslib import defaults
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.components.fx_options.call_put import FXCall, FXOption, FXPut
from rateslib.instruments.components.protocols import _KWArgs

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CalInput,
        CurvesT_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXVolStrat_,
        Sequence,
        Solver_,
        VolT_,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FXOptionStrat(FXOption):
    """
    A custom option strategy composed of a list of :class:`~rateslib.instruments.FXOption`,
    or other :class:`~rateslib.instruments.FXOptionStrat` objects, of the same currency ``pair``.

    Parameters
    ----------
    options: list
        The *FXOptions* or *FXOptionStrats* which make up the strategy.
    rate_weight: list
        The multiplier for the *'pips_or_%'* metric that sums the options to a final *rate*.
        E.g. A *RiskReversal* uses [-1.0, 1.0] for a sale and a purchase.
        E.g. A *Straddle* uses [1.0, 1.0] for summing two premium purchases.
    rate_weight_vol: list
        The multiplier for the *'vol'* metric that sums the options to a final *rate*.
        E.g. A *RiskReversal* uses [-1.0, 1.0] to obtain the vol difference between two options.
        E.g. A *Straddle* uses [0.5, 0.5] to obtain the volatility at the strike of each option.
    """

    _greeks: dict[str, Any] = {}
    _strat_elements: tuple[FXOption | FXOptionStrat, ...]

    @property
    def kwargs(self) -> _KWArgs:
        """The :class:`~rateslib.instruments.protocols._KWArgs` of the *Instrument*."""
        return self._kwargs

    def __init__(
        self,
        options: Sequence[FXOption | FXOptionStrat],
        rate_weight: list[float],
        rate_weight_vol: list[float],
        metric: str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        vol: FXVolStrat_ = NoInput(0),
    ):
        if len(options) != len(rate_weight) or len(options) != len(rate_weight_vol):
            raise ValueError(
                "`rate_weight` and `rate_weight_vol` must have same length as `options`.",
            )
        self._kwargs = _KWArgs(
            spec=NoInput(0),
            user_args=dict(
                rate_weight=rate_weight,
                rate_weight_vol=rate_weight_vol,
                instruments=tuple(options),
                metric=metric,
                vol=vol,
                pair=options[0].kwargs.leg1["pair"],
            ),
            default_args=dict(
                metric="vol",
            ),
            meta_args=["metric", "vol", "instruments", "rate_weight", "rate_weight_vol"],
        )
        self.kwargs.leg2["premium_ccy"] = self.instruments[0].kwargs.leg2["premium_ccy"]
        self.kwargs.meta["curves"] = self._parse_curves(curves)

    # @property
    # def _vol_agg(self) -> FXVolStrat_:
    #     """Aggregate the `vol` metric on contained options into a container"""
    #
    #     def vol_attr(obj: FXOption | FXOptionStrat) -> FXVolStrat_:
    #         if isinstance(obj, FXOption):
    #             return obj.vol
    #         else:
    #             return obj._vol_agg
    #
    #     return [vol_attr(obj) for obj in self._strat_elements]
    #
    # def _parse_vol_sequence(self, vol: FXVolStrat_) -> ListFXVol_:
    #     """
    #     This function exists to determine a recursive list
    #
    #     This function must exist to parse an input sequence of given vol values for each
    #     *Instrument* in the strategy to a list that will be applied sequentially to value
    #     each of those *Instruments*.
    #
    #     If a sub-sequence, e.g BrokerFly is a strategy of strategies then this function will
    #     be repeatedly called within each strategy.
    #     """
    #     ret: ListFXVol_ = []
    #     if isinstance(
    #         vol,
    #         str
    #         | float
    #         | Dual
    #         | Dual2
    #         | Variable
    #         | FXDeltaVolSurface
    #         | FXDeltaVolSmile
    #         | FXSabrSmile
    #         | FXSabrSurface
    #         | NoInput,
    #     ):
    #         for obj in self.periods:
    #             if isinstance(obj, FXOptionStrat):
    #                 ret.append(obj._parse_vol_sequence(vol))
    #             else:
    #                 ret.append(vol)
    #
    #     elif isinstance(vol, Sequence):
    #         if len(vol) != len(self.periods):
    #             raise ValueError(
    #                 "`vol` as sequence must have same length as its contained "
    #                 f"strategy elements: {len(self.periods)}"
    #             )
    #         else:
    #             for obj, vol_ in zip(self.periods, vol, strict=True):
    #                 if isinstance(obj, FXOptionStrat):
    #                     ret.append(obj._parse_vol_sequence(vol_))
    #                 else:
    #                     assert isinstance(vol_, str) or not isinstance(vol_, Sequence)  # noqa: S101
    #                     ret.append(vol_)
    #     return ret
    #
    # def _get_fxvol_maybe_from_solver_recursive(
    #     self, vol: FXVolStrat_, solver: Solver_
    # ) -> ListFXVol_:
    #     """
    #     Function must parse a ``vol`` input in combination with ``vol_agg`` attribute to yield
    #     a Sequence of vols applied to the various levels of associated *Options* or *OptionStrats*
    #     """
    #     vol_ = self._parse_vol_sequence(vol)  # vol_ is properly nested for one vol per option
    #     ret: ListFXVol_ = []
    #     for obj, vol__ in zip(self.periods, vol_, strict=False):
    #         if isinstance(obj, FXOptionStrat):
    #             ret.append(obj._get_fxvol_maybe_from_solver_recursive(vol__, solver))
    #         else:
    #             assert isinstance(vol__, str) or not isinstance(vol__, Sequence)  # noqa: S101
    #             ret.append(_get_fxvol_maybe_from_solver(vol_attr=obj.vol, vol=vol__, solver=solver))
    #     return ret

    def _parse_vol(self, vol: FXVolStrat_) -> FXVolStrat_:  # type: ignore[override]
        if isinstance(vol, list | tuple):
            return vol
        else:
            return (vol,) * len(self.instruments)

    @property
    def instruments(self) -> tuple[FXOption | FXOptionStrat, ...]:
        return self.kwargs.meta["instruments"]  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def rate(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: FXVolStrat_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return various pricing metrics of the *FXOptionStrat*.

        Parameters
        ----------
        curves : list of Curve
            Curves for discounting cashflows. List follows the structure used by IRDs and
            should be given as:
            `[None, Curve for ccy1, None, Curve for ccy2]`
        solver : Solver, optional
            The numerical :class:`Solver` that constructs *Curves*, *Smiles* or *Surfaces* from
            calibrating instruments.
        fx: FXForwards
            The object to project the relevant forward and spot FX rates.
        base: str, optional
            Not used by the `rate` method.
        vol: float, Dual, Dual2, FXDeltaVolSmile or FXDeltaVolSurface, or Sequence of such, optional
            The volatility used in calculation. See notes.
        metric: str in {"pips_or_%", "vol", "premium"}, optional
            The pricing metric type to return. See notes for
            :meth:`FXOption.rate <rateslib.instruments.FXOption.rate>`

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        If the ``vol`` option is given as a Sequence of volatility values, these should be
        ordered according to each *FXOption* or *FXOptionStrat* contained on the *Instrument*.
        For nested *FXOptionStrat* use nested sequences.

        For example, for an *FXBrokerFly*, which contains an *FXStrangle* and an *FXStraddle*,
        ``vol`` may be entered as `[[12, 11], 10]` which are values of 12% and 11% on the
        *Strangle* options and 10% for the two *Straddle* options, or just `"fx_surface1"` which
        will determine all volatilities from an FXDeltaVolSurface associated with a *Solver*,
        with id: "fx_surface1".

        """
        vol_: FXVolStrat_ = self._parse_vol(vol)
        metric_: str = _drb(self.kwargs.meta["metric"], metric)
        map_ = {
            "pips_or_%": self.kwargs.meta["rate_weight"],
            "vol": self.kwargs.meta["rate_weight_vol"],
            "premium": [1.0] * len(self.instruments),
            "single_vol": self.kwargs.meta["rate_weight_vol"],
        }
        weights = map_[metric_]

        _: DualTypes = 0.0
        for option, vol__, weight in zip(self.instruments, vol_, weights, strict=True):  # type: ignore[misc, arg-type]
            _ += (
                option.rate(
                    curves=curves,
                    solver=solver,
                    fx=fx,
                    base=base,
                    vol=vol__,  # type: ignore[arg-type]
                    metric=metric_,
                    settlement=settlement,
                    forward=forward,
                )
                * weight
            )
        return _

    def npv(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: FXVolStrat_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *FXOptionStrat*.

        Parameters
        ----------
        curves : list of Curve
            Curves for discounting cashflows. List follows the structure used by IRDs and
            should be given as: `[None, Curve for ccy1, None, Curve for ccy2]`
        solver : Solver, optional
            The numerical :class:`Solver` that constructs *Curves*, *Smiles* or *Surfaces* from
            calibrating instruments.
        fx: FXForwards
            The object to project the relevant forward and spot FX rates.
        base: str, optional
            3-digit currency in which to express values.
        local: bool, optional
            If `True` will return a dict identifying NPV by local currencies on each
            period.
        vol: float, Dual, Dual2, FXDeltaVolSmile or FXDeltaVolSurface, or Sequence of such, optional
            The volatility used in calculation.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        If the ``vol`` option is given as a Sequence of volatility values, these should be
        ordered according to each *FXOption* or *FXOptionStrat* contained on the *Instrument*.
        For nested *FXOptionStrat* use nested sequences.
        """
        vol_ = self._parse_vol(vol)

        results = [
            option.npv(
                curves=curves,
                solver=solver,
                fx=fx,
                base=base,
                local=local,
                vol=vol__,  # type: ignore[arg-type]
                forward=forward,
                settlement=settlement,
            )
            for (option, vol__) in zip(self.instruments, vol_, strict=True)  # type: ignore[arg-type]
        ]

        if local:
            df = DataFrame(results).fillna(0.0)
            df_sum = df.sum()
            _: DualTypes | dict[str, DualTypes] = df_sum.to_dict()
        else:
            _ = sum(results)  # type: ignore[arg-type]
        return _

    def cashflows(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: FXVolStrat_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return self._cashflows_from_instruments(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            settlement=settlement,
            forward=forward,
            base=base,
        )

    def _plot_payoff(
        self,
        window: list[float] | NoInput = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: FXVolStrat_ = NoInput(0),
    ) -> tuple[Any, Any]:
        vol_: FXVolStrat_ = self._parse_vol(vol)

        y = None
        for inst, vol__ in zip(self.instruments, vol_, strict=True):  # type: ignore[misc, arg-type]
            x, y_ = inst._plot_payoff(
                window=window,
                curves=curves,
                solver=solver,
                fx=fx,
                base=base,
                local=local,
                vol=vol__,  # type: ignore[arg-type]
            )
            if y is None:
                y = y_
            else:
                y += y_

        return x, y

    def analytic_greeks(
        self,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVolStrat_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return aggregated greeks of the *FXOptionStrat*.

        Parameters
        ----------
        curves : list of Curve
            Curves for discounting cashflows. List follows the structure used by IRDs and
            should be given as:
            `[None, Curve for domestic ccy, None, Curve for foreign ccy]`
        solver : Solver, optional
            The numerical :class:`Solver` that constructs *Curves*, *Smiles* or *Surfaces* from
            calibrating instruments.
        fx: FXForwards
            The object to project the relevant forward and spot FX rates.
        base: str, optional
            Not used by `analytic_greeks`.
        vol: float, Dual, Dual2, FXDeltaVolSmile or FXDeltaVolSurface, or Sequence of such, optional
            The volatility used in calculation.

        Returns
        -------
        dict

        Notes
        -----
        If the ``vol`` option is given as a Sequence of volatility values, these should be
        ordered according to each *FXOption* or *FXOptionStrat* contained on the *Instrument*.
        For nested *FXOptionStrat* use nested sequences.
        """

        # implicitly call set_pricing_mid for unpriced parameters
        # this is important for Strategies whose options are
        # dependent upon each other, e.g. Strangle. (RR and Straddle do not have
        # interdependent options)
        self.rate(curves=curves, solver=solver, fx=fx, base=base, vol=vol)

        vol_: FXVolStrat_ = self._parse_vol(vol=vol)
        gks = []
        for inst, vol_i in zip(self.instruments, vol_, strict=True):  # type: ignore[misc, arg-type]
            if isinstance(inst, FXOptionStrat):
                gks.append(
                    inst.analytic_greeks(
                        curves=curves,
                        solver=solver,
                        fx=fx,
                        base=base,
                        vol=vol_i,
                    )
                )
            else:  # option is FXOption
                # by calling on the OptionPeriod directly the strike is maintained from rate call.
                gks.append(
                    inst._analytic_greeks_set_metrics(
                        curves=curves,
                        solver=solver,
                        fx=fx,
                        base=base,
                        vol=vol_i,  # type: ignore[arg-type]
                        set_metrics=False,  # already done in the rate call above
                    )
                )

        _unit_attrs = ["delta", "gamma", "vega", "vomma", "vanna", "_kega", "_kappa", "__bs76"]
        _: dict[str, Any] = {}
        for attr in _unit_attrs:
            _[attr] = sum(gk[attr] * self.kwargs.meta["rate_weight"][i] for i, gk in enumerate(gks))

        _notional_attrs = [
            f"delta_{self.kwargs.leg1['pair'][:3]}",
            f"gamma_{self.kwargs.leg1['pair'][:3]}_1%",
            f"vega_{self.kwargs.leg1['pair'][3:]}",
        ]
        for attr in _notional_attrs:
            _[attr] = sum(gk[attr] * self.kwargs.meta["rate_weight"][i] for i, gk in enumerate(gks))

        _.update(
            {
                "__class": "FXOptionStrat",
                "__options": gks,
                "__delta_type": gks[0]["__delta_type"],
                "__notional": self.kwargs.leg1["notional"],
            },
        )
        return _


class FXRiskReversal(FXOptionStrat):
    """
    An *FX Risk Reversal* :class:`~rateslib.instruments.components.FXOptionStrat`.

    A *RiskReversal* is composed of a lower strike :class:`~rateslib.instruments.components.FXPut`
    and a higher strike :class:`~rateslib.instruments.components.FXCall`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments.components import FXRiskReversal
       from datetime import datetime as dt

    .. ipython:: python

       fxc = FXRiskReversal(
           expiry="3m",
           strike=["-25d", "25d"],
           eval_date=dt(2020, 1, 1),
           spec="eurusd_call",
       )
       fxc.cashflows()

    .. rubric:: Pricing

    The pricing mirrors that for an :class:`~rateslib.instruments.components.FXCall`.
    Allowable inputs are:

    .. code-block:: python

       curves = [rate_curve, disc_curve]  #  two curves are applied in the given order
       curves = {"rate_curve": rate_curve, "disc_curve": disc_curve}  # dict form is explicit

    Any *FXOption* also requires an :class:`~rateslib.fx.FXForwards` as input to the ``fx``
    argument.

    A ``vol`` argument must be provided to each *Instrument*. This can either be a single
    value universally used for all, or an individual item as part of a sequence. Allowed
    inputs are:

    .. code-block:: python

       vol = 12.0 | vol_obj  # a single item universally applied
       vol = [12.0, 13.0]  # values for the Put and Call respectively

    The pricing ``metric`` will return the following calculations:

    - *'vol'*: the implied volatility value of the option from a volatility object.
    - *'premium'*: the cash premium amount applicable to the 'payment' date.
    - *'pips_or_%'*: if the premium currency is LHS of ``pair`` this is a % of notional, whilst if
      the premium currency is RHS this gives a number of pips of the FX rate.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .

        .. note::

           The following define **fx option** and generalised **settlement** parameters.

    expiry: datetime, str, :red:`required`
        The expiry of the option. If given in string tenor format, e.g. "1M" requires an
        ``eval_date``. See **Notes**.
    strike: 2-tuple of float, Variable, str, :red:`required`
        The strike of the put and the call in order.
    pair: str, :red:`required`
        The currency pair for the FX rate which settles the option, in 3-digit codes, e.g. "eurusd".
        May be included as part of ``spec``.
    notional: float, :green:`optional (set by 'defaults')`
        The notional amount of each option expressed in units of LHS of ``pair``.
    eval_date: datetime, :green:`optional`
        Only required if ``expiry`` is given as string tenor.
        Should be entered as today (also called horizon) and **not** spot. Spot is derived
        from ``delivery_lag`` and ``calendar``.
    modifier : str, :green:`optional (set by 'defaults')`
        The modification rule, in {"F", "MF", "P", "MP"} for date evaluation.
    eom: bool, :green:`optional (set by 'defaults')`
        Whether to use end-of-month rolls when expiry is given as a month or year tenor.
    calendar : calendar or str, :green:`optional`
        The holiday calendar object to use. If str, looks up named calendar from
        static data.
    delivery_lag: int, :green:`optional (set by 'defaults')`
        The number of business days after expiry that the physical settlement of the FX
        exchange occurs.
    payment_lag: int or datetime, :green:`optional (set by 'defaults')`
        The number of business days after expiry to pay premium. If a *datetime* is given this will
        set the premium date explicitly.
    premium_ccy: str, :green:`optional (set as RHS of 'pair')`
        The currency in which the premium is paid. Can *only* be one of the two currencies
        in `pair`.
    delta_type: FXDeltaMethod, str, :green:`optional (set by 'defaults')`
        When deriving strike from delta use the equation associated with *'spot'* or *'forward'*
        delta. If premium currency is LHS of ``pair`` then this will produce
        **premium adjusted** delta values. If the `premium_ccy` is RHS of ``pair`` then delta values
        are **unadjusted**.

        .. note::

           The following define additional **rate** parameters.

    premium: 2-tuple of float, :green:`optional`
        The amount paid for the put and call in order. If not given assumes unpriced
        *Options* and sets this as mid-market premium during pricing.
    option_fixings: 2-tuple of float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of each option's :class:`~rateslib.data.fixings.FXFixing`. If a scalar, is used
        directly. If a string identifier, links to the central ``fixings`` object and data loader.

        .. note::

           The following are **meta parameters**.

    metric : str, :green:`optional (set as "pips_or_%")`
        The pricing metric returned by the ``rate`` method. See **Pricing**.
    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    vol: str, Smile, Surface, float, Dual, Dual2, Variable, Sequence
        Pricing objects passed directly to the *Instrument's* methods' ``vol`` argument. See
        **Pricing**.
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.

    Notes
    -----
    Buying a *Risk Reversal* equates to selling a lower strike :class:`~rateslib.instruments.FXPut`
    and buying a higher strike :class:`~rateslib.instruments.FXCall`. The ``notional`` of each are
    the same, and should be entered as a single value.
    A positive *notional* will indicate a sale of the *Put* and a purchase of the *Call*.

    When supplying ``strike`` as a string delta the strike will be determined at price time from
    the provided volatility.

    This class is an alias constructor for an
    :class:`~rateslib.instruments.FXOptionStrat` where the number
    of options and their definitions and nominals have been specifically overloaded for
    convenience.
    """

    _rate_scalar = 100.0

    def __init__(
        self,
        expiry: datetime | str,
        strike: tuple[DualTypes | str, DualTypes | str],
        pair: str_ = NoInput(0),
        *,
        notional: DualTypes_ = NoInput(0),
        eval_date: datetime | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        modifier: str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        delivery_lag: int_ = NoInput(0),
        premium: tuple[DualTypes_, DualTypes_] = (NoInput(0), NoInput(0)),
        premium_ccy: str_ = NoInput(0),
        payment_lag: str | datetime_ = NoInput(0),
        option_fixings: DualTypes_ = NoInput(0),
        delta_type: str_ = NoInput(0),
        metric: str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        spec: str_ = NoInput(0),
    ) -> None:
        vol_ = self._parse_vol(vol)
        notional_ = _drb(defaults.notional, notional)
        options = [
            FXPut(
                pair=pair,
                expiry=expiry,
                delivery_lag=delivery_lag,
                payment_lag=payment_lag,
                calendar=calendar,
                modifier=modifier,
                eom=eom,
                eval_date=eval_date,
                strike=strike[0],
                notional=-notional_,
                option_fixings=option_fixings[0]
                if isinstance(option_fixings, tuple | list)
                else option_fixings,
                delta_type=delta_type,
                premium=premium[0],
                premium_ccy=premium_ccy,
                curves=curves,
                vol=vol_[0],  # type: ignore[index]
                metric=NoInput(0),
                spec=spec,
            ),
            FXCall(
                pair=pair,
                expiry=expiry,
                delivery_lag=delivery_lag,
                payment_lag=payment_lag,
                calendar=calendar,
                modifier=modifier,
                eom=eom,
                eval_date=eval_date,
                strike=strike[1],
                notional=notional_,
                option_fixings=option_fixings[1]
                if isinstance(option_fixings, tuple | list)
                else option_fixings,
                delta_type=delta_type,
                premium=premium[1],
                premium_ccy=premium_ccy,
                curves=curves,
                vol=vol_[1],  # type: ignore[index]
                metric=NoInput(0),
                spec=spec,
            ),
        ]
        super().__init__(
            options=options,
            rate_weight=[-1.0, 1.0],
            rate_weight_vol=[-1.0, 1.0],
            metric=metric,
            curves=curves,
            vol=vol_,
        )
        self.kwargs.leg1["notional"] = notional_

    def _parse_vol(self, vol: FXVolStrat_) -> FXVolStrat_:  # type: ignore[override]
        if isinstance(vol, list | tuple):
            return vol
        else:
            return (vol,) * 2
