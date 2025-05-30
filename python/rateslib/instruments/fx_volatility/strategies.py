from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pandas import DataFrame

from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.default import NoInput, _drb
from rateslib.dual import Dual, Dual2, Variable, dual_log, newton_1dim
from rateslib.dual.utils import _dual_float, _set_ad_order_objects
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile, FXSabrSurface
from rateslib.instruments.fx_volatility.vanilla import FXCall, FXOption, FXPut
from rateslib.instruments.utils import (
    _get_curves_fx_and_base_maybe_from_solver,
    _get_fxvol_maybe_from_solver,
)
from rateslib.periods.utils import _validate_fx_as_forwards
from rateslib.splines import evaluate

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        NPV,
        Any,
        Curve,
        Curves_,
        DualTypes,
        DualTypes_,
        FXForwards,
        FXOptionPeriod,
        FXVol,
        FXVolOption,
        FXVolStrat_,
        ListFXVol_,
        Solver_,
        datetime,
        str_,
    )


class FXOptionStrat:
    """
    Create a custom option strategy composed of a list of :class:`~rateslib.instruments.FXOption`,
    or other :class:`~rateslib.instruments.FXOptionStrat` objects.

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
    vol: FXVolStrat_
    curves: Curves_
    kwargs: dict[str, Any]

    def __init__(
        self,
        options: list[FXOption | FXOptionStrat],
        rate_weight: list[float],
        rate_weight_vol: list[float],
    ):
        self._strat_elements = tuple(options)
        self.rate_weight = rate_weight
        self.rate_weight_vol = rate_weight_vol
        if len(self.periods) != len(self.rate_weight) or len(self.periods) != len(
            self.rate_weight_vol,
        ):
            raise ValueError(
                "`rate_weight` and `rate_weight_vol` must have same length as `options`.",
            )

    @property
    def _vol_agg(self) -> FXVolStrat_:
        """Aggregate the `vol` metric on contained options into a container"""

        def vol_attr(obj: FXOption | FXOptionStrat) -> FXVolStrat_:
            if isinstance(obj, FXOption):
                return obj.vol
            else:
                return obj._vol_agg

        return [vol_attr(obj) for obj in self._strat_elements]

    def _parse_vol_sequence(self, vol: FXVolStrat_) -> ListFXVol_:
        """
        This function exists to determine a recursive list

        This function must exist to parse an input sequence of given vol values for each
        *Instrument* in the strategy to a list that will be applied sequentially to value
        each of those *Instruments*.

        If a sub-sequence, e.g BrokerFly is a strategy of strategies then this function will
        be repeatedly called within each strategy.
        """
        ret: ListFXVol_ = []
        if isinstance(
            vol,
            str
            | float
            | Dual
            | Dual2
            | Variable
            | FXDeltaVolSurface
            | FXDeltaVolSmile
            | FXSabrSmile
            | FXSabrSurface
            | NoInput,
        ):
            for obj in self.periods:
                if isinstance(obj, FXOptionStrat):
                    ret.append(obj._parse_vol_sequence(vol))
                else:
                    ret.append(vol)

        elif isinstance(vol, Sequence):
            if len(vol) != len(self.periods):
                raise ValueError(
                    "`vol` as sequence must have same length as its contained "
                    f"strategy elements: {len(self.periods)}"
                )
            else:
                for obj, vol_ in zip(self.periods, vol, strict=True):
                    if isinstance(obj, FXOptionStrat):
                        ret.append(obj._parse_vol_sequence(vol_))
                    else:
                        assert isinstance(vol_, str) or not isinstance(vol_, Sequence)  # noqa: S101
                        ret.append(vol_)
        return ret

    def _get_fxvol_maybe_from_solver_recursive(
        self, vol: FXVolStrat_, solver: Solver_
    ) -> ListFXVol_:
        """
        Function must parse a ``vol`` input in combination with ``vol_agg`` attribute to yield
        a Sequence of vols applied to the various levels of associated *Options* or *OptionStrats*
        """
        vol_ = self._parse_vol_sequence(vol)  # vol_ is properly nested for one vol per option
        ret: ListFXVol_ = []
        for obj, vol__ in zip(self.periods, vol_, strict=False):
            if isinstance(obj, FXOptionStrat):
                ret.append(obj._get_fxvol_maybe_from_solver_recursive(vol__, solver))
            else:
                assert isinstance(vol__, str) or not isinstance(vol__, Sequence)  # noqa: S101
                ret.append(_get_fxvol_maybe_from_solver(vol_attr=obj.vol, vol=vol__, solver=solver))
        return ret

    @property
    def periods(self) -> list[FXOption | FXOptionStrat]:
        return list(self._strat_elements)

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVolStrat_ = NoInput(0),
        metric: str_ = NoInput(0),  # "pips_or_%",
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
        vol_: ListFXVol_ = self._get_fxvol_maybe_from_solver_recursive(vol, solver)
        metric_: str = _drb(self.kwargs["metric"], metric)
        map_ = {
            "pips_or_%": self.rate_weight,
            "vol": self.rate_weight_vol,
            "premium": [1.0] * len(self.periods),
            "single_vol": self.rate_weight_vol,
        }
        weights = map_[metric_]

        _: DualTypes = 0.0
        for option, vol__, weight in zip(self.periods, vol_, weights, strict=True):
            _ += option.rate(curves, solver, fx, base, vol__, metric_) * weight  # type: ignore[arg-type]
        return _

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: FXVolStrat_ = NoInput(0),
    ) -> NPV:
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
        vol_: ListFXVol_ = self._get_fxvol_maybe_from_solver_recursive(vol, solver)

        results = [
            option.npv(curves, solver, fx, base, local, vol__)  # type: ignore[arg-type]
            for (option, vol__) in zip(self.periods, vol_, strict=True)
        ]

        if local:
            df = DataFrame(results).fillna(0.0)
            df_sum = df.sum()
            _: NPV = df_sum.to_dict()
        else:
            _ = sum(results)  # type: ignore[arg-type]
        return _

    def _plot_payoff(
        self,
        window: list[float] | NoInput = NoInput(0),
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: FXVolStrat_ = NoInput(0),
    ) -> tuple[Any, Any]:
        vol_: ListFXVol_ = self._get_fxvol_maybe_from_solver_recursive(vol, solver)

        y = None
        for option, vol__ in zip(self.periods, vol_, strict=True):
            x, y_ = option._plot_payoff(window, curves, solver, fx, base, local, vol__)  # type: ignore[arg-type]
            if y is None:
                y = y_
            else:
                y += y_

        return x, y

    def analytic_greeks(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
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
        self.rate(curves, solver, fx, base, vol)

        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.kwargs["pair"][3:],
        )
        vol_: ListFXVol_ = self._get_fxvol_maybe_from_solver_recursive(vol, solver)

        gks = []
        for option, vol_i in zip(self.periods, vol_, strict=True):
            if isinstance(option, FXOptionStrat):
                gks.append(
                    option.analytic_greeks(
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
                    option._option_periods[0].analytic_greeks(
                        disc_curve=_validate_obj_not_no_input(curves_[1], "curves_[1]"),
                        disc_curve_ccy2=_validate_obj_not_no_input(curves_[3], "curves_[3]"),
                        fx=_validate_fx_as_forwards(fx_),
                        base=base_,
                        vol=vol_i,  # type: ignore[arg-type]
                    ),
                )

        _unit_attrs = ["delta", "gamma", "vega", "vomma", "vanna", "_kega", "_kappa", "__bs76"]
        _: dict[str, Any] = {}
        for attr in _unit_attrs:
            _[attr] = sum(gk[attr] * self.rate_weight[i] for i, gk in enumerate(gks))

        _notional_attrs = [
            f"delta_{self.kwargs['pair'][:3]}",
            f"gamma_{self.kwargs['pair'][:3]}_1%",
            f"vega_{self.kwargs['pair'][3:]}",
        ]
        for attr in _notional_attrs:
            _[attr] = sum(gk[attr] * self.rate_weight[i] for i, gk in enumerate(gks))

        _.update(
            {
                "__class": "FXOptionStrat",
                "__options": gks,
                "__delta_type": gks[0]["__delta_type"],
                "__notional": self.kwargs["notional"],
            },
        )
        return _


class FXRiskReversal(FXOptionStrat, FXOption):
    """
    Create an *FX Risk Reversal* option strategy.

    A *RiskReversal* is composed of a lower strike :class:`~rateslib.instruments.FXPut` and a
    higher strike :class:`~rateslib.instruments.FXCall`.

    For additional arguments see :class:`~rateslib.instruments.FXOption`.

    Parameters
    ----------
    args: tuple
        Positional arguments to :class:`~rateslib.instruments.FXOption`.
    strike: 2-element sequence
        The first element is applied to the lower strike put and the
        second element applied to the higher strike call, e.g. `["-25d", "25d"]`.
    premium: 2-element sequence, optional
        The premiums associated with each option of the risk reversal.
    metric: str, optional
        The default metric to apply in the method :meth:`~rateslib.instruments.FXOptionStrat.rate`
    kwargs: tuple
        Keyword arguments to :class:`~rateslib.instruments.FXOption`.

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

    rate_weight = [-1.0, 1.0]
    rate_weight_vol = [-1.0, 1.0]
    _rate_scalar = 100.0
    periods: list[FXOption]  # type: ignore[assignment]
    vol: FXVolStrat_

    def __init__(
        self,
        *args: Any,
        strike: tuple[str | DualTypes_, str | DualTypes_] = (NoInput(0), NoInput(0)),
        premium: tuple[DualTypes_, DualTypes_] = (NoInput(0), NoInput(0)),
        metric: str = "vol",
        **kwargs: Any,
    ) -> None:
        super(FXOptionStrat, self).__init__(  # type: ignore[misc]
            *args,
            strike=list(strike),  # type: ignore[arg-type]
            premium=list(premium),  # type: ignore[arg-type]
            **kwargs,
        )
        self.kwargs["metric"] = metric
        self._strat_elements = (
            FXPut(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"][0],
                notional=-self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][0],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
            FXCall(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"][1],
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][1],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
        )

    def _validate_strike_and_premiums(self) -> None:
        """called as part of init, specific validation rules for straddle"""
        if any(_ is NoInput.blank for _ in self.kwargs["strike"]):
            raise ValueError(
                "`strike` for FXRiskReversal must be set to list of 2 numeric or string values.",
            )
        for k, p in zip(self.kwargs["strike"], self.kwargs["premium"], strict=False):
            if isinstance(k, str) and p != NoInput.blank:
                raise ValueError(
                    "FXRiskReversal with string delta as `strike` cannot be initialised with a "
                    "known `premium`.\n"
                    "Either set `strike` as a defined numeric value, or remove the `premium`.",
                )


class FXStraddle(FXOptionStrat, FXOption):
    """
    Create an *FX Straddle* option strategy.

    An *FXStraddle* is composed of an :class:`~rateslib.instruments.FXCall`
    and an :class:`~rateslib.instruments.FXPut` at the same strike.

    For additional arguments see :class:`~rateslib.instruments.FXOption`.

    Parameters
    ----------
    args: tuple
        Positional arguments to :class:`~rateslib.instruments.FXOption`.
    premium: 2-element sequence, optional
        The premiums associated with each option of the straddle.
    metric: str, optional
        The default metric to apply in the method :meth:`~rateslib.instruments.FXOptionStrat.rate`
    kwargs: tuple
        Keyword arguments to :class:`~rateslib.instruments.FXOption`.

    Notes
    -----
    Buying a *Straddle* equates to buying an :class:`~rateslib.instruments.FXCall`
    and an :class:`~rateslib.instruments.FXPut` at the same strike. The ``notional`` of each are
    the same, and is input as a single value.

    ``strike`` should be supplied as a single value.
    When providing a string delta the strike will be determined at price time from
    the provided volatility and FX forward market.

    This class is essentially an alias constructor for an
    :class:`~rateslib.instruments.FXOptionStrat` where the number
    of options and their definitions have been specifically overloaded for convenience.
    """

    rate_weight = [1.0, 1.0]
    rate_weight_vol = [0.5, 0.5]
    _rate_scalar = 100.0
    periods: list[FXOption]  # type: ignore[assignment]
    vol: FXVolStrat_

    def __init__(
        self,
        *args: Any,
        premium: tuple[DualTypes_, DualTypes_] = (NoInput(0), NoInput(0)),
        metric: str = "vol",
        **kwargs: Any,
    ) -> None:
        super(FXOptionStrat, self).__init__(*args, premium=list(premium), **kwargs)  # type: ignore[misc, arg-type]
        self.kwargs["metric"] = metric
        self._strat_elements = (
            FXPut(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"],
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][0],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
            FXCall(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"],
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][1],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
        )

    def _set_notionals(self, notional: DualTypes) -> None:
        """
        Set the notionals on each option period. Mainly used by Brokerfly for vega neutral
        strangle and straddle.
        """
        for option in self.periods:
            option.periods[0].notional = notional

    def _validate_strike_and_premiums(self) -> None:
        """called as part of init, specific validation rules for straddle"""
        if isinstance(self.kwargs["strike"], NoInput):
            raise ValueError("`strike` for FXStraddle must be set to numeric or string value.")
        if isinstance(self.kwargs["strike"], str) and self.kwargs["premium"] != [
            NoInput.blank,
            NoInput.blank,
        ]:
            raise ValueError(
                "FXStraddle with string delta as `strike` cannot be initialised with a known "
                "`premium`.\nEither set `strike` as a defined numeric value, or remove "
                "the `premium`.",
            )


class FXStrangle(FXOptionStrat, FXOption):
    """
    Create an *FX Strangle* option strategy.

    An *FXStrangle* is composed of a lower strike :class:`~rateslib.instruments.FXPut` and
    a higher strike :class:`~rateslib.instruments.FXCall`.

    For additional arguments see :class:`~rateslib.instruments.FXOption`.

    Parameters
    ----------
    args: tuple
        Positional arguments to :class:`~rateslib.instruments.FXOption`.
    strike: 2-element sequence
        The first element is applied to the lower strike *Put* and the
        second element applied to the higher strike *Call*, e.g. `["-25d", "25d"]`.
    premium: 2-element sequence, optional
        The premiums associated with each *FXOption* of the *Strangle*.
    metric: str, optional
        The default metric to apply in the method :meth:`~rateslib.instruments.FXOptionStrat.rate`
    kwargs: tuple
        Keyword arguments to :class:`~rateslib.instruments.FXOption`.

    Notes
    -----
    Buying a *Strangle* equates to buying a lower strike :class:`~rateslib.instruments.FXPut`
    and buying a higher strike :class:`~rateslib.instruments.FXCall`. The ``notional`` is provided
    as a single input and is applied to both *FXOptions*.

    When supplying ``strike`` as a string delta the strike will be determined at price time from
    the provided volatility.

    This class is essentially an alias constructor for an
    :class:`~rateslib.instruments.FXOptionStrat` where the number
    of options and their definitions and nominals have been specifically overloaded for
    convenience.

    .. warning::

       The default ``metric`` for an *FXStrangle* is *'single_vol'*, which requires
       an iterative algorithm to solve.
       For defined strikes it is accurate but for strikes defined by delta it
       will return an iterated solution within 0.1 pips. This means it is both slower
       than other instruments and inexact.

    """

    rate_weight = [1.0, 1.0]
    rate_weight_vol = [0.5, 0.5]
    _rate_scalar = 100.0
    periods: list[FXOption]  # type: ignore[assignment]
    vol: FXVolStrat_

    def __init__(
        self,
        *args: Any,
        strike: tuple[str | DualTypes_, str | DualTypes_] = (NoInput(0), NoInput(0)),
        premium: tuple[DualTypes_, DualTypes_] = (NoInput(0), NoInput(0)),
        metric: str = "single_vol",
        **kwargs: Any,
    ) -> None:
        super(FXOptionStrat, self).__init__(  # type: ignore[misc]
            *args,
            strike=list(strike),  # type: ignore[arg-type]
            premium=list(premium),  # type: ignore[arg-type]
            **kwargs,
        )
        self.kwargs["metric"] = metric
        self._is_fixed_delta = [
            isinstance(self.kwargs["strike"][0], str)
            and self.kwargs["strike"][0][-1].lower() == "d"
            and self.kwargs["strike"][0] != "atm_forward",
            isinstance(self.kwargs["strike"][1], str)
            and self.kwargs["strike"][1][-1].lower() == "d"
            and self.kwargs["strike"][1] != "atm_forward",
        ]
        self._strat_elements = (
            FXPut(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"][0],
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][0],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
            FXCall(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"][1],
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][1],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
        )

    def _validate_strike_and_premiums(self) -> None:
        """called as part of init, specific validation rules for strangle"""
        if any(isinstance(_, NoInput) for _ in self.kwargs["strike"]):
            raise ValueError(
                "`strike` for FXStrangle must be set to list of 2 numeric or string values.",
            )
        for k, p in zip(self.kwargs["strike"], self.kwargs["premium"], strict=False):
            if isinstance(k, str) and not isinstance(p, NoInput):
                raise ValueError(
                    "FXStrangle with string delta as `strike` cannot be initialised with a "
                    "known `premium`.\n"
                    "Either set `strike` as a defined numeric value, or remove the `premium`.",
                )

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVolStrat_ = NoInput(0),
        metric: str_ = NoInput(0),  # "pips_or_%",
    ) -> DualTypes:
        """
        Returns the rate of the *FXStrangle* according to a pricing metric.

        For parameters see :meth:`FXOptionStrat.rate <rateslib.instruments.FXOptionStrat.rate>`.

        Notes
        ------

        .. warning::

           The default ``metric`` for an *FXStrangle* is *'single_vol'*, which requires an
           iterative algorithm to solve.
           For defined strikes it is usually very accurate but for strikes defined by delta it
           will return a solution within 0.01 pips. This means it is both slower than other
           instruments and inexact.

           The ``metric`` *'vol'* is not sensible to use with an *FXStrangle*, although it will
           return the arithmetic average volatility across both options, *'single_vol'* is the
           more standardised choice.
        """
        return self._rate(curves, solver, fx, base, vol, metric)

    def _rate(
        self,
        curves: Curves_,
        solver: Solver_,
        fx: FX_,
        base: str_,
        vol: FXVolStrat_,
        metric: str_,
        record_greeks: bool = False,
    ) -> DualTypes:
        metric = _drb(self.kwargs["metric"], metric).lower()
        if metric != "single_vol" and not any(self._is_fixed_delta):
            # the strikes are explicitly defined and independent across options.
            # can evaluate separately, therefore the default method will suffice.
            return super().rate(curves, solver, fx, base, vol, metric)
        else:
            # must perform single vol evaluation to determine mkt convention strikes
            single_vol = self._rate_single_vol(curves, solver, fx, base, vol, record_greeks)
            if metric == "single_vol":
                return single_vol
            else:
                # return the premiums using the single_vol as the volatility
                return super().rate(curves, solver, fx, base, vol=single_vol, metric=metric)

    def _rate_single_vol(
        self,
        curves: Curves_,
        solver: Solver_,
        fx: FX_,
        base: str_,
        vol: FXVolStrat_,
        record_greeks: bool,
    ) -> DualTypes:
        """
        Solve the single vol rate metric for a strangle using iterative market convergence routine.
        """
        # Get curves and vol
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.kwargs["pair"][3:],
        )
        vol_: ListFXVol_ = self._get_fxvol_maybe_from_solver_recursive(vol, solver)
        # type assignment, instead of using assert
        vol_0: FXVolOption = vol_[0]  # type: ignore[assignment]
        vol_1: FXVolOption = vol_[1]  # type: ignore[assignment]
        curves_1: Curve = _validate_obj_not_no_input(curves_[1], "curves_[1]")
        curves_3: Curve = _validate_obj_not_no_input(curves_[3], "curves_[3]")
        fxf: FXForwards = _validate_fx_as_forwards(fx_)

        # Get initial data from objects in their native AD order
        spot: datetime = fxf.pairs_settlement[self.kwargs["pair"]]
        w_spot: DualTypes = curves_1[spot]
        w_deli: DualTypes = curves_1[self.kwargs["delivery"]]
        f_d: DualTypes = fxf.rate(self.kwargs["pair"], self.kwargs["delivery"])
        f_t: DualTypes = fxf.rate(self.kwargs["pair"], spot)
        z_w_0 = 1.0 if "forward" in self.kwargs["delta_type"] else w_deli / w_spot
        f_0 = f_d if "forward" in self.kwargs["delta_type"] else f_t

        eta1 = None
        fzw1zw0: DualTypes = 0.0
        if isinstance(
            vol_[0], FXDeltaVolSurface | FXDeltaVolSmile
        ):  # multiple Vol objects cannot be used, will derive conventions from the first one found.
            eta1 = -0.5 if "_pa" in vol_[0].meta.delta_type else 0.5
            z_w_1 = 1.0 if "forward" in vol_[0].meta.delta_type else w_deli / w_spot
            fzw1zw0 = f_0 * z_w_1 / z_w_0

        # Determine the initial guess for Newton type iterations

        _ad = _set_ad_order_objects([0] * 5, [vol_0, vol_1, curves_1, curves_3, fxf])
        gks: list[dict[str, Any]] = [
            self.periods[0]._analytic_greeks_reduced(
                curves=[NoInput(0), curves_1, NoInput(0), curves_3],
                solver=NoInput(0),
                fx=fxf,
                base=base,
                vol=vol_0,
            ),
            self.periods[1]._analytic_greeks_reduced(
                curves=[NoInput(0), curves_1, NoInput(0), curves_3],
                solver=NoInput(0),
                fx=fxf,
                base=base,
                vol=vol_1,
            ),
        ]

        g0: DualTypes = gks[0]["__vol"] * gks[0]["vega"] + gks[1]["__vol"] * gks[1]["vega"]
        g0 /= gks[0]["vega"] + gks[1]["vega"]

        put_op_period: FXOptionPeriod = self.periods[0]._option_periods[0]
        call_op_period: FXOptionPeriod = self.periods[1]._option_periods[0]

        def root1d(
            tgt_vol: DualTypes, fzw1zw0: DualTypes, as_float: bool
        ) -> tuple[DualTypes, DualTypes]:
            if not as_float:
                # reset objects to their original order and perform final iterations
                _set_ad_order_objects(_ad, [vol_0, vol_1, curves_1, curves_3, fxf])

            # Determine the greeks of the options with the current tgt_vol iterate
            gks = [
                self.periods[0]._analytic_greeks_reduced(
                    curves=[NoInput(0), curves_1, NoInput(0), curves_3],
                    solver=NoInput(0),
                    fx=fxf,
                    base=base,
                    vol=tgt_vol * 100.0,
                ),
                self.periods[1]._analytic_greeks_reduced(
                    curves=[NoInput(0), curves_1, NoInput(0), curves_3],
                    solver=NoInput(0),
                    fx=fxf,
                    base=base,
                    vol=tgt_vol * 100.0,
                ),
            ]

            # Also determine the greeks of these options measured with the market smile vol.
            # (note the strikes have been set by previous call, call OptionPeriods direct
            # to avoid re-determination)
            s_gks = [
                put_op_period._analytic_greeks(
                    curves_1, curves_3, fxf, base_, vol_0, _reduced=True
                ),
                call_op_period._analytic_greeks(
                    curves_1, curves_3, fxf, base_, vol_1, _reduced=True
                ),
            ]

            # The value of the root function is derived from the 4 previous calculated prices
            f0 = s_gks[0]["__bs76"] + s_gks[1]["__bs76"] - gks[0]["__bs76"] - gks[1]["__bs76"]

            dc1_dvol1_0 = _d_c_hat_d_sigma_hat(gks[0], self._is_fixed_delta[0])
            dcmkt_dvol1_0 = _d_c_mkt_d_sigma_hat(
                gks[0],
                s_gks[0],
                self.kwargs["expiry"],
                vol_0,
                eta1,
                self._is_fixed_delta[0],
                fzw1zw0,
                fxf,
            )
            dc1_dvol1_1 = _d_c_hat_d_sigma_hat(gks[1], self._is_fixed_delta[1])
            dcmkt_dvol1_1 = _d_c_mkt_d_sigma_hat(
                gks[1],
                s_gks[1],
                self.kwargs["expiry"],
                vol_1,
                eta1,
                self._is_fixed_delta[1],
                fzw1zw0,
                fxf,
            )
            f1 = dcmkt_dvol1_0 + dcmkt_dvol1_1 - dc1_dvol1_0 - dc1_dvol1_1

            return f0, f1

        root_solver = newton_1dim(
            root1d,
            g0,
            args=(fzw1zw0,),
            pre_args=(True,),  # solve `as_float` in iterations
            final_args=(False,),  # capture AD in final iterations
            raise_on_fail=True,
            max_iter=10,
            func_tol=1e-6,
        )
        tgt_vol: DualTypes = root_solver["g"] * 100.0

        if record_greeks:  # this needs to be explicitly called since it degrades performance
            self._greeks["strangle"] = {
                "single_vol": {
                    "FXPut": self.periods[0].analytic_greeks(curves, solver, fxf, base, tgt_vol),
                    "FXCall": self.periods[1].analytic_greeks(curves, solver, fxf, base, tgt_vol),
                },
                "market_vol": {
                    "FXPut": put_op_period.analytic_greeks(curves_1, curves_3, fxf, base, vol_0),
                    "FXCall": call_op_period.analytic_greeks(curves_1, curves_3, fxf, base, vol_1),
                },
            }

        return tgt_vol


# Calculations related to Strange:single_vol


def _d_c_hat_d_sigma_hat(
    g: dict[str, Any],  # greeks
    fixed_delta: bool,
) -> DualTypes:
    """
    Return the total derivative of option priced with single vol with respect to single
    vol.

    Parameters
    ----------
    g: dict
        The dict of greeks for the given option period measured against the tgt, single vol.
    fixed_delta: bool
        Whether the given FXOption is defined by fixed delta or an explicit strike.

    Returns
    -------
    DualTypes
    """
    if not fixed_delta:
        # kega is 0.0
        return g["vega"]  # type: ignore[no-any-return]
    else:
        return g["_kappa"] * g["_kega"] + g["vega"]  # type: ignore[no-any-return]


def _d_c_mkt_d_sigma_hat(
    g: dict[str, Any],  # greeks
    sg: dict[str, Any],  # smile_greeks
    expiry: datetime,
    vol: FXVol,
    eta1: float | None,
    fixed_delta: bool,
    fzw1zw0: DualTypes | None,
    fxf: FXForwards,
) -> DualTypes:
    """
    Return the total derivative of option priced with mkt vol with respect to single
    vol.

    Parameters
    ----------
    g: dict
        The dict of greeks for the given option period measured against the tgt, single vol.
    sg: dict
        The dict of greeks for the given option period measured against the smile.
    expiry: datetime
        The expiry of the Option.
    vol: VolObj
        The smile object.
    eta1: float | None
        The delta type of the Smile if available
    fixed_delta: bool
        Whether the option is defined by fixed delta or an explicit strike.
    fxf: FXForwards,
        Used by SabrSurface to forecast multiple forward rates for cross-sectional smiles before
        interpolation.

    Returns
    -------
    DualTypes
    """
    if not fixed_delta:
        return 0.0  # kega is zero and the mkt vol has no sensitivity to vol_hat.
    else:
        if isinstance(vol, FXDeltaVolSurface | FXDeltaVolSmile):
            if isinstance(vol, FXDeltaVolSurface):
                vol = vol.get_smile(expiry)

            dvol_ddeltaidx = evaluate(vol.nodes.spline.spline, sg["_delta_index"], 1) * 0.01

            ddeltaidx_dvol1 = sg["gamma"] * fzw1zw0
            if eta1 < 0:  # type: ignore[operator]
                # premium adjusted vol smile
                ddeltaidx_dvol1 += sg["_delta_index"]
            ddeltaidx_dvol1 *= g["_kega"] / sg["__strike"]

            _ = dual_log(sg["__strike"] / sg["__forward"]) / sg["__vol"]
            _ += eta1 * sg["__vol"] * sg["__sqrt_t"] ** 2
            _ *= dvol_ddeltaidx * sg["gamma"] * fzw1zw0
            ddeltaidx_dvol1 /= 1 + _

            dvol_dvol1: DualTypes = dvol_ddeltaidx * ddeltaidx_dvol1
        elif isinstance(vol, FXSabrSmile):
            dvol_dk = vol._d_sabr_d_k_or_f(
                k=sg["__strike"],
                f=sg["__forward"],
                expiry=expiry,
                as_float=False,
                derivative=1,
            )[1]
            dvol_dvol1 = dvol_dk * g["_kega"]
        elif isinstance(vol, FXSabrSurface):
            dvol_dk = vol._d_sabr_d_k_or_f(
                k=sg["__strike"],
                f=fxf,
                expiry=expiry,
                as_float=False,
                derivative=1,
            )[1]
            dvol_dvol1 = dvol_dk * g["_kega"]
        else:
            dvol_dvol1 = 0.0

        return sg["_kappa"] * g["_kega"] + sg["vega"] * dvol_dvol1  # type: ignore[no-any-return]


class FXBrokerFly(FXOptionStrat, FXOption):
    """
    Create an *FX BrokerFly* option strategy.

    An *FXBrokerFly* is composed of an :class:`~rateslib.instruments.FXStrangle` and an
    :class:`~rateslib.instruments.FXStraddle`, in that order.

    For additional arguments see :class:`~rateslib.instruments.FXOption`.

    Parameters
    ----------
    args: tuple
        Positional arguments to :class:`~rateslib.instruments.FXOption`.
    strike: 2-element sequence
        The first element should be a 2-element sequence of strikes of the *FXStrangle*.
        The second element should be a single element for the strike of the *FXStraddle*.
        call, e.g. `[["-25d", "25d"], "atm_delta"]`.
    premium: 2-element sequence, optional
        The premiums associated with each option of the strategy;
        The first element contains 2 values for the premiums of each *FXOption* in the *Strangle*.
        The second element contains 2 values for the premiums of each *FXOption* in the *Straddle*.
    notional: 2-element sequence, optional
        The first element is the notional associated with the *Strangle*. If the second element
        is *None*, it will be implied in a vega neutral sense at price time.
    metric: str, optional
        The default metric to apply in the method :meth:`~rateslib.instruments.FXOptionStrat.rate`
    kwargs: tuple
        Keyword arguments to :class:`~rateslib.instruments.FXOption`.

    Notes
    -----
    Buying a *BrokerFly* equates to buying an :class:`~rateslib.instruments.FXStrangle` and
    selling a :class:`~rateslib.instruments.FXStraddle`, where the convention is to set the
    notional on the *Straddle* such that the entire strategy is *vega* neutral at inception.

    When supplying ``strike`` as a string delta the strike will be determined at price time from
    the provided volatility.

    .. warning::

       The default ``metric`` for an *FXBrokerFly* is *'single_vol'*, which requires an iterative
       algorithm to solve.
       For defined strikes it is accurate but for strikes defined by delta it
       will return a solution within 0.1 pips. This means it is both slower than other instruments
       and inexact.

    """

    rate_weight = [1.0, 1.0]
    rate_weight_vol = [1.0, -1.0]
    _rate_scalar = 100.0

    periods: list[FXOptionStrat]  # type: ignore[assignment]
    vol: FXVolStrat_

    def __init__(
        self,
        *args: Any,
        strike: tuple[tuple[DualTypes | str_, DualTypes | str_], DualTypes | str_] = (
            (NoInput(0), NoInput(0)),
            NoInput(0),
        ),
        premium: tuple[tuple[DualTypes_, DualTypes_], tuple[DualTypes_, DualTypes_]] = (
            (NoInput(0), NoInput(0)),
            (NoInput(0), NoInput(0)),
        ),
        notional: tuple[DualTypes_, DualTypes_] = (NoInput(0), NoInput(0)),
        metric: str = "single_vol",
        **kwargs: Any,
    ) -> None:
        super(FXOptionStrat, self).__init__(  # type: ignore[misc]
            *args,
            premium=list(premium),  # type: ignore[arg-type]
            strike=list(strike),  # type: ignore[arg-type]
            notional=list(notional),  # type: ignore[arg-type]
            **kwargs,
        )
        self.kwargs["notional"][1] = (
            NoInput(0) if self.kwargs["notional"][1] is None else self.kwargs["notional"][1]
        )
        self.kwargs["metric"] = metric
        self._strat_elements = (
            FXStrangle(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"][0],
                notional=self.kwargs["notional"][0],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][0],
                premium_ccy=self.kwargs["premium_ccy"],
                metric=self.kwargs["metric"],
                curves=self.curves,
                vol=self.vol,
            ),
            FXStraddle(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"][1],
                notional=self.kwargs["notional"][1],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][1],
                premium_ccy=self.kwargs["premium_ccy"],
                metric="vol" if self.kwargs["metric"] == "single_vol" else self.kwargs["metric"],
                curves=self.curves,
                vol=self.vol,
            ),
        )

    def _maybe_set_vega_neutral_notional(
        self, curves: Curves_, solver: Solver_, fx: FX_, base: str_, vol: ListFXVol_, metric: str_
    ) -> None:
        """
        Calculate the vega of the strangle and then set the notional on the straddle
        to yield a vega neutral strategy.

        Notional is set as a fixed quantity, collapsing any AD sensitivities in accordance
        with the general principle for determining risk sensitivities of unpriced instruments.

        This is only applied if ``metric`` is a cash based quantity, {"pips_or_%", "premium"}
        """
        if isinstance(self.kwargs["notional"][1], NoInput) and metric in ["pips_or_%", "premium"]:
            self.periods[0]._rate(  # type: ignore[attr-defined]
                curves,
                solver,
                fx,
                base,
                vol=vol[0],
                metric="single_vol",
                record_greeks=True,
            )
            self._greeks["straddle"] = self.periods[1].analytic_greeks(
                curves,
                solver,
                fx,
                base,
                vol=vol[1],
            )
            strangle_vega = self._greeks["strangle"]["market_vol"]["FXPut"]["vega"]
            strangle_vega += self._greeks["strangle"]["market_vol"]["FXCall"]["vega"]
            straddle_vega = self._greeks["straddle"]["vega"]
            scalar = strangle_vega / straddle_vega
            self.periods[1].kwargs["notional"] = _dual_float(
                self.periods[0].periods[0].periods[0].notional * -scalar,  # type: ignore[union-attr]
            )
            self.periods[1]._set_notionals(self.periods[1].kwargs["notional"])  # type: ignore[attr-defined]
            # BrokerFly -> Strangle -> FXPut -> FXPutPeriod

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVolStrat_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Returns the rate of the *FXBrokerFly* according to a pricing metric.

        For parameters see :meth:`FXOptionStrat.rate <rateslib.instruments.FXOptionStrat.rate>`.

        Notes
        ------

        .. warning::

           The default ``metric`` for an *FXBrokerFly* is *'single_vol'*, which requires an
           iterative algorithm to solve.
           For defined strikes it is usually very accurate but for strikes defined by delta it
           will return a solution within 0.01 pips. This means it is both slower than other
           instruments and inexact.

           The ``metric`` *'vol'* is not sensible to use with an *FXBrokerFly*, although it will
           return the arithmetic average volatility across both strategies, *'single_vol'* is the
           more standardised choice.
        """
        vol_ = self._get_fxvol_maybe_from_solver_recursive(vol, solver)
        # if not isinstance(vol, list):
        #     vol = [[vol, vol], vol]
        # else:
        #     vol = [
        #         [vol[0], vol[2]],
        #         vol[1],
        #     ]  # restructure to pass to Strangle and Straddle separately

        temp_metric = _drb(self.kwargs["metric"], metric)
        self._maybe_set_vega_neutral_notional(curves, solver, fx, base, vol_, temp_metric.lower())

        if temp_metric == "pips_or_%":
            straddle_scalar = (
                self.periods[1].periods[0].periods[0].notional  # type: ignore[union-attr]
                / self.periods[0].periods[0].periods[0].notional  # type: ignore[union-attr]
            )
            weights: Sequence[DualTypes] = [1.0, straddle_scalar]
        elif temp_metric == "premium":
            weights = self.rate_weight
        else:
            weights = self.rate_weight_vol
        _: DualTypes = 0.0

        for option_strat, vol__, weight in zip(self.periods, vol_, weights, strict=False):
            _ += option_strat.rate(curves, solver, fx, base, vol__, metric) * weight
        return _

    def analytic_greeks(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVolStrat_ = NoInput(0),
    ) -> dict[str, Any]:
        # implicitly call set_pricing_mid for unpriced parameters
        self.rate(curves, solver, fx, base, vol, metric="pips_or_%")
        # curves, fx, base = _get_curves_fx_and_base_maybe_from_solver(
        #     self.curves, solver, curves, fx, base, self.kwargs["pair"][3:]
        # )
        vol_ = self._get_fxvol_maybe_from_solver_recursive(vol, solver)
        # if not isinstance(vol, list):
        #     vol = [[vol, vol], vol]
        # else:
        #     vol = [[vol[0], vol[2]], vol[1]]  # restructure for strangle / straddle

        # TODO: this meth can be optimised because it calculates greeks at multiple times in frames
        g_grks = self.periods[0].analytic_greeks(curves, solver, fx, base, vol_[0])
        d_grks = self.periods[1].analytic_greeks(curves, solver, fx, base, vol_[1])
        sclr = abs(
            self.periods[1].periods[0].periods[0].notional  # type: ignore[union-attr]
            / self.periods[0].periods[0].periods[0].notional,  # type: ignore[union-attr]
        )

        _unit_attrs = ["delta", "gamma", "vega", "vomma", "vanna", "_kega", "_kappa", "__bs76"]
        _: dict[str, Any] = {}
        for attr in _unit_attrs:
            _[attr] = g_grks[attr] - sclr * d_grks[attr]

        _notional_attrs = [
            f"delta_{self.kwargs['pair'][:3]}",
            f"gamma_{self.kwargs['pair'][:3]}_1%",
            f"vega_{self.kwargs['pair'][3:]}",
        ]
        for attr in _notional_attrs:
            _[attr] = g_grks[attr] - d_grks[attr]

        _.update(
            {
                "__class": "FXOptionStrat",
                "__strategies": {"FXStrangle": g_grks, "FXStraddle": d_grks},
                "__delta_type": g_grks["__delta_type"],
                "__notional": self.kwargs["notional"],
            },
        )
        return _

    def _plot_payoff(
        self,
        range: list[float] | NoInput = NoInput(0),  # noqa: A002
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: FXVolStrat_ = NoInput(0),
    ) -> tuple[Any, Any]:
        vol_ = self._get_fxvol_maybe_from_solver_recursive(vol, solver)
        self._maybe_set_vega_neutral_notional(curves, solver, fx, base, vol_, metric="pips_or_%")
        return super()._plot_payoff(range, curves, solver, fx, base, local, vol_)
