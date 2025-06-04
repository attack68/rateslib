from __future__ import annotations

import json
import warnings
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import combinations, product
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from pandas import DataFrame, Series

from rateslib import defaults
from rateslib.calendars import add_tenor
from rateslib.curves import Curve, LineCurve, MultiCsaCurve, ProxyCurve
from rateslib.default import NoInput, PlotOutput, _drb, plot
from rateslib.dual import Dual, Dual2, Variable, gradient
from rateslib.fx.fx_rates import FXRates
from rateslib.mutability import (
    _clear_cache_post,
    _new_state_post,
    _validate_states,
    _WithCache,
    _WithState,
)

if TYPE_CHECKING:
    from rateslib.typing import CalInput, Number, datetime_
DualTypes: TypeAlias = (
    "Dual | Dual2 | Variable | float"  # required for non-cyclic import on _WithCache
)


"""
.. ipython:: python
   :suppress:

   from rateslib.curves import Curve
   from datetime import datetime as dt
"""


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FXForwards(_WithState, _WithCache[tuple[str, datetime], DualTypes]):
    """
    Class for storing and calculating FX forward rates.

    Parameters
    ----------
    fx_rates : FXRates, or list of such
        An ``FXRates`` object with an associated settlement date. If multiple settlement
        dates are relevant, e.g. GBPUSD (T+2) and USDCAD(T+1), then a list of
        ``FXRates`` object is allowed to create a no arbitrage framework.
    fx_curves : dict
        A dict of DF ``Curve`` objects defined by keys of two currency labels. First, by
        the currency in which cashflows occur (3-digit code), combined with the
        currency by which the future cashflow is collateralised in a derivatives sense
        (3-digit code). There must also be a curve in each currency for
        local discounting, i.e. where the cashflow and collateral currency are the
        same. See examples.
    base : str, optional
        The base currency (3-digit code). If not given defaults to the base currency
        of the first ``fx_rates`` object.

    Notes
    -----

    .. math::

       f_{DOMFOR,i} &= \\text{Forward domestic-foreign FX rate fixing on maturity date, }m_i \\\\
       F_{DOMFOR,0} &= \\text{Immediate settlement market domestic-foreign FX rate} \\\\
       v_{dom:dom,i} &= \\text{Local domestic-currency DF on maturity date, }m_i \\\\
       w_{dom:for,i} &= \\text{XCS adjusted domestic-currency DF on maturity date, }m_i \\\\

    Examples
    --------
    The most basic ``FXForwards`` object is created from a spot ``FXRates`` object and
    two local currency discount curves.

    .. ipython:: python

       from rateslib.fx import FXRates, FXForwards
       from rateslib.curves import Curve

    .. ipython:: python

       fxr = FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3))
       eur_local = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.91})
       usd_local = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.95})
       fxf = FXForwards(fxr, {"usdusd": usd_local, "eureur": eur_local, "eurusd": eur_local})

    Note that in the above the ``eur_local`` curve has also been used as the curve
    for EUR cashflows collateralised in USD, which is necessary for calculation
    of forward FX rates and cross-currency basis. With this assumption the
    cross-currency basis is implied to be zero at all points along the curve.

    Attributes
    ----------
    fx_rates : FXRates or list
    fx_curves : dict
    immediate : datetime
    currencies: dict
    q : int
    currencies_list : list
    transform : ndarray
    base : str
    fx_rates_immediate : FXRates
    """

    _mutable_by_association = True

    # @_new_state_post # handled internally
    @_clear_cache_post
    def update(self, fx_rates: list[dict[str, float]] | NoInput = NoInput(0)) -> None:
        """
        Update the FXForward object with the latest FX rates and FX curves values.

        The update method is primarily used to allow synchronous updating within a
        ``Solver``.

        Parameters
        ----------
        fx_rates: list of dict, optional
            A list of dictionaries with new rates to update the associated
            :class:`~rateslib.fx.FXRates` objects associated with the *FXForwards* object.

        Returns
        -------
        None

        Notes
        -----
        An *FXForwards* object contains associations to external objects, those being
        :class:`~rateslib.fx.FXRates` and :class:`~rateslib.curves.Curve`, and its purpose is
        to be able to combine those objects to yield FX forward rates.

        When those external objects have themselves been updated the *FXForwards* class
        will detect this via *rateslib's* cache management and will automatically update
        the *FXForwards* object. Manually calling this update on the *FXForwards* class
        also allows those associated *FXRates* classes to be updated with new market data.

        .. ipython:: python

           fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3), base="usd")
           fx_curves = {
               "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
               "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
               "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
           }
           fxf = FXForwards(fxr, fx_curves)
           fxf.rate("eurusd", dt(2022, 8, 15))

        .. ipython:: python

           fxr.update({"eurusd": 2.0})  # <-- update the associated FXRates object.
           fxf.rate("eurusd", dt(2022, 8, 15))  # <-- rate has changed, fxf has auto-updated.

        It is possible to update an *FXRates* object directly from the *FXForwards* object, via
        the ``fx_rates`` argument.

        .. ipython:: python

           fxf.update([{"eurusd": 1.50}])
           fxf.rate("eurusd", dt(2022, 8, 15))

        The :class:`~rateslib.solver.Solver` also automatically updates *FXForwards* objects
        when it mutates and solves the *Curves*.
        """
        # does not require cache validation because resets the cache_id at end of method.
        if not isinstance(fx_rates, NoInput):
            self_fx_rates = self.fx_rates if isinstance(self.fx_rates, list) else [self.fx_rates]
            if not isinstance(fx_rates, list) or len(self_fx_rates) != len(fx_rates):
                raise ValueError(
                    "`fx_rates` must be a list of dicts with length equal to the number of FXRates "
                    f"objects associated with the *FXForwards* object: {len(self_fx_rates)}."
                )
            for fxr_obj, fxr_up in zip(self_fx_rates, fx_rates, strict=True):
                fxr_obj.update(fxr_up)

        if self._state != self._get_composited_state():
            self._calculate_immediate_rates(base=self.base, init=False)
            self._set_new_state()

    @_new_state_post
    @_clear_cache_post
    def __init__(
        self,
        fx_rates: FXRates | list[FXRates],
        fx_curves: dict[str, Curve],
        base: str | NoInput = NoInput(0),
    ) -> None:
        self._ad = 1
        self._validate_fx_curves(fx_curves)
        self._fx_proxy_curves: dict[str, ProxyCurve] = {}
        self.fx_rates: FXRates | list[FXRates] = fx_rates
        self._calculate_immediate_rates(base, init=True)
        assert self.currencies_list == self.fx_rates_immediate.currencies_list  # noqa: S101

    @property
    def fx_proxy_curves(self) -> dict[str, ProxyCurve]:
        """
        A dict of cached :class:`~rateslib.curves.ProxyCurve` associated with this object.
        """
        return self._fx_proxy_curves

    def _get_composited_state(self) -> int:
        self_fx_rates = [self.fx_rates] if not isinstance(self.fx_rates, list) else self.fx_rates
        total = sum(curve._state for curve in self.fx_curves.values()) + sum(
            fxr._state for fxr in self_fx_rates
        )
        return hash(total)

    def _validate_state(self) -> None:
        if self._state != self._get_composited_state():
            self.update()

    def _validate_fx_curves(self, fx_curves: dict[str, Curve]) -> None:
        self.fx_curves: dict[str, Curve] = {k.lower(): v for k, v in fx_curves.items()}

        self.terminal: datetime = datetime(2200, 1, 1)
        for flag, (k, curve) in enumerate(self.fx_curves.items()):
            curve._meta = replace(curve._meta, _collateral=k[3:6])  # label curves with collateral

            if flag == 0:
                self.immediate: datetime = curve.nodes.keys[0]
            elif self.immediate != curve.nodes.keys[0]:
                raise ValueError("`fx_curves` do not have the same initial date.")
            if isinstance(curve, LineCurve):
                raise TypeError("`fx_curves` must be DF based, not type LineCurve.")
            if curve.nodes.final < self.terminal:
                self.terminal = curve.nodes.final

    def _calculate_immediate_rates(self, base: str | NoInput, init: bool) -> None:
        if not isinstance(self.fx_rates, list):
            # if in initialisation phase (and not update phase) populate immutable values
            if init:
                self.currencies = self.fx_rates.currencies
                self.q = len(self.currencies.keys())
                self.currencies_list: list[str] = self.fx_rates.currencies_list
                self.transform = _get_curves_indicator_array(
                    self.q,
                    self.currencies,
                    self.fx_curves,
                )
                self._paths = _create_initial_mapping(self.transform)
                self.base: str = self.fx_rates.base if isinstance(base, NoInput) else base
                self.pairs = self.fx_rates.pairs
                self.variables = tuple(f"fx_{pair}" for pair in self.pairs)
                self.pairs_settlement = self.fx_rates.pairs_settlement
            self.fx_rates_immediate = self._calculate_immediate_rates_same_settlement_frame()
        else:
            # Get values for the first FXRates in the list
            sub_curves = self._get_curves_for_currencies(
                self.fx_curves,
                self.fx_rates[0].currencies_list,
            )
            acyclic_fxf: FXForwards = FXForwards(
                fx_rates=self.fx_rates[0],
                fx_curves=sub_curves,
            )
            settlement_pairs = dict.fromkeys(self.fx_rates[0].pairs, self.fx_rates[0].settlement)

            # Now iterate through the remaining FXRates objects and patch them into the fxf
            for fx_rates_obj in self.fx_rates[1:]:
                # create sub FXForwards for each FXRates instance and re-combine.
                # This reuses the arg validation of a single FXRates object and
                # dependency of FXRates with fx_curves.

                # calculate additional FX rates from previous objects
                # in the same settlement frame.
                overlapping_currencies = [
                    ccy
                    for ccy in fx_rates_obj.currencies_list
                    if ccy in acyclic_fxf.currencies_list
                ]
                pre_currencies = [
                    ccy
                    for ccy in acyclic_fxf.currencies_list
                    if ccy not in fx_rates_obj.currencies_list
                ]
                pre_rates = {
                    f"{overlapping_currencies[0]}{ccy}": acyclic_fxf._rate_without_validation(
                        f"{overlapping_currencies[0]}{ccy}",
                        fx_rates_obj.settlement,
                    )
                    for ccy in pre_currencies
                }
                combined_fx_rates = FXRates(
                    fx_rates={**fx_rates_obj.fx_rates, **pre_rates},
                    settlement=fx_rates_obj.settlement,
                )
                sub_curves = self._get_curves_for_currencies(
                    self.fx_curves,
                    fx_rates_obj.currencies_list + pre_currencies,
                )
                acyclic_fxf = FXForwards(fx_rates=combined_fx_rates, fx_curves=sub_curves)
                settlement_pairs.update(
                    dict.fromkeys(fx_rates_obj.pairs, fx_rates_obj.settlement),
                )

            if not isinstance(base, NoInput):
                acyclic_fxf.base = base.lower()

            for attr in [
                "currencies",
                "q",
                "currencies_list",
                "transform",
                "base",
                "fx_rates_immediate",
                "pairs",
                "_paths",
            ]:
                setattr(self, attr, getattr(acyclic_fxf, attr))
            self.pairs_settlement = settlement_pairs

    def _calculate_immediate_rates_same_settlement_frame(self) -> FXRates:
        """
        Calculate the immediate FX rates values given current Curves and input FXRates obj.

        Notes
        -----
        Searches the non-diagonal elements of transformation matrix, once it has
        found a pair uses the relevant curves and the FX rate to determine the
        immediate FX rate for that pair.
        """
        # this method can only be performed on an FXForwards object that is associated to a
        # single FXRates obj (hence the use of the acyclic_fxf)
        # since this is an internal method this line is used for testing
        assert not isinstance(self.fx_rates, list)  # noqa: S101

        fx_rates_immediate: dict[str, DualTypes] = {}
        for row in range(self.q):
            for col in range(self.q):
                if row == col or self.transform[row, col] == 0:
                    continue
                cash_ccy = self.currencies_list[row]
                coll_ccy = self.currencies_list[col]
                settlement = self.fx_rates.settlement
                if isinstance(settlement, NoInput) or settlement is None:
                    raise ValueError(
                        "`fx_rates` as FXRates supplied to FXForwards must contain a "
                        "`settlement` argument.",
                    )
                v_i = self.fx_curves[f"{coll_ccy}{coll_ccy}"][settlement]
                v_0 = self.fx_curves[f"{coll_ccy}{coll_ccy}"][self.immediate]
                w_i = self.fx_curves[f"{cash_ccy}{coll_ccy}"][settlement]
                w_0 = self.fx_curves[f"{cash_ccy}{coll_ccy}"][self.immediate]
                pair = f"{cash_ccy}{coll_ccy}"
                fx_rates_immediate.update(
                    {pair: self.fx_rates.fx_array[row, col] * v_i / w_i * w_0 / v_0}
                )

        fx_rates_immediate_ = FXRates(fx_rates_immediate, self.immediate, self.currencies_list[0])
        return fx_rates_immediate_.restate(self.fx_rates.pairs, keep_ad=True)

    def __repr__(self) -> str:
        if len(self.currencies_list) > 5:
            return (
                f"<rl.FXForwards:[{','.join(self.currencies_list[:2])},"
                f"+{len(self.currencies_list) - 2} others] at {hex(id(self))}>"
            )
        else:
            return f"<rl.FXForwards:[{','.join(self.currencies_list)}] at {hex(id(self))}>"

    @staticmethod
    def _get_curves_for_currencies(
        fx_curves: dict[str, Curve], currencies: list[str]
    ) -> dict[str, Curve]:
        """produces a complete subset of fx curves given a list of currencies"""
        ps = product(currencies, currencies)
        ret = {p[0] + p[1]: fx_curves[p[0] + p[1]] for p in ps if p[0] + p[1] in fx_curves}
        return ret

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    @_validate_states
    def rate(
        self,
        pair: str,
        settlement: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the fx forward rate for a currency pair.

        Parameters
        ----------
        pair : str
            The FX pair in usual domestic:foreign convention (6 digit code).
        settlement : datetime, optional
            The settlement date of currency exchange. If not given defaults to
            immediate settlement.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        Uses the formula,

        .. math::

           f_{DOMFOR, i} = \\frac{w_{dom:for, i}}{v_{for:for, i}} F_{DOMFOR,0} = \\frac{v_{dom:dom, i}}{w_{for:dom, i}} F_{DOMFOR,0}

        where :math:`v` is a local currency discount curve and :math:`w` is a discount
        curve collateralised with an alternate currency.

        If required curves do not exist in the relevant currencies then forwards rates are chained
        using those calculable from available curves. The chain is found using a search algorithm.

        .. math::

           f_{DOMFOR, i} = f_{DOMALT, i} ...  f_{ALTFOR, i}

        """  # noqa: E501
        return self._rate_without_validation(pair, settlement)

    def _rate_without_validation(self, pair: str, settlement: datetime_ = NoInput(0)) -> DualTypes:
        settlement_: datetime = _drb(self.immediate, settlement)
        if defaults.curve_caching and (pair, settlement_) in self._cache:
            return self._cache[(pair, settlement_)]

        if settlement_ < self.immediate:
            raise ValueError("`settlement` cannot be before immediate FX rate date.")

        if settlement_ == self.immediate:
            # get FX rate directly from the immediate object
            return self._cached_value((pair, settlement_), self.fx_rates_immediate.rate(pair))
        elif isinstance(self.fx_rates, FXRates) and settlement_ == self.fx_rates.settlement:
            # get FX rate directly from the spot object
            return self._cached_value((pair, settlement_), self.fx_rates.rate(pair))

        ccy_lhs = pair[0:3].lower()
        ccy_rhs = pair[3:6].lower()
        if ccy_lhs == ccy_rhs:
            return 1.0  # then return identity

        if (self.currencies[ccy_lhs], self.currencies[ccy_rhs]) not in self._paths:
            # then paths have not been recursively determined, so determine them and cache now.
            self._paths = _recursive_pair_population(self.transform, self._paths)[1]

        via_idx = self._paths[(self.currencies[ccy_lhs], self.currencies[ccy_rhs])]
        if via_idx == -1:
            # then a rate is directly available
            return self._rate_direct(ccy_lhs, ccy_rhs, settlement_)
        else:
            # recursively determine from FX-crosses
            via_ccy = self.currencies_list[via_idx]
            ret = self.rate(f"{ccy_lhs}{via_ccy}", settlement_) * self.rate(
                f"{via_ccy}{ccy_rhs}", settlement_
            )
            return self._cached_value((pair, settlement_), ret)

    def _rate_direct(
        self,
        ccy_lhs: str,
        ccy_rhs: str,
        settlement: datetime,
    ) -> DualTypes:
        """Return a forward FX rate conditional on curves existing directly between the
        given currency indexes."""
        ccy_lhs_idx = self.currencies[ccy_lhs]
        ccy_rhs_idx = self.currencies[ccy_rhs]
        if self.transform[ccy_lhs_idx, ccy_rhs_idx] == 1:
            # f_ab = w_ab / v_bb * F_ab
            w_ab = self.fx_curves[f"{ccy_lhs}{ccy_rhs}"][settlement]
            v_bb = self.fx_curves[f"{ccy_rhs}{ccy_rhs}"][settlement]
            scalar = w_ab / v_bb
        elif self.transform[ccy_rhs_idx, ccy_lhs_idx] == 1:
            # f_ab = v_aa / w_ba * F_ab
            v_aa = self.fx_curves[f"{ccy_lhs}{ccy_lhs}"][settlement]
            w_ba = self.fx_curves[f"{ccy_rhs}{ccy_lhs}"][settlement]
            scalar = v_aa / w_ba
        else:
            raise ValueError("`fx_curves` do not exist to create a direct FX rate for the pair.")
        f = self.fx_rates_immediate.rate(f"{ccy_lhs}{ccy_rhs}")
        return self._cached_value((f"{ccy_lhs}{ccy_rhs}", settlement), scalar * f)

    @_validate_states
    def positions(
        self, value: Number, base: str | NoInput = NoInput(0), aggregate: bool = False
    ) -> Series[float] | DataFrame:
        """
        Convert a base value with FX rate sensitivities into an array of cash positions
        by settlement date.

        Parameters
        ----------
        value : float or Dual
            The amount expressed in base currency to convert to cash positions.
        base : str, optional
            The base currency in which ``value`` is given (3-digit code). If not given
            assumes the ``base`` of the object.
        aggregate : bool, optional
            Whether to aggregate positions across all settlement dates and yield
            a single column Series.

        Returns
        -------
        DataFrame or Series

        Examples
        --------
        .. ipython:: python

           fxr1 = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
           fxr2 = FXRates({"usdcad": 1.1}, settlement=dt(2022, 1, 2))
           fxf = FXForwards(
               fx_rates=[fxr1, fxr2],
               fx_curves={
                   "usdusd": Curve({dt(2022, 1, 1):1.0, dt(2022, 2, 1): 0.999}),
                   "eureur": Curve({dt(2022, 1, 1):1.0, dt(2022, 2, 1): 0.999}),
                   "cadcad": Curve({dt(2022, 1, 1):1.0, dt(2022, 2, 1): 0.999}),
                   "usdeur": Curve({dt(2022, 1, 1):1.0, dt(2022, 2, 1): 0.999}),
                   "cadusd": Curve({dt(2022, 1, 1):1.0, dt(2022, 2, 1): 0.999}),
               }
           )
           fxf.positions(
               value=Dual(100000, ["fx_eurusd", "fx_usdcad"], [-100000, -150000]),
               base="usd",
           )

        """
        if isinstance(value, float | int):
            value = Dual(value, [], [])
        base_: str = self.base if isinstance(base, NoInput) else base.lower()
        _ = np.array(
            [0 if ccy != base_ else float(value) for ccy in self.currencies_list],
        )  # this is an NPV so is assumed to be immediate settlement

        if isinstance(self.fx_rates, list):
            fx_rates = self.fx_rates
        else:
            fx_rates = [self.fx_rates]

        dates = list({fxr.settlement for fxr in fx_rates})
        if self.immediate not in dates:
            dates.insert(0, self.immediate)
        df = DataFrame(0.0, index=self.currencies_list, columns=dates)
        df.loc[base_, self.immediate] = float(value)
        for pair in value.vars:
            if pair[:3] == "fx_":
                dom_, for_ = pair[3:6], pair[6:9]
                for fxr in fx_rates:
                    if dom_ in fxr.currencies_list and for_ in fxr.currencies_list:
                        delta = gradient(value, [pair])[0]
                        _ = fxr._get_positions_from_delta(delta, pair[3:], base_)
                        _ = Series(_, index=fxr.currencies_list, name=fxr.settlement)
                        df = df.add(_.to_frame(), fill_value=0.0)

        if aggregate:
            _s: Series[float] = df.sum(axis=1).rename(dates[0])
            return _s
        else:
            _d: DataFrame = df.sort_index(axis=1)
            return _d

    @_validate_states
    def convert(
        self,
        value: DualTypes,
        domestic: str,
        foreign: str | NoInput = NoInput(0),
        settlement: datetime | NoInput = NoInput(0),
        value_date: datetime | NoInput = NoInput(0),
        collateral: str | NoInput = NoInput(0),
        on_error: str = "ignore",
    ) -> DualTypes | None:
        """
        Convert an amount of a domestic currency, as of a settlement date
        into a foreign currency, valued on another date.

        Parameters
        ----------
        value : float or Dual
            The amount of the domestic currency to convert.
        domestic : str
            The domestic currency (3-digit code).
        foreign : str, optional
            The foreign currency to convert to (3-digit code). Uses instance
            ``base`` if not given.
        settlement : datetime, optional
            The date of the assumed domestic currency cashflow. If not given is
            assumed to be ``immediate`` settlement.
        value_date : datetime, optional
            The date for which the domestic cashflow is to be projected to. If not
            given is assumed to be equal to the ``settlement``.
        collateral : str, optional
            The collateral currency to project the cashflow if ``value_date`` is
            different to ``settlement``. If they are the same this is not needed.
            If not given defaults to ``domestic``.
        on_error : str in {"ignore", "warn", "raise"}
            The action taken if either ``domestic`` or ``foreign`` are not contained
            in the FX framework. `"ignore"` and `"warn"` will still return `None`.

        Returns
        -------
        Dual or None

        Examples
        --------

        .. ipython:: python

           fxr1 = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
           fxr2 = FXRates({"usdcad": 1.1}, settlement=dt(2022, 1, 2))
           fxf = FXForwards(
               fx_rates=[fxr1, fxr2],
               fx_curves={
                   "usdusd": Curve({dt(2022, 1, 1):1.0, dt(2022, 2, 1): 0.999}),
                   "eureur": Curve({dt(2022, 1, 1):1.0, dt(2022, 2, 1): 0.999}),
                   "cadcad": Curve({dt(2022, 1, 1):1.0, dt(2022, 2, 1): 0.999}),
                   "usdeur": Curve({dt(2022, 1, 1):1.0, dt(2022, 2, 1): 0.999}),
                   "cadusd": Curve({dt(2022, 1, 1):1.0, dt(2022, 2, 1): 0.999}),
               }
           )
           fxf.convert(1000, "usd", "cad")

        """
        foreign_ = _drb(self.base, foreign).lower()
        domestic_ = domestic.lower()
        collateral_ = _drb(domestic_, collateral).lower()
        for ccy in [domestic_, foreign_]:
            if ccy not in self.currencies:
                if on_error == "ignore":
                    return None
                elif on_error == "warn":
                    warnings.warn(
                        f"'{ccy}' not in FXForwards.currencies: returning None.",
                        UserWarning,
                    )
                    return None
                else:
                    raise ValueError(f"'{ccy}' not in FXForwards.currencies.")

        settlement_: datetime = _drb(self.immediate, settlement)
        value_date_: datetime = _drb(settlement_, value_date)

        fx_rate: DualTypes = self.rate(domestic_ + foreign_, settlement_)
        if value_date_ == settlement_:
            return fx_rate * value
        else:
            crv = self.curve(foreign_, collateral_)
            return fx_rate * value * crv[settlement_] / crv[value_date_]

    @_validate_states
    # this is technically unnecessary since calls pre-cached method: convert
    def convert_positions(
        self,
        array: np.ndarray[tuple[int], np.dtype[np.float64]]
        | list[float]
        | DataFrame
        | Series[float],
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Convert an input of currency cash positions into a single base currency value.

        Parameters
        ----------
        array : list, 1d ndarray of floats, or Series, or DataFrame
            The cash positions to simultaneously convert to base currency value.
            If a DataFrame, must be indexed by currencies (3-digit lowercase) and the
            column headers must be settlement dates.
            If a Series, must be indexed by currencies (3-digit lowercase).
            If a 1d array or sequence, must
            be ordered by currency as defined in the attribute ``FXForward.currencies``.
        base : str, optional
            The currency to convert to (3-digit code). Uses instance ``base`` if not
            given.

        Returns
        -------
        Dual

        Examples
        --------

        .. ipython:: python

           fxr = FXRates({"usdnok": 8.0}, settlement=dt(2022, 1, 1))
           usdusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
           noknok = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995})
           fxf = FXForwards(fxr, {"usdusd": usdusd, "noknok": noknok, "nokusd": noknok})
           fxf.currencies
           fxf.convert_positions([0, 1000000], "usd")

        .. ipython:: python

           fxr.convert_positions(Series([1000000, 0], index=["nok", "usd"]), "usd")

        .. ipython:: python

           positions = DataFrame(index=["usd", "nok"], data={
               dt(2022, 6, 2): [0, 1000000],
               dt(2022, 9, 7): [0, -1000000],
           })
           fxf.convert_positions(positions, "usd")
        """
        base = _drb(self.base, base).lower()

        if isinstance(array, Series):
            array_: DataFrame = array.to_frame(name=self.immediate)
        elif isinstance(array, DataFrame):
            array_ = array
        else:
            array_ = DataFrame({self.immediate: np.asarray(array)}, index=self.currencies_list)

        # j = self.currencies[base]
        # return np.sum(array_ * self.fx_array[:, j])
        sum_: DualTypes = 0.0
        for d in array_.columns:
            d_sum: DualTypes = 0.0
            for ccy in array_.index:
                # typing d is a datetime by default.
                value_: DualTypes | None = self.convert(array_.loc[ccy, d], ccy, base, d)  # type: ignore[arg-type]
                d_sum += 0.0 if value_ is None else value_
            if abs(d_sum) < 1e-2:
                sum_ += d_sum
            else:  # only discount if there is a real value
                value_ = self.convert(d_sum, base, base, d, self.immediate)  # type: ignore[arg-type]
                sum_ += 0.0 if value_ is None else value_
        return sum_

    @_validate_states
    def swap(
        self,
        pair: str,
        settlements: list[datetime],
    ) -> DualTypes:
        """
        Return the FXSwap mid-market rate for the given currency pair.

        Parameters
        ----------
        pair : str
            The FX pair in usual domestic:foreign convention (6-digit code).
        settlements : list of datetimes,
            The settlement date of currency exchanges.

        Returns
        -------
        Dual
        """
        fx0 = self._rate_without_validation(pair, settlements[0])
        fx1 = self._rate_without_validation(pair, settlements[1])
        return (fx1 - fx0) * 10000

    @_validate_states
    def _full_curve(self, cashflow: str, collateral: str) -> Curve:
        """
        Calculate a cash collateral curve.

        Parameters
        ----------
        cashflow : str
            The currency in which cashflows are represented (3-digit code).
        collateral : str
            The currency of the CSA against which cashflows are collateralised (3-digit
            code).

        Returns
        -------
        Curve

        Notes
        -----
        Uses the formula,

        .. math::

           w_{DOM:FOR,i} = \\frac{f_{DOMFOR,i}}{F_{DOMFOR,0}} v_{FOR:FOR,i}

        The returned curve has each DF uniquely specified on each date.
        """
        cash_ccy, coll_ccy = cashflow.lower(), collateral.lower()
        cash_idx, coll_idx = self.currencies[cash_ccy], self.currencies[coll_ccy]
        end = self.fx_curves[f"{coll_ccy}{coll_ccy}"].nodes.final
        days = (end - self.immediate).days
        nodes = {
            k: (
                self._rate_without_validation(f"{cash_ccy}{coll_ccy}", k)
                / self.fx_rates_immediate.fx_array[cash_idx, coll_idx]
                * self.fx_curves[f"{coll_ccy}{coll_ccy}"][k]
            )
            for k in [self.immediate + timedelta(days=i) for i in range(days + 1)]
        }
        c_: Curve = Curve(nodes)
        return c_

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    # @_validate_states: function does not determine values, just links to contained objects.
    def curve(
        self,
        cashflow: str,
        collateral: str | list[str] | tuple[str, ...],
        convention: str | NoInput = NoInput(1),  # will inherit from available curve
        modifier: str | NoInput = NoInput(1),  # will inherit from available curve
        calendar: CalInput = NoInput(1),  # will inherit from available curve
        id: str | NoInput = NoInput(0),  # noqa: A002
    ) -> Curve:
        """
        Return a cash collateral *Curve*.

        Parameters
        ----------
        cashflow : str
            The currency in which cashflows are represented (3-digit code).
        collateral : str, or list/tuple of such
            The currency of the CSA against which cashflows are collateralised (3-digit
            code). If a list or tuple will return a CompositeCurve in multi-CSA mode.
        convention : str
            The day count convention used for calculating rates. If `None` defaults
            to the convention in the local cashflow currency.
        modifier : str, optional
            The modification rule, in {"F", "MF", "P", "MP"}, for determining rates.
            If `False` will default to the modifier in the local cashflow currency.
        calendar : calendar or str, optional
            The holiday calendar object to use. If str, lookups named calendar
            from static data. Used for determining rates. If `False` will
            default to the calendar in the local cashflow currency.
        id : str, optional
            The identifier attached to any constructed :class:`~rateslib.fx.ProxyCurve`.

        Returns
        -------
        Curve, ProxyCurve or MultiCsaCurve

        Notes
        -----
        If the :class:`~rateslib.curves.Curve` already exists within the attribute
        ``fx_curves`` that *Curve* will be returned.

        If a :class:`~rateslib.curves.ProxyCurve` already exists with the attribute
        ``fx_proxy_curves`` that *Curve* will be returned.

        Otherwise, creates and returns a ``ProxyCurve`` which determines rates
        and DFs via the chaining method and the below formula,

        .. math::

           w_{dom:for,i} = \\frac{f_{DOMFOR,i}}{F_{DOMFOR,0}} v_{for:for,i}

        The returned curve contains contrived methods to calculate rates and DFs
        from the combination of curves and FX rates that are available within
        the given :class:`FXForwards` instance.

        For multiple collateral currencies returns a :class:`~rateslib.curves.MultiCsaCurve`.
        """
        if isinstance(collateral, list | tuple):
            # TODO add this curve to fx_proxy_curves and lexsort the collateral
            curves = []
            for coll in collateral:
                curves.append(self.curve(cashflow, coll, convention, modifier, calendar))
            curve = MultiCsaCurve(curves=curves, id=id)
            curve._meta = replace(curve.meta, _collateral=",".join([_.lower() for _ in collateral]))
            return curve

        cash_ccy, coll_ccy = cashflow.lower(), collateral.lower()
        pair = f"{cash_ccy}{coll_ccy}"

        if pair in self.fx_curves:
            return self.fx_curves[pair]
        elif pair in self._fx_proxy_curves:
            return self._fx_proxy_curves[pair]
        else:
            curve_ = ProxyCurve(
                cashflow=cash_ccy,
                collateral=coll_ccy,
                fx_forwards=self,
                convention=convention,
                modifier=modifier,
                calendar=calendar,
                id=id,
            )
            self._fx_proxy_curves[pair] = curve_
            return curve_

    @_validate_states
    def plot(
        self,
        pair: str,
        right: datetime | str | NoInput = NoInput(0),
        left: datetime | str | NoInput = NoInput(0),
        fx_swap: bool = False,
    ) -> PlotOutput:
        """
        Plot given forward FX rates.

        Parameters
        ----------
        pair : str
            The FX pair to determine rates for (6-digit code).
        right : datetime or str, optional
            The right bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the terminal date of the FXForwards object.
        left : datetime or str, optional
            The left bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the immediate FX settlement date.
        fx_swap : bool
            Whether to plot as the FX rate or as FX swap points relative to the
            initial FX rate on the left side of the chart.
            Default is `False`.

        Returns
        -------
        (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D
        """
        if isinstance(left, NoInput):
            left_: datetime = self.immediate
        elif isinstance(left, str):
            left_ = add_tenor(self.immediate, left, "NONE", NoInput(0))
        elif isinstance(left, datetime):
            left_ = left
        else:
            raise ValueError("`left` must be supplied as datetime or tenor string.")

        if isinstance(right, NoInput):
            right_: datetime = self.terminal
        elif isinstance(right, str):
            right_ = add_tenor(self.immediate, right, "NONE", NoInput(0))
        elif isinstance(right, datetime):
            right_ = right
        else:
            raise ValueError("`right` must be supplied as datetime or tenor string.")

        points: int = (right_ - left_).days
        x = [left_ + timedelta(days=i) for i in range(points)]
        rates: list[DualTypes] = [self._rate_without_validation(pair, _) for _ in x]
        if not fx_swap:
            y: list[list[DualTypes]] = [rates]
        else:
            y = [[(rate - rates[0]) * 10000 for rate in rates]]
        return plot([x] * len(y), y)

    @_clear_cache_post
    def _set_ad_order(self, order: int) -> None:
        # does not require cache validation because updates the cache_id at end of method
        self._ad = order
        for curve in self.fx_curves.values():
            curve._set_ad_order(order)

        if isinstance(self.fx_rates, list):
            for fx_rates in self.fx_rates:
                fx_rates._set_ad_order(order)
        else:
            self.fx_rates._set_ad_order(order)
        self.fx_rates_immediate._set_ad_order(order)

    @_validate_states
    def to_json(self) -> str:
        if isinstance(self.fx_rates, list):
            fx_rates: list[str] | str = [_.to_json() for _ in self.fx_rates]
        else:
            fx_rates = self.fx_rates.to_json()
        container = {
            "base": self.base,
            "fx_rates": fx_rates,
            "fx_curves": {k: v.to_json() for k, v in self.fx_curves.items()},
        }
        return json.dumps(container, default=str)

    @classmethod
    def from_json(cls, fx_forwards: str, **kwargs) -> FXForwards:  # type: ignore[no-untyped-def]
        """
        Loads an FXForwards object from JSON.

        Parameters
        ----------
        fx_forwards : str
            JSON string describing the FXForwards class. Typically constructed with
            :meth:`to_json`.

        Returns
        -------
        FXForwards

        Notes
        -----
        This method also creates new ``FXRates`` and ``Curve`` objects from JSON.
        These new objects can be accessed from the attributes of the ``FXForwards``
        instance.
        """
        from rateslib.serialization import from_json

        serial = json.loads(fx_forwards)

        if isinstance(serial["fx_rates"], list):
            fx_rates = [from_json(_) for _ in serial["fx_rates"]]
        else:
            fx_rates = from_json(serial["fx_rates"])

        fx_curves = {k: from_json(v) for k, v in serial["fx_curves"].items()}
        base = serial["base"]
        return FXForwards(fx_rates, fx_curves, base)

    def __eq__(self, other: Any) -> bool:
        """Test two FXForwards are identical"""
        if type(self) is not type(other):
            return False
        for attr in ["base"]:
            if getattr(self, attr, None) != getattr(other, attr, None):
                return False
        if self.fx_rates_immediate != other.fx_rates_immediate:
            return False

        # it is sufficient to check that FX immediate and curves are equivalent.

        # if type(self.fx_rates) != type(other.fx_rates):
        #     return False
        # if isinstance(self.fx_rates, list):
        #     if len(self.fx_rates) != len(other.fx_rates):
        #         return False
        #     for i in range(len(self.fx_rates)):
        #         # this tests FXRates are also ordered in the same on each object
        #         if self.fx_rates[i] != other.fx_rates[i]:
        #             return False
        # else:
        #     if self.fx_rates != other.fx_rates:
        #         return False

        for k, curve in self.fx_curves.items():
            if k not in other.fx_curves:
                return False
            if curve != other.fx_curves[k]:
                return False

        return True

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    # @_validate_state: unused because it is redirected to a cache_validated method (to_json)
    def copy(self) -> FXForwards:
        """
        An FXForwards copy creates a new object with copied references.
        """
        return self.from_json(self.to_json())


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def forward_fx(
    date: datetime,
    curve_domestic: Curve,
    curve_foreign: Curve,
    fx_rate: DualTypes,
    fx_settlement: datetime | NoInput = NoInput(0),
) -> DualTypes:
    """
    Return a forward FX rate based on interest rate parity.

    Parameters
    ----------
    date : datetime
        The target date to determine the adjusted FX rate for.
    curve_domestic : Curve
        The discount curve for the domestic currency. Should be collateral adjusted.
    curve_foreign : Curve
        The discount curve for the foreign currency. Should be collateral consistent
        with ``domestic curve``.
    fx_rate : float or Dual
        The known FX rate, typically spot FX given with a spot settlement date.
    fx_settlement : datetime, optional
        The date the given ``fx_rate`` will settle, i.e. spot T+2. If `None` is assumed
        to be immediate settlement, i.e. date upon which both ``curves`` have a DF
        of precisely 1.0. Method is more efficient if ``fx_rate`` is given for
        immediate settlement.

    Returns
    -------
    float, Dual, Dual2

    Notes
    -----
    We use the formula,

    .. math::

       (EURUSD) f_i = \\frac{(EUR:USD-CSA) w^*_i}{(USD:USD-CSA) v_i} F_0 = \\frac{(EUR:EUR-CSA) v^*_i}{(USD:EUR-CSA) w_i} F_0

    where :math:`w` is a collateral adjusted discount curve and :math:`v` is the
    locally derived discount curve in a given currency, and `*` denotes the domestic
    currency. :math:`F_0` is the immediate FX rate, i.e. aligning with the initial date
    on curves such that discounts factors are precisely 1.0.

    This implies that given the dates and rates supplied,

    .. math::

       f_i = \\frac{w^*_iv_j}{v_iw_j^*} f_j = \\frac{v^*_iw_j}{w_iv_j^*} f_j

    where `j` denotes the settlement date provided.

    Examples
    --------
    Using this function directly.

    .. ipython:: python

       domestic_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.96})
       foreign_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
       forward_fx(
           date=dt(2022, 7, 1),
           curve_domestic=domestic_curve,
           curve_foreign=foreign_curve,
           fx_rate=2.0,
           fx_settlement=dt(2022, 1, 3)
       )

    Best practice is to use :class:`FXForwards` classes but this method provides
    an efficient alternative and is occasionally used internally in the library.

    .. ipython:: python

       fxr = FXRates({"usdgbp": 2.0}, settlement=dt(2022, 1, 3))
       fxf = FXForwards(fxr, {
           "usdusd": domestic_curve,
           "gbpgbp": foreign_curve,
           "gbpusd": foreign_curve,
       })
       fxf.rate("usdgbp", dt(2022, 7, 1))
    """  # noqa: E501
    if date == fx_settlement:  # noqa: SIM114
        return fx_rate  # noqa: SIM114
    elif date == curve_domestic.nodes.initial and isinstance(fx_settlement, NoInput):  # noqa: SIM114
        return fx_rate  # noqa: SIM114

    _: DualTypes = curve_domestic[date] / curve_foreign[date]
    if not isinstance(fx_settlement, NoInput):
        _ *= curve_foreign[fx_settlement] / curve_domestic[fx_settlement]
    # else: fx_settlement is deemed to be immediate hence DF are both equal to 1.0
    _ *= fx_rate
    return _


def _get_curves_indicator_array(
    q: int, currencies: dict[str, int], fx_curves: dict[str, Curve]
) -> np.ndarray[tuple[int, int], np.dtype[np.int_]]:
    """
    Constructs an indicator array identifying which cash-collateral curves are available in the
    `fx_curves` dictionary.
    """
    # Define the transformation matrix with unit elements in each valid pair.
    T = np.zeros((q, q), dtype=int)
    for k, _ in fx_curves.items():
        cash, coll = k[:3].lower(), k[3:].lower()
        try:
            cash_idx, coll_idx = currencies[cash], currencies[coll]
        except KeyError:
            raise ValueError(f"`fx_curves` contains an unexpected currency: {cash} or {coll}")
        T[cash_idx, coll_idx] = 1

    _validate_curves_indicator_array(T)
    return T


def _validate_curves_indicator_array(T: np.ndarray[tuple[int, int], np.dtype[np.int_]]) -> None:
    """
    Performs checks to ensure the indicator array of cash-collateral curves contains the
    appropriate number of curves required by an FXForwards object.
    """
    q = T.shape[0]
    if T.sum() > (2 * q) - 1:
        raise ValueError(
            f"`fx_curves` is overspecified. {2 * q - 1} curves are expected "
            f"but {T.sum()} provided.",
        )
    elif T.sum() < (2 * q) - 1:
        raise ValueError(
            f"`fx_curves` is underspecified. {2 * q - 1} curves are expected "
            f"but {T.sum()} provided.",
        )
    elif T.diagonal().sum() != q:
        raise ValueError(
            "`fx_curves` must contain local cash-collateral curves for each and every currency."
        )
    elif np.linalg.matrix_rank(T) != q:
        raise ValueError("`fx_curves` contains co-dependent rates.")


def _recursive_pair_population(
    arr: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    mapping: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.int_]], dict[tuple[int, int], int]]:
    """
    Recursively scan through an indicator matrix and populate new entries.

    This identifies existing FX pairs and attempts to derive new FX pairs from those values.

    Parameters
    ----------
    arr: 2d-ndarray
        An square indicator matrix consisting only of zeros and ones.

    Notes
    -----
    ``arr`` should satify the following:

    - be a square matrix,
    - be an indicator matrix containing only zero and ones,
    - have unit diagonal,
    - sum to 2n - 1, so that the correct number of prior rates are supplied,
    - be a full rank matrix so no pairs are degenerate
    """
    # Build the initial mapping if none exists
    if mapping is None:
        _mapping: dict[tuple[int, int], int] = _create_initial_mapping(arr)
    else:
        _mapping = mapping

    # loop through currencies and find new pairs
    _arr = arr.copy()
    for i in range(len(_arr)):
        ccy_idxs = [_ for _ in range(len(_arr)) if _arr[i, _] == 1]
        pairs = combinations(ccy_idxs, 2)
        for pair in pairs:
            if _arr[pair[0], pair[1]] == 1 and _arr[pair[1], pair[0]] == 1:
                # then the rate and its inverse are already attainable
                continue
            elif _arr[pair[0], pair[1]] == 1:
                # then the inverse is directly attainable
                _mapping[pair[1], pair[0]] = _mapping[pair[0], pair[1]]
                _arr[pair[1], pair[0]] = 1
            elif _arr[pair[1], pair[0]] == 1:
                # then the inverse is directly attainable
                _mapping[pair[0], pair[1]] = _mapping[pair[1], pair[0]]
                _arr[pair[0], pair[1]] = 1
            else:
                _arr[pair[0], [pair[1]]] = 1
                _arr[pair[1], [pair[0]]] = 1
                _mapping[(pair[0], pair[1])] = i
                _mapping[(pair[1], pair[0])] = i

    if np.all(_arr == arr) or np.sum(_arr, axis=None) == len(_arr) ** 2:
        return _arr, _mapping
    else:
        return _recursive_pair_population(_arr, _mapping)


def _create_initial_mapping(
    arr: np.ndarray[tuple[int, int], np.dtype[np.int_]],
) -> dict[tuple[int, int], int]:
    """Detect the mappings immediately available and denote these with the value '-1'."""
    _mapping: dict[tuple[int, int], int] = {}
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i == j:
                continue
            if arr[i, j] == 1:
                _mapping[(i, j)] = -1
                _mapping[(j, i)] = -1
    return _mapping
