from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta
from itertools import product

import numpy as np
from pandas import DataFrame, Series
from pandas.tseries.offsets import CustomBusinessDay

from rateslib.calendars import add_tenor
from rateslib.curves import Curve, LineCurve, MultiCsaCurve, ProxyCurve
from rateslib.default import NoInput, plot
from rateslib.dual import Dual, DualTypes, gradient
from rateslib.fx.fx_rates import FXRates

"""
.. ipython:: python
   :suppress:

   from rateslib.curves import Curve
   from datetime import datetime as dt
"""


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FXForwards:
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

    def update(
        self,
        fx_rates: FXRates | list[FXRates] | NoInput = NoInput(0),
        fx_curves: dict | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        """
        Update the FXForward object with the latest FX rates and FX curves values.

        The update method is primarily used to allow synchronous updating within a
        ``Solver``.

        Parameters
        ----------
        fx_rates : FXRates, or list of such, optional
            An ``FXRates`` object with an associated settlement date. If multiple
            settlement dates are relevant, e.g. GBPUSD (T+2) and USDCAD(T+1), then a
            list of ``FXRates`` object is allowed to create a no arbitrage framework.
        fx_curves : dict, optional
            A dict of DF ``Curve`` objects defined by keys of two currency labels.
            First, by the currency in which cashflows occur (3-digit code), combined
            with the currency by which the future cashflow is collateralised in a
            derivatives sense (3-digit code). There must also be a curve in each
            currency for local discounting, i.e. where the cashflow and collateral
            currency are the same. See examples of instance instantiation.
        base : str, optional
            The base currency (3-digit code). If not given defaults to the base
            currency of the first given ``fx_rates`` object.

        Returns
        -------
        None

        Notes
        -----
        .. warning::

           **Rateslib** is an object-oriented library that uses complex associations. It
           is best practice to create objects and any associations and then use the
           ``update`` methods to push new market data to them. Recreating objects with
           new data will break object-oriented associations and possibly lead to
           undetected market data based pricing errors.

        Do **not** do this..

        .. ipython:: python

           fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3), base="usd")
           fx_curves = {
               "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
               "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
               "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
           }
           fxf = FXForwards(fxr, fx_curves)
           id(fxr) == id(fxf.fx_rates)  #  <- these objects are associated
           fxr = FXRates({"eurusd": 1.06}, settlement=dt(2022, 1, 3), base="usd")
           id(fxr) == id(fxf.fx_rates)  #  <- this association is broken by new instance
           fxf.rate("eurusd", dt(2022, 1, 3))  # <- wrong price because it is broken

        Instead **do this**..

        .. ipython:: python

           fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3), base="usd")
           fx_curves = {
               "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
               "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
               "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
           }
           fxf = FXForwards(fxr, fx_curves)
           fxr.update({"eurusd": 1.06})
           fxf.update()
           id(fxr) == id(fxf.fx_rates)  #  <- this association is maintained
           fxf.rate("eurusd", dt(2022, 1, 3))  # <- correct new price

        For regular use, an ``FXForwards`` class has its associations, with ``FXRates``
        and ``Curve`` s, set at instantiation. This means that the most common
        form of this method will be to call it with no new arguments, but after
        either one of the ``FXRates`` or ``Curve`` objects has itself been updated.

        Examples
        --------
        Updating a component ``FXRates`` instance before updating the ``FXForwards``.

        .. ipython:: python

           uu_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.96}, id="uu")
           fx_curves = {
               "usdusd": uu_curve,
               "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="ee"),
               "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.991}, id="eu"),
           }
           fx_rates = FXRates({"usdeur": 0.9}, dt(2022, 1, 3))
           fxf = FXForwards(fx_rates, fx_curves)
           fxf.rate("usdeur", dt(2022, 7, 15))
           fx_rates.update({"usdeur": 1.0})
           fxf.update()
           fxf.rate("usdeur", dt(2022, 7, 15))

        Updating an ``FXForwards`` instance with a new ``FXRates`` instance.

        .. ipython:: python

           fxf = FXForwards(FXRates({"usdeur": 0.9}, dt(2022, 1, 3)), fx_curves)
           fxf.update(FXRates({"usdeur": 1.0}, dt(2022, 1, 3)))
           fxf.rate("usdeur", dt(2022, 7, 15))

        Updating a ``Curve`` component before updating the ``FXForwards``.

        .. ipython:: python

           fxf = FXForwards(FXRates({"usdeur": 0.9}, dt(2022, 1, 3)), fx_curves)
           uu_curve.nodes[dt(2023, 1, 1)] = 0.98
           fxf.update()
           fxf.rate("usdeur", dt(2022, 7, 15))

        """
        if isinstance(fx_curves, dict):
            self.fx_curves = {k.lower(): v for k, v in fx_curves.items()}

            self.terminal = datetime(2200, 1, 1)
            for flag, (k, curve) in enumerate(self.fx_curves.items()):
                curve.collateral = k[3:6]  # label curves with collateral

                if flag == 0:
                    self.immediate: datetime = curve.node_dates[0]
                elif self.immediate != curve.node_dates[0]:
                    raise ValueError("`fx_curves` do not have the same initial date.")
                if isinstance(curve, LineCurve):
                    raise TypeError("`fx_curves` must be DF based, not type LineCurve.")
                if curve.node_dates[-1] < self.terminal:
                    self.terminal = curve.node_dates[-1]

        if fx_rates is not NoInput.blank:
            self.fx_rates = fx_rates

        if isinstance(self.fx_rates, list):
            for flag, fx_rates_obj in enumerate(self.fx_rates):
                # create sub FXForwards for each FXRates instance and re-combine.
                # This reuses the arg validation of a single FXRates object and
                # dependency of FXRates with fx_curves.
                if flag == 0:
                    sub_curves = self._get_curves_for_currencies(
                        self.fx_curves, fx_rates_obj.currencies_list
                    )
                    acyclic_fxf: FXForwards = FXForwards(
                        fx_rates=fx_rates_obj,
                        fx_curves=sub_curves,
                    )
                    settlement_pairs = {
                        pair: fx_rates_obj.settlement for pair in fx_rates_obj.pairs
                    }
                else:
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
                        f"{overlapping_currencies[0]}{ccy}": acyclic_fxf.rate(
                            f"{overlapping_currencies[0]}{ccy}", fx_rates_obj.settlement
                        )
                        for ccy in pre_currencies
                    }
                    combined_fx_rates = FXRates(
                        fx_rates={**fx_rates_obj.fx_rates, **pre_rates},
                        settlement=fx_rates_obj.settlement,
                    )
                    sub_curves = self._get_curves_for_currencies(
                        self.fx_curves, fx_rates_obj.currencies_list + pre_currencies
                    )
                    acyclic_fxf = FXForwards(fx_rates=combined_fx_rates, fx_curves=sub_curves)
                    settlement_pairs.update(
                        {pair: fx_rates_obj.settlement for pair in fx_rates_obj.pairs}
                    )

            if base is not NoInput.blank:
                acyclic_fxf.base = base.lower()

            for attr in [
                "currencies",
                "q",
                "currencies_list",
                "transform",
                "base",
                "fx_rates_immediate",
                "pairs",
            ]:
                setattr(self, attr, getattr(acyclic_fxf, attr))
            self.pairs_settlement = settlement_pairs
        else:
            self.currencies = self.fx_rates.currencies
            self.q = len(self.currencies.keys())
            self.currencies_list: list[str] = list(self.currencies.keys())
            self.transform = self._get_forwards_transformation_matrix(
                self.q, self.currencies, self.fx_curves
            )
            self.base: str = self.fx_rates.base if base is NoInput.blank else base
            self.pairs = self.fx_rates.pairs
            self.variables = tuple(f"fx_{pair}" for pair in self.pairs)
            self.fx_rates_immediate = self._update_fx_rates_immediate()
            self.pairs_settlement = self.fx_rates.pairs_settlement

    def __init__(
        self,
        fx_rates: FXRates | list[FXRates],
        fx_curves: dict,
        base: str | NoInput = NoInput(0),
    ):
        self._ad = 1
        self.update(fx_rates, fx_curves, base)

    @staticmethod
    def _get_curves_for_currencies(fx_curves, currencies):
        ps = product(currencies, currencies)
        ret = {p[0] + p[1]: fx_curves[p[0] + p[1]] for p in ps if p[0] + p[1] in fx_curves}
        return ret

    @staticmethod
    def _get_forwards_transformation_matrix(q, currencies, fx_curves):
        """
        Performs checks to ensure FX forwards can be generated from provided DF curves.

        The transformation matrix has cash currencies by row and collateral currencies
        by column.
        """
        # Define the transformation matrix with unit elements in each valid pair.
        T = np.zeros((q, q))
        for k, _ in fx_curves.items():
            cash, coll = k[:3].lower(), k[3:].lower()
            try:
                cash_idx, coll_idx = currencies[cash], currencies[coll]
            except KeyError:
                raise ValueError(f"`fx_curves` contains an unexpected currency: {cash} or {coll}")
            T[cash_idx, coll_idx] = 1

        if T.sum() > (2 * q) - 1:
            raise ValueError(
                f"`fx_curves` is overspecified. {2 * q - 1} curves are expected "
                f"but {len(fx_curves.keys())} provided."
            )
        elif T.sum() < (2 * q) - 1:
            raise ValueError(
                f"`fx_curves` is underspecified. {2 * q -1} curves are expected "
                f"but {len(fx_curves.keys())} provided."
            )
        elif np.linalg.matrix_rank(T) != q:
            raise ValueError("`fx_curves` contains co-dependent rates.")
        return T

    @staticmethod
    def _get_recursive_chain(
        T: np.ndarray,
        start_idx: int,
        search_idx: int,
        traced_paths: list[int] = [],
        recursive_path: list[dict] = [],
    ) -> tuple[bool, list[dict]]:
        """
        Recursively calculate map from a cash currency to another via collateral curves.

        Parameters
        ----------
        T : ndarray
            The transformation mapping of cash and collateral currencies.
        start_idx : int
            The index of the currency as the starting point of this search.
        search_idx : int
            The index of the currency identifying the termination of search.
        traced_paths : list[int]
            The index of currencies that have already been exhausted within the search.
        recursive_path : list[dict]
            The path taken from the original start to the current search start location.

        Returns
        -------
        bool, path

        Notes
        -----
        The return path outlines the route taken from the ``start_idx`` to the
        ``search_idx`` detailing each step as either traversing a row or column.

        Examples
        --------
        .. ipython:: python

           T = np.array([[1,1,1,0], [0,1,0,1],[0,0,1,0],[0,0,0,1]])
           FXForwards._get_recursive_chain(T, 0, 3)

        """
        recursive_path = recursive_path.copy()
        traced_paths = traced_paths.copy()
        if len(traced_paths) == 0:
            traced_paths.append(start_idx)

        # try row:
        row_paths = np.where(T[start_idx, :] == 1)[0]
        col_paths = np.where(T[:, start_idx] == 1)[0]
        if search_idx in row_paths:
            recursive_path.append({"row": search_idx})
            return True, recursive_path
        if search_idx in col_paths:
            recursive_path.append({"col": search_idx})
            return True, recursive_path

        for axis, paths in [("row", row_paths), ("col", col_paths)]:
            for path_idx in paths:
                if path_idx == start_idx:
                    pass
                elif path_idx != search_idx and path_idx not in traced_paths:
                    recursive_path_app = recursive_path + [{axis: path_idx}]
                    traced_paths_app = traced_paths + [path_idx]
                    recursion = FXForwards._get_recursive_chain(
                        T, path_idx, search_idx, traced_paths_app, recursive_path_app
                    )
                    if recursion[0]:
                        return recursion

        return False, recursive_path

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def _update_fx_rates_immediate(self):
        """
        Find the immediate FX rates values.

        Notes
        -----
        Searches the non-diagonal elements of transformation matrix, once it has
        found a pair uses the relevant curves and the FX rate to determine the
        immediate FX rate for that pair.
        """
        fx_rates_immediate = {}
        for row in range(self.q):
            for col in range(self.q):
                if row == col or self.transform[row, col] == 0:
                    continue
                cash_ccy = self.currencies_list[row]
                coll_ccy = self.currencies_list[col]
                settlement = self.fx_rates.settlement
                if settlement is NoInput.blank or settlement is None:
                    raise ValueError(
                        "`fx_rates` as FXRates supplied to FXForwards must contain a "
                        "`settlement` argument."
                    )
                v_i = self.fx_curves[f"{coll_ccy}{coll_ccy}"][settlement]
                w_i = self.fx_curves[f"{cash_ccy}{coll_ccy}"][settlement]
                pair = f"{cash_ccy}{coll_ccy}"
                fx_rates_immediate.update({pair: self.fx_rates.fx_array[row, col] * v_i / w_i})

        fx_rates_immediate = FXRates(fx_rates_immediate, self.immediate, self.base)
        return fx_rates_immediate.restate(self.fx_rates.pairs, keep_ad=True)

    def rate(
        self,
        pair: str,
        settlement: datetime | NoInput = NoInput(0),
        path: list[dict] | NoInput = NoInput(0),
        return_path: bool = False,
    ) -> DualTypes | tuple[DualTypes, list[dict]]:
        """
        Return the fx forward rate for a currency pair.

        Parameters
        ----------
        pair : str
            The FX pair in usual domestic:foreign convention (6 digit code).
        settlement : datetime, optional
            The settlement date of currency exchange. If not given defaults to
            immediate settlement.
        path : list of dict, optional
            The chain of currency collateral curves to traverse to calculate the rate.
            This is calculated automatically and this argument is provided for
            internal calculation to avoid repeatedly calculating the same path. Use of
            this argument in normal circumstances is not recommended.
        return_path : bool
            If `True` returns the path in a tuple alongside the rate. Use of this
            argument in normal circumstances is not recommended.

        Returns
        -------
        float, Dual, Dual2 or tuple

        Notes
        -----
        Uses the formula,

        .. math::

           f_{DOMFOR, i} = \\frac{w_{dom:for, i}}{v_{for:for, i}} F_{DOMFOR,0} = \\frac{v_{dom:dom, i}}{w_{for:dom, i}} F_{DOMFOR,0}

        where :math:`v` is a local currency discount curve and :math:`w` is a discount
        curve collateralised with an alternate currency.

        Where curves do not exist in the relevant currencies we chain rates available
        given the available curves.

        .. math::

           f_{DOMFOR, i} = f_{DOMALT, i} ...  f_{ALTFOR, i}

        """  # noqa: E501

        def _get_d_f_idx_and_path(pair, path: list[dict] | None) -> tuple[int, int, list[dict]]:
            domestic, foreign = pair[:3].lower(), pair[3:].lower()
            d_idx: int = self.fx_rates_immediate.currencies[domestic]
            f_idx: int = self.fx_rates_immediate.currencies[foreign]
            if path is NoInput.blank:
                path = self._get_recursive_chain(self.transform, f_idx, d_idx)[1]
            return d_idx, f_idx, path

        # perform a fast conversion if settlement aligns with known dates,
        if settlement is NoInput.blank:
            settlement = self.immediate
        elif settlement < self.immediate:  # type: ignore[operator]
            raise ValueError("`settlement` cannot be before immediate FX rate date.")

        if settlement == self.fx_rates_immediate.settlement:
            rate_ = self.fx_rates_immediate.rate(pair)
            if return_path:
                _, _, path = _get_d_f_idx_and_path(pair, path)
                return rate_, path
            return rate_
        elif isinstance(self.fx_rates, FXRates) and settlement == self.fx_rates.settlement:
            rate_ = self.fx_rates.rate(pair)
            if return_path:
                _, _, path = _get_d_f_idx_and_path(pair, path)
                return rate_, path
            return rate_

        # otherwise must rely on curves and path search which is slower
        d_idx, f_idx, path = _get_d_f_idx_and_path(pair, path)
        rate_, current_idx = 1.0, f_idx
        for route in path:
            if "col" in route:
                coll_ccy = self.currencies_list[current_idx]
                cash_ccy = self.currencies_list[route["col"]]
                w_i = self.fx_curves[f"{cash_ccy}{coll_ccy}"][settlement]
                v_i = self.fx_curves[f"{coll_ccy}{coll_ccy}"][settlement]
                rate_ *= self.fx_rates_immediate.fx_array[route["col"], current_idx]
                rate_ *= w_i / v_i
                current_idx = route["col"]
            elif "row" in route:
                coll_ccy = self.currencies_list[route["row"]]
                cash_ccy = self.currencies_list[current_idx]
                w_i = self.fx_curves[f"{cash_ccy}{coll_ccy}"][settlement]
                v_i = self.fx_curves[f"{coll_ccy}{coll_ccy}"][settlement]
                rate_ *= self.fx_rates_immediate.fx_array[route["row"], current_idx]
                rate_ *= v_i / w_i
                current_idx = route["row"]

        if return_path:
            return rate_, path
        return rate_

    def positions(self, value, base: str | NoInput = NoInput(0), aggregate: bool = False):
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
        if isinstance(value, (float, int)):
            value = Dual(value, [], [])
        base = self.base if base is NoInput.blank else base.lower()
        _ = np.array(
            [0 if ccy != base else float(value) for ccy in self.currencies_list]
        )  # this is an NPV so is assumed to be immediate settlement

        if isinstance(self.fx_rates, list):
            fx_rates = self.fx_rates
        else:
            fx_rates = [self.fx_rates]

        dates = list({fxr.settlement for fxr in fx_rates})
        if self.immediate not in dates:
            dates.insert(0, self.immediate)
        df = DataFrame(0.0, index=self.currencies_list, columns=dates)
        df.loc[base, self.immediate] = float(value)
        for pair in value.vars:
            if pair[:3] == "fx_":
                dom_, for_ = pair[3:6], pair[6:9]
                for fxr in fx_rates:
                    if dom_ in fxr.currencies_list and for_ in fxr.currencies_list:
                        delta = gradient(value, [pair])[0]
                        _ = fxr._get_positions_from_delta(delta, pair[3:], base)
                        _ = Series(_, index=fxr.currencies_list, name=fxr.settlement)
                        df = df.add(_.to_frame(), fill_value=0.0)

        if aggregate:
            _ = df.sum(axis=1).rename(dates[0])
            return _
        else:
            return df.sort_index(axis=1)

    def convert(
        self,
        value: DualTypes,
        domestic: str,
        foreign: str | NoInput = NoInput(0),
        settlement: datetime | NoInput = NoInput(0),
        value_date: datetime | NoInput = NoInput(0),
        collateral: str | NoInput = NoInput(0),
        on_error: str = "ignore",
    ):
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
        foreign = self.base if foreign is NoInput.blank else foreign.lower()
        domestic = domestic.lower()
        collateral = domestic if collateral is NoInput.blank else collateral.lower()
        for ccy in [domestic, foreign]:
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

        if settlement is NoInput.blank:
            settlement = self.immediate
        if value_date is NoInput.blank:
            value_date = settlement

        fx_rate: DualTypes = self.rate(domestic + foreign, settlement)
        if value_date == settlement:
            return fx_rate * value
        else:
            crv = self.curve(foreign, collateral)
            return fx_rate * value * crv[settlement] / crv[value_date]

    def convert_positions(
        self,
        array: np.ndarray | list | DataFrame | Series,
        base: str | NoInput = NoInput(0),
    ):
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
        base = self.base if base is NoInput.blank else base.lower()

        if isinstance(array, Series):
            array_ = array.to_frame(name=self.immediate)
        elif isinstance(array, DataFrame):
            array_ = array
        else:
            array_ = DataFrame({self.immediate: np.asarray(array)}, index=self.currencies_list)

        # j = self.currencies[base]
        # return np.sum(array_ * self.fx_array[:, j])
        sum = 0
        for d in array_.columns:
            d_sum = 0
            for ccy in array_.index:
                d_sum += self.convert(array_.loc[ccy, d], ccy, base, d)
            if abs(d_sum) < 1e-2:
                sum += d_sum
            else:  # only discount if there is a real value
                sum += self.convert(d_sum, base, base, d, self.immediate)
        return sum

    def swap(
        self,
        pair: str,
        settlements: list[datetime],
        path: list[dict] | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the FXSwap mid-market rate for the given currency pair.

        Parameters
        ----------
        pair : str
            The FX pair in usual domestic:foreign convention (6-digit code).
        settlements : list of datetimes,
            The settlement date of currency exchanges.
        path : list of dict, optional
            The chain of currency collateral curves to traverse to calculate the rate.
            This is calculated automatically and this argument is provided for
            internal calculation to avoid repeatedly calculating the same path. Use of
            this argument in normal circumstances is not recommended.

        Returns
        -------
        Dual
        """
        fx0: DualTypes = self.rate(pair, settlements[0], path)
        fx1: DualTypes = self.rate(pair, settlements[1], path)
        return (fx1 - fx0) * 10000

    def _full_curve(self, cashflow: str, collateral: str):
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
        path = self._get_recursive_chain(self.transform, coll_idx, cash_idx)[1]
        end = list(self.fx_curves[f"{coll_ccy}{coll_ccy}"].nodes.keys())[-1]
        days = (end - self.immediate).days
        nodes = {
            k: (
                self.rate(f"{cash_ccy}{coll_ccy}", k, path=path)
                / self.fx_rates_immediate.fx_array[cash_idx, coll_idx]
                * self.fx_curves[f"{coll_ccy}{coll_ccy}"][k]
            )
            for k in [self.immediate + timedelta(days=i) for i in range(days + 1)]
        }
        return Curve(nodes)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def curve(
        self,
        cashflow: str,
        collateral: str,
        convention: str | NoInput = NoInput(0),
        modifier: str | bool = False,
        calendar: CustomBusinessDay | str | bool = False,
        id: str | NoInput = NoInput(0),
    ):
        """
        Return a cash collateral curve.

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
        Curve, ProxyCurve or CompositeCurve

        Notes
        -----
        If the curve already exists within the attribute ``fx_curves`` that curve
        will be returned.

        Otherwise, returns a ``ProxyCurve`` which determines and rates
        and DFs via the chaining method and the below formula,

        .. math::

           w_{dom:for,i} = \\frac{f_{DOMFOR,i}}{F_{DOMFOR,0}} v_{for:for,i}

        The returned curve contains contrived methods to calculate rates and DFs
        from the combination of curves and FX rates that are available within
        the given :class:`FXForwards` instance.
        """
        if isinstance(collateral, (list, tuple)):
            curves = []
            for coll in collateral:
                curves.append(self.curve(cashflow, coll, convention, modifier, calendar))
            _ = MultiCsaCurve(curves=curves, id=id)
            _.collateral = ",".join([__.lower() for __ in collateral])
            return _

        cash_ccy, coll_ccy = cashflow.lower(), collateral.lower()
        pair = f"{cash_ccy}{coll_ccy}"
        if pair in self.fx_curves:
            return self.fx_curves[pair]

        return ProxyCurve(
            cashflow=cash_ccy,
            collateral=coll_ccy,
            fx_forwards=self,
            convention=convention,
            modifier=modifier,
            calendar=calendar,
            id=id,
        )

    def plot(
        self,
        pair: str,
        right: datetime | str | NoInput = NoInput(0),
        left: datetime | str | NoInput = NoInput(0),
        fx_swap: bool = False,
    ):
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
        if left is NoInput.blank:
            left_: datetime = self.immediate
        elif isinstance(left, str):
            left_ = add_tenor(self.immediate, left, "NONE", NoInput(0))
        elif isinstance(left, datetime):
            left_ = left
        else:
            raise ValueError("`left` must be supplied as datetime or tenor string.")

        if right is NoInput.blank:
            right_: datetime = self.terminal
        elif isinstance(right, str):
            right_ = add_tenor(self.immediate, right, "NONE", NoInput(0))
        elif isinstance(right, datetime):
            right_ = right
        else:
            raise ValueError("`right` must be supplied as datetime or tenor string.")

        points: int = (right_ - left_).days
        x = [left_ + timedelta(days=i) for i in range(points)]
        _, path = self.rate(pair, x[0], return_path=True)
        rates: list[DualTypes] = [self.rate(pair, _, path=path) for _ in x]
        if not fx_swap:
            y: list[DualTypes] = [rates]
        else:
            y = [(rate - rates[0]) * 10000 for rate in rates]
        return plot(x, y)

    def _set_ad_order(self, order):
        self._ad = order
        for curve in self.fx_curves.values():
            curve._set_ad_order(order)

        if isinstance(self.fx_rates, list):
            for fx_rates in self.fx_rates:
                fx_rates._set_ad_order(order)
        else:
            self.fx_rates._set_ad_order(order)
        self.fx_rates_immediate._set_ad_order(order)

    def to_json(self):
        if isinstance(self.fx_rates, list):
            fx_rates = [_.to_json() for _ in self.fx_rates]
        else:
            fx_rates = self.fx_rates.to_json()
        container = {
            "base": self.base,
            "fx_rates": fx_rates,
            "fx_curves": {k: v.to_json() for k, v in self.fx_curves.items()},
        }
        return json.dumps(container, default=str)

    @classmethod
    def from_json(cls, fx_forwards, **kwargs):
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
        from rateslib.json import from_json

        serial = json.loads(fx_forwards)

        if isinstance(serial["fx_rates"], list):
            fx_rates = [from_json(_) for _ in serial["fx_rates"]]
        else:
            fx_rates = from_json(serial["fx_rates"])

        fx_curves = {k: Curve.from_json(v) for k, v in serial["fx_curves"].items()}
        base = serial["base"]
        return FXForwards(fx_rates, fx_curves, base)

    def __eq__(self, other):
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

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
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
) -> Dual:
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
    if date == fx_settlement:
        return fx_rate
    elif date == curve_domestic.node_dates[0] and fx_settlement is NoInput.blank:
        return fx_rate

    _ = curve_domestic[date] / curve_foreign[date]
    if fx_settlement is not NoInput.blank:
        _ *= curve_foreign[fx_settlement] / curve_domestic[fx_settlement]
    # else: fx_settlement is deemed to be immediate hence DF are both equal to 1.0
    _ *= fx_rate
    return _
