from __future__ import annotations  # type hinting

from rateslib.dual import set_order_convert, dual_exp, dual_inv_norm_cdf, DualTypes
from rateslib.splines import PPSplineF64, PPSplineDual, PPSplineDual2
from rateslib.default import plot, NoInput
from uuid import uuid4
import numpy as np
from typing import Union
from datetime import datetime


# class FXMoneyVolSmile:
#
#     _ini_solve = 0  # All node values are solvable
#
#     def __init__(
#         self,
#         nodes: dict,
#         id: Union[str, NoInput] = NoInput(0),
#         ad: int = 0,
#     ):
#         self.id = uuid4().hex[:5] + "_" if id is NoInput.blank else id  # 1 in a million clash
#         self.nodes = nodes
#         self.node_keys = list(self.nodes.keys())
#         self.n = 5
#         if len(self.node_keys) != 5:
#             raise ValueError(
#                 "`FXVolSmile` currently designed only for 5 `nodes` and degrees of freedom."
#             )
#
#         l_bnd = 2 * self.node_keys[0] - self.node_keys[2]
#         r_bnd = 2 * self.node_keys[-1] - self.node_keys[2]
#         c = self.node_keys[2]
#         mid_l = 0.5 * (self.node_keys[0] + self.node_keys[1])
#         mid_r = 0.5 * (self.node_keys[3] + self.node_keys[4])
#         self.t = [l_bnd] * 4 + [mid_l, c, mid_r] + [r_bnd] * 4
#         self.u_max = r_bnd
#         self.u_min = l_bnd
#
#         self._set_ad_order(ad)  # includes csolve
#
#     def csolve(self):
#         """
#         Solves **and sets** the coefficients, ``c``, of the :class:`PPSpline`.
#
#         Returns
#         -------
#         None
#
#         Notes
#         -----
#         Only impacts curves which have a knot sequence, ``t``, and a ``PPSpline``.
#         Only solves if ``c`` not given at curve initialisation.
#
#         Uses the ``spline_endpoints`` attribute on the class to determine the solving
#         method.
#         """
#         # Get the Spline classs by data types
#         if self.ad == 0:
#             Spline = PPSplineF64
#         elif self.ad == 1:
#             Spline = PPSplineDual
#         else:
#             Spline = PPSplineDual2
#
#         tau = list(self.nodes.keys())
#         y = list(self.nodes.values())
#
#         # Left side constraint
#         tau.insert(0, self.t[0])
#         y.insert(0, set_order_convert(0.0, self.ad, None))
#         left_n = 2
#
#         tau.append(self.t[-1])
#         y.append(set_order_convert(0.0, self.ad, None))
#         right_n = 2
#
#         self.spline = Spline(4, self.t, None)
#         self.spline.csolve(tau, y, left_n, right_n, False)
#         return None
#
#     def __iter__(self):
#         raise TypeError("`FXVolSmile` is not iterable.")
#
#     def __getitem__(self, item):
#         return self.spline.ppev_single(item)
#
#     def _set_ad_order(self, order: int):
#         if order == getattr(self, "ad", None):
#             return None
#         elif order not in [0, 1, 2]:
#             raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")
#
#         self.ad = order
#         self.nodes = {
#             k: set_order_convert(v, order, [f"{self.id}{i}"])
#             for i, (k, v) in enumerate(self.nodes.items())
#         }
#         self.csolve()
#         return None
#
#     def plot(
#         self,
#         comparators: list[FXMoneyVolSmile] = [],
#         difference: bool = False,
#         labels: list[str] = [],
#     ):
#         """
#         Plot given forward tenor rates from the curve.
#
#         Parameters
#         ----------
#         tenor : str
#             The tenor of the forward rates to plot, e.g. "1D", "3M".
#         right : datetime or str, optional
#             The right bound of the graph. If given as str should be a tenor format
#             defining a point measured from the initial node date of the curve.
#             Defaults to the final node of the curve minus the ``tenor``.
#         left : datetime or str, optional
#             The left bound of the graph. If given as str should be a tenor format
#             defining a point measured from the initial node date of the curve.
#             Defaults to the initial node of the curve.
#         comparators: list[Curve]
#             A list of curves which to include on the same plot as comparators.
#         difference : bool
#             Whether to plot as comparator minus base curve or outright curve levels in
#             plot. Default is `False`.
#         labels : list[str]
#             A list of strings associated with the plot and comparators. Must be same
#             length as number of plots.
#
#         Returns
#         -------
#         (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D
#         """
#         x = np.linspace(self.t[0], self.t[-1], 501)
#         vols = self.spline.ppev(x)
#         if not difference:
#             y = [vols.tolist()]
#             if comparators is not None:
#                 for comparator in comparators:
#                     y.append(comparator.spline.ppev(x).tolist())
#         elif difference and len(comparators) > 0:
#             y = []
#             for comparator in comparators:
#                 diff = [comparator.spline.ppev(x) - vols]
#                 y.append(diff)
#         return plot(x, y, labels)


class FXDeltaVolSmile:
    """
    Create an *FX Volatility Smile* at a given expiry indexed by delta percent.

    Parameters
    -----------
    nodes: dict[float, DualTypes]
        Key-value pairs for a delta amount and associated volatility. Use either 3 nodes or 5 nodes. See examples.
    eval_date: datetime
        Acts as the initial node of a *Curve*. Should be assigned today's immediate date.
    expiry: datetime
        The expiry date of the options associated with this *Smile*
    delta_type: str in {"spot", "spot_pa", "forward", "forward_pa"}
        The type of delta calculation that is used on the options to attain a delta which is referenced by the
        node keys.
    id: str, optional
        The unique identifier to distinguish between *Smiles* in a multicurrency framework and/or *Surface*.
    ad: int, optional
        Sets the automatic differentiation order. Defines whether to convert node
        values to float, :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2`. It is advised against
        using this setting directly. It is mainly used internally.
    """

    _ini_solve = 0  # All node values are solvable

    def __init__(
        self,
        nodes: dict,
        eval_date: datetime,
        expiry: datetime,
        delta_type: str,
        id: Union[str, NoInput] = NoInput(0),
        ad: int = 0,
    ):
        self.id = uuid4().hex[:5] + "_" if id is NoInput.blank else id  # 1 in a million clash
        self.nodes = nodes
        self.node_keys = list(self.nodes.keys())
        self.n = len(self.node_keys)
        self.eval_date = eval_date
        self.expiry = expiry
        self.t_expiry = (expiry - eval_date).days / 365.0
        self.t_expiry_sqrt = self.t_expiry ** 0.5
        self.delta_type = _validate_delta_type(delta_type)

        if self.n == 3:
            self.t = [0., 0., 0., 0., 0.5, 1., 1., 1., 1.]
        elif self.n == 5:
            self.t = [0., 0., 0., 0., 0.2, 0.5, 0.8, 1., 1., 1., 1.]
        else:
            raise ValueError(
                "`FXDeltaVolSmile` currently designed only for 3 or 5 `nodes` and degrees of freedom."
            )

        self._set_ad_order(ad)  # includes csolve

    def csolve(self):
        """
        Solves **and sets** the coefficients, ``c``, of the :class:`PPSpline`.

        Returns
        -------
        None

        Notes
        -----
        Only impacts curves which have a knot sequence, ``t``, and a ``PPSpline``.
        Only solves if ``c`` not given at curve initialisation.

        Uses the ``spline_endpoints`` attribute on the class to determine the solving
        method.
        """
        # Get the Spline classs by data types
        if self.ad == 0:
            Spline = PPSplineF64
        elif self.ad == 1:
            Spline = PPSplineDual
        else:
            Spline = PPSplineDual2

        tau = list(self.nodes.keys())
        y = list(self.nodes.values())

        # Left side constraint
        tau.insert(0, self.t[0])
        y.insert(0, set_order_convert(0.0, self.ad, None))
        left_n = 2

        tau.append(self.t[-1])
        y.append(set_order_convert(0.0, self.ad, None))
        right_n = 2

        self.spline = Spline(4, self.t, None)
        self.spline.csolve(tau, y, left_n, right_n, False)
        return None

    def __iter__(self):
        raise TypeError("`FXVolSmile` is not iterable.")

    def __getitem__(self, item):
        return self.spline.ppev_single(item)

    def convert_delta(
        self,
        delta: float,
        delta_type: str,
        phi: float,
        w_deli: Union[DualTypes, NoInput] = NoInput(0),
        w_spot: Union[DualTypes, NoInput] = NoInput(0),
    ):
        """
        Convert the given delta into a call delta equivalent of the type associated with the *Smile*.

        Parameters
        ----------
        delta: float
            The delta to obtain a volatility for.
        delta_type: str in {"spot", "spot_pa", "forward", "forward_pa"}
            The delta type the given delta is expressed in.
        phi: float
            Whether the given delta is assigned to a put or call option.
        w_deli: DualTypes, optional
            Required only for spot/forward conversions.
        w_spot: DualTypes, optional
            Required only for spot/forward conversions.

        Returns
        -------
        DualTypes
        """
        delta_type = _validate_delta_type(delta_type)

        if "_pa" in self.delta_type or "_pa" in delta_type:
            raise NotImplementedError("Cannot currently convert to/from premium adjusted deltas.")

        # If put delta convert to equivalent call delta
        if phi < 0:
            if delta_type == "spot":
                delta = w_deli / w_spot + delta
            else:
                delta = 1.0 + delta

        # If delta types of Smile and given do not align make conversion
        if self.delta_type == delta_type:
            return delta
        elif self.delta_type == "forward" and delta_type == "spot":
            return delta * w_spot / w_deli
        else:  # self.delta_type == "spot" and delta_type == "forward":
            return delta * w_deli / w_spot

    def get(
        self,
        delta: float,
        delta_type: str,
        phi: float,
        w_deli: Union[DualTypes, NoInput] = NoInput(0),
        w_spot: Union[DualTypes, NoInput] = NoInput(0),
    ):
        """
        Return a volatility for a provided delta.

        This function is more explicit than the `__getitem__` method of the *Smile* because it
        permits certain forward/spot delta conversions and put/call option delta conversions.

        Parameters
        ----------
        delta: float
            The delta to obtain a volatility for.
        delta_type: str in {"spot", "spot_pa", "forward", "forward_pa"}
            The delta type the given delta is expressed in.
        phi: float
            Whether the given delta is assigned to a put or call option.
        w_deli: DualTypes, optional
            Required only for spot/forward conversions.
        w_spot: DualTypes, optional
            Required only for spot/forward conversions.

        Returns
        -------
        DualTypes
        """
        return self[self.convert_delta(delta, delta_type, phi, w_deli, w_spot)]

    def _set_ad_order(self, order: int):
        if order == getattr(self, "ad", None):
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self.ad = order
        self.nodes = {
            k: set_order_convert(v, order, [f"{self.id}{i}"])
            for i, (k, v) in enumerate(self.nodes.items())
        }
        self.csolve()
        return None

    def plot(
        self,
        comparators: list[FXDeltaVolSmile] = [],
        difference: bool = False,
        labels: list[str] = [],
        x_axis: str = "delta",
    ):
        """
        Plot given forward tenor rates from the curve.

        Parameters
        ----------
        tenor : str
            The tenor of the forward rates to plot, e.g. "1D", "3M".
        right : datetime or str, optional
            The right bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the final node of the curve minus the ``tenor``.
        left : datetime or str, optional
            The left bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the initial node of the curve.
        comparators: list[Curve]
            A list of curves which to include on the same plot as comparators.
        difference : bool
            Whether to plot as comparator minus base curve or outright curve levels in
            plot. Default is `False`.
        labels : list[str]
            A list of strings associated with the plot and comparators. Must be same
            length as number of plots.
        x_axis : str in {"delta", "moneyness"}
            If "delta" the vol is shown relative to its native delta values.
            If "moneyness" the delta values are converted to :math:`K/f_d`.

        Returns
        -------
        (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D
        """
        # reversed for intuitive strike direction
        x = np.linspace(self.t[-1], self.t[0], 101)
        vols = self.spline.ppev(x)
        if x_axis == "moneyness":
            x, vols = x[1:-1], vols[1:-1]
            x_as_u = [
                dual_exp(-dual_inv_norm_cdf(_1)*_2 * self.t_expiry_sqrt / 100. + 0.0005 * _2 * _2 * self.t_expiry)
                for (_1, _2) in zip(x, vols)
            ]

        if not difference:
            y = [vols.tolist()]
            if comparators is not None:
                for comparator in comparators:
                    y.append(comparator.spline.ppev(x).tolist())
        elif difference and len(comparators) > 0:
            y = []
            for comparator in comparators:
                diff = [comparator.spline.ppev(x) - vols]
                y.append(diff)

        # reverse for intuitive strike direction
        if x_axis == "moneyness":
            return plot(x_as_u, y, labels)
        return plot(x, y, labels)


def _validate_delta_type(delta_type: str):
    if delta_type.lower() not in ["spot", "spot_pa", "forward", "forward_pa"]:
        raise ValueError("`delta_type` must be in {'spot', 'spot_pa', 'forward', 'forward_pa'}.")
    return delta_type.lower()