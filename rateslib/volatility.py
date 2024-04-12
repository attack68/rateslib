from __future__ import annotations  # type hinting

from rateslib.dual import set_order_convert
from rateslib.splines import PPSplineF64, PPSplineDual, PPSplineDual2
from rateslib.default import plot, NoInput
from uuid import uuid4
import numpy as np
from typing import Union


class FXVolSmile:

    _ini_solve = 0  # All node values are solvable

    def __init__(
        self,
        nodes: dict,
        id: Union[str, NoInput] = NoInput(0),
        ad: int = 0,
    ):
        self.id = uuid4().hex[:5] + "_" if id is NoInput.blank else id  # 1 in a million clash
        self.nodes = nodes
        self.node_keys = list(self.nodes.keys())
        self.n = 5
        if len(self.node_keys) != 5:
            raise ValueError(
                "`FXVolSmile` currently designed only for 5 `nodes` and degrees of freedom."
            )

        l_bnd = 2 * self.node_keys[0] - self.node_keys[2]
        r_bnd = 2 * self.node_keys[-1] - self.node_keys[2]
        c = self.node_keys[2]
        mid_l = 0.5 * (self.node_keys[0] + self.node_keys[1])
        mid_r = 0.5 * (self.node_keys[3] + self.node_keys[4])
        self.t = [l_bnd] * 4 + [mid_l, c, mid_r] + [r_bnd] * 4

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
        comparators: list[FXVolSmile] = [],
        difference: bool = False,
        labels: list[str] = [],
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

        Returns
        -------
        (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D
        """
        x = np.linspace(self.t[0], self.t[-1], 501)
        vols = self.spline.ppev(x)
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
        return plot(x, y, labels)
