from __future__ import annotations  # type hinting

from rateslib.dual import (
    set_order_convert,
    dual_exp,
    dual_inv_norm_cdf,
    DualTypes,
    dual_norm_cdf,
    dual_log,
    dual_norm_pdf,
    Dual,
    Dual2,
)
from rateslib.splines import PPSplineF64, PPSplineDual, PPSplineDual2, evaluate
from rateslib.default import plot, NoInput
from rateslib.solver import newton_1dim
from rateslib.rateslibrs import index_left_f64
from uuid import uuid4
import numpy as np
from typing import Union
from datetime import datetime
from pandas import DataFrame
from pytz import UTC
from rateslib.default import _drb

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
    r"""
    Create an *FX Volatility Smile* at a given expiry indexed by delta percent.

    Parameters
    ----------
    nodes: dict[float, DualTypes]
        Key-value pairs for a delta index amount and associated volatility. See examples.
    eval_date: datetime
        Acts as the initial node of a *Curve*. Should be assigned today's immediate date.
    expiry: datetime
        The expiry date of the options associated with this *Smile*
    delta_type: str in {"spot", "spot_pa", "forward", "forward_pa"}
        The type of delta calculation that is used on the options to attain a delta which
        is referenced by the node keys.
    id: str, optional
        The unique identifier to distinguish between *Smiles* in a multicurrency framework
        and/or *Surface*.
    ad: int, optional
        Sets the automatic differentiation order. Defines whether to convert node
        values to float, :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2`. It is advised against
        using this setting directly. It is mainly used internally.

    Notes
    -----
    The *delta* axis of this *Smile* is a **negated put delta**, i.e. 0.25 corresponds to a put
    delta of -0.25. This permits increasing strike for increasing delta index.
    For a 'forward' delta type 0.25 corresponds to a call delta of 0.75 via
    put-call delta parity. For a 'spot' delta type it would not because under a 'spot' delta
    type put-call delta parity is not 1.0, but related to the spot versus forward interest rates.

    The **interpolation function** between nodes is a **cubic spline**.

    - For an *unadjusted* ``delta_type`` the range of the delta index is set to [0,1], and the
      cubic spline is **natural** with second order derivatives set to zero at the endpoints.

    - For *premium adjusted* ``delta_types`` the range of the delta index is in [0, *d*] where *d*
      is set large enough to encompass 99.99% of all possible values. The right endpoint is clamped
      with a first derivative of zero to avoid uncontrolled behaviour. The value of *d* is derived
      using :math:`d = e^{\sigma \sqrt{t} (3.75 + \frac{1}{2} \sigma \sqrt{t})}`

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
        self.t_expiry_sqrt = self.t_expiry**0.5

        self.delta_type = _validate_delta_type(delta_type)

        if "_pa" in self.delta_type:
            vol = list(self.nodes.values())[-1] / 100.0
            upper_bound = dual_exp(
                vol * self.t_expiry_sqrt * (3.75 - 0.5 * vol * self.t_expiry_sqrt)
            )
            self.plot_upper_bound = dual_exp(
                vol * self.t_expiry_sqrt * (3.25 - 0.5 * vol * self.t_expiry_sqrt)
            )
            self._right_n = 1  # right hand spline endpoint will be constrained by derivative
        else:
            upper_bound = 1.0
            self.plot_upper_bound = 1.0
            self._right_n = 2  # right hand spline endpoint will be constrained by derivative

        if self.n in [1, 2]:
            self.t = [0.0] * 4 + [float(upper_bound)] * 4
        else:
            self.t = [0.0] * 4 + self.node_keys[1:-1] + [float(upper_bound)] * 4

        self._set_ad_order(ad)  # includes csolve

    def __iter__(self):
        raise TypeError("`FXDeltaVolSmile` is not iterable.")

    def __getitem__(self, item):
        """
        Get a value from the DeltaVolSmile given an item which is a delta_index.
        """
        if item > self.t[-1]:
            # raise ValueError(
            #     "Cannot index the FXDeltaVolSmile for a delta index out of bounds.\n"
            #     f"Got: {item}, valid range: [{self.t[0]}, {self.t[-1]}]"
            # )
            return self.spline.ppev_single(self.t[-1])
        elif item < self.t[0]:
            # raise ValueError(
            #     "Cannot index the FXDeltaVolSmile for a delta index out of bounds.\n"
            #     f"Got: {item}, valid range: [{self.t[0]}, {self.t[-1]}]"
            # )
            return self.spline.ppev_single(self.t[0])
        else:
            return evaluate(self.spline, item, 0)

    def _get_index(self, delta_index: float, expiry: NoInput(0)):
        """
        Return a volatility from a given delta index
        Used internally alongside Surface, where a surface also requires an expiry.
        """
        return self[delta_index]

    def get(
        self,
        delta: float,
        delta_type: str,
        phi: float,
        w_deli: Union[DualTypes, NoInput] = NoInput(0),
        w_spot: Union[DualTypes, NoInput] = NoInput(0),
        u: Union[DualTypes, NoInput] = NoInput(0),
    ):
        """
        Return a volatility for a provided real option delta.

        This function is more explicit than the `__getitem__` method of the *Smile* because it
        permits certain forward/spot delta conversions and put/call option delta conversions,
        and also converts to the index delta of the *Smile*.

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
        u: DualTypes, optional
            Required only for premium adjustment / unadjusted conversions.

        Returns
        -------
        DualTypes
        """
        return self[self._convert_delta(delta, delta_type, phi, w_deli, w_spot, u)]

    def get_from_strike(
        self,
        k: DualTypes,
        phi: float,
        f: DualTypes,
        w_deli: Union[DualTypes, NoInput] = NoInput(0),
        w_spot: Union[DualTypes, NoInput] = NoInput(0),
        expiry: Union[datetime, NoInput(0)] = NoInput(0)
    ) -> tuple:
        """
        Given an option strike return associated delta and vol values.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        phi: float
            Whether the option is call (1.0) or a put (-1.0).
        f: float, Dual, Dual2
            The forward rate at delivery of the option.
        w_deli: DualTypes, optional
            Required only for spot/forward conversions.
        w_spot: DualTypes, optional
            Required only for spot/forward conversions.
        expiry: datetime, optional
            If given, performs a check to ensure consistency of valuations. Raises if expiry requested and expiry
            of the *Smile* do not match. Used internally.

        Returns
        -------
        tuple of float, Dual, Dual2 : (delta index, vol, k)

        Notes
        -----
        This function will return a delta index associated with the *FXDeltaVolSmile* and the volatility attributed
        to the delta at that point. Recall that the delta index is the negated put option delta for the given strike
        ``k``.
        """
        expiry = _drb(self.expiry, expiry)
        if self.expiry != expiry:
            raise ValueError(
                "`expiry` of VolSmile and OptionPeriod do not match: calculation aborted "
                "due to potential pricing errors."
            )

        u = k / f  # moneyness
        eta, z_w, z_u = _delta_type_constants(self.delta_type, w_deli / w_spot, u)

        # Variables are passed to these functions so that iteration can take place using float
        # which is faster and then a final iteration at the fixed point can be included with Dual
        # variables to capture fixed point sensitivity.
        def root(delta, u, sqrt_t, z_u, z_w, ad):
            # Function value
            delta_index = self._delta_index_from_call_or_put_delta(delta, phi, z_w, u)
            vol_ = self[delta_index] / 100.0
            vol_ = float(vol_) if ad == 0 else vol_
            vol_sqrt_t = sqrt_t * vol_
            d_plus_min = -dual_log(u) / vol_sqrt_t + eta * vol_sqrt_t
            f0 = delta - z_w * z_u * phi * dual_norm_cdf(phi * d_plus_min)
            # Derivative
            dvol_ddelta = -1.0 * evaluate(self.spline, delta_index, 1) / 100.0
            dvol_ddelta = float(dvol_ddelta) if ad == 0 else dvol_ddelta
            dd_ddelta = dvol_ddelta * (dual_log(u) * sqrt_t / vol_sqrt_t**2 + eta * sqrt_t)
            f1 = 1 - z_w * z_u * dual_norm_pdf(phi * d_plus_min) * dd_ddelta
            return f0, f1

        # Initial approximation is obtained through the closed form solution of the delta given
        # an approximated delta at close to the base of the smile.
        avg_vol = float(list(self.nodes.values())[int(self.n / 2)]) / 100.0
        d_plus_min = -dual_log(float(u)) / (
            avg_vol * float(self.t_expiry_sqrt)
        ) + eta * avg_vol * float(self.t_expiry_sqrt)
        delta_0 = float(z_u) * phi * float(z_w) * dual_norm_cdf(phi * d_plus_min)

        solver_result = newton_1dim(
            root,
            delta_0,
            args=(u, self.t_expiry_sqrt, z_u, z_w),
            pre_args=(0,),
            final_args=(1,),
            conv_tol=1e-13,
        )
        delta = solver_result["g"]

        # # Explicit Final iteration method
        # root_solver = _newton(
        #     root, root_deriv, delta_0, args=(float(u), float(self.t_expiry_sqrt), float(z_u), float(z_w), 0), tolerance=1e-15,
        # )
        # # Final iteration of fixed point to capture AD sensitivity
        # root_solver = _newton(
        #     root, root_deriv, root_solver[0], args=(u, self.t_expiry_sqrt, z_u, z_w, 1), max_iter=1, tolerance=1e-10
        # )
        # delta = root_solver[0]

        # # Analytical determination of AD sensitivity
        # h = root(root_solver[0], u, self.t_expiry_sqrt, z_u, z_w, 1)
        # dh = root_deriv(root_solver[0], u, self.t_expiry_sqrt, z_u, z_w, 1)
        #
        # vars = list(set(h.vars).union(dh.vars))
        # dual = (-1.0 / float(dh)) * gradient(h, vars) # + (float(h) / float(dh) ** 2) * gradient(dh, vars)
        # delta = Dual(root_solver[0], vars, dual.tolist())

        if phi > 0:
            delta_index = -1.0 * self._call_to_put_delta(delta, self.delta_type, z_w, u)
        else:
            delta_index = -1.0 * delta

        return (delta_index, self[delta_index], k)

    def _convert_delta(
        self,
        delta,
        delta_type,
        phi,
        w_deli,
        w_spot,
        u,
    ):
        """
        Convert the given option delta into a delta index associated with the *Smile*.

        Parameters
        ----------
        delta: DualTypes
            The delta to convert to an equivalent Smile delta index
        delta_type: str in {"spot", "spot_pa", "forward", "forward_pa"}
            The delta type the given delta is expressed in.
        phi: float
            Whether the given delta is assigned to a put or call option.
        w_deli: DualTypes, optional
            Required only for spot/forward conversions.
        w_spot: DualTypes, optional
            Required only for spot/forward conversions.
        u: DualTypes, optional
            Required only for premium adjustment / unadjusted conversions.

        Returns
        -------
        DualTypes
        """
        z_w = NoInput(0) if (w_deli is NoInput(0) or w_spot is NoInput(0)) else w_deli / w_spot
        eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
        eta_1, z_w_1, z_u_1 = _delta_type_constants(self.delta_type, z_w, u)

        if phi > 0:
            delta = delta - z_w_0 * z_u_0

        if eta_0 == eta_1:  # premium adjustment types are same so closed form (=> z_u_0 == z_u_1)
            if z_w_1 == z_w_0:
                return -delta
            else:
                return -delta * z_w_1 / z_w_0
        else:  # root solver
            phi_inv = dual_inv_norm_cdf(-delta / (z_w_0 * z_u_0))

            def root(delta_idx, z_1, eta_0, eta_1, sqrt_t, ad):
                # Function value
                vol_ = self[delta_idx] / 100.0
                vol_ = float(vol_) if ad == 0 else vol_
                _ = phi_inv - (eta_1 - eta_0) * vol_ * sqrt_t
                f0 = delta_idx - z_1 * dual_norm_cdf(_)
                # Derivative
                dvol_ddelta_idx = evaluate(self.spline, delta_idx, 1) / 100.0
                dvol_ddelta_idx = float(dvol_ddelta_idx) if ad == 0 else dvol_ddelta_idx
                f1 = 1 - z_1 * dual_norm_pdf(_) * (eta_1 - eta_0) * sqrt_t * dvol_ddelta_idx
                return f0, f1

            solver_result = newton_1dim(
                f=root,
                g0=min(-delta, float(w_deli / w_spot)),
                args=(z_u_1 * z_w_1, eta_0, eta_1, self.t_expiry_sqrt),
                pre_args=(0,),
                final_args=(1,),
            )
            return solver_result["g"]

    def _call_to_put_delta(
        self,
        delta: DualTypes,
        delta_type: DualTypes,
        z_w: Union[DualTypes, NoInput] = NoInput(0),
        u: Union[DualTypes, NoInput] = NoInput(0),
    ):
        """
        Convert a call delta to a put delta.

        This is required because the delta_index of the *Smile* uses negated put deltas

        Parameters
        ----------
        delta: DualTypes
            The expressed option delta.
        delta_type: str in {"forward", "spot", "forward_pa", "spot_pa"}
            The type the delta is expressed in.
        z_w: DualTypes
            The spot/forward conversion factor defined by: `w_deli / w_spot`.
        u: DualTypes
            Moneyness defined by: `k/f_d`

        Returns
        -------
        float, Dual, Dual2
        """
        if delta_type == "forward":
            return delta - 1.0
        elif delta_type == "spot":
            return delta - z_w
        elif delta_type == "forward_pa":
            return delta - u
        elif delta_type == "spot_pa":
            return delta - z_w * u
        else:
            raise ValueError("`delta_type` must be in {'forward', 'spot', 'forward_pa', 'spot_pa'}")

    def _delta_index_from_call_or_put_delta(
        self,
        delta: DualTypes,
        phi: float,
        z_w: Union[DualTypes, NoInput] = NoInput(0),
        u: Union[DualTypes, NoInput] = NoInput(0),
    ):
        """
        Get the *Smile* index delta given an option delta of the same type as the *Smile*.

        Note: This is required because the delta_index of the *Smile* uses negated put deltas.

        Parameters
        ----------
        delta: DualTypes
            The expressed option delta. This MUST be given in the same type as the *Smile*.
        phi: float
            Whether a call (1.0) or a put (-1.0)
        z_w: DualTypes
            The spot/forward conversion factor defined by: `w_deli / w_spot`.
        u: DualTypes
            Moneyness defined by: `k/f_d`

        Returns
        -------
        float, Dual, Dual2
        """
        # if call then must convert to put delta using delta parity equations
        if phi > 0:
            if self.delta_type == "forward":
                put_delta = delta - 1.0
            elif self.delta_type == "spot":
                put_delta = delta - z_w
            elif self.delta_type == "forward_pa":
                put_delta = delta - u
            else:  # self.delta_type == "spot_pa":
                put_delta = delta - z_w * u
        else:
            put_delta = delta
        return -1.0 * put_delta

    def _csolve_n1(self):
        # create a straight line by converting from one to two nodes with the first at tau=0.
        tau = list(self.nodes.keys())
        tau.insert(0, self.t[0])
        y = list(self.nodes.values()) * 2

        # Left side constraint
        tau.insert(0, self.t[0])
        y.insert(0, set_order_convert(0.0, self.ad, None))
        left_n = 2

        tau.append(self.t[-1])
        y.append(set_order_convert(0.0, self.ad, None))
        right_n = self._right_n
        return tau, y, left_n, right_n

    def _csolve_n_other(self):
        tau = list(self.nodes.keys())
        y = list(self.nodes.values())

        # Left side constraint
        tau.insert(0, self.t[0])
        y.insert(0, set_order_convert(0.0, self.ad, None))
        left_n = 2

        tau.append(self.t[-1])
        y.append(set_order_convert(0.0, self.ad, None))
        right_n = self._right_n
        return tau, y, left_n, right_n

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

        if self.n == 1:
            tau, y, left_n, right_n = self._csolve_n1()
        else:
            tau, y, left_n, right_n = self._csolve_n_other()

        self.spline = Spline(4, self.t, None)
        self.spline.csolve(tau, y, left_n, right_n, False)

        # self._create_approx_spline_conversions(Spline)
        return None

    # def _build_datatable(self):
    #     """
    #     With the given (Delta, Vol)
    #     """
    #     N_ROWS = 101  # Must be odd to have explicit midpoint (0, 1, 2, 3, 4) = 2
    #     MID = int((N_ROWS - 1) / 2)
    #
    #     # Choose an appropriate distribution of forward delta:
    #     delta = np.linspace(0, 1, N_ROWS)
    #     delta[0] = 0.0001
    #     delta[-1] = 0.9999
    #
    #     # Derive the vol directly from the spline
    #     vol = self.spline.ppev(delta)
    #
    #     # Derive d_plus from forward delta, using symmetry to reduce calculations
    #     _ = np.array([dual_inv_norm_cdf(_) for _ in delta[: MID + 1]])
    #     d_plus = np.concatenate((-1.0 * _, _[:-1][::-1]))
    #
    #     data = DataFrame(
    #         data={
    #             "index_delta": delta,
    #             "put_delta_forward": delta * -1.0,
    #             "vol": vol,
    #             "d_plus": d_plus,
    #         },
    #     )
    #     data["vol_sqrt_t"] = data["vol"] * self.t_expiry_sqrt / 100.0
    #     data["d_min"] = data["d_plus"] - data["vol_sqrt_t"]
    #     data["log_moneyness"] = (0.5 * data["vol_sqrt_t"] - data["d_plus"]) * data["vol_sqrt_t"]
    #     data["moneyness"] = data["log_moneyness"].map(dual_exp)
    #     data["put_delta_forward_pa"] = (data["d_min"].map(dual_norm_cdf) - 1.0) * data["moneyness"]
    #     return data

    # def _create_approx_spline_conversions(
    #     self, spline_class: Union[PPSplineF64, PPSplineDual, PPSplineDual2]
    # ):
    #     """
    #     Create approximation splines for (U, Vol) pairs and (Delta, U) pairs given the (Delta, Vol) spline.
    #
    #     U is moneyness i.e.: U = K / f
    #     """
    #     # TODO: this only works for forward unadjusted delta because no spot conversion takes place
    #     # Create approximate (K, Delta) curve via interpolation
    #     delta = np.array(
    #         [
    #             0.00001,
    #             0.05,
    #             0.1,
    #             0.15,
    #             0.2,
    #             0.25,
    #             0.3,
    #             0.35,
    #             0.4,
    #             0.45,
    #             0.5,
    #             0.55,
    #             0.6,
    #             0.65,
    #             0.7,
    #             0.75,
    #             0.8,
    #             0.85,
    #             0.9,
    #             0.95,
    #             0.99999,
    #         ]
    #     )
    #     vols = self.spline.ppev(delta).tolist()
    #     u = [
    #         dual_exp(
    #             -dual_inv_norm_cdf(_1) * _2 * self.t_expiry_sqrt / 100.0
    #             + 0.0005 * _2 * _2 * self.t_expiry
    #         )
    #         for (_1, _2) in zip(delta, vols)
    #     ][::-1]
    #
    #     self.spline_u_delta_approx = spline_class(t=[u[0]] * 4 + u[2:-2] + [u[-1]] * 4, k=4)
    #     self.spline_u_delta_approx.csolve(u, delta.tolist()[::-1], 0, 0, False)
    #     return None

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
        x = np.linspace(self.plot_upper_bound, self.t[0], 301)
        vols = self.spline.ppev(x)
        if x_axis == "moneyness":
            x, vols = x[40:-40], vols[40:-40]
            x_as_u = [
                dual_exp(
                    _2
                    * self.t_expiry_sqrt
                    / 100.0
                    * (dual_inv_norm_cdf(_1) * _2 * self.t_expiry_sqrt * _2 / 100.0)
                )
                for (_1, _2) in zip(x, vols)
            ]

        if not difference:
            y = [vols]
            if comparators is not None:
                for comparator in comparators:
                    y.append(comparator.spline.ppev(x))
        elif difference and len(comparators) > 0:
            y = []
            for comparator in comparators:
                diff = [comparator.spline.ppev(x) - vols]
                y.append(diff)

        # reverse for intuitive strike direction
        if x_axis == "moneyness":
            return plot(x_as_u, y, labels)
        return plot(x, y, labels)

    def _set_node_vector(self, vector, ad):
        """Update the node values in a Solver. ``ad`` in {1, 2}."""
        DualType = Dual if ad == 1 else Dual2
        DualArgs = ([],) if ad == 1 else ([], [])
        base_obj = DualType(0.0, [f"{self.id}{i}" for i in range(self.n)], *DualArgs)
        ident = np.eye(self.n)

        for i, k in enumerate(self.node_keys):
            self.nodes[k] = DualType.vars_from(
                base_obj,
                vector[i].real,
                base_obj.vars,
                ident[i, :].tolist(),
                *DualArgs[1:],
            )
        self.csolve()

    def _get_node_vector(self):
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array(list(self.nodes.values()))

    def _get_node_vars(self):
        """Get the variable names of elements updated by a Solver"""
        return tuple((f"{self.id}{i}" for i in range(self.n)))


class FXDeltaVolSurface:
    r"""
    Create an *FX Volatility Surface* parametrised by cross-sectional *Smiles* at different expiries.

    Parameters
    ----------
    delta_indexes: list[float]
        Axis values representing the delta indexes on each cross-sectional *Smile*.
    expiries: list[datetime]
        Datetimes representing the expiries of each cross-sectional *Smile*, in ascending order.
    node_values: 2d-shape of float, Dual, Dual2
        An array of values representing each node value on each cross-sectional *Smile*. Should be an array of size:
        (length of ``expiries``, length of ``delta_indexes``).
    eval_date: datetime
        Acts as the initial node of a *Curve*. Should be assigned today's immediate date.
    delta_type: str in {"spot", "spot_pa", "forward", "forward_pa"}
        The type of delta calculation that is used as the *Smiles* definition to obtain a delta index which
        is referenced by the node keys.
    id: str, optional
        The unique identifier to label the *Surface* and its variables.
    ad: int, optional
        Sets the automatic differentiation order. Defines whether to convert node
        values to float, :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2`. It is advised against
        using this setting directly. It is mainly used internally.

    Notes
    -----
    See :class:`~rateslib.fx_volatility.FXDeltaVolSmile` for a description of delta indexes and *Smile* construction.

    Interpolation along the expiry axis occurs by performing total linear variance interpolation for each *delta
    index* and then dynamically constructing a *Smile* with the usual cubic interpolation.

    See :ref:`constructing FX volatility surfaces <c-fx-smile-doc>` for more details.

    """

    _ini_solve = 0

    def __init__(
        self,
        delta_indexes: Union[list, NoInput] = NoInput(0),
        expiries: Union[list, NoInput] = NoInput(0),
        node_values: Union[list, NoInput] = NoInput(0),
        eval_date: Union[datetime, NoInput] = NoInput(0),
        delta_type: Union[str, NoInput] = NoInput(0),
        id: Union[str, NoInput] = NoInput(0),
        ad: int = 0,
    ):
        node_values = np.asarray(node_values)
        self.id = uuid4().hex[:5] + "_" if id is NoInput.blank else id  # 1 in a million clash
        self.eval_date = eval_date
        self.eval_posix = self.eval_date.replace(tzinfo=UTC).timestamp()
        self.expiries = expiries
        self.expiries_posix = [_.replace(tzinfo=UTC).timestamp() for _ in self.expiries]
        for idx in range(1, len(self.expiries)):
            if self.expiries[idx - 1] >= self.expiries[idx]:
                raise ValueError(
                    "Surface `expiries` are not sorted or contain duplicates.\n"
                )

        self.delta_indexes = delta_indexes
        self.delta_type = _validate_delta_type(delta_type)
        self.smiles = [
            FXDeltaVolSmile(
                nodes={
                    k: v for k, v in zip(self.delta_indexes, node_values[i, :])
                },
                expiry=expiry,
                eval_date=self.eval_date,
                delta_type=self.delta_type,
                id=f"{self.id}_{i}_",
            ) for i, expiry in enumerate(self.expiries)
        ]
        self.n = len(self.expiries) * len(self.delta_indexes)

        self._set_ad_order(ad)  # includes csolve on each smile

    def _set_ad_order(self, order: int):
        self.ad = order
        for smile in self.smiles:
            smile._set_ad_order(order)

    def _set_node_vector(self, vector: np.array, ad: int):
        m = len(self.delta_indexes)
        for i in range(int(len(vector) / m)):
            # smiles are indexed by expiry, shortest first
            self.smiles[i]._set_node_vector(vector[i*m:i*m+m], ad)

    def _get_node_vector(self):
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array([list(_.nodes.values()) for _ in self.smiles]).ravel()

    def _get_node_vars(self):
        """Get the variable names of elements updated by a Solver"""
        vars = ()
        for smile in self.smiles:
            vars += tuple((f"{smile.id}{i}" for i in range(smile.n)))
        return vars

    def get_smile(self, expiry: datetime):
        """
        Construct a *DeltaVolSmile* with linear total variance interpolation over delta indexes.

        Parameters
        ----------
        expiry: datetime
            The expiry for the *Smile* as cross-section of *Surface*.

        Returns
        -------
        FXDeltaVolSmile
        """
        expiry_posix = expiry.replace(tzinfo=UTC).timestamp()
        e_idx = index_left_f64(self.expiries_posix, expiry_posix)
        if abs(expiry_posix - self.expiries_posix[e_idx]) < 1e-15:
            # expiry aligns with a known smile
            return self.smiles[e_idx]
        elif expiry_posix > self.expiries_posix[-1]:
            # use the data from the last smile
            _ = FXDeltaVolSmile(
                nodes={
                    k: v for k, v in zip(self.delta_indexes, self.smiles[-1].nodes.values())
                },
                eval_date=self.eval_date,
                expiry=expiry,
                ad=self.ad,
                delta_type=self.delta_type,
                id=self.smiles[-1].id + "_ext"
            )
            return _
        elif expiry <= self.eval_date:
            raise ValueError("`expiry` before the `eval_date` of the Surface is invalid.")
        elif expiry_posix < self.expiries_posix[0]:
            # use the data from the first smile
            _ = FXDeltaVolSmile(
                nodes={
                    k: v for k, v in zip(self.delta_indexes, self.smiles[0].nodes.values())
                },
                eval_date=self.eval_date,
                expiry=expiry,
                ad=self.ad,
                delta_type=self.delta_type,
                id=self.smiles[-1].id + "_ext"
            )
            return _
        else:
            def t_var_interp(ep1, vol1, ep2, vol2, ep_t):
                """
                Return the volatility of an intermediate timestamp via total linear variance interpolation.

                Parameters
                ----------
                ep1: float
                    The left side expiry in posix timestamp.
                vol1: float, Dual, Dual2
                    The left side vol value.
                ep2: float
                    The right side expiry in posix timestamp.
                vol2: float, Dual, Dual2
                    The right side vol value.
                ep_t: float
                    The posix timestamp for the interpolated time value.

                Returns
                -------
                float, Dual, Dual2
                """
                # 86400 posix seconds per day
                # 31536000 posix seconds per 365 day year
                t_var_1 = (ep1 - self.eval_posix) * vol1 **2
                t_var_2 = (ep2 - self.eval_posix) * vol2 **2
                _ = t_var_1 + (t_var_2 - t_var_1) * (ep_t - ep1) / (ep2 - ep1)
                _ /= (ep_t - self.eval_posix)
                return _ ** 0.5

            _ = FXDeltaVolSmile(
                nodes={
                    k: t_var_interp(self.expiries_posix[e_idx], vol1, self.expiries_posix[e_idx+1], vol2, expiry_posix)
                    for k, vol1, vol2 in zip(
                        self.delta_indexes,
                        self.smiles[e_idx].nodes.values(),
                        self.smiles[e_idx+1].nodes.values()
                    )
                },
                eval_date=self.eval_date,
                expiry=expiry,
                ad=self.ad,
                delta_type=self.delta_type,
                id=self.smiles[e_idx].id + "_" + self.smiles[e_idx+1].id + "_intp"
            )
            return _

    def get_from_strike(
        self,
        k: DualTypes,
        phi: float,
        f: DualTypes,
        w_deli: Union[DualTypes, NoInput] = NoInput(0),
        w_spot: Union[DualTypes, NoInput] = NoInput(0),
        expiry: Union[datetime, NoInput(0)] = NoInput(0)
    ) -> tuple:
        """
        Given an option strike and expiry return associated delta and vol values.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        phi: float
            Whether the option is call (1.0) or a put (-1.0).
        f: float, Dual, Dual2
            The forward rate at delivery of the option.
        w_deli: DualTypes, optional
            Required only for spot/forward conversions.
        w_spot: DualTypes, optional
            Required only for spot/forward conversions.
        expiry: datetime
            Required to produce the cross-sectional *Smile* on the *Surface*.

        Returns
        -------
        tuple of float, Dual, Dual2 : (delta index, vol, k)

        Notes
        -----
        This function will return a delta index associated with the *FXDeltaVolSmile* and the volatility attributed
        to the delta at that point. Recall that the delta index is the negated put option delta for the given strike
        ``k``.
        """
        if expiry is NoInput.blank:
            raise ValueError(
                "`expiry` required to get cross-section of FXDeltaVolSurface."
            )
        smile = self.get_smile(expiry)
        return smile.get_from_strike(k, phi, f, w_deli, w_spot, expiry)

    def _get_index(self, delta_index: float, expiry: datetime):
        """
        Return a volatility from a given delta index.
        Used internally alongside Surface, where a surface also requires an expiry.
        """
        return self.get_smile(expiry)[delta_index]


def _validate_delta_type(delta_type: str):
    if delta_type.lower() not in ["spot", "spot_pa", "forward", "forward_pa"]:
        raise ValueError("`delta_type` must be in {'spot', 'spot_pa', 'forward', 'forward_pa'}.")
    return delta_type.lower()


# def _convert_same_adjustment_delta(
#     delta: float,
#     from_delta_type: str,
#     to_delta_type: str,
#     w_deli: Union[DualTypes, NoInput] = NoInput(0),
#     w_spot: Union[DualTypes, NoInput] = NoInput(0),
# ):
#     """
#     Convert a delta of one type to another, preserving its unadjusted or premium adjusted nature.
#
#     Parameters
#     ----------
#     delta: float
#         The delta to obtain a volatility for.
#     from_delta_type: str in {"spot", "forward"}
#         The delta type the given delta is expressed in.
#     to_delta_type: str in {"spot", "forward"}
#         The delta type the given delta is to be converted to
#     w_deli: DualTypes, optional
#         Required only for spot/forward conversions.
#     w_spot: DualTypes, optional
#         Required only for spot/forward conversions.
#
#     Returns
#     -------
#     DualTypes
#     """
#     if ("_pa" in from_delta_type and "_pa" not in to_delta_type) or (
#         "_pa" not in from_delta_type and "_pa" in to_delta_type
#     ):
#         raise ValueError(
#             "Can only convert between deltas of the same premium type, i.e. adjusted or unadjusted."
#         )
#
#     if from_delta_type == to_delta_type:
#         return delta
#     elif "forward" in to_delta_type and "spot" in from_delta_type:
#         return delta * w_spot / w_deli
#     else:  # to_delta_type == "spot" and from_delta_type == "forward":
#         return delta * w_deli / w_spot


#
# def _get_pricing_params_from_delta_vol(
#     delta,
#     delta_type,
#     vol: Union[DualTypes, FXDeltaVolSmile],
#     t_e,
#     phi,
#     w_deli: Union[DualTypes, NoInput] = NoInput(0),
#     w_spot: Union[DualTypes, NoInput] = NoInput(0),
# ):
#     if isinstance(vol, FXDeltaVolSmile):
#         vol_ = vol.get(delta, delta_type, phi, w_deli, w_spot)
#     else:  # vol is DualTypes
#         vol_ = vol
#
#     if "_pa" in delta_type:
#         return _get_pricing_params_from_delta_vol_adjusted_fixed_vol(
#             delta,
#             delta_type,
#             vol_,
#             t_e,
#             phi,
#             w_deli,
#             w_spot,
#         )
#     else:
#         return _get_pricing_params_from_delta_vol_unadjusted_fixed_vol(
#             delta,
#             delta_type,
#             vol_,
#             t_e,
#             phi,
#             w_deli,
#             w_spot,
#         )
#
#
# def _get_pricing_params_from_delta_vol_unadjusted_fixed_vol(
#     delta,
#     delta_type,
#     vol: DualTypes,
#     t_e,
#     phi,
#     w_deli: Union[DualTypes, NoInput] = NoInput(0),
#     w_spot: Union[DualTypes, NoInput] = NoInput(0),
# ) -> dict:
#     _ = {"delta": delta, "delta_type": delta_type, "vol": vol}
#     if "spot" in delta_type:
#         _["d_plus"] = phi * dual_inv_norm_cdf(phi * delta * w_spot / w_deli)
#     else:
#         _["d_plus"] = phi * dual_inv_norm_cdf(phi * delta)
#
#     _["vol_sqrt_t"] = vol * t_e**0.5 / 100.0
#     _["d_min"] = _["d_plus"] - _["vol_sqrt_t"]
#     _["ln_u"] = (0.5 * _["vol_sqrt_t"] - _["d_plus"]) * _["vol_sqrt_t"]
#     _["u"] = dual_exp(_["ln_u"])
#     return _
#
#
# def _get_pricing_params_from_delta_vol_adjusted_fixed_vol(
#     delta: DualTypes,
#     delta_type: str,
#     vol: DualTypes,
#     t_e: DualTypes,
#     phi: float,
#     w_deli: Union[DualTypes, NoInput] = NoInput(0),
#     w_spot: Union[DualTypes, NoInput] = NoInput(0),
# ) -> dict:
#     """
#     Iterative algorithm.
#
#     AD is preserved by newton_root.
#     """
#     # TODO can get unadjusted prcing params for out of bounds adjusted delta e.g. -1.5
#     _ = _get_pricing_params_from_delta_vol_unadjusted_fixed_vol(
#         delta, delta_type, vol, t_e, phi, w_deli, w_spot
#     )
#
#     if "spot" in delta_type:
#         z_w = w_deli / w_spot
#     else:
#         z_w = 1.0
#
#     def root(u, delta, vol_sqrt_t, z):
#         d_min = -dual_log(u) / vol_sqrt_t - 0.5 * vol_sqrt_t
#         f0 = delta - z * u * phi * dual_norm_cdf(phi * d_min)
#         f1 = z * (
#             -phi * dual_norm_cdf(phi * d_min) + u * dual_norm_pdf(phi * d_min) / (u * vol_sqrt_t)
#         )
#         return f0, f1
#
#     root_solver = newton_root(root, _["u"], args=(delta, _["vol_sqrt_t"], z_w))
#
#     _ = {"delta": delta, "delta_type": delta_type, "vol": vol}
#     _["u"] = root_solver["g"]
#     if "spot" in delta_type:
#         _["d_min"] = phi * dual_inv_norm_cdf(phi * delta * w_spot / (w_deli * _["u"]))
#     else:
#         _["d_min"] = phi * dual_inv_norm_cdf(phi * delta / _["u"])
#
#     _["vol_sqrt_t"] = vol * t_e**0.5
#     _["d_plus"] = _["d_min"] + _["vol_sqrt_t"]
#     _["ln_u"] = dual_log(_["u"])
#     return _


def _black76(
    F: DualTypes,
    K: DualTypes,
    t_e: float,
    v1: NoInput,
    v2: DualTypes,
    vol: DualTypes,
    phi: float,
):
    """
    Option price in points terms for immediate premium settlement.

    Parameters
    -----------
    F: float, Dual, Dual2
        The forward price for settlement at the delivery date.
    K: float, Dual, Dual2
        The strike price of the option.
    t_e: float
        The annualised time to expiry.
    v1: float
        Not used. The discounting rate on ccy1 side.
    v2: float, Dual, Dual2
        The discounting rate to delivery on ccy2, at the appropriate collateral rate.
    vol: float, Dual, Dual2
        The volatility measured over the period until expiry.
    phi: float
        Whether to calculate for call (1.0) or put (-1.0).

    Returns
    --------
    float, Dual, Dual2
    """
    vs = vol * t_e**0.5
    d1 = _d_plus(K, F, vs)
    d2 = d1 - vs
    Nd1, Nd2 = dual_norm_cdf(phi * d1), dual_norm_cdf(phi * d2)
    _ = phi * (F * Nd1 - K * Nd2)
    # Spot formulation instead of F (Garman Kohlhagen formulation)
    # https://quant.stackexchange.com/a/63661/29443
    # r1, r2 = dual_log(df1) / -t, dual_log(df2) / -t
    # S_imm = F * df2 / df1
    # d1 = (dual_log(S_imm / K) + (r2 - r1 + 0.5 * vol ** 2) * t) / vs
    # d2 = d1 - vs
    # Nd1, Nd2 = dual_norm_cdf(d1), dual_norm_cdf(d2)
    # _ = df1 * S_imm * Nd1 - K * df2 * Nd2
    return _ * v2


def _d_plus_min(K: DualTypes, f: DualTypes, vol_sqrt_t: DualTypes, eta: float) -> DualTypes:
    # AD preserving calculation of d_plus in Black-76 formula  (eta should +/- 0.5)
    return dual_log(f / K) / vol_sqrt_t + eta * vol_sqrt_t


def _d_plus_min_u(u: DualTypes, vol_sqrt_t: DualTypes, eta: float) -> DualTypes:
    # AD preserving calculation of d_plus in Black-76 formula  (eta should +/- 0.5)
    return -dual_log(u) / vol_sqrt_t + eta * vol_sqrt_t


def _d_min(K: DualTypes, f: DualTypes, vol_sqrt_t: DualTypes) -> DualTypes:
    return _d_plus_min(K, f, vol_sqrt_t, -0.5)


def _d_plus(K: DualTypes, f: DualTypes, vol_sqrt_t: DualTypes) -> DualTypes:
    return _d_plus_min(K, f, vol_sqrt_t, +0.5)


def _delta_type_constants(delta_type, w, u):
    """
    Get the values: (eta, z_w, z_u) for the type of expressed delta

    w: should be input as w_deli / w_spot
    u: should be input as K / f_d
    """
    if delta_type == "forward":
        return (0.5, 1.0, 1.0)
    elif delta_type == "spot":
        return (0.5, w, 1.0)
    elif delta_type == "forward_pa":
        return (-0.5, 1.0, u)
    else:  # "spot_pa"
        return (-0.5, w, u)


FXVols = Union[FXDeltaVolSmile, FXDeltaVolSurface]
FXVolObj = (FXDeltaVolSmile, FXDeltaVolSurface)