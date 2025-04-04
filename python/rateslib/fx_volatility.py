from __future__ import annotations  # type hinting

from datetime import datetime, timedelta
from datetime import datetime as dt
from typing import Any, TypeAlias
from uuid import uuid4

import numpy as np
from pandas import Series
from pytz import UTC

from rateslib import defaults
from rateslib.calendars import get_calendar
from rateslib.default import (
    NoInput,
    PlotOutput,
    _drb,
    plot,
    plot3d,
)
from rateslib.dual import (
    Dual,
    Dual2,
    Variable,
    dual_exp,
    dual_inv_norm_cdf,
    dual_log,
    dual_norm_cdf,
    dual_norm_pdf,
    newton_1dim,
    set_order_convert,
)
from rateslib.dual.utils import _dual_float
from rateslib.mutability import (
    _clear_cache_post,
    _new_state_post,
    _validate_states,
    _WithCache,
    _WithState,
)
from rateslib.rs import index_left_f64
from rateslib.splines import PPSplineDual, PPSplineDual2, PPSplineF64, evaluate

# if TYPE_CHECKING:
#     from rateslib.typing import DualTypes
DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"  # if not defined causes _WithCache failure

TERMINAL_DATE = dt(2100, 1, 1)


class FXDeltaVolSmile(_WithState, _WithCache[float, DualTypes]):
    r"""
    Create an *FX Volatility Smile* at a given expiry indexed by delta percent.

    See also the :ref:`FX Vol Surfaces section in the user guide <c-fx-smile-doc>`.

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

    @_new_state_post
    def __init__(
        self,
        nodes: dict[float, DualTypes],
        eval_date: datetime,
        expiry: datetime,
        delta_type: str,
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int = 0,
    ):
        self.id: str = (
            uuid4().hex[:5] + "_" if isinstance(id, NoInput) else id
        )  # 1 in a million clash
        self.eval_date: datetime = eval_date
        self.expiry: datetime = expiry
        self.t_expiry: float = (expiry - eval_date).days / 365.0
        self.t_expiry_sqrt: float = self.t_expiry**0.5
        self.delta_type: str = _validate_delta_type(delta_type)

        self.__set_nodes__(nodes, ad)

    def __iter__(self) -> Any:
        raise TypeError("`FXDeltaVolSmile` is not iterable.")

    def __getitem__(self, item: DualTypes) -> DualTypes:
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

    def _get_index(
        self, delta_index: DualTypes, expiry: datetime | NoInput = NoInput(0)
    ) -> DualTypes:
        """
        Return a volatility from a given delta index
        Used internally alongside Surface, where a surface also requires an expiry.
        """
        return self[delta_index]

    def get(
        self,
        delta: DualTypes,
        delta_type: str,
        phi: float,
        w_deli: DualTypes | NoInput = NoInput(0),
        w_spot: DualTypes | NoInput = NoInput(0),
        u: DualTypes | NoInput = NoInput(0),
    ) -> DualTypes:
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
        f: DualTypes,
        w_deli: DualTypes | NoInput,
        w_spot: DualTypes | NoInput,
        expiry: datetime | NoInput = NoInput(0),
    ) -> tuple[DualTypes, DualTypes, DualTypes]:
        """
        Given an option strike return associated delta and vol values.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        f: float, Dual, Dual2
            The forward rate at delivery of the option.
        w_deli: DualTypes, optional
            Required only for spot/forward conversions.
        w_spot: DualTypes, optional
            Required only for spot/forward conversions.
        expiry: datetime, optional
            If given, performs a check to ensure consistency of valuations. Raises if expiry
            requested and expiry of the *Smile* do not match. Used internally.

        Returns
        -------
        tuple of float, Dual, Dual2 : (delta index, vol, k)

        Notes
        -----
        This function will return a delta index associated with the *FXDeltaVolSmile* and the
        volatility attributed to the delta at that point. Recall that the delta index is the
        negated put option delta for the given strike ``k``.
        """
        expiry = _drb(self.expiry, expiry)
        if self.expiry != expiry:
            raise ValueError(
                "`expiry` of VolSmile and OptionPeriod do not match: calculation aborted "
                "due to potential pricing errors.",
            )

        u: DualTypes = k / f  # moneyness
        w: DualTypes | NoInput = (
            NoInput(0)
            if isinstance(w_deli, NoInput) or isinstance(w_spot, NoInput)
            else w_deli / w_spot
        )
        eta, z_w, z_u = _delta_type_constants(self.delta_type, w, u)

        # Variables are passed to these functions so that iteration can take place using float
        # which is faster and then a final iteration at the fixed point can be included with Dual
        # variables to capture fixed point sensitivity.
        def root(
            delta: DualTypes,
            u: DualTypes,
            sqrt_t: DualTypes,
            z_u: DualTypes,
            z_w: DualTypes,
            ad: int,
        ) -> tuple[DualTypes, DualTypes]:
            # Function value
            delta_index = -delta
            vol_ = self[delta_index] / 100.0
            vol_ = _dual_float(vol_) if ad == 0 else vol_
            vol_sqrt_t = sqrt_t * vol_
            d_plus_min = -dual_log(u) / vol_sqrt_t + eta * vol_sqrt_t
            f0 = delta + z_w * z_u * dual_norm_cdf(-d_plus_min)
            # Derivative
            dvol_ddelta = -1.0 * evaluate(self.spline, delta_index, 1) / 100.0
            dvol_ddelta = _dual_float(dvol_ddelta) if ad == 0 else dvol_ddelta
            dd_ddelta = dvol_ddelta * (dual_log(u) * sqrt_t / vol_sqrt_t**2 + eta * sqrt_t)
            f1 = 1 - z_w * z_u * dual_norm_pdf(-d_plus_min) * dd_ddelta
            return f0, f1

        # Initial approximation is obtained through the closed form solution of the delta given
        # an approximated delta at close to the base of the smile.
        avg_vol = _dual_float(list(self.nodes.values())[int(self.n / 2)]) / 100.0
        d_plus_min = -dual_log(_dual_float(u)) / (
            avg_vol * _dual_float(self.t_expiry_sqrt)
        ) + eta * avg_vol * _dual_float(self.t_expiry_sqrt)
        delta_0 = -_dual_float(z_u) * _dual_float(z_w) * dual_norm_cdf(-d_plus_min)

        solver_result = newton_1dim(
            root,
            delta_0,
            args=(u, self.t_expiry_sqrt, z_u, z_w),
            pre_args=(0,),
            final_args=(1,),
            conv_tol=1e-13,
        )
        delta = solver_result["g"]
        delta_index = -delta
        return delta_index, self[delta_index], k

    def _convert_delta(
        self,
        delta: DualTypes,
        delta_type: str,
        phi: float,
        w_deli: DualTypes | NoInput,
        w_spot: DualTypes | NoInput,
        u: DualTypes | NoInput,
    ) -> DualTypes:
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
        z_w = (
            NoInput(0)
            if (isinstance(w_deli, NoInput) or isinstance(w_spot, NoInput))
            else w_deli / w_spot
        )
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

            def root(
                delta_idx: DualTypes,
                z_1: DualTypes,
                eta_0: float,
                eta_1: float,
                sqrt_t: DualTypes,
                ad: int,
            ) -> tuple[DualTypes, DualTypes]:
                # Function value
                vol_ = self[delta_idx] / 100.0
                vol_ = _dual_float(vol_) if ad == 0 else vol_
                _ = phi_inv - (eta_1 - eta_0) * vol_ * sqrt_t
                f0 = delta_idx - z_1 * dual_norm_cdf(_)
                # Derivative
                dvol_ddelta_idx = evaluate(self.spline, delta_idx, 1) / 100.0
                dvol_ddelta_idx = _dual_float(dvol_ddelta_idx) if ad == 0 else dvol_ddelta_idx
                f1 = 1 - z_1 * dual_norm_pdf(_) * (eta_1 - eta_0) * sqrt_t * dvol_ddelta_idx
                return f0, f1

            g0: DualTypes = min(-delta, _dual_float(w_deli / w_spot))  # type: ignore[operator, assignment]
            solver_result = newton_1dim(
                f=root,
                g0=g0,
                args=(z_u_1 * z_w_1, eta_0, eta_1, self.t_expiry_sqrt),
                pre_args=(0,),
                final_args=(1,),
            )
            ret: DualTypes = solver_result["g"]
            return ret

    def _delta_index_from_call_or_put_delta(
        self,
        delta: DualTypes,
        phi: float,
        z_w: DualTypes | NoInput = NoInput(0),
        u: DualTypes | NoInput = NoInput(0),
    ) -> DualTypes:
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
                put_delta = delta - z_w  # type: ignore[operator]
            elif self.delta_type == "forward_pa":
                put_delta = delta - u  # type: ignore[operator]
            else:  # self.delta_type == "spot_pa":
                put_delta = delta - z_w * u  # type: ignore[operator]
        else:
            put_delta = delta
        return -1.0 * put_delta

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
    #     data["put_delta_forward_pa"] = (data["d_min"].map(dual_norm_cdf)-1.0) * data["moneyness"]
    #     return data

    # def _create_approx_spline_conversions(
    #     self, spline_class: Union[PPSplineF64, PPSplineDual, PPSplineDual2]
    # ):
    #     """
    #     Create approximation splines for (U, Vol) pairs and (Delta, U) pairs given the
    #     (Delta, Vol) spline.
    #
    #     U is moneyness i.e.: U = K / f
    #     """
    #     # TODO: this only works for forward unadjusted delta because no spot conversion takes
    #     # place
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

    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[np.object_]]:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array(list(self.nodes.values()))

    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        return tuple(f"{self.id}{i}" for i in range(self.n))

    # Plotting

    def plot(
        self,
        comparators: list[FXDeltaVolSmile] | NoInput = NoInput(0),
        difference: bool = False,
        labels: list[str] | NoInput = NoInput(0),
        x_axis: str = "delta",
    ) -> PlotOutput:
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
        comparators = _drb([], comparators)
        labels = _drb([], labels)
        x: list[float] = np.linspace(_dual_float(self.plot_upper_bound), self.t[0], 301)  # type: ignore[assignment]
        vols: list[float] | list[Dual] | list[Dual2] = self.spline.ppev(x)
        if x_axis == "moneyness":
            x, vols = x[40:-40], vols[40:-40]
            x_as_u: list[float] | list[Dual] | list[Dual2] = [  # type: ignore[assignment]
                dual_exp(
                    _2  # type: ignore[operator]
                    * self.t_expiry_sqrt
                    / 100.0
                    * (dual_inv_norm_cdf(_1) * _2 * self.t_expiry_sqrt * _2 / 100.0),  # type: ignore[operator]
                )
                for (_1, _2) in zip(x, vols, strict=True)
            ]

        if difference and not isinstance(comparators, NoInput):
            y: list[list[float] | list[Dual] | list[Dual2]] = []
            for comparator in comparators:
                diff = [(y_ - v_) for y_, v_ in zip(comparator.spline.ppev(x), vols, strict=True)]  # type: ignore[operator]
                y.append(diff)
        else:  # not difference:
            y = [vols]
            if not isinstance(comparators, NoInput):
                for comparator in comparators:
                    y.append(comparator.spline.ppev(x))

        # reverse for intuitive strike direction
        if x_axis == "moneyness":
            return plot(x_as_u, y, labels)
        return plot(x, y, labels)

    # Mutation

    def __set_nodes__(self, nodes: dict[float, DualTypes], ad: int) -> None:
        # self.ad = None

        self.nodes = nodes
        self.node_keys = list(self.nodes.keys())
        self.n = len(self.node_keys)
        if "_pa" in self.delta_type:
            vol = list(self.nodes.values())[-1] / 100.0
            upper_bound = dual_exp(
                vol * self.t_expiry_sqrt * (3.75 - 0.5 * vol * self.t_expiry_sqrt),
            )
            self.plot_upper_bound = dual_exp(
                vol * self.t_expiry_sqrt * (3.25 - 0.5 * vol * self.t_expiry_sqrt),
            )
            self._right_n = 1  # right hand spline endpoint will be constrained by derivative
        else:
            upper_bound = 1.0
            self.plot_upper_bound = 1.0
            self._right_n = 2  # right hand spline endpoint will be constrained by derivative

        if self.n in [1, 2]:
            self.t = [0.0] * 4 + [_dual_float(upper_bound)] * 4
        else:
            self.t = [0.0] * 4 + self.node_keys[1:-1] + [_dual_float(upper_bound)] * 4

        self._set_ad_order(ad)  # includes _csolve()

    def _csolve_n1(self) -> tuple[list[float], list[DualTypes], int, int]:
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

    def _csolve_n_other(self) -> tuple[list[float], list[DualTypes], int, int]:
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

    def _csolve(self) -> None:
        # Get the Spline classs by data types
        if self.ad == 0:
            Spline: type[PPSplineF64] | type[PPSplineDual] | type[PPSplineDual2] = PPSplineF64
        elif self.ad == 1:
            Spline = PPSplineDual
        else:
            Spline = PPSplineDual2

        if self.n == 1:
            tau, y, left_n, right_n = self._csolve_n1()
        else:
            tau, y, left_n, right_n = self._csolve_n_other()

        self.spline: PPSplineF64 | PPSplineDual | PPSplineDual2 = Spline(4, self.t, None)
        self.spline.csolve(tau, y, left_n, right_n, False)  # type: ignore[arg-type]

    @_new_state_post
    @_clear_cache_post
    def csolve(self) -> None:
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
        self._csolve()

    @_new_state_post
    @_clear_cache_post
    def _set_node_vector(
        self, vector: np.ndarray[tuple[int, ...], np.dtype[np.object_]], ad: int
    ) -> None:
        """
        Update the node values in a Solver. ``ad`` in {1, 2}.
        Only the real values in vector are used, dual components are dropped and restructured.
        """
        DualType: type[Dual] | type[Dual2] = Dual if ad == 1 else Dual2
        DualArgs: tuple[list[float]] | tuple[list[float], list[float]] = (
            ([],) if ad == 1 else ([], [])
        )
        base_obj = DualType(0.0, [f"{self.id}{i}" for i in range(self.n)], *DualArgs)
        ident = np.eye(self.n)

        for i, k in enumerate(self.node_keys):
            self.nodes[k] = DualType.vars_from(
                base_obj,  # type: ignore[arg-type]
                vector[i].real,
                base_obj.vars,
                ident[i, :].tolist(),  # type: ignore[arg-type]
                *DualArgs[1:],
            )
        self._csolve()

    @_clear_cache_post
    def _set_ad_order(self, order: int) -> None:
        if order == getattr(self, "ad", None):
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self.ad = order
        self.nodes = {
            k: set_order_convert(v, order, [f"{self.id}{i}"])
            for i, (k, v) in enumerate(self.nodes.items())
        }
        self._csolve()

    @_new_state_post
    @_clear_cache_post
    def update(
        self,
        nodes: dict[float, DualTypes],
    ) -> None:
        """
        Update a *Smile* with new, manually passed nodes.

        For arguments see :class:`~rateslib.fx_volatility.FXDeltaVolSmile`

        Returns
        -------
        None

        Notes
        -----

        .. warning::

           *Rateslib* is an object-oriented library that uses complex associations. Although
           Python may not object to directly mutating attributes of a *Smile* instance, this
           should be avoided in *rateslib*. Only use official ``update`` methods to mutate the
           values of an existing *Smile* instance.
           This class is labelled as a **mutable on update** object.

        """
        if any(isinstance(_, Dual2) for _ in nodes.values()):
            ad_: int = 2
        elif any(isinstance(_, Dual) for _ in nodes.values()):
            ad_ = 1
        elif any(isinstance(_, Variable) for _ in nodes.values()):
            ad_ = defaults._global_ad_order
        else:
            ad_ = 0
        self.__set_nodes__(nodes, ad_)  # this will also perform `csolve` and `clear_cache`.

    @_new_state_post
    @_clear_cache_post
    def update_node(self, key: float, value: DualTypes) -> None:
        """
        Update a single node value on the *Curve*.

        Parameters
        ----------
        key: float
            The node date to update. Must exist in ``nodes``.
        value: float, Dual, Dual2, Variable
            Value to update on the *Curve*.

        Returns
        -------
        None

        Notes
        -----

        .. warning::

           *Rateslib* is an object-oriented library that uses complex associations. Although
           Python may not object to directly mutating attributes of a *Curve* instance, this
           should be avoided in *rateslib*. Only use official ``update`` methods to mutate the
           values of an existing *Curve* instance.
           This class is labelled as a **mutable on update** object.

        """
        if key not in self.nodes:
            raise KeyError("`key` is not in Curve ``nodes``.")
        self.nodes[key] = value
        self._csolve()

    # Serialization


class FXDeltaVolSurface(_WithState, _WithCache[datetime, FXDeltaVolSmile]):
    r"""
    Create an *FX Volatility Surface* parametrised by cross-sectional *Smiles* at different
    expiries.

    See also the :ref:`FX Vol Surfaces section in the user guide <c-fx-smile-doc>`.

    Parameters
    ----------
    delta_indexes: list[float]
        Axis values representing the delta indexes on each cross-sectional *Smile*.
    expiries: list[datetime]
        Datetimes representing the expiries of each cross-sectional *Smile*, in ascending order.
    node_values: 2d-shape of float, Dual, Dual2
        An array of values representing each node value on each cross-sectional *Smile*. Should be
        an array of size: (length of ``expiries``, length of ``delta_indexes``).
    eval_date: datetime
        Acts as the initial node of a *Curve*. Should be assigned today's immediate date.
    delta_type: str in {"spot", "spot_pa", "forward", "forward_pa"}
        The type of delta calculation that is used as the *Smiles* definition to obtain a delta
        index which is referenced by the node keys.
    weights: Series, optional
        Weights used for temporal volatility interpolation. See notes.
    id: str, optional
        The unique identifier to label the *Surface* and its variables.
    ad: int, optional
        Sets the automatic differentiation order. Defines whether to convert node
        values to float, :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2`. It is advised against
        using this setting directly. It is mainly used internally.

    Notes
    -----
    See :class:`~rateslib.fx_volatility.FXDeltaVolSmile` for a description of delta indexes and
    *Smile* construction.

    **Temporal Interpolation**

    Interpolation along the expiry axis occurs by performing total linear variance interpolation
    for each *delta index* and then dynamically constructing a *Smile* with the usual cubic
    interpolation.

    If ``weights`` are given this uses the scaling approach of forward volatility (as demonstrated
    in Clark's *FX Option Pricing*) for calendar days (different options 'cuts' and timezone are
    not implemented). A datetime indexed `Series` must be provided, where any calendar date that
    is not included will be assigned the default weight of 1.0.

    See :ref:`constructing FX volatility surfaces <c-fx-smile-doc>` for more details.

    """

    _ini_solve = 0
    _mutable_by_association = True

    def __init__(
        self,
        delta_indexes: list[float],
        expiries: list[datetime],
        node_values: list[DualTypes],
        eval_date: datetime,
        delta_type: str,
        weights: Series[float] | NoInput = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int = 0,
    ):
        self.id: str = (
            uuid4().hex[:5] + "_" if isinstance(id, NoInput) else id
        )  # 1 in a million clash
        self.delta_indexes: list[float] = delta_indexes
        self.delta_type: str = _validate_delta_type(delta_type)

        self.expiries: list[datetime] = expiries
        self.expiries_posix: list[float] = [
            _.replace(tzinfo=UTC).timestamp() for _ in self.expiries
        ]
        for idx in range(1, len(self.expiries)):
            if self.expiries[idx - 1] >= self.expiries[idx]:
                raise ValueError("Surface `expiries` are not sorted or contain duplicates.\n")

        self.eval_date: datetime = eval_date
        self.eval_posix: float = self.eval_date.replace(tzinfo=UTC).timestamp()

        node_values_: np.ndarray[tuple[int, ...], np.dtype[np.object_]] = np.asarray(node_values)
        self.smiles = [
            FXDeltaVolSmile(
                nodes=dict(zip(self.delta_indexes, node_values_[i, :], strict=False)),
                expiry=expiry,
                eval_date=self.eval_date,
                delta_type=self.delta_type,
                id=f"{self.id}_{i}_",
            )
            for i, expiry in enumerate(self.expiries)
        ]
        self.n: int = len(self.expiries) * len(self.delta_indexes)

        self.weights = self._validate_weights(weights)
        self.weights_cum = (
            NoInput(0) if isinstance(self.weights, NoInput) else self.weights.cumsum()
        )

        self._set_ad_order(ad)  # includes csolve on each smile
        self._set_new_state()

    def _get_composited_state(self) -> int:
        return hash(sum(smile._state for smile in self.smiles))

    def _validate_state(self) -> None:
        if self._state != self._get_composited_state():
            # If any of the associated curves have been mutated then the cache is invalidated
            self._clear_cache()
            self._set_new_state()

    @_clear_cache_post
    def _set_ad_order(self, order: int) -> None:
        self.ad = order
        for smile in self.smiles:
            smile._set_ad_order(order)

    @_new_state_post
    @_clear_cache_post
    def _set_node_vector(
        self, vector: np.ndarray[tuple[int, ...], np.dtype[np.object_]], ad: int
    ) -> None:
        m = len(self.delta_indexes)
        for i in range(int(len(vector) / m)):
            # smiles are indexed by expiry, shortest first
            self.smiles[i]._set_node_vector(vector[i * m : i * m + m], ad)

    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[np.object_]]:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array([list(_.nodes.values()) for _ in self.smiles]).ravel()

    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        vars_: tuple[str, ...] = ()
        for smile in self.smiles:
            vars_ += tuple(f"{smile.id}{i}" for i in range(smile.n))
        return vars_

    @_validate_states
    def get_smile(self, expiry: datetime) -> FXDeltaVolSmile:
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
        if defaults.curve_caching and expiry in self._cache:
            return self._cache[expiry]

        expiry_posix = expiry.replace(tzinfo=UTC).timestamp()
        e_idx = index_left_f64(self.expiries_posix, expiry_posix)
        if expiry == self.expiries[0]:
            smile = self.smiles[0]
        elif abs(expiry_posix - self.expiries_posix[e_idx + 1]) < 1e-10:
            # expiry aligns with a known smile
            smile = self.smiles[e_idx + 1]
        elif expiry_posix > self.expiries_posix[-1]:
            # use the data from the last smile
            smile = FXDeltaVolSmile(
                nodes={
                    k: self._t_var_interp(
                        expiry_index=e_idx,
                        expiry=expiry,
                        expiry_posix=expiry_posix,
                        vol1=vol1,
                        vol2=vol1,
                        bounds_flag=1,
                    )
                    for k, vol1 in zip(
                        self.delta_indexes, self.smiles[e_idx + 1].nodes.values(), strict=False
                    )
                },
                eval_date=self.eval_date,
                expiry=expiry,
                ad=self.ad,
                delta_type=self.delta_type,
                id=self.smiles[e_idx + 1].id + "_ext",
            )
        elif expiry <= self.eval_date:
            raise ValueError("`expiry` before the `eval_date` of the Surface is invalid.")
        elif expiry_posix < self.expiries_posix[0]:
            # use the data from the first smile
            smile = FXDeltaVolSmile(
                nodes={
                    k: self._t_var_interp(
                        expiry_index=e_idx,
                        expiry=expiry,
                        expiry_posix=expiry_posix,
                        vol1=vol1,
                        vol2=vol1,
                        bounds_flag=-1,
                    )
                    for k, vol1 in zip(
                        self.delta_indexes, self.smiles[0].nodes.values(), strict=False
                    )
                },
                eval_date=self.eval_date,
                expiry=expiry,
                ad=self.ad,
                delta_type=self.delta_type,
                id=self.smiles[0].id + "_ext",
            )
        else:
            ls, rs = self.smiles[e_idx], self.smiles[e_idx + 1]  # left_smile, right_smile
            smile = FXDeltaVolSmile(
                nodes={
                    k: self._t_var_interp(
                        expiry_index=e_idx,
                        expiry=expiry,
                        expiry_posix=expiry_posix,
                        vol1=vol1,
                        vol2=vol2,
                        bounds_flag=0,
                    )
                    for k, vol1, vol2 in zip(
                        self.delta_indexes,
                        ls.nodes.values(),
                        rs.nodes.values(),
                        strict=False,
                    )
                },
                eval_date=self.eval_date,
                expiry=expiry,
                ad=self.ad,
                delta_type=self.delta_type,
                id=ls.id + "_" + rs.id + "_intp",
            )

        return self._cached_value(expiry, smile)

    def _t_var_interp(
        self,
        expiry_index: int,
        expiry: datetime,
        expiry_posix: float,
        vol1: DualTypes,
        vol2: DualTypes,
        bounds_flag: int,
    ) -> DualTypes:
        """
        Return the volatility of an intermediate timestamp via total linear variance interpolation.
        Possibly scaled by time weights if weights is available.

        Parameters
        ----------
        expiry_index: int
            The index defining the interval within which expiry falls.
        expiry: datetime
            The target expiry to be interpolated.
        expiry_posix: float
            The pre-calculated posix timestamp for expiry.
        vol1: float, Dual, DUal2
            The volatility of the left side
        vol2: float, Dual, Dual2
            The volatility on the right side
        bounds_flag: int
            -1: left side extrapolation, 0: normal interpolation, 1: right side extrapolation

        Notes
        -----
        This function performs different interpolation if weights are given or not. ``bounds_flag``
        is used to parse the inputs when *Smiles* to the left and/or right are not available.
        """
        # 86400 posix seconds per day
        # 31536000 posix seconds per 365 day year
        if isinstance(self.weights_cum, NoInput):  # weights must also be NoInput
            if bounds_flag == 0:
                ep1 = self.expiries_posix[expiry_index]
                ep2 = self.expiries_posix[expiry_index + 1]
            elif bounds_flag == -1:
                # left side extrapolation
                ep1 = self.eval_posix
                ep2 = self.expiries_posix[expiry_index]
            else:  # bounds_flag == 1:
                # right side extrapolation
                ep1 = self.expiries_posix[expiry_index + 1]
                ep2 = TERMINAL_DATE.replace(tzinfo=UTC).timestamp()

            t_var_1 = (ep1 - self.eval_posix) * vol1**2
            t_var_2 = (ep2 - self.eval_posix) * vol2**2
            _: DualTypes = t_var_1 + (t_var_2 - t_var_1) * (expiry_posix - ep1) / (ep2 - ep1)
            _ /= expiry_posix - self.eval_posix
        else:
            if bounds_flag == 0:
                t1 = self.weights_cum[self.expiries[expiry_index]]
                t2 = self.weights_cum[self.expiries[expiry_index + 1]]
            elif bounds_flag == -1:
                # left side extrapolation
                t1 = 0.0
                t2 = self.weights_cum[self.expiries[expiry_index]]
            else:  # bounds_flag == 1:
                # right side extrapolation
                t1 = self.weights_cum[self.expiries[expiry_index + 1]]
                t2 = self.weights_cum[TERMINAL_DATE]

            t = self.weights_cum[expiry]
            t_var_1 = t1 * vol1**2
            t_var_2 = t2 * vol2**2
            _ = t_var_1 + (t_var_2 - t_var_1) * (t - t1) / (t2 - t1)
            _ *= 86400.0 / (
                expiry_posix - self.eval_posix
            )  # scale by real cal days and not adjusted weights
        return _**0.5

    # _validate_states not required since called by `get_smile` internally
    def get_from_strike(
        self,
        k: DualTypes,
        f: DualTypes,
        w_deli: DualTypes | NoInput = NoInput(0),
        w_spot: DualTypes | NoInput = NoInput(0),
        expiry: datetime | NoInput = NoInput(0),
    ) -> tuple[DualTypes, DualTypes, DualTypes]:
        """
        Given an option strike and expiry return associated delta and vol values.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
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
        This function will return a delta index associated with the *FXDeltaVolSmile* and the
        volatility attributed to the delta at that point. Recall that the delta index is the
        negated put option delta for the given strike ``k``.
        """
        if isinstance(expiry, NoInput):
            raise ValueError("`expiry` required to get cross-section of FXDeltaVolSurface.")
        smile = self.get_smile(expiry)
        return smile.get_from_strike(k, f, w_deli, w_spot, expiry)

    # _validate_states not required since called by `get_smile` internally
    def _get_index(self, delta_index: DualTypes, expiry: datetime) -> DualTypes:
        """
        Return a volatility from a given delta index.
        Used internally alongside Surface, where a surface also requires an expiry.
        """
        return self.get_smile(expiry)[delta_index]

    def plot(self) -> PlotOutput:
        plot_upper_bound = max([_.plot_upper_bound for _ in self.smiles])
        deltas = np.linspace(0.0, plot_upper_bound, 20)
        vols = np.array([[_._get_index(d, NoInput(0)) for d in deltas] for _ in self.smiles])
        expiries = [(_ - self.eval_posix) / (365 * 24 * 60 * 60.0) for _ in self.expiries_posix]
        return plot3d(deltas, expiries, vols)  # type: ignore[arg-type, return-value]

    def _validate_weights(self, weights: Series[float] | NoInput) -> Series[float] | NoInput:
        if isinstance(weights, NoInput):
            return weights

        w: Series[float] = Series(
            1.0, index=get_calendar("all").cal_date_range(self.eval_date, TERMINAL_DATE)
        )
        w.update(weights)
        # restrict to sorted and filtered for outliers
        w = w.sort_index()
        w = w[self.eval_date :]  # type: ignore[misc]

        node_points: list[datetime] = [self.eval_date] + self.expiries + [TERMINAL_DATE]
        for i in range(len(self.expiries) + 1):
            s, e = node_points[i] + timedelta(days=1), node_points[i + 1]
            days = (e - s).days + 1
            w[s:e] = (  # type: ignore[misc]
                w[s:e] * days / w[s:e].sum()  # type: ignore[misc]
            )  # scale the weights to allocate the correct time between nodes.
        w[self.eval_date] = 0.0
        return w


class FXSabrSmile(_WithState, _WithCache[float, DualTypes]):
    r"""
    Create an *FX Volatility Smile* at a given expiry indexed by strike using SABR parameters.

    .. warning::

       This class is in beta status.

    Parameters
    ----------
    nodes: dict[str, DualTypes]
        The parameters for the SABR model. Keys must be *'alpha', 'beta', 'rho', 'nu'*. See below.
    eval_date: datetime
        Acts as the initial node of a *Curve*. Should be assigned today's immediate date.
    expiry: datetime
        The expiry date of the options associated with this *Smile*
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
    The keys for ``nodes`` are described as the following:

    - ``alpha`` (DualTypes): The initial volatility parameter (e.g. 0.10 for 10%) of the SABR model.
    - ``beta`` (float): The scaling parameter between normal (0) and lognormal (1)
      of the SABR model in [0, 1].
    - ``rho`` (DualTypes): The correlation between spot and volatility of the SABR model,
      e.g. -0.10.
    - ``nu`` (DualTypes): The volatility of volatility parameter of the SABR model, e.g. 0.80.

    """

    @_new_state_post
    def __init__(
        self,
        nodes: dict[str, DualTypes],
        eval_date: datetime,
        expiry: datetime,
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int = 0,
    ):
        self.id: str = (
            uuid4().hex[:5] + "_" if isinstance(id, NoInput) else id
        )  # 1 in a million clash

        self.eval_date: datetime = eval_date
        self.expiry: datetime = expiry
        self.t_expiry: float = (expiry - eval_date).days / 365.0
        self.t_expiry_sqrt: float = self.t_expiry**0.5

        self.nodes = nodes
        for _ in ["alpha", "beta", "rho", "nu"]:
            if _ not in self.nodes:
                raise ValueError(
                    f"'{_}' is a required SABR parameter that must be included in ``nodes``"
                )
        self._set_ad_order(ad)

    def __iter__(self) -> Any:
        raise TypeError("`FXSabrSmile` is not iterable.")

    def get_from_strike(
        self,
        k: DualTypes,
        f: DualTypes,
        w_deli: NoInput = NoInput(0),
        w_spot: NoInput = NoInput(0),
        expiry: datetime | NoInput = NoInput(0),
    ) -> tuple[DualTypes, DualTypes, DualTypes]:
        """
        Given an option strike return the volatility.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        f: float, Dual, Dual2
            The forward rate at delivery of the option.
        w_deli: DualTypes, optional
            Not used by *SabrSmile*
        w_spot: DualTypes, optional
            Not used by *SabrSmile*
        expiry: datetime, optional
            If given, performs a check to ensure consistency of valuations. Raises if expiry
            requested and expiry of the *Smile* do not match. Used internally.

        Returns
        -------
        tuple of DualTypes : (placeholder, vol, k)

        Notes
        -----
        This function returns a tuple consistent with an
        :class:`~rateslib.fx_volatility.FXDeltaVolSmile`, however since the *FXSabrSmile* has no
        concept of a `delta index` the first element returned is always zero and can be
        effectively ignored.
        """
        expiry = _drb(self.expiry, expiry)
        if self.expiry != expiry:
            raise ValueError(
                "`expiry` of VolSmile and OptionPeriod do not match: calculation aborted "
                "due to potential pricing errors.",
            )
        vol_ = _sabr(
            k,
            f,
            self.t_expiry,
            self.nodes["alpha"],
            self.nodes["beta"],  # type: ignore[arg-type]
            self.nodes["rho"],
            self.nodes["nu"],
        )
        return 0.0, vol_ * 100.0, k

    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[np.object_]]:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array([self.nodes["alpha"], self.nodes["rho"], self.nodes["nu"]])

    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        return tuple(f"{self.id}{i}" for i in range(3))

    @_new_state_post
    @_clear_cache_post
    def _set_node_vector(
        self, vector: np.ndarray[tuple[int, ...], np.dtype[np.object_]], ad: int
    ) -> None:
        """
        Update the node values in a Solver. ``ad`` in {1, 2}.
        Only the real values in vector are used, dual components are dropped and restructured.
        """
        DualType: type[Dual] | type[Dual2] = Dual if ad == 1 else Dual2
        DualArgs: tuple[list[float]] | tuple[list[float], list[float]] = (
            ([],) if ad == 1 else ([], [])
        )
        base_obj = DualType(0.0, [f"{self.id}{i}" for i in range(3)], *DualArgs)
        ident = np.eye(3)

        self.nodes["alpha"] = DualType.vars_from(
            base_obj,  # type: ignore[arg-type]
            vector[0].real,
            base_obj.vars,
            ident[0, :].tolist(),  # type: ignore[arg-type]
            *DualArgs[1:],
        )
        self.nodes["rho"] = DualType.vars_from(
            base_obj,  # type: ignore[arg-type]
            vector[1].real,
            base_obj.vars,
            ident[1, :].tolist(),  # type: ignore[arg-type]
            *DualArgs[1:],
        )
        self.nodes["nu"] = DualType.vars_from(
            base_obj,  # type: ignore[arg-type]
            vector[2].real,
            base_obj.vars,
            ident[2, :].tolist(),  # type: ignore[arg-type]
            *DualArgs[1:],
        )

    @_clear_cache_post
    def _set_ad_order(self, order: int) -> None:
        """This does not alter the beta node, since that is not varied by a Solver.
        beta values that are AD sensitive should be given as a Variable and not Dual/Dual2.
        """
        if order == getattr(self, "ad", None):
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self.ad = order

        self.nodes["alpha"] = set_order_convert(self.nodes["alpha"], order, [f"{self.id}0"])
        self.nodes["rho"] = set_order_convert(self.nodes["rho"], order, [f"{self.id}1"])
        self.nodes["nu"] = set_order_convert(self.nodes["nu"], order, [f"{self.id}2"])

    @_new_state_post
    @_clear_cache_post
    def update_node(self, key: str, value: DualTypes) -> None:
        """
        Update a single node value on the *SABRSmile*.

        Parameters
        ----------
        key: str in {"alpha", "beta", "rho", "nu"}
            The node value to update.
        value: float, Dual, Dual2, Variable
            Value to update on the *Smile*.

        Returns
        -------
        None

        Notes
        -----

        .. warning::

           *Rateslib* is an object-oriented library that uses complex associations. Although
           Python may not object to directly mutating attributes of a *Curve* instance, this
           should be avoided in *rateslib*. Only use official ``update`` methods to mutate the
           values of an existing *Curve* instance.
           This class is labelled as a **mutable on update** object.

        """
        if key not in self.nodes:
            raise KeyError("`key` is not in ``nodes``.")
        self.nodes[key] = value
        self._set_ad_order(self.ad)


def _validate_delta_type(delta_type: str) -> str:
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
#             "Can only convert between deltas of the same premium type, i.e. adjusted or "
#             "unadjusted."
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
) -> DualTypes:
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
    _: DualTypes = phi * (F * Nd1 - K * Nd2)
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


def _delta_type_constants(
    delta_type: str, w: DualTypes | NoInput, u: DualTypes | NoInput
) -> tuple[float, DualTypes, DualTypes]:
    """
    Get the values: (eta, z_w, z_u) for the type of expressed delta

    w: should be input as w_deli / w_spot
    u: should be input as K / f_d
    """
    if delta_type == "forward":
        return 0.5, 1.0, 1.0
    elif delta_type == "spot":
        return 0.5, w, 1.0  # type: ignore[return-value]
    elif delta_type == "forward_pa":
        return -0.5, 1.0, u  # type: ignore[return-value]
    else:  # "spot_pa"
        return -0.5, w, u  # type: ignore[return-value]


def _sabr(
    k: DualTypes,
    f: DualTypes,
    t: DualTypes,
    a: DualTypes,
    b: float,
    p: DualTypes,
    v: DualTypes,
) -> DualTypes:
    """
    Calculate the SABR vol. For formula see for example I. Clark "Foreign Exchange Option
    Pricing" section 3.10.
    """
    c1 = (f * k) ** ((1.0 - b) / 2.0)
    c2 = (f * k) ** (1.0 - b)
    l1 = dual_log(f / k)

    z = v / a * c1 * l1
    chi = dual_log(((1 - 2 * p * z + z * z) ** 0.5 + z - p) / (1 - p))

    _: DualTypes = a / (c1 * (1 + ((1 - b) ** 2 / 24.0) * l1**2 + ((1 - b) ** 4 / 1920) * l1**4))

    if abs(z) > 1e-14:
        _ *= z / chi
    else:
        # this is an approximation to avoid 0/0 yet preserve the result of 1.0 and maintain
        # AD sensitivity, rather than just omitting the multiplication
        _ *= (z + 1e-12) / (chi + 1e-12)

    _ *= (
        1
        + (
            (1 - b) ** 2 / 24.0 * a**2 / c2
            + 0.25 * (p * b * v * a) / c1
            + (2 - 3 * p * p) * v * v / 24
        )
        * t
    )
    return _


def _d_sabr_d_k(
    k: DualTypes,
    f: DualTypes,
    t: DualTypes,
    a: DualTypes,
    b: float,
    p: DualTypes,
    v: DualTypes,
) -> DualTypes:
    """
    Calculate the derivative of the SABR function with respect to k.

    This function was composed using the Python package `sympy` in the following way:

    SABR(k) = A(k) * B(k) * C(k) * D(k)

    .. code::

       import sympy as sym
       k = sym.Symbol("k")
       f = sym.Symbol("f")
       t = sym.Symbol("t")
       a = sym.Symbol("a")
       b = sym.Symbol("b")
       v = sym.Symbol("v")
       p = sym.Symbol("p")

       A = a/((f*k)**((1-b)/2)*(1+(1-b)/24*sym.log(f/k)**2+(1-b)**4/1920*sym.log(f/k)**4))

       z = v/a*(f*k)**((1-b)/2)*sym.log(f/k)
       B = z

       chi = sym.log(((1-2*p*z+z*z)**(1/2)+z-p)/(1-p))
       C = chi**-1

       D = 1+((1-b)**2/24*a**2/(f*k)**(1-b)+(1/4)*(p*b*v*a)/(f*k)**((1-b)/2)+(2-3*p*p)*v*v/24)*t

       sym.cse(sym.diff(A * B * C * D, k))

    """
    x0 = 1 / k
    x1 = v**2
    x2 = f * k
    x3 = b / 2 - 1 / 2
    x4 = x2**x3
    x5 = 0.25 * a * b * p * v * x4
    x6 = a**2
    x7 = b - 1
    x8 = x2**x7
    x9 = -x7
    x10 = x6 * x8 * x9**2 / 24
    x11 = t * (x1 * (2 - 3 * p**2) / 24 + x10 + x5) + 1
    x12 = 1 / 24 - b / 24
    x13 = dual_log(f * x0)
    x14 = x13**2
    x15 = x9**4
    x16 = x12 * x14 + x13**4 * x15 / 1920 + 1
    x17 = 1 / x16
    x18 = 1 / a
    x19 = 1 / x4
    x20 = v * x19
    x21 = x18 * x20
    x22 = x13 * x21
    x23 = p * x22
    x24 = 1 / x6
    x25 = 1 / x8
    x26 = x1 * x24 * x25
    x27 = (x14 * x26 - 2 * x23 + 1) ** 0.5
    x28 = -p + x22 + x27
    x29 = dual_log(x28 / (1 - p))
    x30 = 1 / x29
    x31 = x17 * x20 * x30 * x4
    x32 = x11 * x31
    x33 = -x3
    x34 = x0 * x13
    x35 = x32 * x34
    x36 = x11 * x13 * x20 * x4
    x37 = x0 * x33
    return (
        t * x13 * x31 * (x0 * x10 * x7 + x0 * x3 * x5)
        - x0 * x32
        - x17
        * x36
        * (
            -x0 * x21
            + x22 * x37
            + (
                1.0 * p * v * x0 * x18 * x19
                + 0.5 * x0 * x1 * x14 * x24 * x25 * x9
                - 1.0 * x23 * x37
                - 1.0 * x26 * x34
            )
            / x27
        )
        / (x28 * x29**2)
        + x3 * x35
        + x33 * x35
        + x30 * x36 * (x0 * x13**3 * x15 / 480 + 2 * x12 * x34) / x16**2
    )


FXVols = FXDeltaVolSmile | FXDeltaVolSurface
FXVolObj = (FXDeltaVolSmile, FXDeltaVolSurface)
