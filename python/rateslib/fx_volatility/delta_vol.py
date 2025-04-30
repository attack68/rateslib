from __future__ import annotations  # type hinting

import warnings
from datetime import datetime
from datetime import datetime as dt
from typing import TYPE_CHECKING, Any, TypeAlias
from uuid import uuid4

import numpy as np
from pandas import Series
from pytz import UTC

from rateslib import defaults
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
    newton_ndim,
    set_order_convert,
)
from rateslib.dual.utils import _dual_float
from rateslib.fx_volatility.utils import (
    _d_plus_min_u,
    _delta_type_constants,
    _moneyness_from_delta_closed_form,
    _t_var_interp,
    _validate_delta_type,
    _validate_smile_plot_comparators,
    _validate_weights,
)
from rateslib.mutability import (
    _clear_cache_post,
    _new_state_post,
    _validate_states,
    _WithCache,
    _WithState,
)
from rateslib.rs import index_left_f64
from rateslib.splines import PPSplineDual, PPSplineDual2, PPSplineF64, evaluate

if TYPE_CHECKING:
    from rateslib.typing import Sequence

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

    @property
    def ad(self) -> int:
        return self._ad

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
        z_w: DualTypes,
    ) -> DualTypes:
        """
        Return a volatility for a provided real option delta.

        This function is more explicit than the `__getitem__` method of the *Smile* because it
        permits forward/spot, adjusted/unadjusted and put/call option delta conversions,
        by deriving an appropriate delta index relevant to that of the *Smile* ``delta_type``.

        Parameters
        ----------
        delta: float
            The delta to obtain a volatility for.
        delta_type: str in {"spot", "spot_pa", "forward", "forward_pa"}
            The delta type the given delta is expressed in.
        phi: float
            Whether the given delta is assigned to a put or call option.
        z_w: DualTypes
            Required only for spot delta types. This is a scaling factor between spot and
            forward rate, equal to :math:`w_(m_{delivery})/w_(m_{spot})`, where *w* is curve
            for the domestic currency collateralised in the foreign currency. If not required
            enter 1.0.

        Returns
        -------
        DualTypes
        """
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, 0.0)  # u: unused
        eta_1, z_w_1, _ = _delta_type_constants(self.delta_type, z_w, 0.0)  # u: unused
        # then delta types are both unadjusted, used closed form.
        if eta_0 == eta_1 and eta_0 == 0.5:
            d_i: DualTypes = (-z_w_1 / z_w_0) * (delta - 0.5 * z_w_0 * (phi + 1.0))
            return self[d_i]
        # then delta types are both adjusted, use 1-d solver.
        elif eta_0 == eta_1 and eta_0 == -0.5:
            u = _moneyness_from_delta_one_dimensional(
                delta,
                delta_type,
                self.delta_type,
                self,
                self.t_expiry,
                z_w,
                phi,
            )
            delta_idx = (-z_w_1 / z_w_0) * (delta - z_w_0 * u * (phi + 1.0) * 0.5)
            return self[delta_idx]
        else:  # delta adjustment types are different, use 2-d solver.
            u, delta_idx = _moneyness_from_delta_two_dimensional(
                delta, delta_type, self, self.t_expiry, z_w, phi
            )
            return self[delta_idx]

    def get_from_strike(
        self,
        k: DualTypes,
        f: DualTypes,
        expiry: datetime | NoInput = NoInput(0),
        w_deli: DualTypes | NoInput = NoInput(0),
        w_spot: DualTypes | NoInput = NoInput(0),
    ) -> tuple[DualTypes, DualTypes, DualTypes]:
        """
        Given an option strike return associated delta and vol values.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        f: float, Dual, Dual2
            The forward rate at delivery of the option.
        expiry: datetime, optional
            Typically used with *Surfaces*.
            If given, performs a check to ensure consistency of valuations. Raises if expiry
            requested and expiry of the *Smile* do not match. Used internally.
        w_deli: DualTypes, optional
            Required only for spot/forward conversions.
        w_spot: DualTypes, optional
            Required only for spot/forward conversions.


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

    # def _delta_index_from_call_or_put_delta(
    #     self,
    #     delta: DualTypes,
    #     phi: float,
    #     z_w: DualTypes | NoInput = NoInput(0),
    #     u: DualTypes | NoInput = NoInput(0),
    # ) -> DualTypes:
    #     """
    #     Get the *Smile* index delta given an option delta of the same type as the *Smile*.
    #
    #     Note: This is required because the delta_index of the *Smile* uses negated put deltas.
    #
    #     Parameters
    #     ----------
    #     delta: DualTypes
    #         The expressed option delta. This MUST be given in the same type as the *Smile*.
    #     phi: float
    #         Whether a call (1.0) or a put (-1.0)
    #     z_w: DualTypes
    #         The spot/forward conversion factor defined by: `w_deli / w_spot`.
    #     u: DualTypes
    #         Moneyness defined by: `k/f_d`
    #
    #     Returns
    #     -------
    #     float, Dual, Dual2
    #     """
    #     # if call then must convert to put delta using delta parity equations
    #     if phi > 0:
    #         if self.delta_type == "forward":
    #             put_delta = delta - 1.0
    #         elif self.delta_type == "spot":
    #             put_delta = delta - z_w  # type: ignore[operator]
    #         elif self.delta_type == "forward_pa":
    #             put_delta = delta - u  # type: ignore[operator]
    #         else:  # self.delta_type == "spot_pa":
    #             put_delta = delta - z_w * u  # type: ignore[operator]
    #     else:
    #         put_delta = delta
    #     return -1.0 * put_delta

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
        labels: list[str] | NoInput = NoInput(0),
        x_axis: str = "delta",
        f: DualTypes | NoInput = NoInput(0),
    ) -> PlotOutput:
        """
        Plot volatilities associated with the *Smile*.

        .. warning::

           If the ``x_axis`` types *'moneyness'* and *'strike'* are used these will be
           generated by assuming that the delta indexes of this *Smile* are **forward, unadjusted**
           types. If this *Smile* has another, actual delta type, then the produced graphs
           will not be correct. This approximation is made to avoid complicated calculations
           involving iterations for each graph point.

        Parameters
        ----------
        comparators: list[Smile]
            A list of Smiles which to include on the same plot as comparators.
        labels : list[str]
            A list of strings associated with the plot and comparators. Must be same
            length as number of plots.
        x_axis : str in {"delta", "moneyness", "strike"}
            *'delta'* is the natural option for this *SMile* type.
            If *'moneyness'* the delta values are converted (see warning), and if *'strike'* then
            those moneyness values are converted using ``f``.
        f: DualTypes, optional
            The FX forward rate at delivery. Required in certain cases to derive the strike on
            the x-axis.

        Returns
        -------
        (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D
        """
        # reversed for intuitive strike direction
        comparators = _drb([], comparators)
        labels = _drb([], labels)

        x_, y_ = self._plot(x_axis, f)

        x = [x_]
        y = [y_]
        if not isinstance(comparators, NoInput):
            for smile in comparators:
                _validate_smile_plot_comparators(smile, (FXDeltaVolSmile, FXSabrSmile))
                x_, y_ = smile._plot(x_axis, f)
                x.append(x_)
                y.append(y_)

        return plot(x, y, labels)

    def _plot(
        self,
        x_axis: str,
        f: DualTypes | NoInput,
    ) -> tuple[list[float], list[DualTypes]]:
        x: list[float] = list(np.linspace(_dual_float(self.plot_upper_bound), self.t[0], 301))
        vols: list[float] | list[Dual] | list[Dual2] = self.spline.ppev(x)
        if x_axis in ["moneyness", "strike"]:
            if self.delta_type != "forward":
                warnings.warn(
                    "FXDeltaVolSmile.plot() approximates 'moneyness' and 'strike' using the "
                    "convention that the Smile has a `delta_type` of 'forward'.\nThe Smile "
                    f"has type: '{self.delta_type}' so this is likely to lead to inexact plots.",
                    UserWarning,
                )

            x = x[40:-40]
            vols = vols[40:-40]
            sq_t = self.t_expiry_sqrt
            x_as_u: list[DualTypes] = [
                dual_exp(_s / 100.0 * sq_t * (dual_inv_norm_cdf(_D) + 0.5 * _s / 100.0 * sq_t))  # type: ignore[operator]
                for (_D, _s) in zip(x, vols, strict=True)
            ]
            if x_axis == "strike":
                if isinstance(f, NoInput):
                    raise ValueError(
                        "`f` (ATM-forward FX rate) is required by `FXDeltaVolSmile.plot` "
                        "to convert 'moneyness' to 'strike'."
                    )
                return ([_ * _dual_float(f) for _ in x_as_u], vols)  # type: ignore[return-value]
            return (x_as_u, vols)  # type: ignore[return-value]
        return (x, vols)  # type: ignore[return-value]

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
                ident[i, :].tolist(),
                *DualArgs[1:],
            )
        self._csolve()

    @_clear_cache_post
    def _set_ad_order(self, order: int) -> None:
        if order == getattr(self, "ad", None):
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self._ad = order
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

    **Calibration**

    *Instruments* that do not match the ``delta_type`` of this *Surface* can still be used within
    a :class:`~rateslib.solver.Solver` to calibrate the surface. This is quite common, when
    *Options* less than or equal to one year expiry might use a *'spot'* delta type whilst longer
    expiries use *'forward'* delta type.

    Internally this is all handled appropriately with necessary conversions, but it is the users
    responsibility to label the *Surface* and *Instrument* with the correct types. Failing to
    take correct delta types into account often introduces a mismatch -
    large enough to be relevant for calibration and pricing, but small enough that it may not be
    noticed at first. Parametrising the *Surface* with a *'forward'* delta type is the
    **recommended**
    choice because it is more standardised and the configuration of which *delta types* to use for
    the *Instruments* can be a separate consideration.

    For performance reasons it is recommended to match unadjusted delta type *Surfaces* with
    calibrating *Instruments* that also have unadjusted delta types. And vice versa with
    premium adjusted
    delta types. However, *rateslib* has internal root solvers which can handle these
    cross-delta type
    specifications, although it degrades the performance of the *Solver* because the calculations
    are made more difficult. Mixing 'spot' and 'forward' is not a difficult distinction to
    refactor and that does not cause performance degradation.

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

        self.weights = _validate_weights(weights, eval_date, expiries)
        self.weights_cum = (
            NoInput(0) if isinstance(self.weights, NoInput) else self.weights.cumsum()
        )

        self._set_ad_order(ad)  # includes csolve on each smile
        self._set_new_state()

    @property
    def ad(self) -> int:
        return self._ad

    def _get_composited_state(self) -> int:
        return hash(sum(smile._state for smile in self.smiles))

    def _validate_state(self) -> None:
        if self._state != self._get_composited_state():
            # If any of the associated curves have been mutated then the cache is invalidated
            self._clear_cache()
            self._set_new_state()

    @_clear_cache_post
    def _set_ad_order(self, order: int) -> None:
        self._ad = order
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
                    k: _t_var_interp(
                        expiries=self.expiries,
                        expiries_posix=self.expiries_posix,
                        expiry=expiry,
                        expiry_posix=expiry_posix,
                        expiry_index=e_idx,
                        eval_posix=self.eval_posix,
                        weights_cum=self.weights_cum,
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
                    k: _t_var_interp(
                        expiries=self.expiries,
                        expiries_posix=self.expiries_posix,
                        expiry=expiry,
                        expiry_posix=expiry_posix,
                        expiry_index=e_idx,
                        eval_posix=self.eval_posix,
                        weights_cum=self.weights_cum,
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
                    k: _t_var_interp(
                        expiries=self.expiries,
                        expiries_posix=self.expiries_posix,
                        expiry=expiry,
                        expiry_posix=expiry_posix,
                        expiry_index=e_idx,
                        eval_posix=self.eval_posix,
                        weights_cum=self.weights_cum,
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

    # _validate_states not required since called by `get_smile` internally
    def get_from_strike(
        self,
        k: DualTypes,
        f: DualTypes,
        expiry: datetime | NoInput = NoInput(0),
        w_deli: DualTypes | NoInput = NoInput(0),
        w_spot: DualTypes | NoInput = NoInput(0),
    ) -> tuple[DualTypes, DualTypes, DualTypes]:
        """
        Given an option strike and expiry return associated delta and vol values.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        f: float, Dual, Dual2
            The forward rate at delivery of the option.
        expiry: datetime
            Required to produce the cross-sectional *Smile* on the *Surface*.
        w_deli: DualTypes, optional
            Required only for spot/forward conversions.
        w_spot: DualTypes, optional
            Required only for spot/forward conversions.

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
        return smile.get_from_strike(k, f, expiry, w_deli, w_spot)

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


def _moneyness_from_atm_delta_one_dimensional(
    delta_type: str,
    vol_delta_type: str,
    vol: DualTypes | FXDeltaVolSmile,
    t_e: DualTypes,
    z_w: DualTypes,
    phi: float,
) -> DualTypes:
    def root1d(
        g: DualTypes,
        delta_type: str,
        vol_delta_type: str,
        phi: float,
        sqrt_t_e: float,
        z_w: DualTypes,
        ad: int,
    ) -> tuple[DualTypes, DualTypes]:
        u = g

        eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
        eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
        dz_u_0_du = 0.5 - eta_0

        delta_idx = z_w_1 * z_u_0 / 2.0
        if isinstance(vol, FXDeltaVolSmile):
            vol_: DualTypes = vol[delta_idx] / 100.0
            dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
        else:
            vol_ = vol / 100.0
            dvol_ddeltaidx = 0.0
        vol_ = _dual_float(vol_) if ad == 0 else vol_
        dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx
        vol_sqrt_t = vol_ * sqrt_t_e

        # Calculate function values
        d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
        _phi0 = dual_norm_cdf(phi * d0)
        f0 = phi * z_w_0 * z_u_0 * (0.5 - _phi0)

        # Calculate derivative values
        ddelta_idx_du = dz_u_0_du * z_w_1 * 0.5

        lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
        dd_du = -1 / (u * vol_sqrt_t) + dvol_ddeltaidx * (lnu + eta_0 * sqrt_t_e) * ddelta_idx_du

        nd0 = dual_norm_pdf(phi * d0)
        f1 = -dz_u_0_du * z_w_0 * phi * _phi0 - z_u_0 * z_w_0 * nd0 * dd_du

        return f0, f1

    if isinstance(vol, FXDeltaVolSmile):
        avg_vol: DualTypes = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
    else:
        avg_vol = vol
    g01 = phi * 0.5 * (z_w if "spot" in delta_type else 1.0)
    g00 = _moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0, phi)

    root_solver = newton_1dim(
        root1d,
        g00,
        args=(delta_type, vol_delta_type, phi, t_e**0.5, z_w),
        pre_args=(0,),
        final_args=(1,),
        raise_on_fail=True,
    )

    u: DualTypes = root_solver["g"]
    return u


def _moneyness_from_delta_one_dimensional(
    delta: DualTypes,
    delta_type: str,
    vol_delta_type: str,
    vol: FXDeltaVolSmile | DualTypes,
    t_e: DualTypes,
    z_w: DualTypes,
    phi: float,
) -> DualTypes:
    def root1d(
        g: DualTypes,
        delta: DualTypes,
        delta_type: str,
        vol_delta_type: str,
        phi: float,
        sqrt_t_e: DualTypes,
        z_w: DualTypes,
        ad: int,
    ) -> tuple[DualTypes, DualTypes]:
        u = g

        eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
        eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
        dz_u_0_du = 0.5 - eta_0

        delta_idx = (-z_w_1 / z_w_0) * (delta - z_w_0 * z_u_0 * (phi + 1.0) * 0.5)
        if isinstance(vol, FXDeltaVolSmile):
            vol_: DualTypes = vol[delta_idx] / 100.0
            dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
        else:
            vol_ = vol / 100.0
            dvol_ddeltaidx = 0.0
        vol_ = _dual_float(vol_) if ad == 0 else vol_
        dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx
        vol_sqrt_t = vol_ * sqrt_t_e

        # Calculate function values
        d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
        _phi0 = dual_norm_cdf(phi * d0)
        f0 = delta - z_w_0 * z_u_0 * phi * _phi0

        # Calculate derivative values
        ddelta_idx_du = dz_u_0_du * z_w_1 * (phi + 1.0) * 0.5

        lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
        dd_du = -1 / (u * vol_sqrt_t) + dvol_ddeltaidx * (lnu + eta_0 * sqrt_t_e) * ddelta_idx_du

        nd0 = dual_norm_pdf(phi * d0)
        f1 = -dz_u_0_du * z_w_0 * phi * _phi0 - z_u_0 * z_w_0 * nd0 * dd_du

        return f0, f1

    if isinstance(vol, FXDeltaVolSmile):
        avg_vol: DualTypes = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
    else:
        avg_vol = vol
    g01 = delta if phi > 0 else max(delta, -0.75)
    g00 = _moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0, phi)

    msg = (
        f"If the delta, {delta:.1f}, is premium adjusted for a call option is it infeasible?"
        if phi > 0
        else ""
    )
    try:
        root_solver = newton_1dim(
            root1d,
            g00,
            args=(delta, delta_type, vol_delta_type, phi, t_e**0.5, z_w),
            pre_args=(0,),
            final_args=(1,),
        )
    except ValueError as e:
        raise ValueError(f"Newton root solver failed, with error: {e.__str__()}.\n{msg}")

    if root_solver["state"] == -1:
        raise ValueError(
            f"Newton root solver failed, after {root_solver['iterations']} iterations.\n{msg}",
        )

    u: DualTypes = root_solver["g"]
    return u


def _moneyness_from_atm_delta_two_dimensional(
    delta_type: str,
    vol: FXDeltaVolSmile,
    t_e: DualTypes,
    z_w: DualTypes,
    phi: float,
) -> tuple[DualTypes, DualTypes]:
    def root2d(
        g: list[DualTypes],
        delta_type: str,
        vol_delta_type: str,
        phi: float,
        sqrt_t_e: DualTypes,
        z_w: DualTypes,
        ad: int,
    ) -> tuple[list[DualTypes], list[list[DualTypes]]]:
        u, delta_idx = g[0], g[1]

        eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
        eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
        dz_u_0_du = 0.5 - eta_0
        dz_u_1_du = 0.5 - eta_1

        vol_ = vol[delta_idx] / 100.0
        vol_ = _dual_float(vol_) if ad == 0 else vol_
        vol_sqrt_t = vol_ * sqrt_t_e

        # Calculate function values
        d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
        _phi0 = dual_norm_cdf(phi * d0)
        f0_0 = phi * z_w_0 * z_u_0 * (0.5 - _phi0)

        d1 = _d_plus_min_u(u, vol_sqrt_t, eta_1)
        _phi1 = dual_norm_cdf(-d1)
        f0_1 = delta_idx - z_w_1 * z_u_1 * _phi1

        # Calculate Jacobian values
        dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
        dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx

        dd_du = -1 / (u * vol_sqrt_t)  # this is the same for 0 or 1 variety
        nd0 = dual_norm_pdf(phi * d0)
        nd1 = dual_norm_pdf(-d1)
        lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
        dd0_ddeltaidx = (lnu + eta_0 * sqrt_t_e) * dvol_ddeltaidx
        dd1_ddeltaidx = (lnu + eta_1 * sqrt_t_e) * dvol_ddeltaidx

        f1_00 = phi * z_w_0 * dz_u_0_du * (0.5 - _phi0) - z_w_0 * z_u_0 * nd0 * dd_du
        f1_10 = -z_w_1 * dz_u_1_du * _phi1 + z_w_1 * z_u_1 * nd1 * dd_du
        f1_01 = -z_w_0 * z_u_0 * nd0 * dd0_ddeltaidx
        f1_11 = 1.0 + z_w_1 * z_u_1 * nd1 * dd1_ddeltaidx

        return [f0_0, f0_1], [[f1_00, f1_01], [f1_10, f1_11]]

    avg_vol = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
    g01 = phi * 0.5 * (z_w if "spot" in delta_type else 1.0)
    g00 = _moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0, phi)

    root_solver = newton_ndim(
        root2d,
        [g00, abs(g01)],
        args=(delta_type, vol.delta_type, phi, t_e**0.5, z_w),
        pre_args=(0,),
        final_args=(1,),
        raise_on_fail=True,
    )

    u, delta_idx = root_solver["g"][0], root_solver["g"][1]
    return u, delta_idx


def _moneyness_from_delta_two_dimensional(
    delta: DualTypes,
    delta_type: str,
    vol: FXDeltaVolSmile,
    t_e: DualTypes,
    z_w: DualTypes,
    phi: float,
) -> tuple[DualTypes, DualTypes]:
    def root2d(
        g: Sequence[DualTypes],
        delta: DualTypes,
        delta_type: str,
        vol_delta_type: str,
        phi: float,
        sqrt_t_e: float,
        z_w: DualTypes,
        ad: int,
    ) -> tuple[list[DualTypes], list[list[DualTypes]]]:
        u, delta_idx = g[0], g[1]

        eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
        eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
        dz_u_0_du = 0.5 - eta_0
        dz_u_1_du = 0.5 - eta_1

        vol_ = vol[delta_idx] / 100.0
        vol_ = _dual_float(vol_) if ad == 0 else vol_
        vol_sqrt_t = vol_ * sqrt_t_e

        # Calculate function values
        d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
        _phi0 = dual_norm_cdf(phi * d0)
        f0_0: DualTypes = delta - z_w_0 * z_u_0 * phi * _phi0

        d1 = _d_plus_min_u(u, vol_sqrt_t, eta_1)
        _phi1 = dual_norm_cdf(-d1)
        f0_1: DualTypes = delta_idx - z_w_1 * z_u_1 * _phi1

        # Calculate Jacobian values
        dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
        dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx

        dd_du = -1 / (u * vol_sqrt_t)
        nd0 = dual_norm_pdf(phi * d0)
        nd1 = dual_norm_pdf(-d1)
        lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
        dd0_ddeltaidx = (lnu + eta_0 * sqrt_t_e) * dvol_ddeltaidx
        dd1_ddeltaidx = (lnu + eta_1 * sqrt_t_e) * dvol_ddeltaidx

        f1_00: DualTypes = -z_w_0 * dz_u_0_du * phi * _phi0 - z_w_0 * z_u_0 * nd0 * dd_du
        f1_10: DualTypes = -z_w_1 * dz_u_1_du * _phi1 + z_w_1 * z_u_1 * nd1 * dd_du
        f1_01: DualTypes = -z_w_0 * z_u_0 * nd0 * dd0_ddeltaidx
        f1_11: DualTypes = 1.0 + z_w_1 * z_u_1 * nd1 * dd1_ddeltaidx

        return [f0_0, f0_1], [[f1_00, f1_01], [f1_10, f1_11]]

    avg_vol = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
    g01 = delta if phi > 0 else max(delta, -0.75)
    g00 = _moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0, phi)

    msg = (
        f"If the delta, {_dual_float(delta):.1f}, is premium adjusted for a "
        "call option is it infeasible?"
        if phi > 0
        else ""
    )
    try:
        root_solver = newton_ndim(
            root2d,
            [g00, abs(g01)],
            args=(delta, delta_type, vol.delta_type, phi, t_e**0.5, z_w),
            pre_args=(0,),
            final_args=(1,),
            raise_on_fail=False,
        )
    except ValueError as e:
        raise ValueError(f"Newton root solver failed, with error: {e.__str__()}.\n{msg}")

    if root_solver["state"] == -1:
        raise ValueError(
            f"Newton root solver failed, after {root_solver['iterations']} iterations.\n{msg}",
        )
    u, delta_idx = root_solver["g"][0], root_solver["g"][1]
    return u, delta_idx


def _moneyness_from_delta_three_dimensional(
    delta_type: str, vol: DualTypes | FXDeltaVolSmile, t_e: DualTypes, z_w: DualTypes, phi: float
) -> tuple[DualTypes, DualTypes, DualTypes]:
    """
    Solve the ATM delta problem where delta is not explicit.

    Book2: section "Strike and Volatility implied from ATM delta" (FXDeltaVolSMile)
    """

    def root3d(
        g: list[DualTypes],
        delta_type: str,
        vol_delta_type: str,
        phi: float,
        sqrt_t_e: DualTypes,
        z_w: DualTypes,
        ad: int,
    ) -> tuple[list[DualTypes], list[list[DualTypes]]]:
        u, delta_idx, delta = g[0], g[1], g[2]

        eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
        eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
        dz_u_0_du = 0.5 - eta_0
        dz_u_1_du = 0.5 - eta_1

        if isinstance(vol, FXDeltaVolSmile):
            vol_: DualTypes = vol[delta_idx] / 100.0
            dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
        else:
            vol_ = vol / 100.0
            dvol_ddeltaidx = 0.0
        vol_ = _dual_float(vol_) if ad == 0 else vol_
        vol_sqrt_t = vol_ * sqrt_t_e

        # Calculate function values
        d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
        _phi0 = dual_norm_cdf(phi * d0)
        f0_0 = delta - z_w_0 * z_u_0 * phi * _phi0

        d1 = _d_plus_min_u(u, vol_sqrt_t, eta_1)
        _phi1 = dual_norm_cdf(-d1)
        f0_1 = delta_idx - z_w_1 * z_u_1 * _phi1

        f0_2 = delta - phi * z_u_0 * z_w_0 / 2.0

        # Calculate Jacobian values
        dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx

        dd_du = -1 / (u * vol_sqrt_t)
        nd0 = dual_norm_pdf(phi * d0)
        nd1 = dual_norm_pdf(-d1)
        lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
        dd0_ddeltaidx = (lnu + eta_0 * sqrt_t_e) * dvol_ddeltaidx
        dd1_ddeltaidx = (lnu + eta_1 * sqrt_t_e) * dvol_ddeltaidx

        f1_00 = -z_w_0 * dz_u_0_du * phi * _phi0 - z_w_0 * z_u_0 * nd0 * dd_du  # dh0/du
        f1_10 = -z_w_1 * dz_u_1_du * _phi1 + z_w_1 * z_u_1 * nd1 * dd_du  # dh1/du
        f1_20 = -phi * z_w_0 * dz_u_0_du / 2.0  # dh2/du
        f1_01 = -z_w_0 * z_u_0 * nd0 * dd0_ddeltaidx  # dh0/ddidx
        f1_11 = 1.0 + z_w_1 * z_u_1 * nd1 * dd1_ddeltaidx  # dh1/ddidx
        f1_21 = 0.0  # dh2/ddidx
        f1_02 = 1.0  # dh0/ddelta
        f1_12 = 0.0  # dh1/ddelta
        f1_22 = 1.0  # dh2/ddelta

        return [f0_0, f0_1, f0_2], [
            [f1_00, f1_01, f1_02],
            [f1_10, f1_11, f1_12],
            [f1_20, f1_21, f1_22],
        ]

    if isinstance(vol, FXDeltaVolSmile):
        avg_vol: DualTypes = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
        vol_delta_type = vol.delta_type
    else:
        avg_vol = vol
        vol_delta_type = delta_type
    g02 = 0.5 * phi * (z_w if "spot" in delta_type else 1.0)
    g01 = g02 if phi > 0 else max(g02, -0.75)
    g00 = _moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0, phi)

    root_solver = newton_ndim(
        root3d,
        [g00, abs(g01), g02],
        args=(delta_type, vol_delta_type, phi, t_e**0.5, z_w),
        pre_args=(0,),
        final_args=(1,),
        raise_on_fail=True,
    )

    u, delta_idx, delta = root_solver["g"][0], root_solver["g"][1], root_solver["g"][1]
    return u, delta_idx, delta
