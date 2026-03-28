# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################


from __future__ import annotations  # type hinting

from datetime import datetime, timezone
from functools import cached_property
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from rateslib.data.fixings import IRSSeries, _get_irs_series
from rateslib.dual import Dual, Dual2, Variable, set_order_convert
from rateslib.dual.utils import _dual_float, _get_order_of, dual_exp, dual_inv_norm_cdf
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import OptionPricingModel, _get_option_pricing_model
from rateslib.mutability import (
    _new_state_post,
)
from rateslib.splines import PPSplineDual, PPSplineDual2, PPSplineF64
from rateslib.splines.evaluate import evaluate
from rateslib.volatility.ir.base import _BaseIRCube, _BaseIRSmile, _WithMutability
from rateslib.volatility.ir.utils import (
    _IRCubeMeta,
    _IRSmileMeta,
    _IRVolPricingParams,
)

UTC = timezone.utc

SPLINE_LOWER = -5000.0
SPLINE_UPPER = 10000.0

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        Arr3dObj,
        DualTypes,
        DualTypes_,
        Iterable,
        Number,
        Sequence,
        Series,
        float_,
        int_,
    )


class _IRSplineSmileNodes:
    """
    A container for data relating to interpolating the `nodes` of a
    :class:`~rateslib.volatility.IRSplineSmile`.
    """

    _nodes: dict[float, DualTypes]
    _spline: _IRVolSpline

    def __init__(self, nodes: dict[float, DualTypes], k: int) -> None:
        self._nodes = dict(sorted(nodes.items()))

        match (self.n, k):
            case (1, _) | (2, _):
                # 1 DoF yields a flat smile, but treat it as a line of zero gradient
                # 2 DoF yields a straight line, usually with some non-zero gradient
                k = 2
                t = [SPLINE_LOWER, SPLINE_LOWER, SPLINE_UPPER, SPLINE_UPPER]
            case (_, 2):
                # 3 or more DoF but piecewise linear endpoints have 2 knots
                t = [SPLINE_LOWER, SPLINE_LOWER] + self.keys[1:-1] + [SPLINE_UPPER, SPLINE_UPPER]
            case (_, 4):
                # 3 or more DoF but piecewise cubic ensure endpoints have 4 knots.
                t = [SPLINE_LOWER] * 4 + self.keys[1:-1] + [SPLINE_UPPER] * 4

        self._spline = _IRVolSpline(t=t, k=k)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _IRSplineSmileNodes):
            return False
        return self._nodes == other._nodes and self.k == other.k

    @property
    def nodes(self) -> dict[float, DualTypes]:
        """The initial nodes dict passed for construction of this class."""
        return self._nodes

    @cached_property
    def keys(self) -> list[float]:
        """A list of the relative strike keys in ``nodes``."""
        return list(self.nodes.keys())

    @cached_property
    def values(self) -> list[DualTypes]:
        """A list of the delta index values in ``nodes``."""
        return list(self.nodes.values())

    @property
    def n(self) -> int:
        """The number of pricing parameters in ``nodes``."""
        return len(self.keys)

    @property
    def k(self) -> int:
        """The order of the interpolating polynomial spline."""
        return self.spline.k

    @property
    def spline(self) -> _IRVolSpline:
        """An instance of :class:`~rateslib.volatility.ir._IRVolSpline`."""
        return self._spline


class _IRVolSpline:
    """
    A container for data relating to interpolating the `nodes` of
    a :class:`~rateslib.volatility.IRSplineSmile` using a PPSpline.
    """

    _k: int
    _t: list[float]
    _spline: PPSplineF64 | PPSplineDual | PPSplineDual2

    def __init__(self, t: list[float], k: int) -> None:
        self._t = t
        self._k = k
        self._spline = PPSplineF64(k, [0.0] * 5, None)  # placeholder: csolve will reengineer

    @property
    def t(self) -> list[float]:
        """The knot sequence of the PPSpline."""
        return self._t

    @property
    def k(self) -> int:
        """The order of the spline."""
        return self._k

    @property
    def spline(self) -> PPSplineF64 | PPSplineDual | PPSplineDual2:
        """An instance of :class:`~rateslib.splines.PPSplineF64`,
        :class:`~rateslib.splines.PPSplineDual` or :class:`~rateslib.splines.PPSplineDual2`"""
        return self._spline

    def evaluate(self, x: DualTypes, m: int = 0) -> Number:
        """Perform the :meth:`~rateslib.splines.evaluate` method on the object's ``spline``."""
        return evaluate(spline=self.spline, x=x, m=m)

    def _csolve_n_other(
        self, nodes: _IRSplineSmileNodes, ad: int
    ) -> tuple[list[float], list[DualTypes], int, int]:
        """
        Solve a spline with more than one node value.
        Premium adjusted delta types have an unbounded right side delta index so a derivative of
        0 is applied to the spline as a boundary condition.
        Premium unadjusted delta types have a right side delta index approximately equal to 1.0.
        Use a natural spline boundary condition here.
        """
        tau = nodes.keys.copy()
        y = nodes.values.copy()

        if self.k == 4:
            # now insert the natural spline 2nd derivative constraint
            y.insert(0, set_order_convert(0.0, ad, None))
            tau.insert(0, SPLINE_LOWER)
            left_n = 2  # natural spline
        else:  # == 2
            left_n = 0

        if self.k == 4:
            tau.append(self.t[-1])
            y.append(set_order_convert(0.0, ad, None))
            right_n = 2  # natural spline
        else:  # == 2
            right_n = 0

        return tau, y, left_n, right_n

    def csolve(self, nodes: _IRSplineSmileNodes, ad: int) -> None:
        """
        Construct a spline of appropriate AD order and solve the spline coefficients for the
        given ``nodes``.

        Parameters
        ----------
        nodes: _IRSplineSmileNodes
            Required information for constructing a PPSpline.
        ad: int
            The AD order of the constructed PPSPline.

        Returns
        -------
        None
        """
        if ad == 0:
            Spline: type[PPSplineF64] | type[PPSplineDual] | type[PPSplineDual2] = PPSplineF64
        elif ad == 1:
            Spline = PPSplineDual
        else:
            Spline = PPSplineDual2

        if nodes.n == 1:
            # one node defines a flat line, all spline coefficients are the equivalent value.
            # no need to solve, just craft the spline directly.
            self._spline = Spline(self.k, self.t, nodes.values * self.k)  # type: ignore[arg-type]
        else:
            tau, y, left_n, right_n = self._csolve_n_other(nodes, ad)
            self._spline = Spline(self.k, self.t, None)
            self._spline.csolve(tau, y, left_n, right_n, False)  # type: ignore[arg-type]

    # def to_json(self) -> str:
    #     """
    #     Serialize this object to JSON format.
    #
    #     The object can be deserialized using the :meth:`~rateslib.serialization.from_json` method.
    #
    #     Returns
    #     -------
    #     str
    #     """
    #     obj = dict(
    #         PyNative=dict(
    #             _FXDeltaVolSpline=dict(
    #                 t=self.t,
    #             )
    #         )
    #     )
    #     return json.dumps(obj)
    #
    # @classmethod
    # def _from_json(cls, loaded_json: dict[str, Any]) -> _FXDeltaVolSpline:
    #     return _FXDeltaVolSpline(
    #         t=loaded_json["t"],
    #     )

    def __eq__(self, other: Any) -> bool:
        """CurveSplines are considered equal if their knot sequence and endpoints are equivalent.
        For the same nodes this will resolve to give the same spline coefficients.
        """
        if not isinstance(other, _IRVolSpline):
            return False
        else:
            return self.t == other.t and self.k == other.k


class IRSplineSmile(_BaseIRSmile, _WithMutability):
    r"""
    Create an *IR Volatility Smile* at a given expiry indexed for a specific IRS tenor
    with normal volatility interpolated by a polynomial spline curve.

    .. warning::

       *Swaptions* and *IR Volatility* are in Beta status introduced in v2.7.0

    .. role:: green

    .. role:: red

    Parameters
    ----------
    nodes: dict[float, float], :red:`required`
        The parameters for the spline. Keys must be basis points relative to the forward rate,
        and values are normal volatility basis points.
    eval_date: datetime, :red:`required`
        Acts like the initial node of a *Curve*. Should be assigned today's immediate date.
    expiry: datetime, :red:`required`
        The expiry date of the options associated with this *Smile*.
    irs_series: IRSSeries, :red:`required`
        The :class:`~rateslib.data.fixings.IRSSeries` that contains the parameters for the
        underlying :class:`~rateslib.instruments.IRS` that the swaptions are settled against.
    tenor: datetime, str, :red:`required`
        The tenor parameter for the underlying :class:`~rateslib.instruments.IRS` that the
        swaptions are settled against.
    k: int in {2, 4}, :green:`optional (set as 2)`
        The order of the interpolating spline, with (2, 4) representing (linear, cubic)
        interpolation respectively.
    id: str, optional, :green:`optional (set as random)`
        The unique identifier to distinguish between *Smiles* in a multicurrency framework
        and/or *Surface*.
    ad: int, :green:`optional (set by default)`
        Sets the automatic differentiation order. Defines whether to convert node
        values to float, :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2`. It is advised against
        using this setting directly. It is mainly used internally.

    Notes
    -----
    The keys for ``nodes`` must be basis points relative to the forward rate. For example

    .. code-block:: python

       nodes = {-200.: 50.0, -100.: 47.0, 0.: 46.0, 100.: 48, 200.: 52.0}

    This means that the volatility model of this spline is naturally dependent on the forward
    *IRS* rate, very similar to an :class:`~rateslib.volatility.FXDeltaVolSmile`, and any type
    SABR type *Smile*.

    The value of ``nodes`` are treated as the parameters that will be calibrated/mutated by
    a :class:`~rateslib.solver.Solver` object. The order of the spline, ``k``, in {2, 4} is a
    hyper-parameter of this model and will not be mutated.

    Examples
    --------
    See :ref:`Constructing a Smile <c-ir-smile-constructing-doc>`.

    """  # noqa: E501

    @_new_state_post
    def __init__(
        self,
        nodes: dict[float, DualTypes],
        eval_date: datetime,
        expiry: datetime | str,
        irs_series: IRSSeries | str,
        tenor: datetime | str,
        *,
        k: int_ = NoInput(0),
        pricing_model: OptionPricingModel | str = "normal_vol",
        shift: DualTypes_ = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int | None = 0,
    ):
        k_ = _drb(2, k)
        del k
        if k_ not in [2, 4]:
            raise ValueError(
                f"`k` must imply linear(2) or cubic(4) spline interpolation. Got {k_}."
            )
        self._id: str = (
            uuid4().hex[:5] + "_" if isinstance(id, NoInput) else id
        )  # 1 in a million clash
        self._meta: _IRSmileMeta = _IRSmileMeta(
            _tenor_input=tenor,
            _irs_series=_get_irs_series(irs_series),
            _eval_date=eval_date,
            _expiry_input=expiry,
            _plot_x_axis="moneyness",
            _plot_y_axis="normal_vol",
            _shift=_drb(0.0, shift),
            _pricing_model=_get_option_pricing_model(pricing_model),
        )

        self._nodes = _IRSplineSmileNodes(nodes=nodes, k=k_)

        self._set_ad_order(ad)

    ### Object unique elements

    @property
    def _n(self) -> int:
        return self.nodes.n

    @property
    def _ini_solve(self) -> int:
        return 0

    @property
    def id(self) -> str:
        """A str identifier to name the *Smile* used in
        :class:`~rateslib.solver.Solver` mappings."""
        return self._id

    @property
    def nodes(self) -> _IRSplineSmileNodes:
        """An instance of :class:`~rateslib.volatility.utils._IRSplineSmileNodes`."""
        return self._nodes

    ### _WithMutability ABCs:

    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[np.object_]]:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array(self.nodes.values)

    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        return tuple(f"{self.id}{i}" for i in range(self._n))

    def _set_node_vector_direct(
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
        base_obj = DualType(0.0, [f"{self.id}{i}" for i in range(self.nodes.n)], *DualArgs)
        ident = np.eye(self.nodes.n)

        nodes_: dict[float, DualTypes] = {}
        for i, k in enumerate(self.nodes.keys):
            nodes_[k] = DualType.vars_from(
                base_obj,  # type: ignore[arg-type]
                vector[i].real,
                base_obj.vars,
                ident[i, :].tolist(),
                *DualArgs[1:],
            )
        self._nodes = _IRSplineSmileNodes(nodes=nodes_, k=self.nodes.k)
        self.nodes.spline.csolve(self.nodes, self.ad)

    def _set_ad_order_direct(self, order: int | None) -> None:
        # -1, -2 force updates to new variables
        if order is None or order == getattr(self, "ad", None):
            if self.nodes.spline.spline.c is None:
                self.nodes.spline.csolve(self.nodes, _get_order_of(self.pricing_params[0]))
            return None
        elif abs(order) not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self._ad = abs(order)
        nodes: dict[float, DualTypes] = {
            k: set_order_convert(v, abs(order), [f"{self.id}{i}"])
            for i, (k, v) in enumerate(self.nodes.nodes.items())
        }
        self._nodes = _IRSplineSmileNodes(nodes=nodes, k=self.nodes.spline.k)
        self.nodes.spline.csolve(self.nodes, self.ad)

    def _set_single_node(self, key: float, value: DualTypes) -> None:
        if key not in self.nodes.keys:
            raise KeyError(f"'{key}' is not in `nodes`.")
        self.nodes._nodes[key] = value
        self.nodes.spline.csolve(self.nodes, self.ad)

    # _BaseIRSmile ABCS:

    def _plot(
        self,
        x_axis: str,
        f: float,
        y_axis: str,
        tgt_shift: float_,
    ) -> tuple[Iterable[float], Iterable[float]]:

        # approximate a range for the x-axis
        shf = _dual_float(self.meta.shift) / 100.0
        sq_t = self._meta.t_expiry_sqrt
        v_ = _dual_float(self.get_from_strike(k=f, f=f).vol) / 100.0
        if self.meta.pricing_model == OptionPricingModel.Black76:
            v_ = v_
        else:
            v_ = v_ / (f + shf)

        x_low = _dual_float(
            dual_exp(0.5 * v_**2 * sq_t**2 - dual_inv_norm_cdf(0.95) * v_ * sq_t) * (f + shf) - shf
        )
        x_top = _dual_float(
            dual_exp(0.5 * v_**2 * sq_t**2 - dual_inv_norm_cdf(0.05) * v_ * sq_t) * (f + shf) - shf
        )

        x = np.linspace(x_low, x_top, 301, dtype=np.float64)
        y: Iterable[float] = [_dual_float(self.get_from_strike(k=_, f=f).vol) for _ in x]

        return self._plot_conversion(
            y_axis=y_axis, x_axis=x_axis, f=f, shift=shf, tgt_shift=_drb(shf, tgt_shift), x=x, y=y
        )

    @property
    def ad(self) -> int:
        return self._ad

    @property
    def pricing_params(self) -> Sequence[float | Dual | Dual2 | Variable]:
        return self.nodes.values

    @property
    def meta(self) -> _IRSmileMeta:
        return self._meta

    def _get_from_strike(self, k: DualTypes, f: DualTypes) -> _IRVolPricingParams:
        """
        Given an option strike return the volatility.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        f: float, Dual, Dual2
            The forward rate at delivery of the option.
        expiry: datetime, optional
            The expiry of the option. Required for temporal interpolation.
        tenor: datetime, optional
            The termination date of the underlying *IRS*, required for parameter interpolation.
        curves: _Curves,
            Pricing objects. See **Pricing** on :class:`~rateslib.instruments.IRSCall`
            for details of allowed inputs.

        Returns
        -------
        _IRVolPricingParams
        """
        vol_ = self.nodes.spline.evaluate(x=(k - f) * 100.0, m=0)
        return _IRVolPricingParams(
            vol=vol_,
            k=k,
            f=f,
            shift=self.meta.shift,
            pricing_model=self.meta.pricing_model,
            t_e=self.meta.t_expiry,
        )


class IRSplineCube(_BaseIRCube[float | Variable], _WithMutability):
    r"""
    Create an *IR Volatility Cube* parametrized by :class:`~rateslib.volatility.IRSplineSmile` at
    different expiries and *IRS* tenors.

    .. warning::

       *Swaptions* and *IR Volatility* are in Beta status introduced in v2.7.0

    See also the :ref:`IR Vol Smiles & Cubes <c-ir-smile-doc>` section in the user guide.

    .. role:: green

    .. role:: red

    Parameters
    ----------
    expiries: list[datetime | str], :red:`required`
        Datetimes representing the expiries of each parametrised *Smile*, in ascending order.
    tenors: list[str], :red:`required`
        The tenors of each underlying *IRS* from each expiry for the parameterised *Smiles*.
    strikes: list[float], :red:`required`
        The indexes for the strike values on each *Smile*, expressed in basis points relative to the
        ATM forward rate.
    eval_date: datetime, :red:`required`
        Acts as the initial node of a *Curve*. Should be assigned today's immediate date.
        If expiry is given as string used to derive the specific date.
    irs_series: str, IRSSeries, :red:`required`
        The :class:`~rateslib.data.fixings.IRSSeries` that contains the parameters for the
        underlying :class:`~rateslib.instruments.IRS` that the swaptions are settled against.
    parameters: float, Dual, Dual2, Variable or 3d-ndarray of such
        The parameters for each *Smile* either adopting a single universal value or as a 3D array
        with axes (expiry, tenor, strike).
    k: int in {2, 4}, :green:`optional (set as 2)`
        The order of the interpolating spline, with (2, 4) representing (linear, cubic)
        interpolation respectively.
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
    See :class:`~rateslib.volatility.FXSabrSmile` for a description of SABR parameters for
    *Smile* construction.

    **Temporal Interpolation**

    Interpolation along the expiry axis occurs by performing total linear variance interpolation
    for a given *strike* measured on neighboring *Smiles*.

    If ``weights`` are given this uses the scaling approach of forward volatility (as demonstrated
    in Clark's *FX Option Pricing*) for calendar days (different options 'cuts' and timezone are
    not implemented). A datetime indexed `Series` must be provided, where any calendar date that
    is not included will be assigned the default weight of 1.0.

    See :ref:`constructing FX volatility surfaces <c-fx-smile-doc>` for more details.

    **Extrapolation**

    When an ``expiry`` is sought that is prior to the first parametrised *Smile expiry* or after the
    final parametrised *Smile expiry* extrapolation is required. This is not recommended,
    however. It would be wiser to create parameterised *Smiles* at *expiries* which suit those
    one wishes to obtian values for.

    When seeking an ``expiry`` beyond the final expiry, a new
    :class:`~rateslib.volatility.SabrSmile` is created at that specific *expiry* using the
    same SABR parameters as matching the final parametrised *Smile*. This will capture the
    evolution of ATM-forward rates through time.

    When seeking an ``expiry`` prior to the first expiry, the volatility found on the first *Smile*
    will be used an interpolated, using total linear variance accooridng to the given ``weights``.
    If ``weights`` are not used then this will return the same value as obtained from that
    first parametrised *Smile*. This does not account any evolution of ATM-forward rates.

    """

    _ini_solve = 0
    _SmileType = IRSplineSmile
    _meta: _IRCubeMeta
    _id: str

    def __init__(
        self,
        expiries: list[datetime | str],
        tenors: list[str],
        strikes: list[float],
        eval_date: datetime,
        irs_series: str | IRSSeries,
        parameters: DualTypes | Arr3dObj,
        shift: DualTypes_ = NoInput(0),
        k: int_ = NoInput(0),
        weights: Series[float] | NoInput = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int = 0,
    ):
        self._id: str = (
            uuid4().hex[:5] + "_" if isinstance(id, NoInput) else id
        )  # 1 in a million clash

        self._meta = _IRCubeMeta(
            _eval_date=eval_date,
            _tenors=tenors,
            _weights=weights,
            _indexes=strikes,
            _expiries=expiries,
            _irs_series=_get_irs_series(irs_series),
            _shift=_drb(0.0, shift),
            _smile_params=dict(k=_drb(2, k)),
        )

        _shape = (self.meta._n_expiries, self.meta._n_tenors, len(strikes))
        self._node_values_: Arr3dObj = np.empty(shape=_shape, dtype=object)
        if isinstance(parameters, float | Dual | Dual2 | Variable):
            self._node_values_.fill(parameters)
        else:
            p = np.asarray(parameters)
            if p.shape != _shape:
                raise ValueError(
                    "If providing `parameters` must be a 3D array-like with shape "
                    "(expiries, tenors, strikes)."
                )
            self._node_values_ = p

        self._set_ad_order(ad)  # includes csolve on each smile
        self._set_new_state()

    @property
    def _n(self) -> int:
        """Number of pricing parameters of the *Cube*."""
        en = self._node_values_.shape[0]
        tn = self._node_values_.shape[1]
        sn = self._node_values_.shape[2]
        return en * tn * sn

    @property
    def id(self) -> str:
        """A str identifier to name the *Surface* used in
        :class:`~rateslib.solver.Solver` mappings."""
        return self._id

    @property
    def meta(self) -> _IRCubeMeta:
        """An instance of :class:`~rateslib.volatility._IRCubeMeta`."""
        return self._meta

    @property
    def pricing_params(self) -> Arr3dObj:
        """The pricing parameters of the *Cube* as 3-d array by (expiry, tenor, strike)."""
        return self._node_values_

    @property
    def ad(self) -> int:
        """Int in {0,1,2} describing the AD order associated with the *Surface*."""
        return self._ad

    def _set_ad_order_direct(self, order: int | None) -> None:
        # -1, and -2 input will force direct vars settings.
        if order is None or order == getattr(self, "ad", None):
            return None
        elif abs(order) not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self._ad = abs(order)
        vec = self._get_node_vector()
        vars_ = self._get_node_vars()
        new_vec = [set_order_convert(v, abs(order), [t]) for v, t in zip(vec, vars_, strict=False)]
        self._node_values_ = np.reshape(
            np.array(new_vec), (self.meta._n_expiries, self.meta._n_tenors, len(self.meta.indexes))
        )
        return None

    def _set_node_vector_direct(
        self, vector: np.ndarray[tuple[int, ...], np.dtype[np.object_]], ad: int
    ) -> None:
        shape = self._node_values_.shape
        if ad == 0:
            self._node_values_ = np.reshape([_dual_float(_) for _ in vector], shape)
        else:
            DualType: type[Dual] | type[Dual2] = Dual if ad == 1 else Dual2
            DualArgs: tuple[list[float]] | tuple[list[float], list[float]] = (
                ([],) if ad == 1 else ([], [])
            )
            vars_ = self._get_node_vars()
            base_obj = DualType(0.0, vars_, *DualArgs)
            ident = np.eye(len(vars_))
            self._node_values_ = np.reshape(
                [
                    DualType.vars_from(
                        base_obj,  #  type: ignore[arg-type]
                        _dual_float(v),
                        base_obj.vars,
                        ident[j, :].tolist(),
                        *DualArgs[1:],
                    )
                    for j, v in enumerate(vector)
                ],
                shape,
            )

    def _set_single_node_direct(
        self, key: tuple[datetime, datetime, float | Variable], value: DualTypes
    ) -> None:
        """
        Update some generic parameters on the *SplineCube*.

        Parameters
        ----------
        key: tuple of (datetime, datetime, float)
            The node value to update, indexed by (expiry, tenor, strike).
        value: Array, float, Dual, Dual2, Variable
            Value to update on the *Cube*.

        Returns
        -------
        None

        Notes
        -----
        This function may update all of the AD variable names to be a consistent pricing object
        familiar to a :class:`~rateslib.solver.Solver`.

        .. warning::

           *Rateslib* is an object-oriented library that uses complex associations. Although
           Python may not object to directly mutating attributes of a *Curve* instance, this
           should be avoided in *rateslib*. Only use official ``update`` methods to mutate the
           values of an existing *Curve* instance.
           This class is labelled as a **mutable on update** object.

        """
        if key[2] not in self.meta.indexes:
            raise KeyError(f"'{key[2]}' is not in `meta.indexes`.")

        tenor_row = self.meta.expiry_dates.index(key[0])
        self._node_values_[
            self.meta.expiry_dates.index(key[0]),
            self.meta.tenor_dates[tenor_row].tolist().index(key[1]),
            self.meta.indexes.index(key[2]),
        ] = value

        self._set_ad_order(self.ad)
        return None
