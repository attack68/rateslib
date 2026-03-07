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
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from rateslib.data.fixings import IRSSeries, _get_irs_series
from rateslib.dual import Dual, Dual2, Variable, set_order_convert
from rateslib.dual.utils import _dual_float, dual_exp, dual_inv_norm_cdf
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import OptionPricingModel, _get_option_pricing_model
from rateslib.mutability import (
    _new_state_post,
)
from rateslib.volatility.ir.base import _BaseIRSmile, _WithMutability
from rateslib.volatility.ir.utils import (
    _IRSmileMeta,
    _IRSplineSmileNodes,
    _IRVolPricingParams,
)

UTC = timezone.utc

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        DualTypes,
        DualTypes_,
        Iterable,
        Sequence,
        float_,
    )


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
        k: int = 2,
        pricing_model: OptionPricingModel | str = "normal_vol",
        shift: DualTypes_ = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int | None = 0,
    ):
        if k not in [2, 4]:
            raise ValueError(f"`k` must imply linear(2) or cubic(4) spline interpolation. Got {k}.")
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

        self._nodes = _IRSplineSmileNodes(nodes=nodes, k=k)

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
        """This does not alter the beta node, since that is not varied by a Solver.
        beta values that are AD sensitive should be given as a Variable and not Dual/Dual2.

        Using `None` allows this Smile to be constructed without overwriting any variable names.
        """
        if order == getattr(self, "ad", None):
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")
        else:
            self._ad = order
            nodes: dict[float, DualTypes] = {
                k: set_order_convert(v, order, [f"{self.id}{i}"])
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
            Pricing objects. See **Pricing** on :class:`~rateslib.instruments.IRCall`
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
            shift=self.meta.rate_shift,
            pricing_model=self.meta.pricing_model,
            t_e=self.meta.t_expiry,
        )
