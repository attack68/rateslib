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
from pandas import DataFrame, Index

from rateslib.curves.interpolation import index_left
from rateslib.data.fixings import IRSSeries, _get_irs_series
from rateslib.dual import (
    Dual,
    Dual2,
    Variable,
    set_order_convert,
)
from rateslib.dual.utils import _dual_float, _to_number, dual_exp, dual_inv_norm_cdf
from rateslib.enums.generics import NoInput, _drb
from rateslib.mutability import (
    _clear_cache_post,
    _new_state_post,
    _WithCache,
    _WithState,
)
from rateslib.volatility.ir.base import _BaseIRSmile
from rateslib.volatility.ir.utils import (
    _bilinear_interp,
    _IRSabrCubeMeta,
    _IRSmileMeta,
    _IRVolPricingParams,
)
from rateslib.volatility.utils import _SabrModel, _SabrSmileNodes

UTC = timezone.utc

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Arr2dObj,
        Arr3dObj,
        CurvesT_,
        DualTypes,
        DualTypes_,
        Number,
        Sequence,
        Series,
        datetime_,
    )


class IRSabrSmile(_BaseIRSmile):
    r"""
    Create an *IR Volatility Smile* at a given expiry indexed for a specific IRS tenor
    using SABR parameters.

    .. role:: green

    .. role:: red

    Parameters
    ----------
    nodes: dict[str, float], :red:`required`
        The parameters for the SABR model. Keys must be *'alpha', 'beta', 'rho', 'nu'*. See below.
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
    shift: float, Variable, :green:`optional (set as zero)`
        The number of basis points to apply to the strike and forward under a 'Black Shifted
        Volatility' model.
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
    The keys for ``nodes`` are described as the following:

    - ``alpha``: The initial volatility parameter (e.g. 0.10 for 10%) of the SABR model,
      in (0, inf).
    - ``beta``: The scaling parameter between normal (0) and lognormal (1)
      of the SABR model in [0, 1].
    - ``rho``: The correlation between spot and volatility of the SABR model,
      e.g. -0.10, in [-1.0, 1.0)
    - ``nu``: The volatility of volatility parameter of the SABR model, e.g. 0.80.

    The parameters :math:`\alpha, \rho, \nu` will be calibrated/mutated by
    a :class:`~rateslib.solver.Solver` object. These should be entered as *float* and the argument
    ``ad`` can be used to automatically tag these as variables.

    The parameter :math:`\beta` will **not** be calibrated/mutated by a
    :class:`~rateslib.solver.Solver`. This value can be entered either as a *float*, or a
    :class:`~rateslib.dual.Variable` to capture exogenous sensivities.

    The arguments ``delivery_lag``, ``calendar`` and ``pair`` are only required if using an
    :class:`~rateslib.fx.FXForwards` object to forecast ATM-forward FX rates for pricing. If
    the forward rates are supplied directly as numeric values these arguments are not required.

    Examples
    --------
    See :ref:`Constructing a Smile <c-fx-smile-constructing-doc>`.

    """

    _ini_solve = 1
    _meta: _IRSmileMeta
    _id: str
    _nodes: _SabrSmileNodes

    @_new_state_post
    def __init__(
        self,
        nodes: dict[str, DualTypes],
        eval_date: datetime,
        expiry: datetime | str,
        irs_series: IRSSeries | str,
        tenor: datetime | str,
        *,
        shift: DualTypes_ = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int | None = 0,
    ):
        self._id: str = (
            uuid4().hex[:5] + "_" if isinstance(id, NoInput) else id
        )  # 1 in a million clash
        self._meta = _IRSmileMeta(
            _tenor_input=tenor,
            _irs_series=_get_irs_series(irs_series),
            _eval_date=eval_date,
            _expiry_input=expiry,
            _plot_x_axis="strike",
            _shift=_drb(0.0, shift),
        )

        try:
            self._nodes: _SabrSmileNodes = _SabrSmileNodes(
                _alpha=_to_number(nodes["alpha"]),
                _beta=nodes["beta"],  # type: ignore[arg-type]
                _rho=_to_number(nodes["rho"]),
                _nu=_to_number(nodes["nu"]),
            )
        except KeyError as e:
            for _ in ["alpha", "beta", "rho", "nu"]:
                if _ not in nodes:
                    raise ValueError(
                        f"'{_}' is a required SABR parameter that must be included in ``nodes``"
                    )
            raise e  # pragma: no cover

        self._set_ad_order(ad)

    @property
    def _n(self) -> int:
        """The number of pricing parameters in ``nodes``."""
        return self.nodes.n

    @property
    def id(self) -> str:
        """A str identifier to name the *Smile* used in
        :class:`~rateslib.solver.Solver` mappings."""
        return self._id

    @property
    def meta(self) -> _IRSmileMeta:  # type: ignore[override]
        """An instance of :class:`~rateslib.volatility.ir.utils._IRSmileMeta`."""
        return self._meta

    @property
    def nodes(self) -> _SabrSmileNodes:
        """An instance of :class:`~rateslib.volatility.utils._SabrSmileNodes`."""
        return self._nodes

    def get_from_strike(
        self,
        k: DualTypes,
        expiry: datetime_ = NoInput(0),
        tenor: datetime_ = NoInput(0),
        f: DualTypes_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
    ) -> _IRVolPricingParams:
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
            Pricing objects. See **Pricing** on :class:`~rateslib.instruments.PayerSwaption`
            for details of allowed inputs.

        Returns
        -------
        _IRVolPricingParams

        """
        if isinstance(expiry, datetime) and self._meta.expiry != expiry:
            raise ValueError(
                "`expiry` of VolSmile and OptionPeriod do not match: calculation aborted "
                "due to potential pricing errors.",
            )

        if isinstance(f, NoInput):
            f_: DualTypes = self.meta.irs_fixing.irs.rate(curves=curves)
        else:
            f_ = f
        del f

        vol_ = _SabrModel._d_sabr_d_k_or_f(
            _to_number(k + self.meta.rate_shift),
            _to_number(f_ + self.meta.rate_shift),
            self._meta.t_expiry,
            self.nodes.alpha,
            self.nodes.beta,
            self.nodes.rho,
            self.nodes.nu,
            derivative=0,
        )[0]
        return _IRVolPricingParams(vol=vol_ * 100.0, k=k, f=f_, shift=self.meta.rate_shift)

    def _d_sabr_d_k_or_f(
        self,
        k: DualTypes,
        f: DualTypes,
        expiry: datetime,
        as_float: bool,
        derivative: int,
    ) -> tuple[DualTypes, DualTypes | None]:
        """Get the derivative of sabr vol with respect to strike

        as_float: bool
            Allow expedited calculation by avoiding dual numbers. Useful during the root solving
            phase of Newton iterations.
        derivative: int
            For with respect to `k` use 1, or `f` use 2.
        """
        t_e = (expiry - self._meta.eval_date).days / 365.0
        K = k + self.meta.rate_shift
        F = f + self.meta.rate_shift
        del k, f

        if as_float:
            k_: Number = _dual_float(K)
            f_: Number = _dual_float(F)
            a_: Number = _dual_float(self.nodes.alpha)
            b_: float | Variable = _dual_float(self.nodes.beta)
            p_: Number = _dual_float(self.nodes.rho)
            v_: Number = _dual_float(self.nodes.nu)
        else:
            k_ = _to_number(K)
            f_ = _to_number(F)
            a_ = self.nodes.alpha  #
            b_ = self.nodes.beta
            p_ = self.nodes.rho
            v_ = self.nodes.nu

        return _SabrModel._d_sabr_d_k_or_f(k_, f_, t_e, a_, b_, p_, v_, derivative)

    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[np.object_]]:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array([self.nodes.alpha, self.nodes.rho, self.nodes.nu])

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

        self._nodes = _SabrSmileNodes(
            _beta=self.nodes.beta,
            _alpha=DualType.vars_from(
                base_obj,  # type: ignore[arg-type]
                vector[0].real,
                base_obj.vars,
                ident[0, :].tolist(),
                *DualArgs[1:],
            ),
            _rho=DualType.vars_from(
                base_obj,  # type: ignore[arg-type]
                vector[1].real,
                base_obj.vars,
                ident[1, :].tolist(),
                *DualArgs[1:],
            ),
            _nu=DualType.vars_from(
                base_obj,  # type: ignore[arg-type]
                vector[2].real,
                base_obj.vars,
                ident[2, :].tolist(),
                *DualArgs[1:],
            ),
        )

    @_clear_cache_post
    def _set_ad_order(self, order: int | None) -> None:
        """This does not alter the beta node, since that is not varied by a Solver.
        beta values that are AD sensitive should be given as a Variable and not Dual/Dual2.

        Using `None` allows this Smile to be constructed without overwriting any variable names.
        """
        if order == getattr(self, "_ad", None):
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self._ad = order

        self._nodes = _SabrSmileNodes(
            _beta=self.nodes.beta,
            _alpha=set_order_convert(self.nodes.alpha, order, [f"{self.id}0"]),
            _rho=set_order_convert(self.nodes.rho, order, [f"{self.id}1"]),
            _nu=set_order_convert(self.nodes.nu, order, [f"{self.id}2"]),
        )

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
        params = ["alpha", "beta", "rho", "nu"]
        if key not in params:
            raise KeyError(f"'{key}' is not in `nodes`.")
        kwargs = {f"_{_}": getattr(self.nodes, _) for _ in params if _ != key}
        kwargs.update({f"_{key}": value})
        self._nodes = _SabrSmileNodes(**kwargs)
        self._set_ad_order(self.ad)

    # Plotting

    def _plot(
        self,
        x_axis: str,
        f: DualTypes_,
    ) -> tuple[list[float], list[DualTypes]]:
        if isinstance(f, NoInput):
            raise ValueError("`f` (ATM-forward FX rate) is required by `FXSabrSmile.plot`.")
        elif isinstance(f, float | Dual | Dual2 | Variable):
            f_: float = _dual_float(f)
        del f

        v_ = _dual_float(self.get_from_strike(k=f_, f=f_).vol) / 100.0
        sq_t = self._meta.t_expiry_sqrt
        x_low = _dual_float(
            dual_exp(0.5 * v_**2 * sq_t**2 - dual_inv_norm_cdf(0.95) * v_ * sq_t) * f_
        )
        x_top = _dual_float(
            dual_exp(0.5 * v_**2 * sq_t**2 - dual_inv_norm_cdf(0.05) * v_ * sq_t) * f_
        )

        x = np.linspace(x_low, x_top, 301, dtype=np.float64)
        u: Sequence[float] = x / f_  # type: ignore[assignment]
        y: list[DualTypes] = [self.get_from_strike(k=_, f=f_).vol for _ in x]
        if x_axis == "moneyness":
            return list(u), y
        else:  # x_axis = "strike"
            return list(x), y


class IRSabrCube(_WithState, _WithCache[tuple[datetime, datetime], IRSabrSmile]):
    r"""
    Create an *FX Volatility Surface* parametrised by cross-sectional *Smiles* at different
    expiries.

    See also the :ref:`FX Vol Surfaces section in the user guide <c-fx-smile-doc>`.

    Parameters
    ----------
    expiries: list[datetime]
       Datetimes representing the expiries of each cross-sectional *Smile*, in ascending order.
    node_values: 2d-shape of float, Dual, Dual2
       An array of values representing each *alpha, beta, rho, nu* node value on each
       cross-sectional *Smile*. Should be an array of size: (length of ``expiries``, 4).
    eval_date: datetime
       Acts as the initial node of a *Curve*. Should be assigned today's immediate date.
    weights: Series, optional
       Weights used for temporal volatility interpolation. See notes.
    delivery_lag: int, optional
        The number of business days after expiry that the physical settlement of the FX
        exchange occurs. Uses ``defaults.fx_delivery_lag``. Used in determination of ATM forward
        rates for different expiries.
    calendar : calendar or str, optional
        The holiday calendar object to use for FX delivery day determination. If str, looks up
        named calendar from static data.
    pair : str, optional
        The FX currency pair used to determine ATM forward rates.
    id: str, optional
       The unique identifier to label the *Surface* and its variables.
    ad: int, optional
       Sets the automatic differentiation order. Defines whether to convert node
       values to float, :class:`~rateslib.dual.Dual` or
       :class:`~rateslib.dual.Dual2`. It is advised against
       using this setting directly. It is mainly used internally.

    Notes
    -----
    See :class:`~rateslib.fx_volatility.FXSabrSmile` for a description of SABR parameters for
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
    :class:`~rateslib.fx_volatility.SabrSmile` is created at that specific *expiry* using the
    same SABR parameters as matching the final parametrised *Smile*. This will capture the
    evolution of ATM-forward rates through time.

    When seeking an ``expiry`` prior to the first expiry, the volatility found on the first *Smile*
    will be used an interpolated, using total linear variance accooridng to the given ``weights``.
    If ``weights`` are not used then this will return the same value as obtained from that
    first parametrised *Smile*. This does not account any evolution of ATM-forward rates.

    """

    _ini_solve = 0
    _meta: _IRSabrCubeMeta
    _id: str

    def __init__(
        self,
        expiries: list[datetime | str],
        tenors: list[str],
        eval_date: datetime,
        beta: DualTypes,
        alphas: DualTypes | Arr2dObj,
        rhos: DualTypes | Arr2dObj,
        nus: DualTypes | Arr2dObj,
        irs_series: str | IRSSeries,
        weights: Series[float] | NoInput = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int = 0,
    ):
        self._id: str = (
            uuid4().hex[:5] + "_" if isinstance(id, NoInput) else id
        )  # 1 in a million clash

        self._meta = _IRSabrCubeMeta(
            _eval_date=eval_date,
            _tenors=tenors,
            _weights=weights,
            _expiries=expiries,
            _irs_series=_get_irs_series(irs_series),
        )

        self._beta = beta
        _shape = (self.meta._n_expiries, self.meta._n_tenors)
        self._node_values_: Arr3dObj = np.empty(shape=_shape + (3,), dtype=object)
        for i, kw in enumerate([alphas, rhos, nus]):
            if isinstance(kw, float | Dual | Dual2 | Variable):
                self._node_values_[:, :, i] = np.full(fill_value=kw, shape=_shape)
            else:
                self._node_values_[:, :, i] = np.asarray(kw)

        self._set_ad_order(ad)  # includes csolve on each smile
        self._set_new_state()

    @property
    def beta(self) -> DualTypes:
        """The *beta*  value of each :class:`~rateslib.volatility.IRSabrSmile` associated with
        this *Cube*."""
        return self._beta

    @property
    def alpha(self) -> DataFrame:
        """The *alpha*  value of each :class:`~rateslib.volatility.IRSabrSmile` associated with
        this *Cube*."""
        return DataFrame(
            index=Index(data=self.meta.expiries, name="expiry"),
            columns=Index(data=self.meta.tenors, name="tenor"),
            data=self._node_values_[:, :, 0],
        )

    @property
    def rho(self) -> DataFrame:
        """The *rho*  value of each :class:`~rateslib.volatility.IRSabrSmile` associated with
        this *Cube*."""
        return DataFrame(
            index=Index(data=self.meta.expiries, name="expiry"),
            columns=Index(data=self.meta.tenors, name="tenor"),
            data=self._node_values_[:, :, 1],
        )

    @property
    def nu(self) -> DataFrame:
        """The *nu*  value of each :class:`~rateslib.volatility.IRSabrSmile` associated with
        this *Cube*."""
        return DataFrame(
            index=Index(data=self.meta.expiries, name="expiry"),
            columns=Index(data=self.meta.tenors, name="tenor"),
            data=self._node_values_[:, :, 2],
        )

    @property
    def _n(self) -> int:
        """Number of pricing parameters of the *Cube*."""
        en = self._node_values_.shape[0]
        tn = self._node_values_.shape[1]
        return en * tn * 3  # alpha, beta, rho

    @property
    def id(self) -> str:
        """A str identifier to name the *Surface* used in
        :class:`~rateslib.solver.Solver` mappings."""
        return self._id

    @property
    def meta(self) -> _IRSabrCubeMeta:
        """An instance of :class:`~rateslib.volatility.fx._FXSabrSurfaceMeta`."""
        return self._meta

    @property
    def ad(self) -> int:
        """Int in {0,1,2} describing the AD order associated with the *Surface*."""
        return self._ad

    @_clear_cache_post
    def _set_ad_order(self, order: int) -> None:
        if order == getattr(self, "ad", None):
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self._ad = order
        vec = self._get_node_vector()
        vars_ = self._get_node_vars()
        new_vec = [set_order_convert(v, order, [t]) for v, t in zip(vec, vars_, strict=False)]
        en = self._node_values_.shape[0]
        tn = self._node_values_.shape[1]
        n = en * tn
        self._node_values_[:, :, 0] = np.reshape(list(new_vec[:n]), (en, tn))
        self._node_values_[:, :, 1] = np.reshape(list(new_vec[n : 2 * n]), (en, tn))
        self._node_values_[:, :, 2] = np.reshape(list(new_vec[2 * n :]), (en, tn))
        return None

    @_new_state_post
    @_clear_cache_post
    def _set_node_vector(
        self, vector: np.ndarray[tuple[int, ...], np.dtype[np.object_]], ad: int
    ) -> None:
        en = self._node_values_.shape[0]
        tn = self._node_values_.shape[1]
        n = en * tn
        if ad == 0:
            self._node_values_[:, :, 0] = np.reshape([_dual_float(_) for _ in vector[:n]], (en, tn))
            self._node_values_[:, :, 1] = np.reshape(
                [_dual_float(_) for _ in vector[n : 2 * n]], (en, tn)
            )
            self._node_values_[:, :, 2] = np.reshape(
                [_dual_float(_) for _ in vector[2 * n :]], (en, tn)
            )
        else:
            DualType: type[Dual] | type[Dual2] = Dual if ad == 1 else Dual2
            DualArgs: tuple[list[float]] | tuple[list[float], list[float]] = (
                ([],) if ad == 1 else ([], [])
            )
            vars_ = self._get_node_vars()
            base_obj = DualType(0.0, vars_, *DualArgs)
            ident = np.eye(len(vars_))
            for i in range(3):
                self._node_values_[:, :, i] = np.reshape(
                    [
                        DualType.vars_from(
                            base_obj,  #  type: ignore[arg-type]
                            _dual_float(vector[n * i + j]),
                            base_obj.vars,
                            ident[n * i + j, :].tolist(),
                            *DualArgs[1:],
                        )
                        for j in range(n)
                    ],
                    (en, tn),
                )

    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[np.object_]]:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.block(
            [
                self._node_values_[:, :, 0].ravel(),  # alphas
                self._node_values_[:, :, 1].ravel(),  # rhos
                self._node_values_[:, :, 2].ravel(),  # nus
            ]
        )

    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        vars_: tuple[str, ...] = ()
        for tag in ["_a_", "_p_", "_v_"]:
            vars_ += tuple(
                f"{self.id}{tag}{i}"
                for i in range(self._node_values_.shape[0] * self._node_values_.shape[1])
            )
        return vars_

    def _bilinear_interpolation(
        self,
        expiry: datetime,
        tenor: datetime,
    ) -> tuple[DualTypes, DualTypes, DualTypes]:
        """
        Linearly interpolate the expiries / tenors array and return interpolated values
        for the alpha, rho and nu parameters.

        Returns
        -------
        (alpha, rho, nu)
        """
        # For out of bounds expiry values convert to boundary expiries with tenor time adjustment
        if expiry < self.meta.expiry_dates[0]:
            return self._bilinear_interpolation(
                expiry=self.meta.expiry_dates[0],
                tenor=tenor + (self.meta.expiry_dates[0] - expiry),
            )
        elif expiry > self.meta.expiry_dates[-1]:
            return self._bilinear_interpolation(
                expiry=self.meta.expiry_dates[-1],
                tenor=tenor - (expiry - self.meta.expiry_dates[-1]),
            )

        e_posix = expiry.replace(tzinfo=UTC).timestamp()
        t_posix = tenor.replace(tzinfo=UTC).timestamp()

        match (self.meta._n_expiries, self.meta._n_tenors):
            case (1, 1):
                # nothing to interpolate: return the only parameters of the surface
                return (
                    self._node_values_[0, 0, 0],
                    self._node_values_[0, 0, 1],
                    self._node_values_[0, 0, 2],
                )

            case (1, _):
                # interpolate only over tenor
                e_l = 0
                e_l_p = 0
                t_posix_1 = t_posix - (e_posix - self.meta.expiries_posix[0])
                t_l_1 = index_left(
                    list_input=self.meta.tenor_dates_posix[0, :],  # type: ignore[arg-type]
                    list_length=self.meta._n_tenors,
                    value=t_posix_1,
                )
                t_l_1_p = t_l_1 + 1
                v_ = (0.0, 0.0)  # only one expiry so no interpolation over that dimension
                t_l_2, t_l_2_p = t_l_1, t_l_1_p
                h_: tuple[float, float] = (
                    (t_posix_1 - self.meta.tenor_dates_posix[e_l, t_l_1])
                    / (
                        self.meta.tenor_dates_posix[e_l, t_l_1_p]
                        - self.meta.tenor_dates_posix[e_l, t_l_1]
                    ),
                ) * 2

            case (_, 1):
                # interpolate only over expiry
                e_l = index_left(
                    list_input=self.meta.expiries_posix,
                    list_length=self.meta._n_expiries,
                    value=e_posix,
                )
                e_l_p = e_l + 1
                t_l_1, t_l_2 = 0, 0
                t_l_1_p, t_l_2_p = 0, 0
                h_ = (0, 0)
                v_ = (
                    (e_posix - self.meta.expiries_posix[e_l])
                    / (self.meta.expiries_posix[e_l_p] - self.meta.expiries_posix[e_l]),
                ) * 2

            case _:
                # perform true bilinear interpolation
                e_l = index_left(
                    list_input=self.meta.expiries_posix,
                    list_length=self.meta._n_expiries,
                    value=e_posix,
                )
                e_l_p = e_l + 1
                v_ = (
                    (e_posix - self.meta.expiries_posix[e_l])
                    / (self.meta.expiries_posix[e_l_p] - self.meta.expiries_posix[e_l]),
                ) * 2

                # these are the relative tenors as measured per each benchmark expiry
                t_posix_1 = t_posix - (e_posix - self.meta.expiries_posix[e_l])
                t_posix_2 = t_posix - (e_posix - self.meta.expiries_posix[e_l_p])

                t_l_1 = index_left(
                    list_input=self.meta.tenor_dates_posix[e_l, :],  # type: ignore[arg-type]
                    list_length=self.meta._n_tenors,
                    value=t_posix_1,
                )
                t_l_1_p = t_l_1 + 1
                t_l_2 = index_left(
                    list_input=self.meta.tenor_dates_posix[e_l_p, :],  # type: ignore[arg-type]
                    list_length=self.meta._n_tenors,
                    value=t_posix_2,
                )
                t_l_2_p = t_l_2 + 1

                h_ = (
                    (t_posix_1 - self.meta.tenor_dates_posix[e_l, t_l_1])
                    / (
                        self.meta.tenor_dates_posix[e_l, t_l_1 + 1]
                        - self.meta.tenor_dates_posix[e_l, t_l_1]
                    ),
                    (t_posix_2 - self.meta.tenor_dates_posix[e_l_p, t_l_2])
                    / (
                        self.meta.tenor_dates_posix[e_l_p, t_l_2 + 1]
                        - self.meta.tenor_dates_posix[e_l_p, t_l_2]
                    ),
                )

        h_ = (min(max(h_[0], 0), 1), min(max(h_[1], 0), 1))
        a = self._node_values_[:, :, 0]
        p = self._node_values_[:, :, 1]
        v = self._node_values_[:, :, 2]

        return tuple(  # type: ignore[return-value]
            [
                _bilinear_interp(
                    tl=param[e_l, t_l_1],
                    tr=param[e_l, t_l_1_p],
                    bl=param[e_l_p, t_l_2],
                    br=param[e_l_p, t_l_2_p],
                    h=h_,
                    v=v_,
                )
                for param in [a, p, v]
            ]
        )

    def get_from_strike(
        self,
        k: DualTypes,
        expiry: datetime,
        tenor: datetime,
        f: DualTypes_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
    ) -> _IRVolPricingParams:
        """
        Given an option strike, expiry and tenor, return the volatility.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        expiry: datetime, optional
            The expiry of the option. Required for temporal interpolation.
        tenor: datetime, optional
            The termination date of the underlying *IRS*, required for parameter interpolation.
        f: float, Dual, Dual2
            The forward rate at delivery of the option.
        curves: _Curves,
            Pricing objects. See **Pricing** on :class:`~rateslib.instruments.PayerSwaption`
            for details of allowed inputs.

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
        if (expiry, tenor) in self._cache:
            smile = self._cache[expiry, tenor]
        else:
            alpha, rho, nu = self._bilinear_interpolation(expiry=expiry, tenor=tenor)
            smile = self._cached_value(
                key=(expiry, tenor),
                val=IRSabrSmile(
                    nodes={
                        "alpha": alpha,
                        "beta": self.beta,
                        "rho": rho,
                        "nu": nu,
                    },
                    eval_date=self.meta.eval_date,
                    expiry=expiry,
                    tenor=tenor,
                    irs_series=self.meta.irs_series,
                    id="UNUSED_VARIABLE_NAME",
                    ad=None,  # ensure variables tags are not overridden by new `id`
                ),
            )
        return smile.get_from_strike(k=k, f=f, curves=curves)

    @_new_state_post
    @_clear_cache_post
    def update_node(self, key: str, value: DualTypes | Arr2dObj) -> None:
        """
        Update some generic parameters on the *SabrCube*.

        Parameters
        ----------
        key: str in {"alpha", "beta", "rho", "nu"}
            The node value to update.
        value: Array, float, Dual, Dual2, Variable
            Value to update on the *Cube*.

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
        params = ["alpha", "beta", "rho", "nu"]
        if key not in params:
            raise KeyError(f"'{key}' is not in `nodes`.")

        for i, key_ in enumerate(["alpha", "rho", "nu"]):
            _shape = (self.meta._n_expiries, self.meta._n_tenors)
            if key == key_:
                if isinstance(value, float | Dual | Dual2 | Variable):
                    self._node_values_[:, :, i] = np.full(fill_value=value, shape=_shape)
                else:
                    self._node_values_[:, :, i] = np.asarray(value)
                return None

        if not isinstance(value, float | Dual | Dual2 | Variable):
            raise ValueError("'beta' must must be a scalar quantity in [0, 1].")
        else:
            self._beta = value

        self._set_ad_order(self.ad)
