from __future__ import annotations  # type hinting

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from pandas import Series
from pytz import UTC

from rateslib import defaults
from rateslib.calendars import get_calendar
from rateslib.default import (
    NoInput,
    _drb,
)
from rateslib.dual import (
    Dual,
    Dual2,
    Variable,
    dual_exp,
    dual_inv_norm_cdf,
    dual_log,
    dual_norm_cdf,
    set_order_convert,
)
from rateslib.dual.utils import _dual_float, _to_number
from rateslib.fx import FXForwards
from rateslib.fx_volatility.base import _BaseSmile
from rateslib.fx_volatility.utils import (
    _d_sabr_d_k_or_f,
    _FXSabrSmileMeta,
    _FXSabrSmileNodes,
    _FXSabrSurfaceMeta,
    _t_var_interp_d_sabr_d_k_or_f,
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

if TYPE_CHECKING:
    from rateslib.typing import CalInput, DualTypes, Number, Sequence, datetime_, int_, str_


class FXSabrSmile(_BaseSmile):
    r"""
    Create an *FX Volatility Smile* at a given expiry indexed by strike using SABR parameters.

    Parameters
    ----------
    nodes: dict[str, float]
        The parameters for the SABR model. Keys must be *'alpha', 'beta', 'rho', 'nu'*. See below.
    eval_date: datetime
        Acts as the initial node of a *Curve*. Should be assigned today's immediate date.
    expiry: datetime
        The expiry date of the options associated with this *Smile*
    id: str, optional
        The unique identifier to distinguish between *Smiles* in a multicurrency framework
        and/or *Surface*.
    delivery_lag: int, optional
        The number of business days after expiry that the physical settlement of the FX
        exchange occurs. Uses ``defaults.fx_delivery_lag``. Used in determination of ATM forward
        rates.
    calendar : calendar or str, optional
        The holiday calendar object to use for FX delivery day determination. If str, looks up
        named calendar from static data.
    pair : str, optional
        The FX currency pair used to determine ATM forward rates.
    ad: int, optional
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
    _meta: _FXSabrSmileMeta
    _id: str
    _nodes: _FXSabrSmileNodes

    @_new_state_post
    def __init__(
        self,
        nodes: dict[str, DualTypes],
        eval_date: datetime,
        expiry: datetime,
        delivery_lag: int_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        pair: str_ = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int = 0,
    ):
        self._id: str = (
            uuid4().hex[:5] + "_" if isinstance(id, NoInput) else id
        )  # 1 in a million clash

        delivery_lag_ = _drb(defaults.fx_delivery_lag, delivery_lag)
        cal_ = get_calendar(calendar)
        self._meta = _FXSabrSmileMeta(
            _eval_date=eval_date,
            _expiry=expiry,
            _plot_x_axis="strike",
            _calendar=cal_,
            _delivery_lag=delivery_lag_,
            _delivery=cal_.lag(expiry, delivery_lag_, True),
            _pair=_drb(None, pair),
        )

        for _ in ["alpha", "beta", "rho", "nu"]:
            if _ not in nodes:
                raise ValueError(
                    f"'{_}' is a required SABR parameter that must be included in ``nodes``"
                )
        self._nodes: _FXSabrSmileNodes = _FXSabrSmileNodes(
            _alpha=_to_number(nodes["alpha"]),
            _beta=nodes["beta"],  # type: ignore[arg-type]
            _rho=_to_number(nodes["rho"]),
            _nu=_to_number(nodes["nu"]),
        )

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
    def meta(self) -> _FXSabrSmileMeta:  # type: ignore[override]
        """An instance of :class:`~rateslib.fx_volatility.utils._FXSabrSmileMeta`."""
        return self._meta

    @property
    def nodes(self) -> _FXSabrSmileNodes:
        """An instance of :class:`~rateslib.fx_volatility.utils._FXSabrSmileNodes`."""
        return self._nodes

    def get_from_strike(
        self,
        k: DualTypes,
        f: DualTypes | FXForwards,
        expiry: datetime_ = NoInput(0),
        w_deli: DualTypes | NoInput = NoInput(0),
        w_spot: DualTypes | NoInput = NoInput(0),
    ) -> tuple[DualTypes, DualTypes, DualTypes]:
        """
        Given an option strike return the volatility.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        f: float, Dual, Dual2
            The forward rate at delivery of the option.
        expiry: datetime, optional
            Typically uses with *Surfaces*.
            If given, performs a check to ensure consistency of valuations. Raises if expiry
            requested and expiry of the *Smile* do not match. Used internally.
        w_deli: DualTypes, optional
            Not used by *SabrSmile*
        w_spot: DualTypes, optional
            Not used by *SabrSmile*

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
        expiry = _drb(self._meta.expiry, expiry)
        if self._meta.expiry != expiry:
            raise ValueError(
                "`expiry` of VolSmile and OptionPeriod do not match: calculation aborted "
                "due to potential pricing errors.",
            )

        if isinstance(f, FXForwards):
            if self._meta.pair is None:
                raise ValueError(
                    "`FXSabrSmile` must be specified with a `pair` argument to use "
                    "`FXForwards` objects for forecasting ATM-forward FX rates."
                )
            f_: DualTypes = f.rate(self._meta.pair, self._meta.delivery)
        elif isinstance(f, float | Dual | Dual2 | Variable):
            f_ = f
        else:
            raise ValueError("`f` (ATM-forward FX rate) must be a value or FXForwards object.")

        vol_ = _d_sabr_d_k_or_f(
            _to_number(k),
            _to_number(f_),
            self._meta.t_expiry,
            self.nodes.alpha,
            self.nodes.beta,
            self.nodes.rho,
            self.nodes.nu,
            derivative=0,
        )[0]
        return 0.0, vol_ * 100.0, k

    def _d_sabr_d_k_or_f(
        self,
        k: DualTypes,
        f: DualTypes | FXForwards,
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
        if isinstance(f, FXForwards):
            f__: DualTypes = f.rate(self._meta.pair, self._meta.delivery)
        else:
            f__ = f  # type: ignore[assignment]

        if as_float:
            k_: Number = _dual_float(k)
            f_: Number = _dual_float(f__)
            a_: Number = _dual_float(self.nodes.alpha)
            b_: float | Variable = _dual_float(self.nodes.beta)
            p_: Number = _dual_float(self.nodes.rho)
            v_: Number = _dual_float(self.nodes.nu)
        else:
            k_ = _to_number(k)
            f_ = _to_number(f__)
            a_ = self.nodes.alpha  #
            b_ = self.nodes.beta
            p_ = self.nodes.rho
            v_ = self.nodes.nu

        return _d_sabr_d_k_or_f(k_, f_, t_e, a_, b_, p_, v_, derivative)

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

        self._nodes = _FXSabrSmileNodes(
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
    def _set_ad_order(self, order: int) -> None:
        """This does not alter the beta node, since that is not varied by a Solver.
        beta values that are AD sensitive should be given as a Variable and not Dual/Dual2.
        """
        if order == getattr(self, "_ad", None):
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self._ad = order

        self._nodes = _FXSabrSmileNodes(
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
            raise KeyError("`key` is not in ``nodes``.")
        kwargs = {f"_{_}": getattr(self.nodes, _) for _ in params if _ != key}
        kwargs.update({f"_{key}": value})
        self._nodes = _FXSabrSmileNodes(**kwargs)
        self._set_ad_order(self.ad)

    # Plotting

    def _plot(
        self,
        x_axis: str,
        f: DualTypes | FXForwards | NoInput,
    ) -> tuple[list[float], list[DualTypes]]:
        if isinstance(f, NoInput):
            raise ValueError("`f` (ATM-forward FX rate) is required by `FXSabrSmile.plot`.")
        elif isinstance(f, FXForwards):
            if self._meta.pair is None:
                raise ValueError(
                    "`FXSabrSmile` must be specified with a `pair` argument to use "
                    "`FXForwards` objects for forecasting ATM-forward FX rates."
                )
            f_: float = _dual_float(f.rate(self._meta.pair, self._meta.delivery))
        elif isinstance(f, float | Dual | Dual2 | Variable):
            f_ = _dual_float(f)
        else:
            raise ValueError("`f` (ATM-forward FX rate) must be a value or FXForwards object.")

        v_ = _dual_float(self.get_from_strike(f_, f_)[1]) / 100.0
        sq_t = self._meta.t_expiry_sqrt
        x_low = _dual_float(
            dual_exp(0.5 * v_**2 * sq_t**2 - dual_inv_norm_cdf(0.95) * v_ * sq_t) * f_
        )
        x_top = _dual_float(
            dual_exp(0.5 * v_**2 * sq_t**2 - dual_inv_norm_cdf(0.05) * v_ * sq_t) * f_
        )

        x = np.linspace(x_low, x_top, 301, dtype=np.float64)
        u: Sequence[float] = x / f_  # type: ignore[assignment]
        y: list[DualTypes] = [self.get_from_strike(_, f_)[1] for _ in x]
        if x_axis == "moneyness":
            return list(u), y
        elif x_axis == "delta":
            # z_w = 1.0  # delta type is assumed to be 'forward' for SabrSmile
            # z_u = 1.0  #  delta type is assumed to be 'unadjusted' for SabrSmile
            eta_1 = 0.5  # for same reason

            sq_t = self._meta.t_expiry_sqrt
            dn = [
                -dual_log(u_) * 100.0 / (s_ * sq_t) + eta_1 * s_ * sq_t / 100.0
                for u_, s_ in zip(u, y, strict=True)
            ]
            delta_index = [dual_norm_cdf(-d_) for d_ in dn]
            return delta_index, y  # type: ignore[return-value]
        else:  # x_axis = "strike"
            return list(x), y


class FXSabrSurface(_WithState, _WithCache[datetime, FXSabrSmile]):
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
    _mutable_by_association = True
    _meta: _FXSabrSurfaceMeta
    _id: str
    _smiles: list[FXSabrSmile]

    def __init__(
        self,
        expiries: list[datetime],
        node_values: list[DualTypes],
        eval_date: datetime,
        weights: Series[float] | NoInput = NoInput(0),
        delivery_lag: int_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        pair: str_ = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int = 0,
    ):
        self._id: str = (
            uuid4().hex[:5] + "_" if isinstance(id, NoInput) else id
        )  # 1 in a million clash

        self._meta = _FXSabrSurfaceMeta(
            _eval_date=eval_date,
            _pair=_drb(None, pair),
            _calendar=get_calendar(calendar),
            _delivery_lag=_drb(defaults.fx_delivery_lag, delivery_lag),
            _weights=_validate_weights(weights, eval_date, expiries),
        )

        self.expiries: list[datetime] = expiries
        self.expiries_posix: list[float] = [
            _.replace(tzinfo=UTC).timestamp() for _ in self.expiries
        ]
        for idx in range(1, len(self.expiries)):
            if self.expiries[idx - 1] >= self.expiries[idx]:
                raise ValueError("Surface `expiries` are not sorted or contain duplicates.\n")

        node_values_: np.ndarray[tuple[int, ...], np.dtype[np.object_]] = np.asarray(node_values)
        self._smiles = [
            FXSabrSmile(
                nodes=dict(zip(["alpha", "beta", "rho", "nu"], node_values_[i, :], strict=True)),
                expiry=expiry,
                eval_date=self._meta.eval_date,
                delivery_lag=delivery_lag,
                calendar=calendar,
                pair=pair,
                id=f"{self.id}_{i}_",
            )
            for i, expiry in enumerate(self.expiries)
        ]

        self._set_ad_order(ad)  # includes csolve on each smile
        self._set_new_state()

    @property
    def _n(self) -> int:
        """Number of pricing parameters of the *Surface*."""
        return len(self.expiries) * 3  # alpha, beta, rho

    @property
    def id(self) -> str:
        """A str identifier to name the *Surface* used in
        :class:`~rateslib.solver.Solver` mappings."""
        return self._id

    @property
    def meta(self) -> _FXSabrSurfaceMeta:
        """An instance of :class:`~rateslib.fx_volatility.utils._FXSabrSurfaceMeta`."""
        return self._meta

    @property
    def ad(self) -> int:
        """Int in {0,1,2} describing the AD order associated with the *Surface*."""
        return self._ad

    @property
    def smiles(self) -> list[FXSabrSmile]:
        """A list of cross-sectional :class:`FXSabrSmile` instances."""
        return self._smiles

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
        m = 3
        for i in range(int(len(vector) / m)):
            # smiles are indexed by expiry, shortest first
            self.smiles[i]._set_node_vector(vector[i * m : i * m + m], ad)

    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[np.object_]]:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array([list(_._get_node_vector()) for _ in self.smiles]).ravel()

    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        vars_: tuple[str, ...] = ()
        for smile in self.smiles:
            vars_ += tuple(f"{smile.id}{i}" for i in range(3))
        return vars_

    # @_validate_states: not required becuase state is validated by interior function
    def get_from_strike(
        self,
        k: DualTypes,
        f: DualTypes | FXForwards,
        expiry: datetime,
        w_deli: DualTypes | NoInput = NoInput(0),
        w_spot: DualTypes | NoInput = NoInput(0),
    ) -> tuple[DualTypes, DualTypes, DualTypes]:
        """
        Given an option strike return the volatility.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        f: float, Dual, Dual2
            The forward rate at delivery of the option.
        expiry: datetime, optional
            The expiry of the option. Required for temporal interpolation between
            cross-sectional *Smiles*.
        w_deli: DualTypes, optional
            Not used by *SabrSurface*
        w_spot: DualTypes, optional
            Not used by *SabrSurface*

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
        vol_ = self._d_sabr_d_k_or_f(k, f, expiry, as_float=False, derivative=0)[0]
        return 0.0, vol_ * 100.0, k

    @_validate_states
    def _d_sabr_d_k_or_f(
        self,
        k: DualTypes,
        f: DualTypes | FXForwards,
        expiry: datetime,
        as_float: bool,
        derivative: int,
    ) -> tuple[DualTypes, DualTypes | None]:
        expiry_posix = expiry.replace(tzinfo=UTC).timestamp()
        e_idx = index_left_f64(self.expiries_posix, expiry_posix)

        if expiry == self.expiries[0]:
            # expiry matches the expiry on the first Smile, call that method directly.
            return self.smiles[0]._d_sabr_d_k_or_f(k, f, expiry, as_float, derivative)
        elif abs(expiry_posix - self.expiries_posix[e_idx + 1]) < 1e-10:
            # expiry matches an expiry of a known Smile (not the first), call method directly.
            return self.smiles[e_idx + 1]._d_sabr_d_k_or_f(k, f, expiry, as_float, derivative)
        elif expiry_posix > self.expiries_posix[-1]:
            # expiry is beyond that of the last known Smile. Construct a new Smile at the expiry
            # by using the SABR parameters of the final Smile. (allows for ATM-forward calculation)
            smile = FXSabrSmile(
                nodes={
                    "alpha": self.smiles[e_idx + 1].nodes.alpha,
                    "beta": self.smiles[e_idx + 1].nodes.beta,
                    "rho": self.smiles[e_idx + 1].nodes.rho,
                    "nu": self.smiles[e_idx + 1].nodes.nu,
                },
                eval_date=self._meta.eval_date,
                expiry=expiry,
                ad=self.ad,
                pair=NoInput(0) if self._meta.pair is None else self._meta.pair,
                delivery_lag=self._meta.delivery_lag,
                calendar=self._meta.calendar,
                id=self.smiles[e_idx + 1].id + "_ext",
            )
            return smile._d_sabr_d_k_or_f(k, f, expiry, as_float, derivative)
        elif expiry <= self._meta.eval_date:
            raise ValueError("`expiry` before the `eval_date` of the Surface is invalid.")
        elif expiry_posix < self.expiries_posix[0]:
            # expiry is before the expiry of the first known Smile.
            # calculate the vol as if it were for expiry on the first Smile and then use
            # temporal interpolation (including weights) to obtain an adjusted volatility.
            vol_, dvol_k_or_f = self.smiles[0]._d_sabr_d_k_or_f(
                k=k,
                f=f,
                expiry=self.smiles[0]._meta.expiry,
                as_float=as_float,
                derivative=derivative,
            )
            return _t_var_interp_d_sabr_d_k_or_f(
                expiries=self.expiries,
                expiries_posix=self.expiries_posix,
                expiry=expiry,
                expiry_posix=expiry_posix,
                expiry_index=e_idx,
                eval_posix=self._meta.eval_posix,
                weights_cum=self.meta.weights_cum,
                vol1=vol_,
                dvol1_dk=dvol_k_or_f,  # type: ignore[arg-type]
                vol2=vol_,
                dvol2_dk=dvol_k_or_f,  # type: ignore[arg-type]
                bounds_flag=-1,
                derivative=derivative > 0,
            )
        else:
            # expiry is sandwiched between two known Smile expiries.
            # Calculate the vol for strike on either of these Smiles and then interpolate
            # for the correct expiry, including weights.
            ls, rs = self.smiles[e_idx], self.smiles[e_idx + 1]  # left_smile, right_smile
            if not isinstance(f, FXForwards):
                raise ValueError(
                    "`f` must be supplied as `FXForwards` in order to calculate"
                    "dynamic ATM-forward rates for temporally-interpolated SABR volatility."
                )
            lvol, d_lvol_dk_or_f = ls._d_sabr_d_k_or_f(
                k=k, f=f, expiry=ls._meta.expiry, as_float=as_float, derivative=derivative
            )
            rvol, d_rvol_dk_or_f = rs._d_sabr_d_k_or_f(
                k=k, f=f, expiry=rs._meta.expiry, as_float=as_float, derivative=derivative
            )
            return _t_var_interp_d_sabr_d_k_or_f(
                expiries=self.expiries,
                expiries_posix=self.expiries_posix,
                expiry=expiry,
                expiry_posix=expiry_posix,
                expiry_index=e_idx,
                eval_posix=self._meta.eval_posix,
                weights_cum=self.meta.weights_cum,
                vol1=lvol,
                dvol1_dk=d_lvol_dk_or_f,  # type: ignore[arg-type]
                vol2=rvol,
                dvol2_dk=d_rvol_dk_or_f,  # type: ignore[arg-type]
                bounds_flag=0,
                derivative=derivative > 0,
            )
