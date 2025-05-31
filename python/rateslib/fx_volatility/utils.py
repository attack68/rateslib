from __future__ import annotations  # type hinting

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from typing import TYPE_CHECKING, TypeAlias

from pandas import Series
from pytz import UTC

from rateslib.calendars import get_calendar
from rateslib.default import (
    NoInput,
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
from rateslib.rs import _sabr_x0 as _rs_sabr_x0
from rateslib.rs import _sabr_x1 as _rs_sabr_x1
from rateslib.rs import _sabr_x2 as _rs_sabr_x2
from rateslib.splines import PPSplineDual, PPSplineDual2, PPSplineF64

if TYPE_CHECKING:
    from rateslib.typing import Any, CalTypes, Number

DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"  # if not defined causes _WithCache failure

TERMINAL_DATE = datetime(2100, 1, 1)


@dataclass(frozen=True)
class _FXDeltaVolSmileMeta:
    """
    An immutable container of meta data associated with a
    :class:`~rateslib.fx_volatility.FXDeltaVolSmile` used to make calculations.
    """

    _eval_date: datetime
    _expiry: datetime
    _plot_x_axis: str
    _delta_type: str

    @property
    def eval_date(self) -> datetime:
        """Evaluation date of the *Smile*."""
        return self._eval_date

    @property
    def expiry(self) -> datetime:
        """Expiry date of the options priced by this *Smile*"""
        return self._expiry

    @property
    def plot_x_axis(self) -> str:
        """The default ``x_axis`` parameter passed to
        :meth:`~rateslib.fx_volatility._BaseSmile.plot`"""
        return self._plot_x_axis

    @property
    def delta_type(self) -> str:
        """The delta type of the delta indexes associated with the ``nodes`` of the *Smile*."""
        return self._delta_type

    @cached_property
    def t_expiry(self) -> float:
        """Calendar days from eval to expiry divided by 365."""
        return (self._expiry - self._eval_date).days / 365.0

    @cached_property
    def t_expiry_sqrt(self) -> float:
        """Square root of ``t_expiry``."""
        ret: float = self.t_expiry**0.5
        return ret


class _FXDeltaVolSmileNodes:
    """
    A container for data relating to interpolating the `nodes` of a
    :class:`~rateslib.fx_volatility.FXDeltaVolSmile`.
    """

    _nodes: dict[float, DualTypes]
    _meta: _FXDeltaVolSmileMeta
    _spline: _FXDeltaVolSpline

    def __init__(self, nodes: dict[float, DualTypes], meta: _FXDeltaVolSmileMeta) -> None:
        self._nodes = nodes
        self._meta = meta

        if "_pa" in self.meta.delta_type:
            vol: DualTypes = self.values[-1] / 100.0
            upper_bound: float = _dual_float(
                dual_exp(
                    vol * self.meta.t_expiry_sqrt * (3.75 - 0.5 * vol * self.meta.t_expiry_sqrt),
                )
            )
        else:
            upper_bound = 1.0

        if self.n in [1, 2]:
            t = [0.0] * 4 + [upper_bound] * 4
        else:
            t = [0.0] * 4 + self.keys[1:-1] + [upper_bound] * 4
        self._spline = _FXDeltaVolSpline(t=t)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _FXDeltaVolSmileNodes):
            return False
        return self._nodes == other._nodes and self._meta == other._meta

    @property
    def plot_upper_bound(self) -> float:
        """The right side delta index bound used in a *'delta' x-axis* plot."""
        if "_pa" in self.meta.delta_type:
            # upper_bound      = exp(vol * t_expiry_sqrt * (3.75 - 0.5 * vol * t_expiry_sqrt)
            # plot_upper_bound = exp(vol * t_expiry_sqrt * (3.25 - 0.5 * vol * t_expiry_sqrt)
            return (
                self.spline.t[-1] - _dual_float(self.values[-1]) * self.meta.t_expiry_sqrt / 200.0
            )
        else:
            return 1.0

    @property
    def meta(self) -> _FXDeltaVolSmileMeta:
        """An instance of :class:`~rateslib.fx_volatility.utils._FXDeltaVolSmileMeta`."""
        return self._meta

    @property
    def nodes(self) -> dict[float, DualTypes]:
        """The initial nodes dict passed for construction of this class."""
        return self._nodes

    @cached_property
    def keys(self) -> list[float]:
        """A list of the delta index keys in ``nodes``."""
        return list(self.nodes.keys())

    @cached_property
    def values(self) -> list[DualTypes]:
        """A list of the delta index values in ``nodes``."""
        return list(self.nodes.values())

    @property
    def n(self) -> int:
        """The number of pricing parameters in ``nodees``."""
        return len(self.keys)

    @property
    def spline(self) -> _FXDeltaVolSpline:
        """An instance of :class:`~rateslib.fx_volatility.utils._FXDeltaVolSpline`."""
        return self._spline


class _FXDeltaVolSpline:
    """
    A container for data relating to interpolating the `nodes` of
    a :class:`~rateslib.fx_volatility.FXDeltaVolSmile` using a cubic PPSpline.
    """

    _t: list[float]
    _spline: PPSplineF64 | PPSplineDual | PPSplineDual2

    def __init__(self, t: list[float]) -> None:
        self._t = t
        self._spline = PPSplineF64(4, [0.0] * 5, None)  # placeholder: csolve will reengineer

    @property
    def t(self) -> list[float]:
        """The knot sequence of the PPSpline."""
        return self._t

    @property
    def spline(self) -> PPSplineF64 | PPSplineDual | PPSplineDual2:
        """An instance of :class:`~rateslib.splines.PPSplineF64`,
        :class:`~rateslib.splines.PPSplineDual` or :class:`~rateslib.splines.PPSplineDual2`"""
        return self._spline

    def _csolve_n_other(
        self, nodes: _FXDeltaVolSmileNodes, ad: int
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

        # left side constraint
        tau.insert(0, self.t[0])
        y.insert(0, set_order_convert(0.0, ad, None))
        left_n = 2  # natural spline

        # right side constraint
        tau.append(self.t[-1])
        y.append(set_order_convert(0.0, ad, None))
        if "_pa" in nodes.meta.delta_type:
            right_n = 1  # 1st derivative at zero
        else:
            right_n = 2  # natural spline
        return tau, y, left_n, right_n

    def csolve(self, nodes: _FXDeltaVolSmileNodes, ad: int) -> None:
        """
        Construct a spline of appropriate AD order and solve the spline coefficients for the
        given ``nodes``.

        Parameters
        ----------
        nodes: _FXDeltaVolSmileNodes
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
            self._spline = Spline(4, self.t, nodes.values * 4)  # type: ignore[arg-type]
        else:
            tau, y, left_n, right_n = self._csolve_n_other(nodes, ad)
            self._spline = Spline(4, self.t, None)
            self._spline.csolve(tau, y, left_n, right_n, False)  # type: ignore[arg-type]

    def to_json(self) -> str:
        """
        Serialize this object to JSON format.

        The object can be deserialized using the :meth:`~rateslib.serialization.from_json` method.

        Returns
        -------
        str
        """
        obj = dict(
            PyNative=dict(
                _FXDeltaVolSpline=dict(
                    t=self.t,
                )
            )
        )
        return json.dumps(obj)

    @classmethod
    def _from_json(cls, loaded_json: dict[str, Any]) -> _FXDeltaVolSpline:
        return _FXDeltaVolSpline(
            t=loaded_json["t"],
        )

    def __eq__(self, other: Any) -> bool:
        """CurveSplines are considered equal if their knot sequence and endpoints are equivalent.
        For the same nodes this will resolve to give the same spline coefficients.
        """
        if not isinstance(other, _FXDeltaVolSpline):
            return False
        else:
            return self.t == other.t


@dataclass(frozen=True)
class _FXDeltaVolSurfaceMeta:
    """
    An immutable container of meta data associated with a
    :class:`~rateslib.fx_volatility.FXDeltaVolSurface` used to make calculations.
    """

    _eval_date: datetime
    _delta_type: str
    _plot_x_axis: str
    _weights: Series[float] | None
    _delta_indexes: list[float]
    _expiries: list[datetime]

    def __post_init__(self) -> None:
        for idx in range(1, len(self.expiries)):
            if self.expiries[idx - 1] >= self.expiries[idx]:
                raise ValueError("Surface `expiries` are not sorted or contain duplicates.\n")

    @property
    def delta_indexes(self) -> list[float]:
        """A list of delta indexes associated with each cross-sectional
        :class:`~rateslib.fx_volatility.FXDeltaVolSmile`."""
        return self._delta_indexes

    @property
    def expiries(self) -> list[datetime]:
        """A list of the expiries of each cross-sectional
        :class:`~rateslib.fx_volatility.FXDeltaVolSmile`."""
        return self._expiries

    @cached_property
    def expiries_posix(self) -> list[float]:
        """A list of the unix timestamps of each date in ``expiries``."""
        return [_.replace(tzinfo=UTC).timestamp() for _ in self.expiries]

    @property
    def weights(self) -> Series[float] | None:
        """Weights used for temporal volatility interpolation."""
        return self._weights

    @cached_property
    def weights_cum(self) -> Series[float] | None:
        """Weight adjusted time to expiry (in calendar days) per date for temporal volatility
        interpolation."""
        if self.weights is None:
            return None
        else:
            return self.weights.cumsum()

    @property
    def eval_date(self) -> datetime:
        """Evaluation date of the *Surface*."""
        return self._eval_date

    @property
    def eval_posix(self) -> float:
        """The unix timestamp of the ``eval_date``."""
        return self.eval_date.replace(tzinfo=UTC).timestamp()

    @property
    def delta_type(self) -> str:
        """The delta type of the delta indexes associated with the ``nodes`` of each
        cross-sectional *Smile*."""
        return self._delta_type

    @property
    def plot_x_axis(self) -> str:
        """The default ``x_axis`` parameter passed to
        :meth:`~rateslib.fx_volatility._BaseSmile.plot`"""
        return self._plot_x_axis


@dataclass(frozen=True)
class _FXSabrSmileMeta:
    """
    An immutable container of meta data associated with a
    :class:`~rateslib.fx_volatility.FXSabrSmile` used to make calculations.
    """

    _eval_date: datetime
    _expiry: datetime
    _pair: str | None
    _calendar: CalTypes
    _delivery: datetime
    _delivery_lag: int
    _plot_x_axis: str

    @property
    def n(self) -> int:
        """The number of pricing parameters."""
        return 4

    @property
    def eval_date(self) -> datetime:
        """Evaluation date of the *Smile*."""
        return self._eval_date

    @property
    def expiry(self) -> datetime:
        """Expiry date of the options priced by this *Smile*"""
        return self._expiry

    @property
    def delivery(self) -> datetime:
        """Delivery date of the forward FX rate applicable to options priced by this *Smile*"""
        return self._delivery

    @property
    def delivery_lag(self) -> int:
        """Business day settlement lag between ``expiry`` and ``delivery``."""
        return self._delivery_lag

    @property
    def plot_x_axis(self) -> str:
        """The default ``x_axis`` parameter passed to
        :meth:`~rateslib.fx_volatility._BaseSmile.plot`"""
        return self._plot_x_axis

    @property
    def calendar(self) -> CalTypes:
        """Settlement calendar used to determine ``delivery`` from ``expiry``."""
        return self._calendar

    @property
    def pair(self) -> str | None:
        """FX pair against which options priced by this *Smile* settle against."""
        return self._pair

    @cached_property
    def t_expiry(self) -> float:
        """Calendar days from eval to expiry divided by 365."""
        return (self.expiry - self.eval_date).days / 365.0

    @cached_property
    def t_expiry_sqrt(self) -> float:
        """Square root of ``t_expiry``."""
        ret: float = self.t_expiry**0.5
        return ret


@dataclass(frozen=True)
class _FXSabrSmileNodes:
    """
    A container for data relating to the SABR parameters of a
    :class:`~rateslib.fx_volatility.FXSabrSmile`.
    """

    _alpha: Number
    _beta: float | Variable
    _rho: Number
    _nu: Number

    @property
    def alpha(self) -> Number:
        """The :math:`\\alpha` parameter of the SABR function."""
        return self._alpha

    @property
    def beta(self) -> float | Variable:
        """The :math:`\\beta` parameter of the SABR function."""
        return self._beta

    @property
    def rho(self) -> Number:
        """The :math:`\\rho` parameter of the SABR function."""
        return self._rho

    @property
    def nu(self) -> Number:
        """The :math:`\\nu` parameter of the SABR function."""
        return self._nu

    @property
    def n(self) -> int:
        return 4


@dataclass(frozen=True)
class _FXSabrSurfaceMeta:
    """
    An immutable container of meta data associated with a
    :class:`~rateslib.fx_volatility.FXSabrSurface` used to make calculations.
    """

    _eval_date: datetime
    _pair: str | None
    _calendar: CalTypes
    _delivery_lag: int
    _weights: Series[float] | None

    @property
    def weights(self) -> Series[float] | None:
        """Weights used for temporal volatility interpolation."""
        return self._weights

    @cached_property
    def weights_cum(self) -> Series[float] | None:
        """Weight adjusted time to expiry (in calendar days) per date for temporal volatility
        interpolation."""
        if self.weights is None:
            return None
        else:
            return self.weights.cumsum()

    @cached_property
    def eval_posix(self) -> float:
        """The unix timestamp of the ``eval_date``."""
        return self.eval_date.replace(tzinfo=UTC).timestamp()

    @property
    def delivery_lag(self) -> int:
        """Business day settlement lag between ``expiry`` and ``delivery``."""
        return self._delivery_lag

    @property
    def eval_date(self) -> datetime:
        """Evaluation date of the *Surface*."""
        return self._eval_date

    @property
    def pair(self) -> str | None:
        """FX pair against which options priced by this *Surface* settle against."""
        return self._pair

    @property
    def calendar(self) -> CalTypes:
        """Settlement calendar used to determine ``delivery`` from ``expiry``."""
        return self._calendar


def _validate_delta_type(delta_type: str) -> str:
    if delta_type.lower() not in ["spot", "spot_pa", "forward", "forward_pa"]:
        raise ValueError("`delta_type` must be in {'spot', 'spot_pa', 'forward', 'forward_pa'}.")
    return delta_type.lower()


def _validate_weights(
    weights: Series[float] | NoInput,
    eval_date: datetime,
    expiries: list[datetime],
) -> Series[float] | None:
    if isinstance(weights, NoInput):
        return None

    w: Series[float] = Series(
        1.0, index=get_calendar("all").cal_date_range(eval_date, TERMINAL_DATE)
    )
    w.update(weights)
    # restrict to sorted and filtered for outliers
    w = w.sort_index()
    w = w[eval_date:]  # type: ignore[misc]

    node_points: list[datetime] = [eval_date] + expiries + [TERMINAL_DATE]
    for i in range(len(expiries) + 1):
        s, e = node_points[i] + timedelta(days=1), node_points[i + 1]
        days = (e - s).days + 1
        w[s:e] = (  # type: ignore[misc]
            w[s:e] * days / w[s:e].sum()  # type: ignore[misc]
        )  # scale the weights to allocate the correct time between nodes.
    w[eval_date] = 0.0
    return w


def _t_var_interp(
    expiries: list[datetime],
    expiries_posix: list[float],
    expiry: datetime,
    expiry_posix: float,
    expiry_index: int,
    eval_posix: float,
    weights_cum: Series[float] | None,
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
    expiries_posix: list[datetime]
        The list of datetimes associated with the expiries of the *Surface*.
    expiries_posix: list[float]
        The list of posix timestamps associated with the expiries of the *Surface*.
    expiry: datetime
        The target expiry to be interpolated.
    expiry_posix: float
        The pre-calculated posix timestamp for expiry.
    eval_posix: float
         The pre-calculated posix timestamp for eval date of the *Surface*
    weights_cum: Series[float] or NoInput
         The cumulative sum of the weights indexes by date
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
    return _t_var_interp_d_sabr_d_k_or_f(
        expiries,
        expiries_posix,
        expiry,
        expiry_posix,
        expiry_index,
        eval_posix,
        weights_cum,
        vol1,
        dvol1_dk=0.0,
        vol2=vol2,
        dvol2_dk=0.0,
        bounds_flag=bounds_flag,
        derivative=False,
    )[0]


def _t_var_interp_d_sabr_d_k_or_f(
    expiries: list[datetime],
    expiries_posix: list[float],
    expiry: datetime,
    expiry_posix: float,
    expiry_index: int,
    eval_posix: float,
    weights_cum: Series[float] | None,
    vol1: DualTypes,
    dvol1_dk: DualTypes,
    vol2: DualTypes,
    dvol2_dk: DualTypes,
    bounds_flag: int,
    derivative: bool,
) -> tuple[DualTypes, DualTypes | None]:
    if weights_cum is None:  # weights must also be NoInput
        if bounds_flag == 0:
            t1 = expiries_posix[expiry_index] - eval_posix
            t2 = expiries_posix[expiry_index + 1] - eval_posix
        elif bounds_flag == -1:
            # left side extrapolation
            t1 = 0.0
            t2 = expiries_posix[expiry_index] - eval_posix
        else:  # bounds_flag == 1:
            # right side extrapolation
            t1 = expiries_posix[expiry_index + 1] - eval_posix
            t2 = TERMINAL_DATE.replace(tzinfo=UTC).timestamp() - eval_posix

        t_hat = expiry_posix - eval_posix
        t = expiry_posix - eval_posix
    else:
        if bounds_flag == 0:
            t1 = weights_cum[expiries[expiry_index]]
            t2 = weights_cum[expiries[expiry_index + 1]]
        elif bounds_flag == -1:
            # left side extrapolation
            t1 = 0.0
            t2 = weights_cum[expiries[expiry_index]]
        else:  # bounds_flag == 1:
            # right side extrapolation
            t1 = weights_cum[expiries[expiry_index + 1]]
            t2 = weights_cum[TERMINAL_DATE]

        t_hat = weights_cum[expiry]  # number of vol weighted calendar days
        t = (expiry_posix - eval_posix) / 86400.0  # number of calendar days

    t_quotient = (t_hat - t1) / (t2 - t1)
    vol = ((t1 * vol1**2 + t_quotient * (t2 * vol2**2 - t1 * vol1**2)) / t) ** 0.5
    if derivative:
        dvol_dk = (
            (t2 / t) * t_quotient * vol2 * dvol2_dk + (t1 / t) * (1 - t_quotient) * vol1 * dvol1_dk
        ) / vol
    else:
        dvol_dk = None
    return vol, dvol_dk


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


def _moneyness_from_atm_delta_closed_form(vol: DualTypes, t_e: DualTypes) -> DualTypes:
    """
    Return `u` given premium unadjusted `delta`, of either 'spot' or 'forward' type.

    This function preserves AD.

    Book2: section "Strike and Volatility implied from ATM delta" (FXDeltaVolSMile)

    Parameters
    -----------
    vol: float, Dual, Dual2
        The volatility (in %, e.g. 10.0) to use in calculations.
    t_e: float,
        The time to expiry.

    Returns
    -------
    float, Dual or Dual2
    """
    return dual_exp((vol / 100.0) ** 2 * t_e / 2.0)


def _moneyness_from_delta_closed_form(
    delta: DualTypes,
    vol: DualTypes,
    t_e: DualTypes,
    z_w_0: DualTypes,
    phi: float,
) -> DualTypes:
    """
    Return `u` given premium unadjusted `delta`, of either 'spot' or 'forward' type.

    This function preserves AD.

    Book2: section "Strike and Volatility implied from a given option's delta" (FXDeltaVolSmile)

    Parameters
    -----------
    delta: float
        The input unadjusted delta for which to determine the moneyness for.
    vol: float, Dual, Dual2
        The volatility (in %, e.g. 10.0) to use in calculations.
    t_e: float, Dual, Dual2
        The time to expiry.
    z_w_0: float, Dual, Dual2
        The scalar for 'spot' or 'forward' delta types.
        If 'forward', this should equal 1.0.
        If 'spot', this should be :math:`w_deli / w_spot`.
    phi: float
        1.0 if is call, -1.0 if is put.

    Returns
    -------
    float, Dual or Dual2
    """
    vol_sqrt_t = vol * t_e**0.5 / 100.0
    _: DualTypes = dual_inv_norm_cdf(phi * delta / z_w_0)
    _ = dual_exp(vol_sqrt_t * (0.5 * vol_sqrt_t - phi * _))
    return _


def _d_sabr_d_k_or_f(
    k: Number,
    f: Number,
    t: Number,
    a: Number,
    b: float | Variable,
    p: Number,
    v: Number,
    derivative: int,
) -> tuple[Number, Number | None]:
    """
    Calculate the SABR function and its derivative with respect to k or f.

    For formula see for example I. Clark "Foreign Exchange Option
    Pricing" section 3.10.

    Rateslib uses the representation sigma(k) = X0 * X1 * X2, with these variables as defined in
    "Coding Interest Rates" chapter 13 to handle AD using dual numbers effectively.

    For no derivative and just the SABR function value use 0.
    For derivatives with respect to `k` use 1.
    For derivatives with respect to `f` use 2.

    See "Coding Interest Rates: FX Swaps and Bonds edition 2"
    """
    b_: Number = _to_number(b)
    X0, dX0 = _sabr_X0(k, f, t, a, b_, p, v, derivative)
    X1, dX1 = _sabr_X1(k, f, t, a, b_, p, v, derivative)
    X2, dX2 = _sabr_X2(k, f, t, a, b_, p, v, derivative)

    if derivative == 0:
        return X0 * X1 * X2, None
    else:
        return X0 * X1 * X2, dX0 * X1 * X2 + X0 * dX1 * X2 + X0 * X1 * dX2  # type: ignore[operator]


def _sabr_X0(
    k: Number,
    f: Number,
    t: Number,
    a: Number,
    b: Number,
    p: Number,
    v: Number,
    derivative: int = 0,
) -> tuple[Number, Number | None]:
    """
    X0 = a / ((fk)^((1-b)/2) * (1 + (1-b)^2/24 ln^2(f/k) + (1-b)^4/1920 ln^4(f/k) )

    If ``derivative`` is 1 also returns dX0/dk, derived using sympy auto code generator.
    If ``derivative`` is 2 also returns dX0/df, derived using sympy auto code generator.
    """
    return _rs_sabr_x0(k, f, t, a, b, p, v, derivative)


def _sabr_X1(
    k: Number,
    f: Number,
    t: Number,
    a: Number,
    b: Number,
    p: Number,
    v: Number,
    derivative: int = 0,
) -> tuple[Number, Number | None]:
    """
    X1 = 1 + t ( (1-b)^2 / 24 * a^2 / (fk)^(1-b) + 1/4 p b v a / (fk)^((1-b)/2) + (2-3p^2)/24 v^2 )

    If ``derivative`` also returns dX0/dk, calculated using sympy.
    """
    return _rs_sabr_x1(k, f, t, a, b, p, v, derivative)


def _sabr_X2(
    k: Number,
    f: Number,
    t: Number,
    a: Number,
    b: Number,
    p: Number,
    v: Number,
    derivative: int = 0,
) -> tuple[Number, Number | None]:
    """
    X2 = z / chi(z)

    z = v / a * (fk) ^((1-b)/2) * ln(f/k)
    chi(z) = ln( (sqrt(1-2pz+z^2) + z -p) / (1-p) )

    If ``derivative`` = 1 also returns dX2/dk, calculated using sympy.
    If ``derivative`` = 2 also returns dX2/df, calculated using sympy.
    """
    return _rs_sabr_x2(k, f, t, a, b, p, v, derivative)
