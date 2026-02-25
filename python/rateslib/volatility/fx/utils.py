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

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import TYPE_CHECKING, TypeAlias

from pandas import Series

from rateslib.dual import (
    Dual,
    Dual2,
    Variable,
    dual_exp,
    dual_inv_norm_cdf,
    set_order_convert,
)
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import (
    NoInput,
)
from rateslib.enums.parameters import FXDeltaMethod
from rateslib.splines import PPSplineDual, PPSplineDual2, PPSplineF64

if TYPE_CHECKING:
    from rateslib.local_types import Any, CalTypes

DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"  # if not defined causes _WithCache failure

UTC = timezone.utc

TERMINAL_DATE = datetime(2100, 1, 1)


@dataclass
class _FXSmileMeta:
    _eval_date: datetime
    _expiry: datetime
    _plot_x_axis: str
    _delta_type: FXDeltaMethod
    _pair: str | None
    _calendar: CalTypes
    _delivery: datetime
    _delivery_lag: int

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
    def delta_type(self) -> FXDeltaMethod:
        """The delta type of the delta indexes associated with the ``nodes`` of the *Smile*."""
        return self._delta_type

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
        return (self._expiry - self._eval_date).days / 365.0

    @cached_property
    def t_expiry_sqrt(self) -> float:
        """Square root of ``t_expiry``."""
        ret: float = self.t_expiry**0.5
        return ret

    @property
    def delivery(self) -> datetime:
        """Delivery date of the forward FX rate applicable to options priced by this *Smile*"""
        return self._delivery

    @property
    def delivery_lag(self) -> int:
        """Business day settlement lag between ``expiry`` and ``delivery``."""
        return self._delivery_lag


class _FXDeltaVolSmileNodes:
    """
    A container for data relating to interpolating the `nodes` of a
    :class:`~rateslib.fx_volatility.FXDeltaVolSmile`.
    """

    _nodes: dict[float, DualTypes]
    _meta: _FXSmileMeta
    _spline: _FXDeltaVolSpline

    def __init__(self, nodes: dict[float, DualTypes], meta: _FXSmileMeta) -> None:
        self._nodes = nodes
        self._meta = meta

        if self.meta.delta_type in [
            FXDeltaMethod.SpotPremiumAdjusted,
            FXDeltaMethod.ForwardPremiumAdjusted,
        ]:
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
        if self.meta.delta_type in [
            FXDeltaMethod.SpotPremiumAdjusted,
            FXDeltaMethod.ForwardPremiumAdjusted,
        ]:
            # upper_bound      = exp(vol * t_expiry_sqrt * (3.75 - 0.5 * vol * t_expiry_sqrt)
            # plot_upper_bound = exp(vol * t_expiry_sqrt * (3.25 - 0.5 * vol * t_expiry_sqrt)
            return (
                self.spline.t[-1] - _dual_float(self.values[-1]) * self.meta.t_expiry_sqrt / 200.0
            )
        else:
            return 1.0

    @property
    def meta(self) -> _FXSmileMeta:
        """An instance of :class:`~rateslib.volatility.fx._FXSmileMeta`."""
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
        """The number of pricing parameters in ``nodes``."""
        return len(self.keys)

    @property
    def spline(self) -> _FXDeltaVolSpline:
        """An instance of :class:`~rateslib.volatility.fx._FXDeltaVolSpline`."""
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
        if nodes.meta.delta_type in [
            FXDeltaMethod.SpotPremiumAdjusted,
            FXDeltaMethod.ForwardPremiumAdjusted,
        ]:
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
    _delta_type: FXDeltaMethod
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
    def delta_type(self) -> FXDeltaMethod:
        """The delta type of the delta indexes associated with the ``nodes`` of each
        cross-sectional *Smile*."""
        return self._delta_type

    @property
    def plot_x_axis(self) -> str:
        """The default ``x_axis`` parameter passed to
        :meth:`~rateslib.fx_volatility._BaseSmile.plot`"""
        return self._plot_x_axis


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
    _expiries: list[datetime]

    def __post_init__(self) -> None:
        for idx in range(1, len(self.expiries)):
            if self.expiries[idx - 1] >= self.expiries[idx]:
                raise ValueError("Surface `expiries` are not sorted or contain duplicates.\n")

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
    def expiries(self) -> list[datetime]:
        """A list of the expiries of each cross-sectional
        :class:`~rateslib.fx_volatility.FXSabrSmile`."""
        return self._expiries

    @cached_property
    def expiries_posix(self) -> list[float]:
        """A list of the unix timestamps of each date in ``expiries``."""
        return [_.replace(tzinfo=UTC).timestamp() for _ in self.expiries]

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


def _delta_type_constants(
    delta_type: FXDeltaMethod, w: DualTypes | NoInput, u: DualTypes | NoInput
) -> tuple[float, DualTypes, DualTypes]:
    """
    Get the values: (eta, z_w, z_u) for the type of expressed delta

    w: should be input as w_deli / w_spot
    u: should be input as K / f_d
    """
    if delta_type == FXDeltaMethod.Forward:
        return 0.5, 1.0, 1.0
    elif delta_type == FXDeltaMethod.Spot:
        return 0.5, w, 1.0  # type: ignore[return-value]
    elif delta_type == FXDeltaMethod.ForwardPremiumAdjusted:
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
