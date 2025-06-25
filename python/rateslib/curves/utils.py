from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

from pytz import UTC

from rateslib import defaults
from rateslib.curves.interpolation import INTERPOLATION, InterpolationFunction
from rateslib.default import NoInput
from rateslib.dual import dual_log, set_order_convert
from rateslib.dual.utils import _to_number
from rateslib.splines import PPSplineDual, PPSplineDual2, PPSplineF64

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        CalTypes,
        DualTypes,
        FXForwards,
        Variable,
        float_,
        str_,
    )  # pragma: no cover


class _CurveType(Enum):
    """
    Enumerable type to define the difference between discount factor based *Curves* and
    values based *Curves*.
    """

    dfs = 0
    values = 1


@dataclass(frozen=True)
class _CurveMeta:
    """
    An immutable container of meta data associated with a
    :class:`~rateslib.curves.Curve` used to make calculations.
    """

    _calendar: CalTypes
    _convention: str
    _modifier: str
    _index_base: float_ | Variable
    _index_lag: int
    _collateral: str | None
    _credit_discretization: int
    _credit_recovery_rate: float | Variable

    @property
    def calendar(self) -> CalTypes:
        """Settlement calendar used to determine fixing dates and tenor end dates."""
        return self._calendar

    @property
    def convention(self) -> str:
        """Day count convention for determining rates and interpolation."""
        return self._convention

    @property
    def modifier(self) -> str:
        """Modification rule for adjusting non-business tenor end dates."""
        return self._modifier

    @property
    def index_base(self) -> Variable | float_:
        """The index value associated with the initial node date of the *Curve*."""
        return self._index_base

    @property
    def index_lag(self) -> int:
        """The number of months by which curve nodes are lagged to determine index values."""
        return self._index_lag

    @property
    def collateral(self) -> str | None:
        """The currency(ies) identified as being the collateral choice for DFs associated with
        the *Curve*."""
        return self._collateral

    @property
    def credit_discretization(self) -> int:
        """A parameter for numerically solving the integral for a *Credit Protection Period*."""
        return self._credit_discretization

    @property
    def credit_recovery_rate(self) -> float | Variable:
        """The recovery rate applied to *Credit Protection Period* cashflows."""
        return self._credit_recovery_rate

    def to_json(self) -> str:
        """
        Serialize this object to JSON format.

        The object can be deserialized using the :meth:`~rateslib.serialization.from_json` method.

        Returns
        -------
        str
        """
        from rateslib.serialization.utils import _obj_to_json

        obj = dict(
            PyNative=dict(
                _CurveMeta=dict(
                    calendar=self.calendar.to_json(),
                    convention=self.convention,
                    modifier=self.modifier,
                    index_base=_obj_to_json(self.index_base),
                    index_lag=self.index_lag,
                    collateral=self.collateral,
                    credit_discretization=self.credit_discretization,
                    credit_recovery_rate=_obj_to_json(self.credit_recovery_rate),
                )
            )
        )
        return json.dumps(obj)

    @classmethod
    def _from_json(cls, loaded_json: dict[str, Any]) -> _CurveMeta:
        from rateslib.serialization import from_json

        return _CurveMeta(
            _convention=loaded_json["convention"],
            _modifier=loaded_json["modifier"],
            _index_lag=loaded_json["index_lag"],
            _collateral=loaded_json["collateral"],
            _index_base=from_json(loaded_json["index_base"]),
            _calendar=from_json(loaded_json["calendar"]),
            _credit_discretization=loaded_json["credit_discretization"],
            _credit_recovery_rate=from_json(loaded_json["credit_recovery_rate"]),
        )


class _CurveSpline:
    """
    A container for data relating to interpolating the `nodes` of
    a *Curve* using a cubic PPSpline.
    """

    _t: list[datetime]
    _spline: PPSplineF64 | PPSplineDual | PPSplineDual2 | None
    _endpoints: tuple[str, str]

    def __init__(self, t: list[datetime], endpoints: tuple[str, str]) -> None:
        self._t = t
        self._endpoints = endpoints
        self._spline = None  # will be set in later in csolve
        if len(self._t) < 10 and "not_a_knot" in self.endpoints:
            raise ValueError(
                "`endpoints` cannot be 'not_a_knot' with only 1 interior breakpoint",
            )

    @property
    def t(self) -> list[datetime]:
        """The knot sequence of the PPSpline."""
        return self._t

    @cached_property
    def t_posix(self) -> list[float]:
        """The knot sequence of the PPSpline converted to float unix timestamps."""
        return [_.replace(tzinfo=UTC).timestamp() for _ in self.t]

    @property
    def spline(self) -> PPSplineF64 | PPSplineDual | PPSplineDual2 | None:
        """An instance of :class:`~rateslib.splines.PPSplineF64`,
        :class:`~rateslib.splines.PPSplineDual` or :class:`~rateslib.splines.PPSplineDual2`.
        """
        return self._spline

    @property
    def endpoints(self) -> tuple[str, str]:
        """The endpoints method used to determine the spline coefficients."""
        return self._endpoints

    # All calling methods should clear the cache and/or set new state after `_csolve`
    def _csolve(self, curve_type: _CurveType, nodes: _CurveNodes, ad: int) -> None:
        t_posix = self.t_posix.copy()
        tau_posix = [k.replace(tzinfo=UTC).timestamp() for k in nodes.keys if k >= self.t[0]]
        if curve_type == _CurveType.dfs:
            # then use log
            y = [dual_log(v) for k, v in nodes.nodes.items() if k >= self.t[0]]
        else:
            # use values directly
            y = [_to_number(v) for k, v in nodes.nodes.items() if k >= self.t[0]]

        # Left side constraint
        if self.endpoints[0].lower() == "natural":
            tau_posix.insert(0, t_posix[0])
            y.insert(0, set_order_convert(0.0, ad, None))
            left_n = 2
        elif self.endpoints[0].lower() == "not_a_knot":
            t_posix.pop(4)
            left_n = 0
        else:
            raise NotImplementedError(
                f"Endpoint method '{self.endpoints[0]}' not implemented.",
            )

        # Right side constraint
        if self.endpoints[1].lower() == "natural":
            tau_posix.append(self.t_posix[-1])
            y.append(set_order_convert(0, ad, None))
            right_n = 2
        elif self.endpoints[1].lower() == "not_a_knot":
            t_posix.pop(-5)
            right_n = 0
        else:
            raise NotImplementedError(
                f"Endpoint method '{self.endpoints[0]}' not implemented.",
            )

        # Get the Spline class by data types
        if ad == 0:
            self._spline = PPSplineF64(4, t_posix, None)
        elif ad == 1:
            self._spline = PPSplineDual(4, t_posix, None)
        else:
            self._spline = PPSplineDual2(4, t_posix, None)

        self._spline.csolve(tau_posix, y, left_n, right_n, False)  # type: ignore[arg-type]

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
                _CurveSpline=dict(
                    t=[_.strftime("%Y-%m-%d") for _ in self.t],
                    endpoints=self.endpoints,
                )
            )
        )
        return json.dumps(obj)

    @classmethod
    def _from_json(cls, loaded_json: dict[str, Any]) -> _CurveSpline:
        return _CurveSpline(
            t=[datetime.strptime(_, "%Y-%m-%d") for _ in loaded_json["t"]],
            endpoints=tuple(loaded_json["endpoints"]),
        )

    def __eq__(self, other: Any) -> bool:
        """CurveSplines are considered equal if their knot sequence and endpoints are equivalent.
        For the same nodes this will resolve to give the same spline coefficients.
        """
        if not isinstance(other, _CurveSpline):
            return False
        else:
            return all(iter([self.t == other.t, self.endpoints == other.endpoints]))


class _CurveInterpolator:
    """
    A container for data relating to interpolating the `nodes` of a :class:`~rateslib.curves.Curve`.
    """

    _local_name: str
    _local_func: InterpolationFunction
    _convention: str
    _spline: _CurveSpline | None

    def __init__(
        self,
        local: str_ | InterpolationFunction,
        t: list[datetime] | NoInput,
        endpoints: tuple[str, str],
        node_dates: list[datetime],
        convention: str,
        curve_type: _CurveType,
    ) -> None:
        if not isinstance(t, NoInput) and local == "spline":
            raise ValueError(
                "When defining 'spline' interpolation, the argument `t` will be "
                "automatically generated.\n"
                f"It should not be specified directly. Got: {t}"
            )

        self._convention = convention
        if isinstance(local, NoInput):
            local = defaults.interpolation[curve_type.name]

        if isinstance(local, str):
            self._local_name = local.lower()
            if self.local_name == "spline":
                # then refactor t
                t = (
                    [node_dates[0], node_dates[0], node_dates[0]]
                    + node_dates
                    + [node_dates[-1], node_dates[-1], node_dates[-1]]
                )

            if self._local_name + "_" + convention in INTERPOLATION:
                self._local_func = INTERPOLATION[self.local_name + "_" + convention]
            else:
                try:
                    self._local_func = INTERPOLATION[self.local_name]
                except KeyError:
                    raise ValueError(
                        f"Curve interpolation: '{self.local_name}' not available.\n"
                        f"Consult the documentation for available methods."
                    )
        else:
            self._local_name = "user_defined_callable"
            self._local_func = local

        if isinstance(t, NoInput):
            self._spline = None
        else:
            self._spline = _CurveSpline(t, endpoints)

    @property
    def local(self) -> str | InterpolationFunction:
        """The local interpolation name or function, if user defined."""
        if self.local_name == "user_defined_callable":
            return self.local_func
        return self.local_name

    @property
    def local_name(self) -> str:
        """The str name of the local interpolation function."""
        return self._local_name

    @property
    def local_func(self) -> InterpolationFunction:
        """The callable used for local interpolation"""
        return self._local_func

    @property
    def spline(self) -> _CurveSpline | None:
        """The :class:`~rateslib.curves.utils._CurveSpline` used for PPSpline interpolation."""
        return self._spline

    @property
    def convention(self) -> str:
        """The day count convention used to adjust interpolation functions."""
        return self._convention

    # All calling methods should clear the cache and/or set new state after `_csolve`
    def _csolve(self, curve_type: _CurveType, nodes: _CurveNodes, ad: int) -> None:
        if self.spline is None:
            return None
        self.spline._csolve(curve_type, nodes, ad)

    def __eq__(self, other: Any) -> bool:
        if (
            not isinstance(other, _CurveInterpolator)
            or self.local_name == "user_defined_callable"
            and self.local_func != other.local_func
        ):
            return False

        return all(iter([self.local_name == other.local_name, self.spline == other.spline]))

    def to_json(self) -> str:
        """
        Serialize this object to JSON format.

        The object can be deserialized using the :meth:`~rateslib.serialization.from_json` method.

        Returns
        -------
        str
        """
        from rateslib.serialization.utils import _obj_to_json

        obj = dict(
            PyNative=dict(
                _CurveInterpolator=dict(
                    local=self.local_name,
                    spline=_obj_to_json(self.spline),
                    convention=self.convention,
                )
            )
        )
        return json.dumps(obj)

    @classmethod
    def _from_json(cls, loaded_json: dict[str, Any]) -> _CurveInterpolator:
        from rateslib.serialization import from_json

        spl = from_json(loaded_json["spline"])

        if loaded_json["local"] == "spline":
            t = NoInput(0)
            node_dates = spl.t[3:-3]
        else:
            t = NoInput(0) if spl is None else spl.t
            node_dates = NoInput(0)

        return _CurveInterpolator(
            local=loaded_json["local"],
            t=t,
            endpoints=NoInput(0) if spl is None else spl.endpoints,  # type: ignore[arg-type]
            node_dates=node_dates,
            convention=loaded_json["convention"],
            curve_type=NoInput(0),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class _ProxyCurveInterpolator:
    """
    A container for data relating to deriving the DFs of a :class:`~rateslib.curves.ProxyCurve`
    from other :class:`~rateslib.curves.Curve` objects and :class:`~rateslib.fx.FXForwards`.
    """

    _fx_forwards: FXForwards
    _cash: str
    _collateral: str

    @property
    def fx_forwards(self) -> FXForwards:
        """The :class:`~rateslib.fx.FXForwards` object containing :class:`~rateslib.fx.FXRates`
        and :class:`~rateslib.curves.Curve` objects."""
        return self._fx_forwards

    @property
    def cash(self) -> str:
        """The currency of the cashflows."""
        return self._cash

    @property
    def collateral(self) -> str:
        """The currency of the collateral assuming PAI."""
        return self._collateral

    @property
    def pair(self) -> str:
        """A pair of currencies representing the cashflow and collateral."""
        return self.cash + self.collateral

    @property
    def cash_index(self) -> int:
        """The index of the cash currency in the :class:`~rateslib.fx.FXForwards` object."""
        return self.fx_forwards.currencies[self.cash]

    @property
    def collateral_index(self) -> int:
        """The index of the collateral currency in the :class:`~rateslib.fx.FXForwards` object."""
        return self.fx_forwards.currencies[self.collateral]

    @property
    def cash_pair(self) -> str:
        """A pair constructed from the cash currency"""
        return self.cash + self.cash

    @property
    def collateral_pair(self) -> str:
        """A pair constructed from the collateral currency"""
        return self.collateral + self.collateral


@dataclass(frozen=True)
class _CurveNodes:
    """
    An immutable container for the pricing parameters of a :class:`~rateslib.curves.Curve`.
    """

    _nodes: dict[datetime, DualTypes]

    def __post_init__(self) -> None:
        for idx in range(1, self.n):
            if self.keys[idx - 1] >= self.keys[idx]:
                raise ValueError(
                    "Curve node dates are not sorted or contain duplicates.\n"
                    "To sort directly use: `dict(sorted(nodes.items()))`",
                )

    @property
    def nodes(self) -> dict[datetime, DualTypes]:
        """The initial nodes dict passed for construction of this class."""
        return self._nodes

    @cached_property
    def keys(self) -> list[datetime]:
        """A list of datetime keys in ``nodes``."""
        return list(self._nodes.keys())

    @cached_property
    def values(self) -> list[DualTypes]:
        """A list of values in ``nodes``."""
        return list(self._nodes.values())

    @property
    def n(self) -> int:
        """Number of parameters contained in ``nodes``."""
        return len(self.keys)

    @cached_property
    def posix_keys(self) -> list[float]:
        """A list of the ``keys`` converted to unix timestamps."""
        return [_.replace(tzinfo=UTC).timestamp() for _ in self.keys]

    @property
    def initial(self) -> datetime:
        """The first node key associated with the *Curve* nodes."""
        return self.keys[0]

    @property
    def final(self) -> datetime:
        """The last node key associated with the *Curve* nodes."""
        return self.keys[-1]

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
                _CurveNodes=dict(
                    _nodes={dt.strftime("%Y-%m-%d"): v.real for dt, v in self._nodes.items()},
                )
            )
        )
        return json.dumps(obj)

    @classmethod
    def _from_json(cls, loaded_json: dict[str, Any]) -> _CurveNodes:
        return _CurveNodes(
            _nodes={datetime.strptime(d, "%Y-%m-%d"): v for d, v in loaded_json["_nodes"].items()}
        )
