from __future__ import annotations

from enum import Enum
from typing import NamedTuple, TYPE_CHECKING
import json
from pytz import UTC
from datetime import datetime

from rateslib import defaults
from rateslib.curves.interpolation import InterpolationFunction, INTERPOLATION
from rateslib.default import NoInput
from rateslib.dual import dual_log, set_order_convert
from rateslib.splines import PPSplineDual, PPSplineDual2, PPSplineF64

if TYPE_CHECKING:
    from rateslib.typing import DualTypes_, CalTypes, Any, str_, DualTypes


class _CurveType(Enum):
    """
    Enumerable type to define the difference between discount factor based Curves and
    values base Curves.
    """

    dfs = 0
    values = 1


class _CurveMeta(NamedTuple):
    calendar: CalTypes
    convention: str
    modifier: str
    index_base: DualTypes_
    index_lag: int
    collateral: str | None

    def to_json(self) -> str:
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
                )
            )
        )
        return json.dumps(obj)

    @classmethod
    def _from_json(cls, loaded_json: dict[str, Any]) -> _CurveMeta:
        from rateslib.serialization import from_json

        return _CurveMeta(
            convention=loaded_json["convention"],
            modifier=loaded_json["modifier"],
            index_lag=loaded_json["index_lag"],
            collateral=loaded_json["collateral"],
            index_base=from_json(loaded_json["index_base"]),
            calendar=from_json(loaded_json["calendar"]),
        )


class _CurveSpline:
    t: list[datetime]
    t_posix: list[float]
    spline: PPSplineF64 | PPSplineDual | PPSplineDual2 | None
    endpoints: tuple[str, str]

    def __init__(self, t: list[datetime], endpoints: tuple[str, str]) -> None:
        self.t = t
        self.t_posix = [_.replace(tzinfo=UTC).timestamp() for _ in self.t]
        self.endpoints = endpoints
        self.spline = None  # will be set in later in csolve
        if len(self.t) < 10 and "not_a_knot" in self.endpoints:
            raise ValueError(
                "`endpoints` cannot be 'not_a_knot' with only 1 interior breakpoint",
            )

    # All calling methods should clear the cache and/or set new state after `_csolve`
    def _csolve(self, curve_type: _CurveType, nodes: dict[datetime, DualTypes], ad: int) -> None:
        t_posix = self.t_posix.copy()
        tau_posix = [k.replace(tzinfo=UTC).timestamp() for k in nodes if k >= self.t[0]]
        if curve_type == _CurveType.dfs:
            # then use log
            y = [dual_log(v) for k, v in nodes.items() if k >= self.t[0]]
        else:
            # use values directly
            y = [v for k, v in nodes.items() if k >= self.t[0]]

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
            self.spline = PPSplineF64(4, t_posix, None)
        elif ad == 1:
            self.spline = PPSplineDual(4, t_posix, None)
        else:
            self.spline = PPSplineDual2(4, t_posix, None)

        self.spline.csolve(tau_posix, y, left_n, right_n, False)  # type: ignore[attr-defined]

    def to_json(self) -> str:
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
            return all(iter([
                self.t == other.t,
                self.endpoints == other.endpoints
            ]))


class _CurveInterpolator:

    local_name: str
    local_func: InterpolationFunction
    convention: str
    spline: _CurveSpline | None

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

        self.convention = convention
        if isinstance(local, NoInput):
           local = defaults.interpolation[curve_type.name]

        if isinstance(local, str):
            self.local_name = local.lower()
            if self.local_name == "spline":
                # then refactor t
                t = (
                    [node_dates[0], node_dates[0], node_dates[0]]
                    + node_dates
                    + [node_dates[-1], node_dates[-1], node_dates[-1]]
                )

            if self.local_name + "_" + convention in INTERPOLATION:
                self.local_func = INTERPOLATION[self.local_name + "_" + convention]
            else:
                try:
                    self.local_func = INTERPOLATION[self.local_name]
                except KeyError:
                    raise ValueError(
                        f"Curve interpolation: '{self.local_name}' not available.\n"
                        f"Consult the documentation for available methods."
                    )
        else:
            self.local_name = "user_defined_callable"
            self.local_func = local

        if isinstance(t, NoInput):
            self.spline = None
        else:
            self.spline = _CurveSpline(t, endpoints)

    @property
    def local(self) -> str | InterpolationFunction:
        if self.local_name == "user_defined_callable":
            return self.local_func
        return self.local_name

    # All calling methods should clear the cache and/or set new state after `_csolve`
    def _csolve(self, curve_type: _CurveType, nodes: dict[datetime, DualTypes], ad: int) -> None:
        if self.spline is None:
            return None
        self.spline._csolve(curve_type, nodes, ad)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _CurveInterpolator):
            return False
        elif self.local_name == "user_defined_callable" and self.local_func != other.local_func:
            return False

        return all(iter([
            self.local_name == other.local_name,
            self.spline == other.spline
        ]))

    def to_json(self) -> str:
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
            t=NoInput(0)
            node_dates=spl.t[3:-3]
        else:
            t=NoInput(0) if spl is None else spl.t
            node_dates=NoInput(0)

        return _CurveInterpolator(
            local=loaded_json["local"],
            t=t,
            endpoints=NoInput(0) if spl is None else spl.endpoints,
            node_dates=node_dates,
            convention=loaded_json["convention"],
            curve_type=NoInput(0),
        )
