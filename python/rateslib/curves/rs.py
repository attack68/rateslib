from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from rateslib import defaults
from rateslib.calendars import _get_modifier, get_calendar  # type: ignore[attr-defined]
from rateslib.calendars.dcfs import _get_convention
from rateslib.default import NoInput, _drb, _make_py_json
from rateslib.dual.utils import _get_adorder
from rateslib.rs import (
    ADOrder,
    FlatBackwardInterpolator,
    FlatForwardInterpolator,
    LinearInterpolator,
    LinearZeroRateInterpolator,
    LogLinearInterpolator,
    NullInterpolator,
    _get_convention_str,
    _get_modifier_str,
)
from rateslib.rs import Curve as CurveObj  # noqa: F401

if TYPE_CHECKING:
    from rateslib.typing import CalInput, CurveInterpolator, DualTypes, Number


class CurveRs:
    def __init__(
        self,
        nodes: dict[datetime, Number],
        *,
        interpolation: str
        | Callable[[datetime, dict[datetime, DualTypes]], DualTypes]
        | NoInput = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        convention: str | NoInput = NoInput(0),
        modifier: str | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        ad: int = 0,
        index_base: float | NoInput = NoInput(0),
    ):
        self._py_interpolator: Callable[[datetime, dict[datetime, DualTypes]], DualTypes] | None = (
            interpolation if callable(interpolation) else None
        )

        self.obj = CurveObj(
            nodes=nodes,
            interpolator=self._validate_interpolator(interpolation),
            ad=_get_adorder(ad),
            id=_drb(uuid4().hex[:5] + "_", id),  # 1 in a million clash
            convention=_get_convention(_drb(defaults.convention, convention)),
            modifier=_get_modifier(_drb(defaults.modifier, modifier), True),
            calendar=get_calendar(calendar, named=True),
            index_base=_drb(None, index_base),
        )

    @property
    def id(self) -> str:
        return self.obj.id

    @property
    def convention(self) -> str:
        return _get_convention_str(self.obj.convention)

    @property
    def modifier(self) -> str:
        return _get_modifier_str(self.obj.modifier)

    @property
    def interpolation(self) -> str:
        return self.obj.interpolation

    @property
    def nodes(self) -> dict[datetime, Number]:
        return self.obj.nodes

    @property
    def ad(self) -> int:
        _ = self.obj.ad
        if _ == ADOrder.One:
            return 1
        elif _ == ADOrder.Two:
            return 2
        return 0

    def _set_ad_order(self, ad: int) -> None:
        self.obj.set_ad_order(_get_adorder(ad))

    @staticmethod
    def _validate_interpolator(
        interpolation: str | Callable[[datetime, dict[datetime, DualTypes]], DualTypes] | NoInput,
    ) -> CurveInterpolator:
        if interpolation is NoInput.blank:
            return _get_interpolator(defaults.interpolation["Curve"])
        elif isinstance(interpolation, str):
            return _get_interpolator(interpolation)
        else:
            return NullInterpolator()

    def to_json(self) -> str:
        return _make_py_json(self.obj.to_json(), "CurveRs")

    @classmethod
    def __init_from_obj__(cls, obj: CurveObj) -> CurveRs:
        new = cls(
            nodes={datetime(2000, 1, 1): 1.0},
            interpolation="linear",
            id="_",
            ad=0,
            index_base=NoInput(0),
        )
        new.obj = obj
        return new

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CurveRs):
            return False
        return self.obj.__eq__(other.obj)

    def __getitem__(self, value: datetime) -> Number:
        return self.obj[value]


def _get_interpolator(name: str) -> CurveInterpolator:
    name_ = name.lower()
    if name_ == "log_linear":
        return LogLinearInterpolator()
    elif name_ == "linear":
        return LinearInterpolator()
    elif name_ == "linear_zero_rate":
        return LinearZeroRateInterpolator()
    elif name_ == "flat_forward":
        return FlatForwardInterpolator()
    elif name_ == "flat_backward":
        return FlatBackwardInterpolator()
    else:
        raise ValueError("Interpolator `name` is invalid.")
