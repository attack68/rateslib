from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from rateslib import defaults
from rateslib.calendars import CalInput, _get_modifier, get_calendar
from rateslib.calendars.dcfs import _get_convention
from rateslib.default import NoInput, _drb
from rateslib.dual import ADOrder, _get_adorder
from rateslib.rs import Curve as CurveObj  # noqa: F401
from rateslib.rs import (
    FlatBackwardInterpolator,
    FlatForwardInterpolator,
    LinearInterpolator,
    LinearZeroRateInterpolator,
    LogLinearInterpolator,
    NullInterpolator,
    _get_convention_str,
    _get_modifier_str,
)


class CurveRs:
    def __init__(
        self,
        nodes: dict,
        *,
        interpolation: str | callable | NoInput = NoInput(0),
        id: str | NoInput = NoInput(0),
        convention: str | NoInput = NoInput(0),
        modifier: str | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        ad: int = 0,
        index_base: float | NoInput = NoInput(0),
    ):
        self._py_interpolator = interpolation if callable(interpolation) else None

        self.obj = CurveObj(
            nodes=nodes,
            interpolator=self._validate_interpolator(interpolation),
            ad=_get_adorder(ad),
            id=_drb(uuid4().hex[:5] + "_", id),  # 1 in a million clash
            convention=_get_convention(_drb(defaults.convention, convention)),
            modifier=_get_modifier(_drb(defaults.modifier, modifier), True),
            calendar=get_calendar(calendar, kind=False, named=True),
            index_base=_drb(None, index_base),
        )

    @property
    def id(self):
        return self.obj.id

    @property
    def convention(self):
        return _get_convention_str(self.obj.convention)

    @property
    def modifier(self):
        return _get_modifier_str(self.obj.modifier)

    @property
    def interpolation(self):
        return self.obj.interpolation

    @property
    def nodes(self):
        return self.obj.nodes

    @property
    def ad(self):
        _ = self.obj.ad
        if _ == ADOrder.One:
            return 1
        elif _ == ADOrder.Two:
            return 2
        return 0

    def _set_ad_order(self, ad: int):
        self.obj.set_ad_order(_get_adorder(ad))
        return None

    @staticmethod
    def _validate_interpolator(interpolation: str | callable | NoInput):
        if interpolation is NoInput.blank:
            return _get_interpolator(defaults.interpolation["Curve"])
        elif isinstance(interpolation, str):
            return _get_interpolator(interpolation)
        else:
            return NullInterpolator()

    def to_json(self):
        return '{"Py":' + self.obj.to_json() + "}"

    @classmethod
    def __init_from_obj__(cls, obj):
        new = cls(
            nodes={datetime(2000, 1, 1): 1.0},
            interpolation="linear",
            id="_",
            ad=0,
            index_base=NoInput(0),
        )
        new.obj = obj
        return new

    def __eq__(self, other):
        if not isinstance(other, CurveRs):
            return False
        return self.obj.__eq__(other.obj)

    def __getitem__(self, value: datetime):
        return self.obj[value]


def _get_interpolator(name: str):
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
