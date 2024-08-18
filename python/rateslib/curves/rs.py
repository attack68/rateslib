from __future__ import annotations

from datetime import datetime as dt
from uuid import uuid4

from rateslib import defaults
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
)


class CurveRs:
    def __init__(
        self,
        nodes: dict,
        *,
        interpolation: str | callable | NoInput = NoInput(0),
        id: str | NoInput = NoInput(0),
        ad: int = 0,
        index_base: float | NoInput = NoInput(0),
    ):
        self._py_interpolator = interpolation if callable(interpolation) else None

        self.obj = CurveObj(
            nodes=nodes,
            interpolator=self._validate_interpolator(interpolation),
            ad=_get_adorder(ad),
            id=_drb(uuid4().hex[:5] + "_", id),  # 1 in a million clash
            index_base=_drb(None, index_base),
        )

    @property
    def id(self):
        return self.obj.id

    @property
    def interpolation(self):
        return self.obj.interpolation

    @property
    def nodes(self):
        return self.obj.nodes

    @property
    def ad(self):
        _ = self.obj.ad
        if _ is ADOrder.One:
            return 1
        elif _ is ADOrder.Two:
            return 2
        return 0

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
            nodes={dt(2000, 1, 1): 1.0},
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
