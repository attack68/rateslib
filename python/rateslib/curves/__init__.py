from typing import TypeAlias

from rateslib.curves.curves import (
    CompositeCurve,
    Curve,
    IndexCurve,
    LineCurve,
    MultiCsaCurve,
    ProxyCurve,
    average_rate,
    index_left,
    interpolate,
)
from rateslib.default import NoInput

Curves: TypeAlias = "list[str | Curve | dict[str, Curve | str] | NoInput] | Curve | str | dict[str, Curve | str] | NoInput"  # noqa: E501
CurveInput: TypeAlias = "Curve | NoInput | str | dict[str, Curve | str]"

CurveOption: TypeAlias = "Curve | dict[str, Curve] | NoInput"
CurvesList: TypeAlias = "tuple[CurveOption, CurveOption, CurveOption, CurveOption]"

__all__ = (
    "CompositeCurve",
    "Curve",
    "IndexCurve",
    "LineCurve",
    "MultiCsaCurve",
    "ProxyCurve",
    "average_rate",
    "index_left",
    "interpolate",
)
