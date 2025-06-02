from __future__ import annotations

from json import dumps, loads

# globals namespace
from typing import TYPE_CHECKING, Any

from rateslib.curves import Curve, LineCurve
from rateslib.curves.rs import CurveRs
from rateslib.curves.utils import _CurveInterpolator, _CurveMeta, _CurveNodes, _CurveSpline
from rateslib.default import NoInput
from rateslib.dual import Variable
from rateslib.fx import FXRates
from rateslib.rs import from_json as from_json_rs

if TYPE_CHECKING:
    pass  # pragma: no cover

NAMES_RsPy: dict[str, Any] = {  # this is a mapping of native Rust obj names to Py obj names
    "FXRates": FXRates,
    "Curve": CurveRs,
}


NAMES_Py: dict[str, Any] = {  # a mapping of native Python classes with a _from_json() method
    "_CurveMeta": _CurveMeta,
    "_CurveSpline": _CurveSpline,
    "_CurveInterpolator": _CurveInterpolator,
    "_CurveNodes": _CurveNodes,
    "Curve": Curve,
    "LineCurve": LineCurve,
    "Variable": Variable,
}


ENUMS_Py: dict[str, Any] = {
    "NoInput": NoInput,
}


def _pynative_from_json(name: str, json: dict[str, Any] | str) -> Any:
    if name in NAMES_Py:
        return NAMES_Py[name]._from_json(json)
    else:
        # is an Enum
        return ENUMS_Py[name](json)


def from_json(json: str) -> Any:
    """
    Create an object from JSON string.

    Parameters
    ----------
    json: str
        JSON string in appropriate format to construct the class.

    Returns
    -------
    Object
    """
    obj = loads(json)
    if isinstance(obj, dict):
        if "PyWrapped" in obj:
            # then object is a Rust struct wrapped by a Python class.
            # determine the Python class name and reconstruct the Python class from the Rust struct.
            class_name = next(iter(obj["PyWrapped"].keys()))
            restructured_json = dumps(obj["PyWrapped"])
            # objs = globals()
            class_obj = NAMES_RsPy[class_name]
            return class_obj.__init_from_obj__(obj=from_json_rs(restructured_json))
        elif "PyNative" in obj:
            # PyNative are objects that are constructed only in Python but do not serialize directly
            # and so are tagged with a serialization flag.
            class_name = next(iter(obj["PyNative"].keys()))
            return _pynative_from_json(name=class_name, json=obj["PyNative"][class_name])
        else:
            # the dict may have been a native Rust object, try loading directly
            # this will raise if all combination exhausted
            return from_json_rs(json)
    else:
        # object is a native Python element
        return obj
